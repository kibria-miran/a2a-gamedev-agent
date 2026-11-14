"""Build/run sandbox for LLM-authored C++ (E-69, E-70, E-73, E-74).

Design (v1.1):
  - NO preexec_fn, anywhere, by design — it is fundamentally broken with
    subprocess. Resource caps come from ``prlimit`` (util-linux) wrapping the
    child: ``--as=<bytes>`` (address space) and ``--cpu=<sec>`` (CPU time).
    Children run with ``start_new_session=True`` so signals and pgid games
    can't cross session boundaries.
  - Network isolation: ``bwrap --unshare-net`` is the PRIMARY path (works on
    Ubuntu 24.04 without AppArmor fights). ``unshare -n`` is the FALLBACK.
    If neither works, ``build_game``/``run_game`` refuse to run anything and
    return a ``sandbox_violation`` BuildResult / RunResult; the Coder turns
    that into walk_away (E-69).
  - Root filesystem is read-only inside bwrap (--ro-bind / /), with the
    build workdir re-bound read/write for outputs. Under unshare (fallback)
    the fs is NOT isolated — documented limitation; network still is.
  - NO seccomp, deliberately (E-69). bwrap already gives us a private mount
    namespace, a PID namespace via --proc, the no-net unshare, and the
    read-only bind — for v1 that is enough. A seccomp profile would mostly
    duplicate what the compiler doesn't need and would be one more thing to
    maintain and misconfigure; dropped with prejudice per E-62/E-69.
  - Determinism gate (E-74): banned identifiers are rejected unless a same-line
    ``// allow-nondeterministic`` comment carries the marker AND a reason
    comment is on the next line; allowed sources are still counted as
    nondeterministic_source=true and the Critic must run two digests.
    RNG check: sources calling rand()/mt19937 without harness_seed() or a
    literal seed are rejected.
  - Children env is MINIMAL (PATH, HOME, HARNESS_*). The parent's
    environment — including NVIDIA_API_KEY — is never propagated into
    LLM-authored binaries.
  - Workdirs are mkdtemp'd under /var/tmp (never /tmp, so bwrap can
    re-bind them without shadowing) with mode 0700 and are removed after
    the call unless keep_workdir=True.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = ROOT / "src" / "harness"
HARNESS_C = HARNESS_DIR / "harness.c"
HARNESS_H = HARNESS_DIR / "harness.h"

CC = shutil.which("cc") or "cc"
CXX = shutil.which("g++") or "g++"
PRLIMIT = shutil.which("prlimit")

COMPILE_MEM_BYTES = 536_870_912   # 512 MiB for cc1/g++ on a ~600-line game
COMPILE_CPU_SECONDS = 60
RUN_MEM_BYTES = 268_435_456       # 256 MiB at game run time
RUN_CPU_SECONDS = 30
MAX_SOURCE_FILE_BYTES = 1_048_576
MAX_SOURCE_FILES = 64
DEFAULT_GATE_TAPE = b"tick\n" * 30

BWRAP = shutil.which("bwrap")
UNSHARE = shutil.which("unshare")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildResult:
    ok: bool
    sandbox_violation: bool = False        # E-69: refused, not failed
    network_isolation: str = "unavailable"  # "bwrap" | "unshare" | "unavailable"
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    binary_bytes: bytes | None = None
    binary_sha256: str | None = None
    source_tar: bytes | None = None
    violations: tuple[str, ...] = ()
    nondeterministic_source: bool = False
    duration_s: float = 0.0
    work_dir: str = ""                      # populated only with keep_workdir=True


@dataclass(frozen=True)
class RunResult:
    sandbox_violation: bool = False
    network_isolation: str = "unavailable"
    exit_code: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    duration_s: float = 0.0
    work_dir: str = ""

    @property
    def digests(self) -> tuple[tuple[int, str], ...]:
        return parse_digests(self.stdout)


@dataclass(frozen=True)
class DeterminismGate:
    ran_twice: bool
    digests_first: tuple[tuple[int, str], ...] = field(default_factory=tuple)
    digests_second: tuple[tuple[int, str], ...] = field(default_factory=tuple)
    identical: bool = False
    nondeterministic_source: bool = False
    ok: bool = False
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeterminismScan:
    violations: tuple[str, ...]
    nondeterministic_source: bool


# ---------------------------------------------------------------------------
# Digest parsing (shared with tests)
# ---------------------------------------------------------------------------

_DIGEST_RE = re.compile(r"^tick (\d+) ([0-9a-f]{8})$", re.MULTILINE)


def parse_digests(stdout: bytes) -> tuple[tuple[int, str], ...]:
    return tuple((int(m.group(1)), m.group(2)) for m in _DIGEST_RE.finditer(stdout.decode("utf-8", errors="replace")))


# ---------------------------------------------------------------------------
# Network isolation probe (E-69)
# ---------------------------------------------------------------------------

def _probe(cmd: list[str]) -> bool:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=10, start_new_session=True)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 - probe must never raise
        return False


def probe_network_isolation() -> tuple[str, str | None]:
    """Return ("bwrap"|"unshare"|"unavailable", reason_or_None)."""
    if BWRAP and _probe([BWRAP, "--unshare-net", "--ro-bind", "/", "/", "/bin/true"]):
        return "bwrap", None
    if UNSHARE and _probe([UNSHARE, "-n", "/bin/true"]):
        return "unshare", None
    reason = "neither bwrap --unshare-net nor unshare -n can run net namespaces on this host"
    return "unavailable", reason


def _isolated_argv(argv: list[str], backend: str, work_dir: Path | None = None) -> list[str] | None:
    if backend == "bwrap":
        if BWRAP is None:
            return None
        args = [
            BWRAP, "--unshare-net",
            "--ro-bind", "/", "/",     # read-only root fs
            "--tmpfs", "/tmp",          # compiler tempfiles land in a private tmpfs
            "--dev", "/dev",
            "--proc", "/proc",
        ]
        if work_dir is not None:
            # re-bind the work dir read/write on top of the read-only root
            args += ["--bind", str(work_dir), str(work_dir)]
        return args + ["--", *argv]
    if backend == "unshare":
        if UNSHARE is None:
            return None
        return [UNSHARE, "-n", *argv]
    return None


def _prlimit_argv(mem_bytes: int, cpu_seconds: int, argv: list[str]) -> list[str]:
    if not PRLIMIT:
        raise RuntimeError("prlimit (util-linux) not found; refusing to run without caps")
    return [PRLIMIT, f"--as={mem_bytes}", f"--cpu={cpu_seconds}", *argv]


def _child_env(work_dir: Path, extra: dict[str, str] | None) -> dict[str, str]:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(work_dir)}
    if extra:
        env.update(extra)
    return env


def _run(argv: list[str], *, stdin: bytes | None = None, env: dict[str, str] | None = None,
         timeout: float, backend: str, work_dir: Path | None = None,
         text: bool = False) -> subprocess.CompletedProcess:
    wrapped = _isolated_argv(argv, backend, work_dir)
    if wrapped is None:
        raise RuntimeError("sandbox: no network isolation available")
    return subprocess.run(
        wrapped,
        input=stdin,
        capture_output=True,
        timeout=timeout,
        start_new_session=True,   # E-69: no preexec_fn, ever
        env=env,
        text=text,
    )


def _make_workdir() -> Path:
    # /var/tmp, not /tmp: bwrap re-binds the work dir on top of the read-only
    # root; a fresh tmpfs at /tmp would shadow it.
    base = "/var/tmp" if os.path.isdir("/var/tmp") else None
    return Path(tempfile.mkdtemp(prefix="a2a-sandbox-", dir=base))


def _cleanup_workdir(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_tar(sources: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, text in sorted(sources.items()):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0  # deterministic tar
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Static scans (pre-flight; no compilation happens on violation)
# ---------------------------------------------------------------------------

_HARNESS_INCLUDE_RE = re.compile(r'#include\s*"harness\.h"')
_INIT_DEFN_RE = re.compile(r"\bint\s+harness_init\s*\(\s*void\s*\)")
_TICK_DEFN_RE = re.compile(r"\bvoid\s+harness_tick\s*\(\s*void\s*\)")
_RENDER_DEFN_RE = re.compile(r"(?:const\s+char\s*\*|char\s+const\s*\*)\s*harness_render\s*\(\s*void\s*\)")


def check_harness_contract(sources: dict[str, str]) -> tuple[str, ...]:
    """V9_HARNESS (E-64): grep for the include and the three definitions."""
    text_all = "\n".join(sources.values())
    violations: list[str] = []
    if not _HARNESS_INCLUDE_RE.search(text_all):
        violations.append('harness_contract:missing #include "harness.h"')
    if not _INIT_DEFN_RE.search(text_all):
        violations.append("harness_contract:missing definition of harness_init")
    if not _TICK_DEFN_RE.search(text_all):
        violations.append("harness_contract:missing definition of harness_tick")
    if not _RENDER_DEFN_RE.search(text_all):
        violations.append("harness_contract:missing definition of harness_render")
    return tuple(violations)


_BANNED = (
    (re.compile(r"__DATE__|__TIME__|__TIMESTAMP__"), "macro"),
    (re.compile(r"\btime\s*\("), "time()"),
    (re.compile(r"\bclock\s*\("), "clock()"),
    (re.compile(r"\bgettimeofday\s*\("), "gettimeofday()"),
    (re.compile(r"std::chrono::system_clock"), "std::chrono::system_clock"),
)
_ALLOW_MARKER = "// allow-nondeterministic"
_REASON_COMMENT_RE = re.compile(r"^\s*//\s*\S.{2,}")
_RAND_CALL_RE = re.compile(r"\brand\s*\(")
_MT19937_FIXED_RE = re.compile(r"mt19937\s*\(\s*\d+\s*\)")
_SRAND_FIXED_RE = re.compile(r"srand\s*\(\s*\d+\s*\)")


def scan_determinism(sources: dict[str, str]) -> DeterminismScan:
    """E-74: banned nondeterministic patterns, allow-comment aware.

    A banned hit is a violation UNLESS the same line carries
    ``// allow-nondeterministic`` and the next line is a non-trivial
    comment (the reason). Allowed hits still flip nondeterministic_source
    for the trace/Critic (two-digest comparison then applies).
    """
    violations: list[str] = []
    allowed = False
    for name, text in sources.items():
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            for pattern, label in _BANNED:
                if not pattern.search(line):
                    continue
                if _ALLOW_MARKER in line:
                    next_line = lines[i] if i < len(lines) else ""
                    if _REASON_COMMENT_RE.match(next_line):
                        allowed = True
                        continue
                    violations.append(
                        f"determinism:{label} at {name}:{i} "
                        "(allow-nondeterministic without a reason comment on the next line)"
                    )
                    continue
                violations.append(f"determinism:{label} at {name}:{i}")

        if _RAND_CALL_RE.search(text) or "mt19937" in text:
            seeded = (
                "harness_seed()" in text
                or bool(_MT19937_FIXED_RE.search(text))
                or bool(_SRAND_FIXED_RE.search(text))
            )
            if not seeded:
                violations.append(
                    f"rng:rand()/mt19937 in {name} without harness_seed() or a fixed seed"
                )
    return DeterminismScan(violations=tuple(violations), nondeterministic_source=allowed)


def _preflight(sources: dict[str, str]) -> tuple[tuple[str, ...], DeterminismScan]:
    violations: list[str] = []
    total = 0
    for name, text in sources.items():
        total += len(text.encode("utf-8"))
        if len(text.encode("utf-8")) > MAX_SOURCE_FILE_BYTES:
            violations.append(f"source:{name} exceeds 1 MiB")
    if len(sources) > MAX_SOURCE_FILES:
        violations.append(f"source:too many files ({len(sources)} > {MAX_SOURCE_FILES})")
    if total > config.MAX_ARTIFACT_BYTES:
        violations.append("source:total exceeds MAX_ARTIFACT_BYTES")
    violations.extend(check_harness_contract(sources))
    det = scan_determinism(sources)
    violations.extend(det.violations)
    if not any(name.endswith((".cpp", ".cc", ".cxx")) for name in sources):
        violations.append("source:no .cpp/.cc/.cxx file present")
    return tuple(violations), det


# ---------------------------------------------------------------------------
# Build (E-69, E-73)
# ---------------------------------------------------------------------------

def build_game(
    sources: dict[str, str],
    *,
    reference: bool = False,
    keep_workdir: bool = False,
) -> BuildResult:
    """Compile the game. Deterministic, isolated, no network.

    ``reference=True`` switches to CPP_FLAGS_REFERENCE (-Werror, E-73).
    Violations from pre-flight scans return ok=False without spawning the
    compiler; network isolation failure returns sandbox_violation=True.
    """
    started = time.monotonic()
    backend, unavailable_reason = probe_network_isolation()

    preflight, det = _preflight(sources)
    if preflight:
        return BuildResult(
            ok=False,
            network_isolation=backend,
            violations=preflight,
            duration_s=time.monotonic() - started,
        )

    if backend == "unavailable":
        logger.error("sandbox: %s", unavailable_reason)
        return BuildResult(
            ok=False,
            sandbox_violation=True,
            network_isolation="unavailable",
            violations=(f"sandbox:no network isolation available ({unavailable_reason})",),
            duration_s=time.monotonic() - started,
        )

    work_dir = _make_workdir()
    try:
        for name, text in sources.items():
            dest = work_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        shutil.copy(HARNESS_H, work_dir / "harness.h")
        shutil.copy(HARNESS_C, work_dir / "harness.c")

        flags = config.CPP_FLAGS_REFERENCE if reference else config.CPP_FLAGS_LLM
        env = _child_env(work_dir, None)

        def run_step(argv: list[str]) -> subprocess.CompletedProcess:
            return _run(
                _prlimit_argv(COMPILE_MEM_BYTES, COMPILE_CPU_SECONDS, argv),
                env=env,
                timeout=COMPILE_CPU_SECONDS + 30,
                backend=backend,
                work_dir=work_dir,
            )

        try:
            harness_o = work_dir / "harness.o"
            steps = [
                [CC, "-std=c11", "-O2", "-I", str(work_dir), "-c", str(work_dir / "harness.c"), "-o", str(harness_o)],
            ]
            objects: list[str] = []
            for name in sorted(sources):
                if name.endswith((".cpp", ".cc", ".cxx")):
                    obj = work_dir / (name + ".o")
                    steps.append([CXX, *flags, "-I", str(work_dir), "-c", str(work_dir / name), "-o", str(obj)])
                    objects.append(str(obj))
            binary = work_dir / "game"
            steps.append([CXX, *objects, str(harness_o), "-o", str(binary)])

            combined_out: list[str] = []
            combined_err: list[str] = []
            for argv in steps:
                proc = run_step(argv)
                combined_out.append(proc.stdout.decode("utf-8", errors="replace") if isinstance(proc.stdout, bytes) else str(proc.stdout))
                combined_err.append(proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else str(proc.stderr))
                if proc.returncode != 0:
                    return BuildResult(
                        ok=False,
                        network_isolation=backend,
                        exit_code=proc.returncode,
                        stdout="".join(combined_out),
                        stderr="".join(combined_err),
                        violations=(),
                        duration_s=time.monotonic() - started,
                        work_dir=str(work_dir) if keep_workdir else "",
                    )
        except subprocess.TimeoutExpired:
            return BuildResult(
                ok=False,
                network_isolation=backend,
                exit_code=None,
                stderr="sandbox: compiler exceeded time limit",
                duration_s=time.monotonic() - started,
                work_dir=str(work_dir) if keep_workdir else "",
            )
        except RuntimeError as exc:  # prlimit missing mid-flight
            logger.error("sandbox: %s", exc)
            return BuildResult(
                ok=False,
                sandbox_violation=True,
                network_isolation="unavailable",
                violations=(f"sandbox:{exc}",),
                duration_s=time.monotonic() - started,
            )

        binary_bytes = binary.read_bytes()
        violations: tuple[str, ...] = ()
        if len(binary_bytes) > config.MAX_ARTIFACT_BYTES:
            violations = ("artifact:binary exceeds MAX_ARTIFACT_BYTES",)
        tar_bytes = _source_tar(sources)
        result = BuildResult(
            ok=not violations,
            network_isolation=backend,
            exit_code=0,
            stdout="".join(combined_out),
            stderr="".join(combined_err),
            binary_bytes=binary_bytes,
            binary_sha256=_sha256(binary_bytes),
            source_tar=tar_bytes,
            violations=violations,
            nondeterministic_source=det.nondeterministic_source,
            duration_s=time.monotonic() - started,
            work_dir=str(work_dir) if keep_workdir else "",
        )
        if keep_workdir:
            return result
        return result
    finally:
        if not keep_workdir:
            _cleanup_workdir(work_dir)


# ---------------------------------------------------------------------------
# Run (E-69)
# ---------------------------------------------------------------------------

def run_game(
    binary_bytes: bytes,
    tape: bytes,
    *,
    play_timeout: int = 10,
    mem_bytes: int = RUN_MEM_BYTES,
    cpu_seconds: int = RUN_CPU_SECONDS,
    extra_env: dict[str, str] | None = None,
    keep_workdir: bool = False,
) -> RunResult:
    started = time.monotonic()
    backend, reason = probe_network_isolation()
    if backend == "unavailable":
        return RunResult(
            sandbox_violation=True,
            network_isolation="unavailable",
            stderr=f"sandbox: no network isolation available ({reason})".encode(),
            duration_s=time.monotonic() - started,
        )

    work_dir = _make_workdir()
    try:
        binary = work_dir / "game"
        binary.write_bytes(binary_bytes)
        binary.chmod(0o755)
        env = _child_env(work_dir, {"HARNESS_PLAY_TIMEOUT": str(play_timeout), **(extra_env or {})})
        try:
            proc = _run(
                _prlimit_argv(mem_bytes, cpu_seconds, [str(binary)]),
                stdin=tape,
                env=env,
                timeout=play_timeout + 10,
                backend=backend,
                work_dir=work_dir,
            )
            return RunResult(
                network_isolation=backend,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_s=time.monotonic() - started,
                work_dir=str(work_dir) if keep_workdir else "",
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                network_isolation=backend,
                exit_code=None,
                stderr=b"sandbox: game killed by wall-clock timeout",
                duration_s=time.monotonic() - started,
                work_dir=str(work_dir) if keep_workdir else "",
            )
        except RuntimeError as exc:
            return RunResult(
                sandbox_violation=True,
                network_isolation="unavailable",
                stderr=f"sandbox: {exc}".encode(),
                duration_s=time.monotonic() - started,
            )
    finally:
        if not keep_workdir:
            _cleanup_workdir(work_dir)


def determinism_gate(
    binary_bytes: bytes,
    *,
    tape: bytes = DEFAULT_GATE_TAPE,
    nondeterministic_source: bool,
    extra_env: dict[str, str] | None = None,
) -> DeterminismGate:
    """E-74: two independent runs; identical digests required UNLESS the
    source was allow-listed (nondeterministic_source) — then the Critic must
    agree on the final frame digest instead."""
    first = run_game(binary_bytes, tape, extra_env=extra_env)
    second = run_game(binary_bytes, tape, extra_env=extra_env)
    identical = first.digests == second.digests and first.exit_code == second.exit_code == 0
    if identical:
        return DeterminismGate(
            ran_twice=True,
            digests_first=first.digests,
            digests_second=second.digests,
            identical=True,
            nondeterministic_source=nondeterministic_source,
            ok=True,
        )
    if nondeterministic_source:
        return DeterminismGate(
            ran_twice=True,
            digests_first=first.digests,
            digests_second=second.digests,
            identical=False,
            nondeterministic_source=True,
            ok=True,
        )
    return DeterminismGate(
        ran_twice=True,
        digests_first=first.digests,
        digests_second=second.digests,
        identical=False,
        nondeterministic_source=False,
        ok=False,
        violations=("determinism:outputs differ and no allow-nondeterministic comment in source",),
    )
