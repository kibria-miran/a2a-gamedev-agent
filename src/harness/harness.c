// harness.c — companion to harness.h (v1.1, frozen for games).
// Implements main(); the game provides harness_init/harness_tick/harness_render.
//
// Exit codes:
//   0    budget exhausted (tty) / tape exhausted or `quit` directive (non-tty),
//        or the game called harness_exit(0)
//   1    harness_init() returned non-zero
//   3    malformed input tape (unknown directive, bad `key` argument)
//   124  HARNESS_PLAY_TIMEOUT watchdog fired (SIGALRM)
//   128+N  terminated by signal N (e.g. 130 = SIGINT); terminal is restored first
//   N    the code passed to harness_exit(N)
//
// stdout carries ONLY frame-digest lines:  "tick N <fnv1a32-8-hex>"
// Everything else (diagnostics, errors) goes to stderr so the Critic can
// parse stdout blindly (E-62/E-75, Critic parsing contract).
//
// Both stdio modes print the identical digest format.
//
// Implementation notes:
//   - Non-tty mode reads the whole tape into memory up front so max_ticks can
//     report the exact number of `tick` directives ("tape length", §8.11).
//     Tapes are a few KB; a hard cap of 1 MiB applies (stderr note, then the
//     truncated tape is executed as read).
//   - Determinism: FNV-1a over bytes; no timestamps, no addresses, no rand()
//     anywhere in the harness. srand() is seeded from the canonical spec JSON
//     (FNV-1a) before harness_init(), so games that call rand() after
//     srand(harness_seed()) replay byte-identically from the same spec.
//   - bwrap/prlimit/seccomp are the sandbox's job (E-69); this file is
//     intentionally free of sandboxing policy so it can be unit-tested.
//   - tcsetattr() is not strictly async-signal-safe per POSIX, but every
//     Unix restores termios this way in practice; without it Ctrl-C in tty
//     mode would leave the user's terminal in raw mode.

#define _GNU_SOURCE /* glibc: usleep, strtok_r, SIGALRM, isatty, ... */
#define _POSIX_C_SOURCE 200809L

#include "harness.h"

#include <fcntl.h>
#include <ctype.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#define DEFAULT_TICKS 600
#define DEFAULT_PLAY_TIMEOUT_SECONDS 30
#define TAPE_CAP_BYTES (1u << 20)   /* 1 MiB */
#define SPEC_CAP_BYTES (4u << 20)   /* 4 MiB */
#define SPEC_FALLBACK "{}"

static struct harness_config_t g_cfg;
static int g_ticks_run = 0;
static int g_active_key = -1;
static int g_pending_key = -1;
static int g_quit = 0;
static int g_exit_requested = 0;
static int g_exit_code = 0;

static struct termios g_orig_termios;
static int g_termios_saved = 0;

/* ---------- FNV-1a 32-bit ---------- */

static uint32_t fnv1a32(const unsigned char* data, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

/* ---------- spec JSON loading ---------- */

static char* read_all_fd(int fd, size_t cap, size_t* out_len, int* overflowed) {
    /* bufcap counts data bytes; the allocation always reserves one extra byte
       for the NUL terminator, so a buffer filled to the cap stays in bounds. */
    size_t len = 0, bufcap = 1 << 16;
    if (bufcap > cap) bufcap = cap;
    char* buf = (char*)malloc(bufcap + 1);
    if (!buf) return NULL;
    for (;;) {
        if (len == bufcap) {
            if (bufcap >= cap) {
                if (overflowed) *overflowed = 1;
                break;
            }
            size_t grown_cap = bufcap * 2;
            if (grown_cap > cap) grown_cap = cap;
            char* grown = (char*)realloc(buf, grown_cap + 1);
            if (!grown) { free(buf); return NULL; }
            buf = grown;
            bufcap = grown_cap;
        }
        ssize_t n = read(fd, buf + len, bufcap - len);
        if (n <= 0) break;
        len += (size_t)n;
    }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    return buf;
}

static char* read_spec_json(size_t* out_len) {
    const char* path = getenv("HARNESS_SPEC_JSON_PATH");
    if (path && *path) {
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            char* buf = read_all_fd(fd, SPEC_CAP_BYTES, out_len, NULL);
            close(fd);
            if (buf) return buf;
        } else {
            fprintf(stderr, "harness: warning: cannot open HARNESS_SPEC_JSON_PATH=%s; using \"{}\"\n", path);
        }
    }
    const char* inline_json = getenv("HARNESS_SPEC_JSON");
    if (inline_json && *inline_json) {
        size_t n = strlen(inline_json);
        char* buf = (char*)malloc(n + 1);
        if (buf) {
            memcpy(buf, inline_json, n + 1);
            if (out_len) *out_len = n;
            return buf;
        }
    }
    if (out_len) *out_len = strlen(SPEC_FALLBACK);
    return strdup(SPEC_FALLBACK);
}

/* Extract `"key": <integer>` from the canonical JSON. Returns dv when absent or non-numeric. */
static int spec_int(const char* json, const char* key, int dv) {
    if (!json) return dv;
    char needle[64];
    snprintf(needle, sizeof needle, "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return dv;
    p += strlen(needle);
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != ':') return dv;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    char* end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) return dv;
    return (int)v;
}

/* win_condition may be a number (0/1/2) or a string (survive/clear/score). */
static int spec_win_condition(const char* json, int dv) {
    if (!json) return dv;
    const char* needle = "\"win_condition\"";
    const char* p = strstr(json, needle);
    if (!p) return dv;
    p += strlen(needle);
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != ':') return dv;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == '"') {
        p++;
        char buf[32];
        size_t i = 0;
        while (*p && *p != '"' && i < sizeof buf - 1) buf[i++] = *p++;
        buf[i] = '\0';
        if (strcmp(buf, "survive") == 0) return 0;
        if (strcmp(buf, "clear") == 0) return 1;
        if (strcmp(buf, "score") == 0) return 2;
        return dv;
    }
    char* end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) return dv;
    return (int)v;
}

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void setup_config(const char* spec, int tape_ticks, int is_tty) {
    g_cfg.grid_w = clamp_int(spec_int(spec, "grid_w", 40), 1, 500);
    g_cfg.grid_h = clamp_int(spec_int(spec, "grid_h", 20), 1, 500);
    g_cfg.tick_ms = clamp_int(spec_int(spec, "tick_ms", 100), 0, 10000);
    g_cfg.enemy_count = clamp_int(spec_int(spec, "enemy_count", 8), 0, 1000);
    g_cfg.win_condition = spec_win_condition(spec, 0);

    int max_ticks = -1;
    const char* mt = getenv("HARNESS_MAX_TICKS");
    if (mt && *mt) {
        char* end = NULL;
        long v = strtol(mt, &end, 10);
        if (end != mt && v >= 0 && v <= 100000) max_ticks = (int)v;
    }
    if (max_ticks >= 0) {
        g_cfg.max_ticks = max_ticks;
    } else if (is_tty) {
        g_cfg.max_ticks = DEFAULT_TICKS;
    } else {
        g_cfg.max_ticks = tape_ticks;
    }
}

const struct harness_config_t* harness_config(void) {
    return &g_cfg;
}

uint32_t harness_seed(void) {
    size_t len = 0;
    char* spec = read_spec_json(&len);
    uint32_t h = fnv1a32((const unsigned char*)(spec ? spec : ""), len);
    free(spec);
    return h;
}

/* ---------- tape handling (non-tty) ---------- */

static char* trim(char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char* end = s + strlen(s);
    while (end > s && isspace((unsigned char)*(end - 1))) {
        *--end = '\0';
    }
    return s;
}

static int count_tick_directives(const char* tape) {
    int count = 0;
    if (!tape) return 0;
    const char* p = tape;
    while (*p) {
        const char* nl = strchr(p, '\n');
        size_t line_len = nl ? (size_t)(nl - p) : strlen(p);
        const char* s = p;
        const char* e = p + line_len;
        while (s < e && isspace((unsigned char)*s)) s++;
        while (e > s && isspace((unsigned char)*(e - 1))) e--;
        if ((size_t)(e - s) == 4 && memcmp(s, "tick", 4) == 0) count++;
        if (!nl) break;
        p = nl + 1;
    }
    return count;
}

static int parse_key_arg(const char* arg) {
    size_t n = strlen(arg);
    if (strcmp(arg, "space") == 0) return ' ';
    if (strcmp(arg, "enter") == 0) return '\n';
    if (n == 1) return (unsigned char)arg[0];
    return -1;
}

/* ---------- digest printing ---------- */

static void print_digest(int n) {
    const char* frame = harness_render();
    if (!frame) frame = "";
    uint32_t h = fnv1a32((const unsigned char*)frame, strlen(frame));
    printf("tick %d %08x\n", n, h);
    fflush(stdout);
}

static void do_tick(void) {
    g_active_key = g_pending_key;
    g_pending_key = -1;
    harness_tick();
    print_digest(g_ticks_run);
    g_ticks_run += 1;
    g_active_key = -1;
}

static int is_directive(const char* line, const char* word) {
    size_t n = strlen(word);
    return strncmp(line, word, n) == 0 && (line[n] == '\0' || isspace((unsigned char)line[n]));
}

static int run_non_tty(char* tape) {
    char* save = NULL;
    char* raw = strtok_r(tape, "\n", &save);
    while (raw != NULL) {
        if (g_exit_requested) break;

        char* line = trim(raw);
        char* next = strtok_r(NULL, "\n", &save);

        if (*line == '\0') {
            raw = next;
            continue;
        }
        if (strcmp(line, "tick") == 0) {
            if (g_ticks_run >= g_cfg.max_ticks) break; /* budget exhausted: stop silently */
            do_tick();
        } else if (strcmp(line, "quit") == 0) {
            g_quit = 1;
            break;
        } else if (is_directive(line, "key")) {
            char* arg = trim(line + 3);
            int key = *arg ? parse_key_arg(arg) : -1;
            if (key < 0) {
                fprintf(stderr, "harness: malformed `key` directive: \"%s\"\n", line);
                return 3;
            }
            g_pending_key = key;
        } else {
            fprintf(stderr, "harness: unknown directive: \"%s\"\n", line);
            return 3;
        }
        raw = next;
    }
    if (g_exit_requested) return g_exit_code;
    if (g_quit) print_digest(g_ticks_run); /* final frame, §8.11 */
    return 0;
}

/* ---------- tty mode ---------- */

static void restore_termios(void) {
    if (g_termios_saved) {
        tcsetattr(STDIN_FILENO, TCSANOW, &g_orig_termios);
        g_termios_saved = 0;
    }
}

static int read_tty_key(void) {
    unsigned char c;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n != 1) return -1;
    if (c == '\r') return '\n';
    return (int)c;
}

static int run_tty(void) {
    if (tcgetattr(STDIN_FILENO, &g_orig_termios) == 0) {
        struct termios t = g_orig_termios;
        t.c_iflag &= (tcflag_t)~(ICRNL | IXON);
        t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
        t.c_cc[VMIN] = 0;
        t.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &t) == 0) {
            g_termios_saved = 1;
        }
    } else {
        fprintf(stderr, "harness: warning: isatty(stdin) but tcgetattr failed; reading tape-like bytes anyway\n");
    }

    /* Test aid (pty digest-identity checks): sleep before the first tick so a
     * forked parent can queue keypresses. Not used in production. */
    const char* delay = getenv("HARNESS_TTY_START_DELAY_MS");
    if (delay && *delay) {
        int ms = atoi(delay);
        if (ms > 0) usleep((useconds_t)ms * 1000);
    }

    while (!g_exit_requested && g_ticks_run < g_cfg.max_ticks) {
        int k = read_tty_key();
        if (k >= 0) g_pending_key = k;   /* last key wins within a tick */
        do_tick();
        if (g_cfg.tick_ms > 0) usleep((useconds_t)g_cfg.tick_ms * 1000);
    }
    restore_termios();
    return g_exit_requested ? g_exit_code : 0;
}

/* ---------- signals / watchdog ---------- */

static void on_alarm(int sig) {
    static const char msg[] = "harness: play timeout (HARNESS_PLAY_TIMEOUT)\n";
    (void)sig;
    restore_termios();
    ssize_t unused = write(STDERR_FILENO, msg, sizeof msg - 1);
    (void)unused;
    _exit(124);
}

static void on_term(int sig) {
    restore_termios();
    _exit(128 + sig);
}

static void install_handlers(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_flags = 0; /* no SA_RESTART: die promptly */

    sa.sa_handler = on_alarm;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGALRM, &sa, NULL);

    sa.sa_handler = on_term;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);

    long timeout = DEFAULT_PLAY_TIMEOUT_SECONDS;
    const char* env_timeout = getenv("HARNESS_PLAY_TIMEOUT");
    if (env_timeout && *env_timeout) {
        char* end = NULL;
        long v = strtol(env_timeout, &end, 10);
        if (end != env_timeout && v > 0 && v <= 24 * 3600) timeout = v;
    }
    alarm((unsigned)timeout);
}

/* ---------- public harness API ---------- */

int harness_poll_key(void) {
    return g_active_key;
}

void harness_exit(int code) {
    g_exit_requested = 1;
    g_exit_code = code;
}

/* ---------- main ---------- */

int main(void) {
    int is_tty = isatty(STDIN_FILENO);

    char* tape = NULL;
    size_t tape_len = 0;
    if (!is_tty) {
        int overflowed = 0;
        tape = read_all_fd(STDIN_FILENO, TAPE_CAP_BYTES, &tape_len, &overflowed);
        if (overflowed) {
            fprintf(stderr, "harness: warning: tape exceeds 1 MiB; running truncated tape\n");
        }
        (void)tape_len;
    }

    size_t spec_len = 0;
    char* spec = read_spec_json(&spec_len);

    int tape_ticks = tape ? count_tick_directives(tape) : DEFAULT_TICKS;
    setup_config(spec, tape_ticks, is_tty);

    install_handlers();

    /* E-74: seed from the canonical spec JSON before the game's init runs. */
    srand(fnv1a32((const unsigned char*)(spec ? spec : SPEC_FALLBACK), spec_len));

    if (harness_init() != 0) {
        fprintf(stderr, "harness: harness_init() returned non-zero; aborting\n");
        free(tape);
        free(spec);
        return 1;
    }

    int rc = is_tty ? run_tty() : run_non_tty(tape);

    restore_termios();
    free(tape);
    free(spec);
    return rc;
}
