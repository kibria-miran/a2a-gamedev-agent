// harness.h — v1.1, frozen. Do not edit; games include this.
// Contract:
//   - main() is provided BY THE HARNESS. The game supplies three functions:
//       harness_init(), harness_tick(), harness_render()
//   - The harness reads a scripted input tape from stdin (non-tty) or
//     raw keypresses (tty). Both modes drive the same game loop.
//   - One "tick N <hash>" line per simulation tick. N is 0-based.
//   - The game exits when the harness's tick budget is exhausted, the
//     tape is exhausted (non-tty), or the game calls harness_exit(code).
//   - RNG MUST be seeded via harness_seed(); the harness calls srand()
//     with it before harness_init().
//
// Non-tty tape format (one directive per line):
//   tick           advance one tick
//   key <c>        deliver keypress c for the next tick
//                  (c is a single ASCII character; "space" = ' ',
//                   "enter" = '\n' are also accepted)
//   quit           stop early (harness still prints the final frame)
//
// Frame digest: FNV-1a 32-bit over the canonical text serialization of
// the visible frame (grid, player, enemies, score). The game provides the
// serialization in harness_render(); the harness hashes it and prints
//   tick N <fnv1a32 as 8 lowercase hex chars>
//
// win_condition codes (harness_config_t.win_condition):
//    0 = survive   1 = clear   2 = score
// If the canonical spec JSON carries a string ("win_condition": "clear"),
// the harness maps survive/clear/score -> 0/1/2; unknown strings map to 0.
//
// Harness-honored environment knobs (testing / sandbox plumbing, not
// part of the game contract):
//   HARNESS_SPEC_JSON_PATH   path to the canonical spec JSON (wins over inline)
//   HARNESS_SPEC_JSON        inline canonical spec JSON
//   HARNESS_MAX_TICKS        override max_ticks (tty mode and spec default)
//   HARNESS_PLAY_TIMEOUT     watchdog seconds (default 30); 124 on expiry
//   HARNESS_TTY_START_DELAY_MS  tty test aid: sleep before first tick

#ifndef HARNESS_H
#define HARNESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Seed derived from the canonical spec JSON (FNV-1a 32-bit).
// Stable across runs; games MUST call srand(harness_seed()).
// (The harness itself already calls srand(harness_seed()) before
//  harness_init(); calling it again with the same value is a no-op.)
uint32_t harness_seed(void);

// Called once at startup. Return 0 on success, non-zero to abort
// (harness exits with code 1, no ticks are run).
// grid_w/grid_h/tick_ms/enemy_count are read from the spec by the harness
// and exposed via harness_config().
int  harness_init(void);

// Advance the simulation by one tick. Read any pending key via
// harness_poll_key(). Called once per "tick" line on the tape.
void harness_tick(void);

// Produce the current visible frame as a deterministic ASCII string
// (no trailing newline). Called by the harness after every tick; hashed
// (FNV-1a 32-bit) and printed as "tick N <hash>".
// MUST NOT include a timestamp, pointer, address, or random bytes.
const char* harness_render(void);

// ---- provided by the harness, usable by games ----

// -1 if no key is pending for the current tick, else the ASCII code.
// The same key is reported for repeated polls within one tick.
int  harness_poll_key(void);

// End the loop cleanly after the current tick finishes.
// Process exits with `code` (no final-frame line is printed).
void harness_exit(int code);

struct harness_config_t {
    int grid_w, grid_h, tick_ms, enemy_count, win_condition; // see contract above
    int max_ticks;                     // tape length, or DEFAULT_TICKS in tty mode
};
const struct harness_config_t* harness_config(void);

#ifdef __cplusplus
}
#endif

#endif // HARNESS_H
