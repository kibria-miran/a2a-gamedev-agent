// Reference invaders — Phase 1 ground truth (E-75 step 4).
//
// This is the game the LLM is benchmarked against (X2). It is written by
// hand, compiled with -Werror (E-73), must survive the determinism gate
// (E-74) and the harness contract (E-64). Nothing here is random unless it
// flows from harness_seed().
//
// Controls (tape; one directive per line):
//   key a / key d   move the cannon left / right
//   key k           fire (also accepted: space)
//
// win_condition (from the spec, via harness_config()):
//   survive (0)  play until the tape budget ends; highest score wins
//   clear   (1)  destroy every invader to win
//   score   (2)  like survive; score is all that matters
//
// The game ends via harness_exit(0) on: all invaders destroyed, player
// destroyed (lives exhausted), or invaders reaching the bottom row. The
// harness ends it with exit 0 on tape/budget exhaustion.
//
// Determinism: every random choice flows from srand(harness_seed()).
// harness_render() output contains no timestamps, addresses, or counters
// other than the game state itself. Frame digests are therefore stable
// across identical runs and replayable from the spec (FNV-1a).

#include "harness.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
// Field geometry, clamped from the spec at init (defensive; the spec lattice
// never produces anything near these bounds).
constexpr int MAX_W = 80;
constexpr int MAX_H = 30;
constexpr int MAX_ENEMIES = 30;
constexpr int ENEMIES_PER_ROW = 10;
constexpr int ROW_STRIDE = 3;   // horizontal spacing between enemies
constexpr int MAX_PLAYER_BULLETS = 4;
constexpr int MAX_ENEMY_BULLETS = 16;
constexpr int ENEMY_MOVE_PERIOD = 2;  // ticks between invader steps
constexpr int ENEMY_FIRE_PERCENT = 2; // per living enemy per tick
constexpr int MAX_ROWS = 3;

struct Point {
    int x;
    int y;
    bool alive;
};

int g_w = 40;
int g_h = 20;

Point g_enemies[MAX_ENEMIES];
int g_enemy_count = 0;
int g_enemies_alive = 0;
int g_enemy_dir = 1;   // +1 right, -1 left
int g_enemy_tick = 0;  // move-period accumulator

Point g_player_bullets[MAX_PLAYER_BULLETS];
Point g_enemy_bullets[MAX_ENEMY_BULLETS];

int g_player_x = 0;
int g_player_y = 0;
int g_score = 0;
int g_lives = 3;

constexpr int EMPTY = 0;

char cell_at(const Point* cells, int count, int x, int y, char glyph) {
    for (int i = 0; i < count; ++i) {
        if (cells[i].alive && cells[i].x == x && cells[i].y == y) return glyph;
    }
    return EMPTY;
}

int enemy_at(int x, int y) {
    for (int i = 0; i < g_enemy_count; ++i) {
        if (g_enemies[i].alive && g_enemies[i].x == x && g_enemies[i].y == y) return i;
    }
    return -1;
}

void move_player_bullets(void) {
    for (auto& b : g_player_bullets) {
        if (!b.alive) continue;
        b.y -= 1;
        if (b.y < 1) {  // above the top border
            b.alive = false;
            continue;
        }
        const int hit = enemy_at(b.x, b.y);
        if (hit >= 0) {
            g_enemies[hit].alive = false;
            --g_enemies_alive;
            g_score += 10;
            b.alive = false;
        }
    }
}

void fire_enemy_bullet(void) {
    // cheapest deterministic spread: uniform pick among living enemies
    int guard = 4 * g_enemy_count + 1;
    while (guard-- > 0) {
        const int i = std::rand() % g_enemy_count;
        if (!g_enemies[i].alive) continue;
        if (g_enemies[i].y >= g_player_y - 1) continue;
        for (auto& b : g_enemy_bullets) {
            if (b.alive) continue;
            b = {g_enemies[i].x, g_enemies[i].y + 1, true};
            return;
        }
        return;
    }
}

void move_enemy_bullets(void) {
    for (auto& b : g_enemy_bullets) {
        if (!b.alive) continue;
        b.y += 1;
        if (b.y >= g_player_y) {
            if (b.x == g_player_x) {
                --g_lives;
                b.alive = false;
                if (g_lives <= 0) harness_exit(0);
                continue;
            }
        }
        if (b.y >= g_h - 1) b.alive = false;
    }
}

void step_enemies(void) {
    g_enemy_tick += 1;
    if (g_enemy_tick < ENEMY_MOVE_PERIOD) return;
    g_enemy_tick = 0;

    // edge probe: does any living enemy sit at the border in the current
    // direction of travel?
    bool at_edge = false;
    for (int i = 0; i < g_enemy_count; ++i) {
        if (!g_enemies[i].alive) continue;
        if (g_enemy_dir > 0 && g_enemies[i].x >= g_w - 2) at_edge = true;
        if (g_enemy_dir < 0 && g_enemies[i].x <= 1) at_edge = true;
    }

    if (at_edge) {
        // drop one row and reverse direction
        for (int i = 0; i < g_enemy_count; ++i) {
            if (g_enemies[i].alive) g_enemies[i].y += 1;
        }
        g_enemy_dir = -g_enemy_dir;
        // invaders reached the player's trench: the run is over
        for (int i = 0; i < g_enemy_count; ++i) {
            if (g_enemies[i].alive && g_enemies[i].y >= g_player_y - 1) {
                harness_exit(0);
            }
        }
        return;
    }

    for (int i = 0; i < g_enemy_count; ++i) {
        if (g_enemies[i].alive) g_enemies[i].x += g_enemy_dir;
    }
}

void maybe_enemy_fire(void) {
    for (int i = 0; i < g_enemy_count; ++i) {
        if (!g_enemies[i].alive) continue;
        if (std::rand() % 100 < ENEMY_FIRE_PERCENT) fire_enemy_bullet();
    }
}

void fire_player(void) {
    for (auto& b : g_player_bullets) {
        if (b.alive) continue;
        b = {g_player_x, g_player_y - 1, true};
        return;
    }
}

void layout_from_config(void) {
    const harness_config_t* cfg = harness_config();
    g_w = cfg->grid_w < 16 ? 16 : (cfg->grid_w > MAX_W ? MAX_W : cfg->grid_w);
    g_h = cfg->grid_h < 10 ? 10 : (cfg->grid_h > MAX_H ? MAX_H : cfg->grid_h);

    int count = cfg->enemy_count;
    if (count < 1) count = 1;
    if (count > MAX_ENEMIES) count = MAX_ENEMIES;

    // fit the formation inside the field, even on the smallest lattice grid
    int per_row = (g_w - 2) / ROW_STRIDE;
    if (per_row < 1) per_row = 1;
    if (per_row > ENEMIES_PER_ROW) per_row = ENEMIES_PER_ROW;
    const int capacity = per_row * MAX_ROWS;
    if (count > capacity) count = capacity;

    g_enemy_count = count;
    g_enemies_alive = count;

    for (int i = 0; i < count; ++i) {
        const int row = i / per_row;
        const int col = i % per_row;
        g_enemies[i] = {2 + col * ROW_STRIDE, 2 + row * 2, true};
    }

    g_player_y = g_h - 2;
    g_player_x = g_w / 2;
}
}  // namespace

int harness_init(void) {
    // mandatory for the determinism gate (E-74): seed from the spec, always
    std::srand(harness_seed());

    std::memset(g_enemies, 0, sizeof g_enemies);
    std::memset(g_player_bullets, 0, sizeof g_player_bullets);
    std::memset(g_enemy_bullets, 0, sizeof g_enemy_bullets);

    layout_from_config();
    return 0;
}

void harness_tick(void) {
    const int key = harness_poll_key();
    if (key == 'a' && g_player_x > 1) g_player_x -= 1;
    if (key == 'd' && g_player_x < g_w - 2) g_player_x += 1;
    if (key == 'k' || key == ' ') fire_player();

    move_player_bullets();
    move_enemy_bullets();
    step_enemies();
    maybe_enemy_fire();

    if (g_enemies_alive == 0) harness_exit(0);  // clear win
}

const char* harness_render(void) {
    static std::string frame;
    frame.clear();
    frame.reserve(static_cast<size_t>((g_w + 4) * (g_h + 1) + 96));

    // top border
    frame += '+';
    frame.append(static_cast<size_t>(g_w), '-');
    frame += "+\n";

    for (int y = 1; y < g_h - 1; ++y) {
        frame += '|';
        for (int x = 0; x < g_w; ++x) {
            char glyph = ' ';
            const int enemy = enemy_at(x, y);
            if (enemy >= 0) {
                glyph = 'i';
            } else if (cell_at(g_player_bullets, MAX_PLAYER_BULLETS, x, y, '|')) {
                glyph = '|';
            } else if (cell_at(g_enemy_bullets, MAX_ENEMY_BULLETS, x, y, '!')) {
                glyph = '!';
            } else if (x == g_player_x && y == g_player_y) {
                glyph = 'A';
            }
            frame += glyph;
        }
        frame += "|\n";
    }

    frame += '+';
    frame.append(static_cast<size_t>(g_w), '-');
    frame += '+';

    char status[96];
    std::snprintf(status, sizeof status,
                  "score=%d lives=%d enemies=%d",
                  g_score, g_lives, g_enemies_alive);
    frame += status;  // no trailing newline (contract)
    return frame.c_str();
}
