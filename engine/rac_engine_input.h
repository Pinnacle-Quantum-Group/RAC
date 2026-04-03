/*
 * rac_engine_input.h — Input System
 * Platform-abstract input polling, action mapping.
 */

#ifndef RAC_ENGINE_INPUT_H
#define RAC_ENGINE_INPUT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Key codes ─────────────────────────────────────────────────────────── */

typedef enum {
    RAC_KEY_UNKNOWN = 0,
    RAC_KEY_A = 'a', RAC_KEY_B = 'b', RAC_KEY_C = 'c', RAC_KEY_D = 'd',
    RAC_KEY_E = 'e', RAC_KEY_F = 'f', RAC_KEY_G = 'g', RAC_KEY_H = 'h',
    RAC_KEY_I = 'i', RAC_KEY_J = 'j', RAC_KEY_K = 'k', RAC_KEY_L = 'l',
    RAC_KEY_M = 'm', RAC_KEY_N = 'n', RAC_KEY_O = 'o', RAC_KEY_P = 'p',
    RAC_KEY_Q = 'q', RAC_KEY_R = 'r', RAC_KEY_S = 's', RAC_KEY_T = 't',
    RAC_KEY_U = 'u', RAC_KEY_V = 'v', RAC_KEY_W = 'w', RAC_KEY_X = 'x',
    RAC_KEY_Y = 'y', RAC_KEY_Z = 'z',
    RAC_KEY_0 = '0', RAC_KEY_1 = '1', RAC_KEY_2 = '2', RAC_KEY_3 = '3',
    RAC_KEY_4 = '4', RAC_KEY_5 = '5', RAC_KEY_6 = '6', RAC_KEY_7 = '7',
    RAC_KEY_8 = '8', RAC_KEY_9 = '9',
    RAC_KEY_SPACE  = ' ',
    RAC_KEY_ESCAPE = 27,
    RAC_KEY_ENTER  = 13,
    RAC_KEY_UP     = 128,
    RAC_KEY_DOWN   = 129,
    RAC_KEY_LEFT   = 130,
    RAC_KEY_RIGHT  = 131,
    RAC_KEY_LSHIFT = 132,
    RAC_KEY_LCTRL  = 133,
} rac_keycode;

/* ── Key state ─────────────────────────────────────────────────────────── */

typedef enum {
    RAC_KEY_STATE_UP       = 0,
    RAC_KEY_STATE_PRESSED  = 1,   /* just pressed this frame */
    RAC_KEY_STATE_HELD     = 2,   /* held down */
    RAC_KEY_STATE_RELEASED = 3,   /* just released this frame */
} rac_key_state;

/* ── Mouse state ───────────────────────────────────────────────────────── */

typedef struct {
    int   x, y;           /* current position */
    int   dx, dy;          /* delta since last frame */
    int   buttons;         /* bitmask: bit0=left, bit1=right, bit2=middle */
    int   wheel;           /* scroll delta */
} rac_mouse_state;

/* ── Gamepad ───────────────────────────────────────────────────────────── */

#define RAC_MAX_GAMEPADS 4

typedef struct {
    int   connected;
    float axes[6];          /* left_x, left_y, right_x, right_y, lt, rt */
    int   buttons;          /* bitmask */
    float deadzone;
} rac_gamepad_state;

/* ── Action mapping ────────────────────────────────────────────────────── */

#define RAC_MAX_ACTIONS 32
#define RAC_ACTION_NAME_LEN 32

typedef struct {
    char          name[RAC_ACTION_NAME_LEN];
    rac_keycode   key;
    int           gamepad_button;   /* -1 = unbound */
    int           active;           /* currently triggered */
} rac_action_binding;

/* ── Input system ──────────────────────────────────────────────────────── */

#define RAC_MAX_KEYS 256

typedef struct {
    /* Key state array */
    rac_key_state   keys[RAC_MAX_KEYS];
    rac_key_state   prev_keys[RAC_MAX_KEYS];

    /* Mouse */
    rac_mouse_state mouse;
    rac_mouse_state prev_mouse;

    /* Gamepad */
    rac_gamepad_state gamepads[RAC_MAX_GAMEPADS];

    /* Action bindings */
    rac_action_binding actions[RAC_MAX_ACTIONS];
    int                num_actions;

    /* Raw terminal input mode */
    int terminal_mode;
} rac_input_system;

/* ── API ───────────────────────────────────────────────────────────────── */

void rac_input_init(rac_input_system *input);
void rac_input_shutdown(rac_input_system *input);

/* Poll for new input events (raw terminal mode) */
void rac_input_poll(rac_input_system *input);

/* Update key states for frame transitions (pressed → held, released → up) */
void rac_input_update(rac_input_system *input);

/* Key queries */
int rac_input_key_pressed(const rac_input_system *input, rac_keycode key);
int rac_input_key_held(const rac_input_system *input, rac_keycode key);
int rac_input_key_released(const rac_input_system *input, rac_keycode key);

/* Mouse queries */
void rac_input_mouse_delta(const rac_input_system *input, int *dx, int *dy);
int  rac_input_mouse_button(const rac_input_system *input, int button);

/* Action mapping */
int  rac_input_bind_action(rac_input_system *input, const char *name,
                           rac_keycode key);
int  rac_input_action_active(const rac_input_system *input, const char *name);

/* Inject key event (for SDL2 or external input backends) */
void rac_input_inject_key(rac_input_system *input, rac_keycode key, int down);
void rac_input_inject_mouse(rac_input_system *input, int x, int y, int buttons);
void rac_input_inject_mouse_delta(rac_input_system *input, int dx, int dy);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_INPUT_H */
