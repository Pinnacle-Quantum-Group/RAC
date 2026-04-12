/*
 * rac_engine_input.c — Input System Implementation
 * Raw terminal input backend + injectable events for SDL2.
 */

#include "rac_engine_input.h"
#include <string.h>
#include <stdio.h>

#ifndef _WIN32
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
static struct termios orig_termios;
static int terminal_raw = 0;
#endif

void rac_input_init(rac_input_system *input)
{
    memset(input, 0, sizeof(*input));

    for (int i = 0; i < RAC_MAX_GAMEPADS; i++)
        input->gamepads[i].deadzone = 0.15f;

#ifndef _WIN32
    /* Set terminal to raw mode for non-blocking key input */
    if (isatty(STDIN_FILENO)) {
        tcgetattr(STDIN_FILENO, &orig_termios);
        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ECHO | ICANON);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        terminal_raw = 1;
        input->terminal_mode = 1;
    }
#endif
}

void rac_input_shutdown(rac_input_system *input)
{
#ifndef _WIN32
    if (terminal_raw) {
        tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
        terminal_raw = 0;
    }
#endif
    input->terminal_mode = 0;
}

void rac_input_poll(rac_input_system *input)
{
    /* Reset deltas */
    input->mouse.dx = 0;
    input->mouse.dy = 0;
    input->mouse.wheel = 0;

#ifndef _WIN32
    if (!input->terminal_mode) return;

    /* Read available bytes from stdin (non-blocking) */
    char buf[32];
    int n = read(STDIN_FILENO, buf, sizeof(buf));
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)buf[i];

        /* Escape sequences for arrow keys */
        if (c == 27 && i + 2 < n && buf[i + 1] == '[') {
            switch (buf[i + 2]) {
                case 'A': rac_input_inject_key(input, RAC_KEY_UP, 1); i += 2; continue;
                case 'B': rac_input_inject_key(input, RAC_KEY_DOWN, 1); i += 2; continue;
                case 'C': rac_input_inject_key(input, RAC_KEY_RIGHT, 1); i += 2; continue;
                case 'D': rac_input_inject_key(input, RAC_KEY_LEFT, 1); i += 2; continue;
            }
        }

        if (c == 27) {
            rac_input_inject_key(input, RAC_KEY_ESCAPE, 1);
        } else {
            /* c is unsigned char ∈ [0, 255]; RAC_MAX_KEYS == 256 so the
             * old `c < RAC_MAX_KEYS` bounds check was a tautology. The
             * type itself enforces the bound. */
            rac_input_inject_key(input, (rac_keycode)c, 1);
        }
    }
#endif
}

void rac_input_update(rac_input_system *input)
{
    /* Transition states */
    memcpy(input->prev_keys, input->keys, sizeof(input->prev_keys));
    input->prev_mouse = input->mouse;

    /* PRESSED → HELD, RELEASED → UP */
    for (int i = 0; i < RAC_MAX_KEYS; i++) {
        if (input->keys[i] == RAC_KEY_STATE_PRESSED)
            input->keys[i] = RAC_KEY_STATE_HELD;
        else if (input->keys[i] == RAC_KEY_STATE_RELEASED)
            input->keys[i] = RAC_KEY_STATE_UP;
    }

    /* Update action bindings */
    for (int i = 0; i < input->num_actions; i++) {
        rac_action_binding *ab = &input->actions[i];
        ab->active = (input->keys[ab->key] == RAC_KEY_STATE_HELD ||
                      input->keys[ab->key] == RAC_KEY_STATE_PRESSED);
    }
}

int rac_input_key_pressed(const rac_input_system *input, rac_keycode key)
{
    if ((int)key >= RAC_MAX_KEYS) return 0;
    return input->keys[key] == RAC_KEY_STATE_PRESSED;
}

int rac_input_key_held(const rac_input_system *input, rac_keycode key)
{
    if ((int)key >= RAC_MAX_KEYS) return 0;
    return input->keys[key] == RAC_KEY_STATE_HELD ||
           input->keys[key] == RAC_KEY_STATE_PRESSED;
}

int rac_input_key_released(const rac_input_system *input, rac_keycode key)
{
    if ((int)key >= RAC_MAX_KEYS) return 0;
    return input->keys[key] == RAC_KEY_STATE_RELEASED;
}

void rac_input_mouse_delta(const rac_input_system *input, int *dx, int *dy)
{
    *dx = input->mouse.dx;
    *dy = input->mouse.dy;
}

int rac_input_mouse_button(const rac_input_system *input, int button)
{
    return (input->mouse.buttons >> button) & 1;
}

int rac_input_bind_action(rac_input_system *input, const char *name,
                          rac_keycode key)
{
    if (input->num_actions >= RAC_MAX_ACTIONS) return -1;
    int id = input->num_actions++;
    rac_action_binding *ab = &input->actions[id];
    strncpy(ab->name, name, RAC_ACTION_NAME_LEN - 1);
    ab->key = key;
    ab->gamepad_button = -1;
    return id;
}

int rac_input_action_active(const rac_input_system *input, const char *name)
{
    for (int i = 0; i < input->num_actions; i++) {
        if (strcmp(input->actions[i].name, name) == 0)
            return input->actions[i].active;
    }
    return 0;
}

void rac_input_inject_key(rac_input_system *input, rac_keycode key, int down)
{
    if ((int)key >= RAC_MAX_KEYS) return;
    if (down) {
        if (input->keys[key] == RAC_KEY_STATE_UP)
            input->keys[key] = RAC_KEY_STATE_PRESSED;
    } else {
        if (input->keys[key] == RAC_KEY_STATE_HELD ||
            input->keys[key] == RAC_KEY_STATE_PRESSED)
            input->keys[key] = RAC_KEY_STATE_RELEASED;
    }
}

void rac_input_inject_mouse(rac_input_system *input, int x, int y, int buttons)
{
    input->mouse.dx = x - input->mouse.x;
    input->mouse.dy = y - input->mouse.y;
    input->mouse.x = x;
    input->mouse.y = y;
    input->mouse.buttons = buttons;
}

void rac_input_inject_mouse_delta(rac_input_system *input, int dx, int dy)
{
    input->mouse.dx += dx;
    input->mouse.dy += dy;
    input->mouse.x += dx;
    input->mouse.y += dy;
}
