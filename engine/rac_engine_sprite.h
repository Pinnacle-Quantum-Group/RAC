/*
 * rac_engine_sprite.h — 2D Sprite System
 * Animated sprite rendering, sprite sheets, billboarding.
 * Renders sprites as textured quads in the software rasterizer.
 */

#ifndef RAC_ENGINE_SPRITE_H
#define RAC_ENGINE_SPRITE_H

#include "rac_engine_render.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Sprite sheet ──────────────────────────────────────────────────────── */

#define RAC_MAX_SPRITES 32
#define RAC_MAX_SPRITE_FRAMES 32

typedef struct {
    uint8_t      *pixels;       /* RGBA8888 data for entire sheet */
    int           sheet_w;      /* total sheet width in pixels */
    int           sheet_h;      /* total sheet height (= frame height) */
    int           frame_w;      /* single frame width */
    int           frame_h;      /* single frame height */
    int           num_frames;
    int           valid;
} rac_sprite_sheet;

/* ── Animation ─────────────────────────────────────────────────────────── */

typedef struct {
    int           start_frame;    /* first frame index in sheet */
    int           num_frames;     /* number of frames in this animation */
    float         fps;            /* playback speed */
    int           looping;
} rac_sprite_anim;

#define RAC_MAX_ANIMS 16

typedef struct {
    int                sheet_id;
    rac_sprite_anim    anims[RAC_MAX_ANIMS];
    int                num_anims;
    int                current_anim;
    float              play_time;
    int                current_frame;
    int                playing;

    /* World-space transform */
    rac_phys_vec3      position;
    float              scale;
    int                flip_x;        /* mirror horizontally */
    int                billboard;     /* always face camera */
    int                visible;
} rac_sprite_instance;

/* ── Sprite registry ───────────────────────────────────────────────────── */

typedef struct {
    rac_sprite_sheet    sheets[RAC_MAX_SPRITES];
    int                 num_sheets;
    rac_sprite_instance instances[RAC_MAX_SPRITES];
    int                 num_instances;
} rac_sprite_registry;

/* ── API ───────────────────────────────────────────────────────────────── */

void rac_sprite_registry_init(rac_sprite_registry *reg);
void rac_sprite_registry_cleanup(rac_sprite_registry *reg);

/* Load sprite sheet from raw file: header(w,h,frames) + RGBA data */
int rac_sprite_load_sheet(rac_sprite_registry *reg, const char *path);

/* Create sheet from embedded RGBA data */
int rac_sprite_create_sheet(rac_sprite_registry *reg,
                            const uint8_t *rgba_data,
                            int frame_w, int frame_h, int num_frames);

/* Create a sprite instance */
int rac_sprite_create(rac_sprite_registry *reg, int sheet_id,
                      rac_phys_vec3 position, float scale);

/* Add animation to an instance */
int rac_sprite_add_anim(rac_sprite_registry *reg, int sprite_id,
                        int start_frame, int num_frames, float fps, int looping);

/* Play an animation */
void rac_sprite_play_anim(rac_sprite_registry *reg, int sprite_id, int anim_id);

/* Update all sprite animations */
void rac_sprite_update(rac_sprite_registry *reg, float dt);

/* Render all visible sprites to framebuffer.
 * Sprites are rendered as billboarded quads in 3D space using the
 * camera's view-projection matrix, with per-pixel alpha testing. */
void rac_sprite_render(rac_sprite_registry *reg,
                       rac_render_state *rs);

/* Render a single sprite frame directly to screen coordinates (2D overlay) */
void rac_sprite_draw_2d(rac_framebuffer *fb,
                        const rac_sprite_sheet *sheet, int frame,
                        int screen_x, int screen_y, int scale,
                        int flip_x);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_SPRITE_H */
