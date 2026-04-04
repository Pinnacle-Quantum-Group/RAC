/*
 * rac_engine_sprite.c — 2D Sprite System Implementation
 *
 * Renders animated sprites as billboarded quads in 3D space or
 * as 2D overlays. All projection math via RAC primitives.
 */

#include "rac_engine_sprite.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Registry ──────────────────────────────────────────────────────────── */

void rac_sprite_registry_init(rac_sprite_registry *reg)
{
    memset(reg, 0, sizeof(*reg));
}

void rac_sprite_registry_cleanup(rac_sprite_registry *reg)
{
    for (int i = 0; i < reg->num_sheets; i++)
        free(reg->sheets[i].pixels);
    memset(reg, 0, sizeof(*reg));
}

/* ── Sheet loading ─────────────────────────────────────────────────────── */

int rac_sprite_load_sheet(rac_sprite_registry *reg, const char *path)
{
    if (reg->num_sheets >= RAC_MAX_SPRITES) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Read header: frame_w(4), frame_h(4), num_frames(4) */
    uint32_t fw, fh, nf;
    if (fread(&fw, 4, 1, f) != 1 || fread(&fh, 4, 1, f) != 1 ||
        fread(&nf, 4, 1, f) != 1) {
        fclose(f);
        return -1;
    }

    int data_size = fw * nf * fh * 4;  /* RGBA */
    uint8_t *pixels = (uint8_t *)malloc(data_size);
    if (!pixels) { fclose(f); return -1; }

    if ((int)fread(pixels, 1, data_size, f) != data_size) {
        free(pixels);
        fclose(f);
        return -1;
    }
    fclose(f);

    int id = reg->num_sheets++;
    rac_sprite_sheet *s = &reg->sheets[id];
    s->pixels = pixels;
    s->frame_w = fw;
    s->frame_h = fh;
    s->sheet_w = fw * nf;
    s->sheet_h = fh;
    s->num_frames = nf;
    s->valid = 1;

    printf("[Sprite] Loaded sheet %d: %dx%d, %d frames (%dx%d each)\n",
           id, s->sheet_w, s->sheet_h, nf, fw, fh);
    return id;
}

int rac_sprite_create_sheet(rac_sprite_registry *reg,
                            const uint8_t *rgba_data,
                            int frame_w, int frame_h, int num_frames)
{
    if (reg->num_sheets >= RAC_MAX_SPRITES) return -1;

    int data_size = frame_w * num_frames * frame_h * 4;
    uint8_t *pixels = (uint8_t *)malloc(data_size);
    if (!pixels) return -1;
    memcpy(pixels, rgba_data, data_size);

    int id = reg->num_sheets++;
    rac_sprite_sheet *s = &reg->sheets[id];
    s->pixels = pixels;
    s->frame_w = frame_w;
    s->frame_h = frame_h;
    s->sheet_w = frame_w * num_frames;
    s->sheet_h = frame_h;
    s->num_frames = num_frames;
    s->valid = 1;
    return id;
}

/* ── Instance management ───────────────────────────────────────────────── */

int rac_sprite_create(rac_sprite_registry *reg, int sheet_id,
                      rac_phys_vec3 position, float scale)
{
    if (reg->num_instances >= RAC_MAX_SPRITES) return -1;
    if (sheet_id < 0 || sheet_id >= reg->num_sheets) return -1;

    int id = reg->num_instances++;
    rac_sprite_instance *si = &reg->instances[id];
    memset(si, 0, sizeof(*si));
    si->sheet_id = sheet_id;
    si->position = position;
    si->scale = scale;
    si->visible = 1;
    si->billboard = 1;
    return id;
}

int rac_sprite_add_anim(rac_sprite_registry *reg, int sprite_id,
                        int start_frame, int num_frames, float fps, int looping)
{
    if (sprite_id < 0 || sprite_id >= reg->num_instances) return -1;
    rac_sprite_instance *si = &reg->instances[sprite_id];
    if (si->num_anims >= RAC_MAX_ANIMS) return -1;

    int id = si->num_anims++;
    si->anims[id].start_frame = start_frame;
    si->anims[id].num_frames = num_frames;
    si->anims[id].fps = fps;
    si->anims[id].looping = looping;
    return id;
}

void rac_sprite_play_anim(rac_sprite_registry *reg, int sprite_id, int anim_id)
{
    if (sprite_id < 0 || sprite_id >= reg->num_instances) return;
    rac_sprite_instance *si = &reg->instances[sprite_id];
    si->current_anim = anim_id;
    si->play_time = 0.0f;
    si->playing = 1;
}

/* ── Animation update ──────────────────────────────────────────────────── */

void rac_sprite_update(rac_sprite_registry *reg, float dt)
{
    for (int i = 0; i < reg->num_instances; i++) {
        rac_sprite_instance *si = &reg->instances[i];
        if (!si->visible || !si->playing) continue;
        if (si->current_anim < 0 || si->current_anim >= si->num_anims) continue;

        rac_sprite_anim *anim = &si->anims[si->current_anim];
        si->play_time += dt;

        float frame_time = 1.0f / anim->fps;
        int frame_idx = (int)(si->play_time / frame_time);

        if (anim->looping) {
            frame_idx = frame_idx % anim->num_frames;
        } else {
            if (frame_idx >= anim->num_frames) {
                frame_idx = anim->num_frames - 1;
                si->playing = 0;
            }
        }

        si->current_frame = anim->start_frame + frame_idx;
    }
}

/* ── 2D sprite drawing (screen-space overlay) ──────────────────────────── */

void rac_sprite_draw_2d(rac_framebuffer *fb,
                        const rac_sprite_sheet *sheet, int frame,
                        int screen_x, int screen_y, int scale,
                        int flip_x)
{
    if (!sheet || !sheet->valid) return;
    if (frame < 0 || frame >= sheet->num_frames) return;
    if (scale < 1) scale = 1;

    int fw = sheet->frame_w;
    int fh = sheet->frame_h;
    int sheet_x_offset = frame * fw;

    for (int sy = 0; sy < fh; sy++) {
        for (int sx = 0; sx < fw; sx++) {
            /* Read pixel from sprite sheet */
            int src_x = flip_x ? (fw - 1 - sx) : sx;
            int src_idx = (sy * sheet->sheet_w + sheet_x_offset + src_x) * 4;
            uint8_t r = sheet->pixels[src_idx + 0];
            uint8_t g = sheet->pixels[src_idx + 1];
            uint8_t b = sheet->pixels[src_idx + 2];
            uint8_t a = sheet->pixels[src_idx + 3];

            /* Alpha test: skip transparent pixels */
            if (a < 128) continue;

            /* Draw scaled pixel block */
            for (int py = 0; py < scale; py++) {
                for (int px = 0; px < scale; px++) {
                    int dx = screen_x + sx * scale + px;
                    int dy = screen_y + sy * scale + py;
                    if (dx >= 0 && dx < fb->width && dy >= 0 && dy < fb->height) {
                        int dst_idx = (dy * fb->width + dx) * 3;
                        /* Alpha blend using shift approximation */
                        if (a >= 240) {
                            fb->pixels[dst_idx + 0] = r;
                            fb->pixels[dst_idx + 1] = g;
                            fb->pixels[dst_idx + 2] = b;
                        } else {
                            /* Simple alpha blend: out = src*a + dst*(1-a) */
                            int inv_a = 255 - a;
                            fb->pixels[dst_idx + 0] = (uint8_t)((r * a + fb->pixels[dst_idx + 0] * inv_a) >> 8);
                            fb->pixels[dst_idx + 1] = (uint8_t)((g * a + fb->pixels[dst_idx + 1] * inv_a) >> 8);
                            fb->pixels[dst_idx + 2] = (uint8_t)((b * a + fb->pixels[dst_idx + 2] * inv_a) >> 8);
                        }
                    }
                }
            }
        }
    }
}

/* ── 3D sprite rendering (billboarded quads) ───────────────────────────── */

void rac_sprite_render(rac_sprite_registry *reg, rac_render_state *rs)
{
    if (!reg || !rs || !rs->camera) return;

    rac_camera *cam = rs->camera;
    rac_framebuffer *fb = rs->fb;

    for (int i = 0; i < reg->num_instances; i++) {
        rac_sprite_instance *si = &reg->instances[i];
        if (!si->visible) continue;
        if (si->sheet_id < 0 || si->sheet_id >= reg->num_sheets) continue;

        rac_sprite_sheet *sheet = &reg->sheets[si->sheet_id];
        if (!sheet->valid) continue;

        /* Project sprite center to screen using rac_dot pairs */
        rac_phys_vec3 pos = si->position;
        rac_mat4 vp = cam->view_proj;

        rac_vec2 px = { pos.x, pos.y };
        rac_vec2 pz = { pos.z, 1.0f };

        rac_vec2 r3xy = { vp.m[3][0], vp.m[3][1] };
        rac_vec2 r3zw = { vp.m[3][2], vp.m[3][3] };
        float cw = rac_dot(r3xy, px) + rac_dot(r3zw, pz);
        if (cw < 0.1f) continue;  /* behind camera */

        float inv_w = 1.0f / cw;

        rac_vec2 r0xy = { vp.m[0][0], vp.m[0][1] };
        rac_vec2 r0zw = { vp.m[0][2], vp.m[0][3] };
        float cx = rac_dot(r0xy, px) + rac_dot(r0zw, pz);

        rac_vec2 r1xy = { vp.m[1][0], vp.m[1][1] };
        rac_vec2 r1zw = { vp.m[1][2], vp.m[1][3] };
        float cy = rac_dot(r1xy, px) + rac_dot(r1zw, pz);

        /* NDC to screen */
        int screen_cx = (int)((cx * inv_w + 1.0f) * 0.5f * (float)fb->width);
        int screen_cy = (int)((1.0f - cy * inv_w) * 0.5f * (float)fb->height);

        /* Sprite screen size based on distance and scale */
        float screen_scale = si->scale * inv_w * 200.0f;
        int pixel_size = (int)screen_scale;
        if (pixel_size < 4) pixel_size = 4;
        if (pixel_size > 400) pixel_size = 400;

        /* Calculate per-pixel scale factor */
        int draw_scale = pixel_size / sheet->frame_w;
        if (draw_scale < 1) draw_scale = 1;

        /* Draw sprite centered at projected position */
        int draw_x = screen_cx - (sheet->frame_w * draw_scale) / 2;
        int draw_y = screen_cy - (sheet->frame_h * draw_scale) / 2;

        rac_sprite_draw_2d(fb, sheet, si->current_frame,
                           draw_x, draw_y, draw_scale, si->flip_x);
    }
}
