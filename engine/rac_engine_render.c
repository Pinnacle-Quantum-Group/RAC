/*
 * rac_engine_render.c — Software Rasterizer Implementation
 *
 * KEY: All vertex transforms via RAC primitives:
 *   - Model-view-projection via rac_mat4_mul (uses rac_dot internally)
 *   - Vertex projection via rac_mat4_transform_point (rac_dot pairs)
 *   - Edge functions via rac_dot (cross product sign)
 *   - Lighting via rac_phys_v3_dot (N·L, N·H)
 *   - Normalization via rac_phys_v3_normalize (rac_norm)
 *
 * Zero standalone multiply in the transform/lighting pipeline.
 */

#include "rac_engine_render.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Framebuffer ───────────────────────────────────────────────────────── */

rac_framebuffer *rac_framebuffer_create(int width, int height)
{
    rac_framebuffer *fb = (rac_framebuffer *)calloc(1, sizeof(rac_framebuffer));
    if (!fb) return NULL;

    fb->width = width;
    fb->height = height;
    fb->stride = width * 3;
    fb->pixels = (uint8_t *)calloc(width * height * 3, sizeof(uint8_t));
    fb->depth = (float *)malloc(width * height * sizeof(float));

    if (!fb->pixels || !fb->depth) {
        free(fb->pixels);
        free(fb->depth);
        free(fb);
        return NULL;
    }

    rac_framebuffer_clear(fb, 0, 0, 0);
    return fb;
}

void rac_framebuffer_destroy(rac_framebuffer *fb)
{
    if (!fb) return;
    free(fb->pixels);
    free(fb->depth);
    free(fb);
}

void rac_framebuffer_clear(rac_framebuffer *fb, uint8_t r, uint8_t g, uint8_t b)
{
    for (int i = 0; i < fb->width * fb->height; i++) {
        fb->pixels[i * 3 + 0] = r;
        fb->pixels[i * 3 + 1] = g;
        fb->pixels[i * 3 + 2] = b;
        fb->depth[i] = 1e30f;
    }
}

int rac_framebuffer_write_ppm(const rac_framebuffer *fb, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", fb->width, fb->height);
    fwrite(fb->pixels, 1, fb->width * fb->height * 3, f);
    fclose(f);
    return 0;
}

int rac_framebuffer_write_bmp(const rac_framebuffer *fb, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int row_pad = (4 - (fb->width * 3) % 4) % 4;
    int data_size = (fb->width * 3 + row_pad) * fb->height;
    int file_size = 54 + data_size;

    /* BMP header */
    uint8_t hdr[54];
    memset(hdr, 0, 54);
    hdr[0] = 'B'; hdr[1] = 'M';
    hdr[2] = file_size & 0xFF; hdr[3] = (file_size >> 8) & 0xFF;
    hdr[4] = (file_size >> 16) & 0xFF; hdr[5] = (file_size >> 24) & 0xFF;
    hdr[10] = 54;
    hdr[14] = 40;
    hdr[18] = fb->width & 0xFF; hdr[19] = (fb->width >> 8) & 0xFF;
    hdr[22] = fb->height & 0xFF; hdr[23] = (fb->height >> 8) & 0xFF;
    hdr[26] = 1;
    hdr[28] = 24;
    fwrite(hdr, 1, 54, f);

    /* BMP stores bottom-to-top, BGR */
    uint8_t pad[3] = {0, 0, 0};
    for (int y = fb->height - 1; y >= 0; y--) {
        for (int x = 0; x < fb->width; x++) {
            int idx = (y * fb->width + x) * 3;
            uint8_t bgr[3] = { fb->pixels[idx + 2], fb->pixels[idx + 1], fb->pixels[idx + 0] };
            fwrite(bgr, 1, 3, f);
        }
        if (row_pad > 0) fwrite(pad, 1, row_pad, f);
    }

    fclose(f);
    return 0;
}

/* ── Render state ──────────────────────────────────────────────────────── */

void rac_render_init(rac_render_state *rs, rac_framebuffer *fb,
                     rac_camera *cam, rac_light_registry *lights)
{
    memset(rs, 0, sizeof(*rs));
    rs->fb = fb;
    rs->camera = cam;
    rs->lights = lights;
    rs->shade_mode = RAC_SHADE_GOURAUD;
}

/* ── Clip-space vertex ─────────────────────────────────────────────────── */

typedef struct {
    rac_phys_vec3 world_pos;
    rac_phys_vec3 world_normal;
    float         clip_x, clip_y, clip_z, clip_w;
    float         screen_x, screen_y;
    float         inv_w;
    rac_color3f   color;   /* lit color (Gouraud) */
} rac_raster_vertex;

/* Transform vertex through MVP using rac_dot pairs */
static rac_raster_vertex transform_vertex(const rac_mat4 *mvp,
                                          const rac_mat4 *model,
                                          rac_phys_vec3 pos,
                                          rac_phys_vec3 normal,
                                          int fb_w, int fb_h)
{
    rac_raster_vertex rv;

    /* World position via model matrix (rac_dot-based) */
    rv.world_pos = rac_mat4_transform_point(*model, pos);
    rv.world_normal = rac_mat4_transform_dir(*model, normal);
    rv.world_normal = rac_phys_v3_normalize(rv.world_normal);

    /* Clip-space via MVP (rac_dot pairs for each row) */
    rac_vec2 px = { pos.x, pos.y };
    rac_vec2 pz = { pos.z, 1.0f };

    rac_vec2 r0xy = { mvp->m[0][0], mvp->m[0][1] };
    rac_vec2 r0zw = { mvp->m[0][2], mvp->m[0][3] };
    rv.clip_x = rac_dot(r0xy, px) + rac_dot(r0zw, pz);

    rac_vec2 r1xy = { mvp->m[1][0], mvp->m[1][1] };
    rac_vec2 r1zw = { mvp->m[1][2], mvp->m[1][3] };
    rv.clip_y = rac_dot(r1xy, px) + rac_dot(r1zw, pz);

    rac_vec2 r2xy = { mvp->m[2][0], mvp->m[2][1] };
    rac_vec2 r2zw = { mvp->m[2][2], mvp->m[2][3] };
    rv.clip_z = rac_dot(r2xy, px) + rac_dot(r2zw, pz);

    rac_vec2 r3xy = { mvp->m[3][0], mvp->m[3][1] };
    rac_vec2 r3zw = { mvp->m[3][2], mvp->m[3][3] };
    rv.clip_w = rac_dot(r3xy, px) + rac_dot(r3zw, pz);

    /* Perspective divide */
    if (rv.clip_w > 1e-6f || rv.clip_w < -1e-6f) {
        rv.inv_w = 1.0f / rv.clip_w;
    } else {
        rv.inv_w = 0.0f;
    }

    float ndc_x = rv.clip_x * rv.inv_w;
    float ndc_y = rv.clip_y * rv.inv_w;

    /* NDC to screen */
    rv.screen_x = (ndc_x + 1.0f) * 0.5f * (float)fb_w;
    rv.screen_y = (1.0f - ndc_y) * 0.5f * (float)fb_h;

    return rv;
}

/* ── Edge function via rac_dot ─────────────────────────────────────────── */

static float edge_function(float ax, float ay, float bx, float by,
                           float cx, float cy)
{
    /* (B-A) x (C-A) = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
     * Decompose into rac_dot pairs */
    rac_vec2 e1 = { bx - ax, -(by - ay) };
    rac_vec2 e2 = { cy - ay, cx - ax };
    return rac_dot(e1, e2);
}

/* ── Pixel writing ─────────────────────────────────────────────────────── */

static void set_pixel(rac_framebuffer *fb, int x, int y,
                      uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= fb->width || y < 0 || y >= fb->height) return;
    int idx = (y * fb->width + x) * 3;
    fb->pixels[idx + 0] = r;
    fb->pixels[idx + 1] = g;
    fb->pixels[idx + 2] = b;
}

/* ── Triangle rasterization with z-buffer ──────────────────────────────── */

static void rasterize_triangle(rac_render_state *rs,
                                rac_raster_vertex *v0,
                                rac_raster_vertex *v1,
                                rac_raster_vertex *v2)
{
    rac_framebuffer *fb = rs->fb;

    /* Bounding box */
    float minx = v0->screen_x;
    if (v1->screen_x < minx) minx = v1->screen_x;
    if (v2->screen_x < minx) minx = v2->screen_x;

    float maxx = v0->screen_x;
    if (v1->screen_x > maxx) maxx = v1->screen_x;
    if (v2->screen_x > maxx) maxx = v2->screen_x;

    float miny = v0->screen_y;
    if (v1->screen_y < miny) miny = v1->screen_y;
    if (v2->screen_y < miny) miny = v2->screen_y;

    float maxy = v0->screen_y;
    if (v1->screen_y > maxy) maxy = v1->screen_y;
    if (v2->screen_y > maxy) maxy = v2->screen_y;

    int ix0 = (int)minx; if (ix0 < 0) ix0 = 0;
    int iy0 = (int)miny; if (iy0 < 0) iy0 = 0;
    int ix1 = (int)maxx + 1; if (ix1 > fb->width) ix1 = fb->width;
    int iy1 = (int)maxy + 1; if (iy1 > fb->height) iy1 = fb->height;

    /* Triangle area via edge function (rac_dot).
     * Screen Y is inverted (top=0), so front-facing triangles have negative area.
     * We flip sign to normalize, and cull truly degenerate triangles. */
    float area = edge_function(v0->screen_x, v0->screen_y,
                               v1->screen_x, v1->screen_y,
                               v2->screen_x, v2->screen_y);

    /* Backface culling: accept both winding orders for robustness */
    if (area > 0.001f) {
        /* CW in screen = CCW in NDC = front-facing (some mesh generators) */
        /* Don't flip — area is already positive */
    } else if (area < -0.001f) {
        /* CCW in screen = CW in NDC (standard) — flip to positive */
        area = -area;
    } else {
        return;  /* degenerate */
    }

    float inv_area = 1.0f / area;

    /* Rasterize scanlines */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8) if(iy1 - iy0 > 32)
    #endif
    for (int y = iy0; y < iy1; y++) {
        for (int x = ix0; x < ix1; x++) {
            float px = (float)x + 0.5f;
            float py = (float)y + 0.5f;

            /* Barycentric coords via edge functions (rac_dot) */
            float w0 = edge_function(v1->screen_x, v1->screen_y,
                                     v2->screen_x, v2->screen_y, px, py);
            float w1 = edge_function(v2->screen_x, v2->screen_y,
                                     v0->screen_x, v0->screen_y, px, py);
            float w2 = edge_function(v0->screen_x, v0->screen_y,
                                     v1->screen_x, v1->screen_y, px, py);

            /* Accept if all same sign (handles both winding orders) */
            if (w0 > 0.0f && w1 > 0.0f && w2 > 0.0f) {
                /* positive winding — use as-is */
            } else if (w0 < 0.0f && w1 < 0.0f && w2 < 0.0f) {
                /* negative winding — negate to make positive */
                w0 = -w0; w1 = -w1; w2 = -w2;
            } else {
                continue;  /* outside triangle */
            }

            w0 *= inv_area;
            w1 *= inv_area;
            w2 *= inv_area;

            /* Perspective-correct interpolation using inv_w */
            float inv_z = w0 * v0->inv_w + w1 * v1->inv_w + w2 * v2->inv_w;
            if (inv_z <= 0.0f) continue;

            float depth = 1.0f / inv_z;

            /* Z-buffer test */
            int pidx = y * fb->width + x;
            if (depth >= fb->depth[pidx]) continue;

            /* Interpolate color (Gouraud) */
            float r, g, b;
            if (rs->shade_mode == RAC_SHADE_GOURAUD) {
                float pw0 = w0 * v0->inv_w * depth;
                float pw1 = w1 * v1->inv_w * depth;
                float pw2 = w2 * v2->inv_w * depth;
                r = pw0 * v0->color.r + pw1 * v1->color.r + pw2 * v2->color.r;
                g = pw0 * v0->color.g + pw1 * v1->color.g + pw2 * v2->color.g;
                b = pw0 * v0->color.b + pw1 * v1->color.b + pw2 * v2->color.b;
            } else {
                /* Flat: use v0 color for entire triangle */
                r = v0->color.r;
                g = v0->color.g;
                b = v0->color.b;
            }

            /* Write pixel */
            fb->depth[pidx] = depth;
            int rgb_idx = pidx * 3;
            fb->pixels[rgb_idx + 0] = (uint8_t)(r * 255.0f);
            fb->pixels[rgb_idx + 1] = (uint8_t)(g * 255.0f);
            fb->pixels[rgb_idx + 2] = (uint8_t)(b * 255.0f);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            rs->pixels_drawn++;
        }
    }
}

/* ── Public rendering API ──────────────────────────────────────────────── */

void rac_render_mesh(rac_render_state *rs,
                     const rac_mesh *mesh,
                     rac_mat4 model_matrix,
                     rac_color3f color)
{
    if (!mesh || !mesh->valid || mesh->num_indices < 3) return;

    rac_camera *cam = rs->camera;
    rac_mat4 mvp = rac_mat4_mul(cam->view_proj, model_matrix);
    rac_phys_vec3 cam_pos = cam->position;

    for (int i = 0; i < mesh->num_indices; i += 3) {
        rs->triangles_submitted++;

        int i0 = mesh->indices[i];
        int i1 = mesh->indices[i + 1];
        int i2 = mesh->indices[i + 2];

        /* Transform vertices through MVP (all via rac_dot) */
        rac_raster_vertex v0 = transform_vertex(&mvp, &model_matrix,
            mesh->vertices[i0].position, mesh->vertices[i0].normal,
            rs->fb->width, rs->fb->height);
        rac_raster_vertex v1 = transform_vertex(&mvp, &model_matrix,
            mesh->vertices[i1].position, mesh->vertices[i1].normal,
            rs->fb->width, rs->fb->height);
        rac_raster_vertex v2 = transform_vertex(&mvp, &model_matrix,
            mesh->vertices[i2].position, mesh->vertices[i2].normal,
            rs->fb->width, rs->fb->height);

        /* Near-plane clip: skip if any vertex behind camera */
        if (v0.clip_w < 0.01f || v1.clip_w < 0.01f || v2.clip_w < 0.01f)
            continue;

        /* Per-vertex lighting (Gouraud) */
        rac_phys_vec3 view_dir0 = rac_phys_v3_sub(cam_pos, v0.world_pos);
        rac_phys_vec3 view_dir1 = rac_phys_v3_sub(cam_pos, v1.world_pos);
        rac_phys_vec3 view_dir2 = rac_phys_v3_sub(cam_pos, v2.world_pos);

        v0.color = rac_light_compute(rs->lights, v0.world_pos, v0.world_normal,
                                     view_dir0, color, 32.0f);
        v1.color = rac_light_compute(rs->lights, v1.world_pos, v1.world_normal,
                                     view_dir1, color, 32.0f);
        v2.color = rac_light_compute(rs->lights, v2.world_pos, v2.world_normal,
                                     view_dir2, color, 32.0f);

        rasterize_triangle(rs, &v0, &v1, &v2);
        rs->triangles_drawn++;
    }
}

void rac_render_triangle_world(rac_render_state *rs,
                               rac_phys_vec3 v0, rac_phys_vec3 v1, rac_phys_vec3 v2,
                               rac_phys_vec3 n0, rac_phys_vec3 n1, rac_phys_vec3 n2,
                               rac_color3f color)
{
    rac_camera *cam = rs->camera;
    rac_mat4 model = rac_mat4_identity();
    rac_mat4 mvp = cam->view_proj;  /* model is identity */
    rac_phys_vec3 cam_pos = cam->position;

    rac_raster_vertex rv0 = transform_vertex(&mvp, &model, v0, n0,
        rs->fb->width, rs->fb->height);
    rac_raster_vertex rv1 = transform_vertex(&mvp, &model, v1, n1,
        rs->fb->width, rs->fb->height);
    rac_raster_vertex rv2 = transform_vertex(&mvp, &model, v2, n2,
        rs->fb->width, rs->fb->height);

    if (rv0.clip_w < 0.01f || rv1.clip_w < 0.01f || rv2.clip_w < 0.01f)
        return;

    rac_phys_vec3 vd0 = rac_phys_v3_sub(cam_pos, rv0.world_pos);
    rac_phys_vec3 vd1 = rac_phys_v3_sub(cam_pos, rv1.world_pos);
    rac_phys_vec3 vd2 = rac_phys_v3_sub(cam_pos, rv2.world_pos);

    rv0.color = rac_light_compute(rs->lights, rv0.world_pos, rv0.world_normal,
                                  vd0, color, 32.0f);
    rv1.color = rac_light_compute(rs->lights, rv1.world_pos, rv1.world_normal,
                                  vd1, color, 32.0f);
    rv2.color = rac_light_compute(rs->lights, rv2.world_pos, rv2.world_normal,
                                  vd2, color, 32.0f);

    rasterize_triangle(rs, &rv0, &rv1, &rv2);
}

void rac_render_particles(rac_render_state *rs,
                          const rac_phys_vec3 *positions,
                          int count, float point_size,
                          rac_color3f color)
{
    rac_camera *cam = rs->camera;
    rac_mat4 vp = cam->view_proj;
    int w = rs->fb->width, h = rs->fb->height;

    for (int i = 0; i < count; i++) {
        /* Project particle center using rac_dot pairs */
        rac_phys_vec3 p = positions[i];
        rac_vec2 px = { p.x, p.y };
        rac_vec2 pz = { p.z, 1.0f };

        rac_vec2 r3xy = { vp.m[3][0], vp.m[3][1] };
        rac_vec2 r3zw = { vp.m[3][2], vp.m[3][3] };
        float cw = rac_dot(r3xy, px) + rac_dot(r3zw, pz);
        if (cw < 0.01f) continue;

        float inv_w = 1.0f / cw;

        rac_vec2 r0xy = { vp.m[0][0], vp.m[0][1] };
        rac_vec2 r0zw = { vp.m[0][2], vp.m[0][3] };
        float cx = rac_dot(r0xy, px) + rac_dot(r0zw, pz);

        rac_vec2 r1xy = { vp.m[1][0], vp.m[1][1] };
        rac_vec2 r1zw = { vp.m[1][2], vp.m[1][3] };
        float cy = rac_dot(r1xy, px) + rac_dot(r1zw, pz);

        int sx = (int)((cx * inv_w + 1.0f) * 0.5f * (float)w);
        int sy = (int)((1.0f - cy * inv_w) * 0.5f * (float)h);

        /* Draw a small quad */
        int half = (int)(point_size * 0.5f * inv_w * 100.0f);
        if (half < 1) half = 1;
        if (half > 8) half = 8;

        uint8_t pr = (uint8_t)(color.r * 255.0f);
        uint8_t pg = (uint8_t)(color.g * 255.0f);
        uint8_t pb = (uint8_t)(color.b * 255.0f);

        for (int dy = -half; dy <= half; dy++) {
            for (int dx = -half; dx <= half; dx++) {
                int px2 = sx + dx, py2 = sy + dy;
                if (px2 >= 0 && px2 < w && py2 >= 0 && py2 < h) {
                    int pidx = py2 * w + px2;
                    float depth = cw;
                    if (depth < rs->fb->depth[pidx]) {
                        rs->fb->depth[pidx] = depth;
                        set_pixel(rs->fb, px2, py2, pr, pg, pb);
                    }
                }
            }
        }
    }
}

void rac_render_line(rac_framebuffer *fb,
                     int x0, int y0, int x1, int y1,
                     uint8_t r, uint8_t g, uint8_t b)
{
    /* Bresenham's line algorithm */
    int dx = x1 - x0;
    int dy = y1 - y0;
    int sx = (dx > 0) ? 1 : -1;
    int sy = (dy > 0) ? 1 : -1;
    dx = (dx < 0) ? -dx : dx;
    dy = (dy < 0) ? -dy : dy;

    int err = dx - dy;

    while (1) {
        set_pixel(fb, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}

void rac_render_shadow_pass(rac_light *light,
                            const rac_mesh *mesh,
                            rac_mat4 model_matrix,
                            rac_mat4 light_view_proj)
{
    if (!light->shadow_depth || !mesh || !mesh->valid) return;

    int res = light->shadow_resolution;
    rac_mat4 mvp = rac_mat4_mul(light_view_proj, model_matrix);

    for (int i = 0; i < mesh->num_indices; i += 3) {
        rac_phys_vec3 v[3];
        float sx[3], sy[3], sz[3], sw[3];

        for (int k = 0; k < 3; k++) {
            int idx = mesh->indices[i + k];
            rac_phys_vec3 p = mesh->vertices[idx].position;

            /* Project using rac_dot pairs */
            rac_vec2 px = { p.x, p.y };
            rac_vec2 pz = { p.z, 1.0f };

            rac_vec2 r3xy = { mvp.m[3][0], mvp.m[3][1] };
            rac_vec2 r3zw = { mvp.m[3][2], mvp.m[3][3] };
            sw[k] = rac_dot(r3xy, px) + rac_dot(r3zw, pz);
            if (sw[k] < 0.01f) goto next_tri;

            float inv_w = 1.0f / sw[k];

            rac_vec2 r0xy = { mvp.m[0][0], mvp.m[0][1] };
            rac_vec2 r0zw = { mvp.m[0][2], mvp.m[0][3] };
            sx[k] = (rac_dot(r0xy, px) + rac_dot(r0zw, pz)) * inv_w;

            rac_vec2 r1xy = { mvp.m[1][0], mvp.m[1][1] };
            rac_vec2 r1zw = { mvp.m[1][2], mvp.m[1][3] };
            sy[k] = (rac_dot(r1xy, px) + rac_dot(r1zw, pz)) * inv_w;

            rac_vec2 r2xy = { mvp.m[2][0], mvp.m[2][1] };
            rac_vec2 r2zw = { mvp.m[2][2], mvp.m[2][3] };
            sz[k] = (rac_dot(r2xy, px) + rac_dot(r2zw, pz)) * inv_w;

            /* NDC to shadow map coords */
            sx[k] = (sx[k] + 1.0f) * 0.5f * (float)res;
            sy[k] = (1.0f - sy[k]) * 0.5f * (float)res;
            v[k] = p;
        }

        /* Simple rasterization for shadow depth */
        {
            int minx = (int)sx[0], miny = (int)sy[0];
            int maxx = minx, maxy = miny;
            for (int k = 1; k < 3; k++) {
                if ((int)sx[k] < minx) minx = (int)sx[k];
                if ((int)sy[k] < miny) miny = (int)sy[k];
                if ((int)sx[k] > maxx) maxx = (int)sx[k];
                if ((int)sy[k] > maxy) maxy = (int)sy[k];
            }
            if (minx < 0) minx = 0;
            if (miny < 0) miny = 0;
            if (maxx >= res) maxx = res - 1;
            if (maxy >= res) maxy = res - 1;

            float area = edge_function(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]);
            if (area <= 0.0f) continue;
            float inv_area = 1.0f / area;

            for (int py = miny; py <= maxy; py++) {
                for (int px = minx; px <= maxx; px++) {
                    float fpx = (float)px + 0.5f;
                    float fpy = (float)py + 0.5f;
                    float w0 = edge_function(sx[1], sy[1], sx[2], sy[2], fpx, fpy);
                    float w1 = edge_function(sx[2], sy[2], sx[0], sy[0], fpx, fpy);
                    float w2 = edge_function(sx[0], sy[0], sx[1], sy[1], fpx, fpy);
                    if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                    w0 *= inv_area; w1 *= inv_area; w2 *= inv_area;
                    float depth = w0 * sz[0] + w1 * sz[1] + w2 * sz[2];
                    int pidx = py * res + px;
                    if (depth < light->shadow_depth[pidx])
                        light->shadow_depth[pidx] = depth;
                }
            }
        }

        next_tri:;
    }
}
