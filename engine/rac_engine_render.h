/*
 * rac_engine_render.h — Software Rasterizer
 * All vertex transforms via rac_rotate/rac_project/rac_dot.
 * Triangle rasterization with z-buffer, flat and Gouraud shading.
 */

#ifndef RAC_ENGINE_RENDER_H
#define RAC_ENGINE_RENDER_H

#include "rac_engine_camera.h"
#include "rac_engine_mesh.h"
#include "rac_engine_light.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Framebuffer ───────────────────────────────────────────────────────── */

typedef struct {
    uint8_t *pixels;       /* RGB888 pixel data */
    float   *depth;        /* z-buffer (per pixel) */
    int      width;
    int      height;
    int      stride;       /* bytes per row (width * 3) */
} rac_framebuffer;

rac_framebuffer *rac_framebuffer_create(int width, int height);
void rac_framebuffer_destroy(rac_framebuffer *fb);
void rac_framebuffer_clear(rac_framebuffer *fb, uint8_t r, uint8_t g, uint8_t b);

/* Write to PPM file */
int rac_framebuffer_write_ppm(const rac_framebuffer *fb, const char *path);

/* Write to BMP file */
int rac_framebuffer_write_bmp(const rac_framebuffer *fb, const char *path);

/* ── Shading mode ──────────────────────────────────────────────────────── */

typedef enum {
    RAC_SHADE_FLAT    = 0,   /* one color per triangle */
    RAC_SHADE_GOURAUD = 1,   /* per-vertex lighting, interpolated */
} rac_shade_mode;

/* ── Render state ──────────────────────────────────────────────────────── */

typedef struct {
    rac_framebuffer     *fb;
    rac_camera          *camera;
    rac_light_registry  *lights;
    rac_shade_mode       shade_mode;

    /* Stats */
    int triangles_submitted;
    int triangles_drawn;
    int pixels_drawn;
} rac_render_state;

void rac_render_init(rac_render_state *rs, rac_framebuffer *fb,
                     rac_camera *cam, rac_light_registry *lights);

/* ── Rendering API ─────────────────────────────────────────────────────── */

/* Render a mesh with a model transform (world matrix).
 * All vertex projection via RAC primitives. */
void rac_render_mesh(rac_render_state *rs,
                     const rac_mesh *mesh,
                     rac_mat4 model_matrix,
                     rac_color3f color);

/* Render a single triangle (already in world space) */
void rac_render_triangle_world(rac_render_state *rs,
                               rac_phys_vec3 v0, rac_phys_vec3 v1, rac_phys_vec3 v2,
                               rac_phys_vec3 n0, rac_phys_vec3 n1, rac_phys_vec3 n2,
                               rac_color3f color);

/* Render particles as points/small quads */
void rac_render_particles(rac_render_state *rs,
                          const rac_phys_vec3 *positions,
                          int count, float point_size,
                          rac_color3f color);

/* Draw a line (Bresenham) */
void rac_render_line(rac_framebuffer *fb,
                     int x0, int y0, int x1, int y1,
                     uint8_t r, uint8_t g, uint8_t b);

/* ── Shadow pass ───────────────────────────────────────────────────────── */

/* Render depth from light's perspective into shadow map */
void rac_render_shadow_pass(rac_light *light,
                            const rac_mesh *mesh,
                            rac_mat4 model_matrix,
                            rac_mat4 light_view_proj);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_RENDER_H */
