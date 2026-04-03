/*
 * rac_engine_light.h — Lighting System
 * Directional, point, ambient lights. Blinn-Phong shading.
 * All dot products via rac_phys_v3_dot, distances via rac_phys_v3_length.
 */

#ifndef RAC_ENGINE_LIGHT_H
#define RAC_ENGINE_LIGHT_H

#include "rac_physics.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Light types ───────────────────────────────────────────────────────── */

typedef enum {
    RAC_LIGHT_DIRECTIONAL = 0,
    RAC_LIGHT_POINT       = 1,
} rac_light_type;

#define RAC_MAX_LIGHTS 8

typedef struct {
    rac_light_type  type;
    int             active;

    /* Directional: direction (normalized), Point: position */
    rac_phys_vec3   direction;
    rac_phys_vec3   position;

    /* Color (RGB, 0-1 range) */
    float           color_r, color_g, color_b;
    float           intensity;

    /* Point light attenuation */
    float           range;
    float           constant_atten;
    float           linear_atten;
    float           quadratic_atten;

    /* Shadow mapping */
    int             cast_shadows;
    float          *shadow_depth;    /* depth buffer for shadow map */
    int             shadow_resolution;
} rac_light;

typedef struct {
    rac_light lights[RAC_MAX_LIGHTS];
    int       num_lights;

    /* Global ambient */
    float     ambient_r, ambient_g, ambient_b;
} rac_light_registry;

/* ── API ───────────────────────────────────────────────────────────────── */

void rac_light_registry_init(rac_light_registry *reg);
int  rac_light_create_directional(rac_light_registry *reg,
                                  rac_phys_vec3 direction,
                                  float r, float g, float b, float intensity);
int  rac_light_create_point(rac_light_registry *reg,
                            rac_phys_vec3 position,
                            float r, float g, float b, float intensity,
                            float range);
void rac_light_set_ambient(rac_light_registry *reg, float r, float g, float b);

/* ── Lighting computation (per-vertex or per-pixel) ────────────────────── */

typedef struct {
    float r, g, b;
} rac_color3f;

/* Compute lit color at a surface point using Blinn-Phong.
 * All dot products via rac_phys_v3_dot, normalization via rac_phys_v3_normalize. */
rac_color3f rac_light_compute(const rac_light_registry *reg,
                              rac_phys_vec3 surface_pos,
                              rac_phys_vec3 surface_normal,
                              rac_phys_vec3 view_dir,
                              rac_color3f surface_color,
                              float shininess);

/* Compute flat shading for a triangle (single color for face) */
rac_color3f rac_light_compute_flat(const rac_light_registry *reg,
                                   rac_phys_vec3 face_center,
                                   rac_phys_vec3 face_normal,
                                   rac_color3f surface_color);

/* ── Shadow map ────────────────────────────────────────────────────────── */

int  rac_light_init_shadow_map(rac_light *light, int resolution);
void rac_light_destroy_shadow_map(rac_light *light);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_LIGHT_H */
