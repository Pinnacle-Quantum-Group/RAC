/*
 * rac_engine_light.c — Lighting System Implementation
 * Blinn-Phong: N·L, N·H via rac_phys_v3_dot, attenuation via rac_norm.
 */

#include "rac_engine_light.h"
#include <stdlib.h>
#include <string.h>

void rac_light_registry_init(rac_light_registry *reg)
{
    memset(reg, 0, sizeof(*reg));
    reg->ambient_r = 0.1f;
    reg->ambient_g = 0.1f;
    reg->ambient_b = 0.1f;
}

int rac_light_create_directional(rac_light_registry *reg,
                                 rac_phys_vec3 direction,
                                 float r, float g, float b, float intensity)
{
    if (reg->num_lights >= RAC_MAX_LIGHTS) return -1;
    int id = reg->num_lights++;
    rac_light *l = &reg->lights[id];
    memset(l, 0, sizeof(*l));

    l->type = RAC_LIGHT_DIRECTIONAL;
    l->active = 1;
    l->direction = rac_phys_v3_normalize(direction);
    l->color_r = r; l->color_g = g; l->color_b = b;
    l->intensity = intensity;
    return id;
}

int rac_light_create_point(rac_light_registry *reg,
                           rac_phys_vec3 position,
                           float r, float g, float b, float intensity,
                           float range)
{
    if (reg->num_lights >= RAC_MAX_LIGHTS) return -1;
    int id = reg->num_lights++;
    rac_light *l = &reg->lights[id];
    memset(l, 0, sizeof(*l));

    l->type = RAC_LIGHT_POINT;
    l->active = 1;
    l->position = position;
    l->color_r = r; l->color_g = g; l->color_b = b;
    l->intensity = intensity;
    l->range = range;
    l->constant_atten = 1.0f;
    l->linear_atten = 2.0f / range;
    l->quadratic_atten = 1.0f / (range * range);
    return id;
}

void rac_light_set_ambient(rac_light_registry *reg, float r, float g, float b)
{
    reg->ambient_r = r;
    reg->ambient_g = g;
    reg->ambient_b = b;
}

static float clamp01(float v)
{
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

rac_color3f rac_light_compute(const rac_light_registry *reg,
                              rac_phys_vec3 surface_pos,
                              rac_phys_vec3 surface_normal,
                              rac_phys_vec3 view_dir,
                              rac_color3f surface_color,
                              float shininess)
{
    rac_color3f result;
    result.r = reg->ambient_r * surface_color.r;
    result.g = reg->ambient_g * surface_color.g;
    result.b = reg->ambient_b * surface_color.b;

    rac_phys_vec3 N = rac_phys_v3_normalize(surface_normal);
    rac_phys_vec3 V = rac_phys_v3_normalize(view_dir);

    for (int i = 0; i < reg->num_lights; i++) {
        const rac_light *light = &reg->lights[i];
        if (!light->active) continue;

        rac_phys_vec3 L;
        float attenuation = 1.0f;

        if (light->type == RAC_LIGHT_DIRECTIONAL) {
            L = rac_phys_v3_negate(light->direction);
        } else {
            /* Point light: direction from surface to light */
            rac_phys_vec3 to_light = rac_phys_v3_sub(light->position, surface_pos);
            float dist = rac_phys_v3_length(to_light);  /* RAC-native via rac_norm */
            if (dist > light->range) continue;
            L = rac_phys_v3_scale(to_light, 1.0f / (dist + 1e-6f));

            /* Attenuation via rac_norm distance */
            attenuation = 1.0f / (light->constant_atten +
                                  light->linear_atten * dist +
                                  light->quadratic_atten * dist * dist);
        }

        /* Diffuse: N·L via rac_phys_v3_dot (RAC-native) */
        float NdotL = rac_phys_v3_dot(N, L);
        if (NdotL <= 0.0f) continue;

        float diff = NdotL * light->intensity * attenuation;

        /* Specular: Blinn-Phong half-vector */
        rac_phys_vec3 H = rac_phys_v3_normalize(rac_phys_v3_add(L, V));
        float NdotH = rac_phys_v3_dot(N, H);
        float spec = 0.0f;
        if (NdotH > 0.0f) {
            /* pow via repeated rac_exp: pow(x,n) = exp(n*ln(x))
             * Approximate: use repeated squaring for small integer shininess */
            spec = NdotH;
            for (int p = 1; p < (int)shininess && p < 64; p++)
                spec = rac_dot((rac_vec2){spec, 0.0f}, (rac_vec2){NdotH, 0.0f});
            spec *= light->intensity * attenuation;
        }

        result.r += (diff * surface_color.r + spec) * light->color_r;
        result.g += (diff * surface_color.g + spec) * light->color_g;
        result.b += (diff * surface_color.b + spec) * light->color_b;
    }

    result.r = clamp01(result.r);
    result.g = clamp01(result.g);
    result.b = clamp01(result.b);
    return result;
}

rac_color3f rac_light_compute_flat(const rac_light_registry *reg,
                                   rac_phys_vec3 face_center,
                                   rac_phys_vec3 face_normal,
                                   rac_color3f surface_color)
{
    /* Simplified: no specular, view direction not needed */
    rac_phys_vec3 dummy_view = rac_phys_v3(0.0f, 0.0f, 1.0f);
    return rac_light_compute(reg, face_center, face_normal, dummy_view,
                             surface_color, 1.0f);
}

int rac_light_init_shadow_map(rac_light *light, int resolution)
{
    light->shadow_depth = (float *)calloc(resolution * resolution, sizeof(float));
    if (!light->shadow_depth) return -1;
    light->shadow_resolution = resolution;
    light->cast_shadows = 1;

    /* Initialize to max depth */
    for (int i = 0; i < resolution * resolution; i++)
        light->shadow_depth[i] = 1e30f;

    return 0;
}

void rac_light_destroy_shadow_map(rac_light *light)
{
    free(light->shadow_depth);
    light->shadow_depth = NULL;
    light->cast_shadows = 0;
}
