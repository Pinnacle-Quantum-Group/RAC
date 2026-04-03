/*
 * rac_engine_ecs.h — Entity-Component System
 * Lightweight ECS: entities are integer IDs, components in typed arrays.
 */

#ifndef RAC_ENGINE_ECS_H
#define RAC_ENGINE_ECS_H

#include "rac_physics.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Limits ────────────────────────────────────────────────────────────── */

#define RAC_ECS_MAX_ENTITIES    16384
#define RAC_ECS_INVALID_ENTITY  UINT32_MAX

/* ── Component bitmask ─────────────────────────────────────────────────── */

typedef enum {
    RAC_COMP_NONE          = 0,
    RAC_COMP_TRANSFORM     = (1 << 0),
    RAC_COMP_RIGIDBODY     = (1 << 1),
    RAC_COMP_MESH_RENDERER = (1 << 2),
    RAC_COMP_CAMERA        = (1 << 3),
    RAC_COMP_LIGHT         = (1 << 4),
    RAC_COMP_AUDIO_SOURCE  = (1 << 5),
    RAC_COMP_PARENT        = (1 << 6),
} rac_component_flag;

/* ── Component data types ──────────────────────────────────────────────── */

typedef struct {
    rac_phys_vec3  position;
    rac_phys_quat  rotation;
    rac_phys_vec3  scale;
} rac_transform_component;

typedef struct {
    int  physics_body_index;   /* index into physics world */
    int  sync_to_transform;    /* 1 = copy physics state to transform each frame */
} rac_rigidbody_component;

typedef struct {
    int  mesh_id;              /* index into engine mesh registry */
    int  visible;
    uint8_t color_r, color_g, color_b;
} rac_mesh_renderer_component;

typedef struct {
    int  camera_id;            /* index into camera registry */
    int  active;
} rac_camera_component;

typedef struct {
    int  light_id;             /* index into light registry */
} rac_light_component;

typedef struct {
    int    source_id;          /* index into audio source registry */
    float  volume;
    int    looping;
} rac_audio_source_component;

typedef struct {
    uint32_t parent_entity;
} rac_parent_component;

/* ── ECS World ─────────────────────────────────────────────────────────── */

typedef struct {
    /* Entity tracking */
    uint32_t  component_masks[RAC_ECS_MAX_ENTITIES];
    int       alive[RAC_ECS_MAX_ENTITIES];
    uint32_t  num_entities;
    uint32_t  next_id;

    /* Component arrays (SoA layout) */
    rac_transform_component      transforms[RAC_ECS_MAX_ENTITIES];
    rac_rigidbody_component      rigidbodies[RAC_ECS_MAX_ENTITIES];
    rac_mesh_renderer_component  mesh_renderers[RAC_ECS_MAX_ENTITIES];
    rac_camera_component         cameras[RAC_ECS_MAX_ENTITIES];
    rac_light_component          lights[RAC_ECS_MAX_ENTITIES];
    rac_audio_source_component   audio_sources[RAC_ECS_MAX_ENTITIES];
    rac_parent_component         parents[RAC_ECS_MAX_ENTITIES];
} rac_ecs_world;

/* ── API ───────────────────────────────────────────────────────────────── */

void      rac_ecs_init(rac_ecs_world *ecs);
uint32_t  rac_ecs_create_entity(rac_ecs_world *ecs);
void      rac_ecs_destroy_entity(rac_ecs_world *ecs, uint32_t entity);
int       rac_ecs_is_alive(const rac_ecs_world *ecs, uint32_t entity);

void      rac_ecs_add_component(rac_ecs_world *ecs, uint32_t entity,
                                rac_component_flag comp);
void      rac_ecs_remove_component(rac_ecs_world *ecs, uint32_t entity,
                                   rac_component_flag comp);
int       rac_ecs_has_component(const rac_ecs_world *ecs, uint32_t entity,
                                rac_component_flag comp);

/* Query: iterate entities matching a component mask */
int       rac_ecs_query(const rac_ecs_world *ecs, uint32_t required_mask,
                        uint32_t *out_entities, int max_results);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_ECS_H */
