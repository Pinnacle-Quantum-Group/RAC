/*
 * rac_engine_scene.h — Scene Graph & Transform System
 * Hierarchical transforms using RAC quaternion math.
 */

#ifndef RAC_ENGINE_SCENE_H
#define RAC_ENGINE_SCENE_H

#include "rac_engine_ecs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── 4x4 transform matrix (row-major) for world transforms ─────────── */

typedef struct {
    float m[4][4];
} rac_mat4;

rac_mat4 rac_mat4_identity(void);
rac_mat4 rac_mat4_from_trs(rac_phys_vec3 pos, rac_phys_quat rot, rac_phys_vec3 scale);
rac_mat4 rac_mat4_mul(rac_mat4 a, rac_mat4 b);
rac_phys_vec3 rac_mat4_transform_point(rac_mat4 m, rac_phys_vec3 p);
rac_phys_vec3 rac_mat4_transform_dir(rac_mat4 m, rac_phys_vec3 d);

/* ── Scene graph ───────────────────────────────────────────────────────── */

#define RAC_SCENE_MAX_NODES RAC_ECS_MAX_ENTITIES
#define RAC_SCENE_NO_PARENT UINT32_MAX

typedef struct {
    /* Parent-child hierarchy (indexed by entity ID) */
    uint32_t  parent[RAC_SCENE_MAX_NODES];
    uint32_t  first_child[RAC_SCENE_MAX_NODES];
    uint32_t  next_sibling[RAC_SCENE_MAX_NODES];

    /* Cached world matrices */
    rac_mat4  world_matrix[RAC_SCENE_MAX_NODES];
    int       dirty[RAC_SCENE_MAX_NODES];

    uint32_t  num_nodes;
} rac_scene_graph;

void rac_scene_init(rac_scene_graph *sg);
void rac_scene_set_parent(rac_scene_graph *sg, uint32_t child, uint32_t parent);
void rac_scene_mark_dirty(rac_scene_graph *sg, uint32_t node);

/* Recompute world matrices from ECS transform components */
void rac_scene_update_transforms(rac_scene_graph *sg, const rac_ecs_world *ecs);

/* Get the world-space position/rotation for an entity */
rac_mat4 rac_scene_get_world_matrix(const rac_scene_graph *sg, uint32_t entity);

/* ── Scene serialization ───────────────────────────────────────────────── */

#define RAC_SCENE_MAGIC 0x52414353  /* 'RACS' */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_entities;
    uint32_t num_parent_links;
} rac_scene_file_header;

int rac_scene_save(const rac_ecs_world *ecs, const rac_scene_graph *sg,
                   const char *path);
int rac_scene_load(rac_ecs_world *ecs, rac_scene_graph *sg,
                   const char *path);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_SCENE_H */
