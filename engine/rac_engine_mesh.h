/*
 * rac_engine_mesh.h — Mesh System
 * Triangle meshes, OBJ loading, procedural generators.
 */

#ifndef RAC_ENGINE_MESH_H
#define RAC_ENGINE_MESH_H

#include "rac_physics.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Vertex format ─────────────────────────────────────────────────────── */

typedef struct {
    rac_phys_vec3 position;
    rac_phys_vec3 normal;
    float         u, v;        /* texture coords */
} rac_vertex;

/* ── Triangle mesh ─────────────────────────────────────────────────────── */

#define RAC_MESH_MAX_VERTICES 65536
#define RAC_MESH_MAX_INDICES  196608

typedef struct {
    rac_vertex   *vertices;
    int          *indices;       /* triangle indices (3 per tri) */
    int           num_vertices;
    int           num_indices;

    /* Per-mesh AABB for frustum culling */
    rac_phys_aabb aabb;

    /* Set after loading */
    int           valid;
} rac_mesh;

/* ── Mesh registry ─────────────────────────────────────────────────────── */

#define RAC_MAX_MESHES 256

typedef struct {
    rac_mesh  meshes[RAC_MAX_MESHES];
    int       num_meshes;
} rac_mesh_registry;

void rac_mesh_registry_init(rac_mesh_registry *reg);
void rac_mesh_registry_cleanup(rac_mesh_registry *reg);

/* ── Mesh creation/destruction ─────────────────────────────────────────── */

int  rac_mesh_create(rac_mesh_registry *reg, int max_verts, int max_indices);
void rac_mesh_destroy(rac_mesh *mesh);
void rac_mesh_compute_aabb(rac_mesh *mesh);
void rac_mesh_compute_normals(rac_mesh *mesh);

/* ── OBJ loader (subset: v, vn, vt, f) ────────────────────────────────── */

int  rac_mesh_load_obj(rac_mesh_registry *reg, const char *path);

/* ── Procedural generators ─────────────────────────────────────────────── */

int  rac_mesh_gen_cube(rac_mesh_registry *reg, float size);
int  rac_mesh_gen_sphere(rac_mesh_registry *reg, float radius, int rings, int sectors);
int  rac_mesh_gen_plane(rac_mesh_registry *reg, float width, float depth, int subdiv);
int  rac_mesh_gen_cylinder(rac_mesh_registry *reg, float radius, float height, int segments);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_MESH_H */
