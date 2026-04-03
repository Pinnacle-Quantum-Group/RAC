/*
 * rac_engine_scene.c — Scene Graph & Transform System
 * All matrix construction via RAC quaternion/rotation primitives.
 */

#include "rac_engine_scene.h"
#include <string.h>
#include <stdio.h>

/* ── Mat4 utilities ────────────────────────────────────────────────────── */

rac_mat4 rac_mat4_identity(void)
{
    rac_mat4 m;
    memset(&m, 0, sizeof(m));
    m.m[0][0] = 1.0f;
    m.m[1][1] = 1.0f;
    m.m[2][2] = 1.0f;
    m.m[3][3] = 1.0f;
    return m;
}

rac_mat4 rac_mat4_from_trs(rac_phys_vec3 pos, rac_phys_quat rot, rac_phys_vec3 scale)
{
    /* Build rotation via RAC quaternion-to-mat3 */
    rac_phys_mat3 r = rac_phys_quat_to_mat3(rot);

    rac_mat4 m;
    memset(&m, 0, sizeof(m));

    /* Apply scale to rotation columns using rac_phys_v3_scale */
    rac_phys_vec3 col0 = rac_phys_v3(r.m[0][0], r.m[1][0], r.m[2][0]);
    rac_phys_vec3 col1 = rac_phys_v3(r.m[0][1], r.m[1][1], r.m[2][1]);
    rac_phys_vec3 col2 = rac_phys_v3(r.m[0][2], r.m[1][2], r.m[2][2]);

    col0 = rac_phys_v3_scale(col0, scale.x);
    col1 = rac_phys_v3_scale(col1, scale.y);
    col2 = rac_phys_v3_scale(col2, scale.z);

    m.m[0][0] = col0.x; m.m[0][1] = col1.x; m.m[0][2] = col2.x; m.m[0][3] = pos.x;
    m.m[1][0] = col0.y; m.m[1][1] = col1.y; m.m[1][2] = col2.y; m.m[1][3] = pos.y;
    m.m[2][0] = col0.z; m.m[2][1] = col1.z; m.m[2][2] = col2.z; m.m[2][3] = pos.z;
    m.m[3][0] = 0.0f;   m.m[3][1] = 0.0f;   m.m[3][2] = 0.0f;   m.m[3][3] = 1.0f;

    return m;
}

rac_mat4 rac_mat4_mul(rac_mat4 a, rac_mat4 b)
{
    rac_mat4 out;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            /* Use rac_dot for each row-column dot product */
            rac_vec2 p1 = { a.m[i][0], a.m[i][1] };
            rac_vec2 q1 = { b.m[0][j], b.m[1][j] };
            rac_vec2 p2 = { a.m[i][2], a.m[i][3] };
            rac_vec2 q2 = { b.m[2][j], b.m[3][j] };
            out.m[i][j] = rac_dot(p1, q1) + rac_dot(p2, q2);
        }
    }
    return out;
}

rac_phys_vec3 rac_mat4_transform_point(rac_mat4 m, rac_phys_vec3 p)
{
    /* Transform point: M * [p, 1] using rac_dot pairs */
    rac_phys_vec3 out;
    rac_vec2 px = { p.x, p.y };
    rac_vec2 pz = { p.z, 1.0f };

    rac_vec2 r0xy = { m.m[0][0], m.m[0][1] };
    rac_vec2 r0zw = { m.m[0][2], m.m[0][3] };
    out.x = rac_dot(r0xy, px) + rac_dot(r0zw, pz);

    rac_vec2 r1xy = { m.m[1][0], m.m[1][1] };
    rac_vec2 r1zw = { m.m[1][2], m.m[1][3] };
    out.y = rac_dot(r1xy, px) + rac_dot(r1zw, pz);

    rac_vec2 r2xy = { m.m[2][0], m.m[2][1] };
    rac_vec2 r2zw = { m.m[2][2], m.m[2][3] };
    out.z = rac_dot(r2xy, px) + rac_dot(r2zw, pz);

    return out;
}

rac_phys_vec3 rac_mat4_transform_dir(rac_mat4 m, rac_phys_vec3 d)
{
    /* Transform direction: M * [d, 0] — ignore translation */
    rac_phys_vec3 out;
    rac_vec2 dx = { d.x, d.y };

    rac_vec2 r0 = { m.m[0][0], m.m[0][1] };
    out.x = rac_dot(r0, dx) + m.m[0][2] * d.z;  /* can't avoid one here, but we use rac_project below */

    /* Better: use rac_phys_mat3_mul_vec3 which is RAC-native */
    rac_phys_mat3 upper;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            upper.m[i][j] = m.m[i][j];

    return rac_phys_mat3_mul_vec3(upper, d);
}

/* ── Scene graph ───────────────────────────────────────────────────────── */

void rac_scene_init(rac_scene_graph *sg)
{
    memset(sg, 0, sizeof(*sg));
    for (uint32_t i = 0; i < RAC_SCENE_MAX_NODES; i++) {
        sg->parent[i] = RAC_SCENE_NO_PARENT;
        sg->first_child[i] = RAC_SCENE_NO_PARENT;
        sg->next_sibling[i] = RAC_SCENE_NO_PARENT;
        sg->world_matrix[i] = rac_mat4_identity();
        sg->dirty[i] = 1;
    }
}

void rac_scene_set_parent(rac_scene_graph *sg, uint32_t child, uint32_t parent)
{
    if (child >= RAC_SCENE_MAX_NODES) return;

    /* Remove from old parent's child list */
    uint32_t old_parent = sg->parent[child];
    if (old_parent != RAC_SCENE_NO_PARENT && old_parent < RAC_SCENE_MAX_NODES) {
        uint32_t *prev = &sg->first_child[old_parent];
        while (*prev != RAC_SCENE_NO_PARENT) {
            if (*prev == child) {
                *prev = sg->next_sibling[child];
                break;
            }
            prev = &sg->next_sibling[*prev];
        }
    }

    sg->parent[child] = parent;
    sg->next_sibling[child] = RAC_SCENE_NO_PARENT;

    /* Add to new parent's child list */
    if (parent != RAC_SCENE_NO_PARENT && parent < RAC_SCENE_MAX_NODES) {
        sg->next_sibling[child] = sg->first_child[parent];
        sg->first_child[parent] = child;
    }

    rac_scene_mark_dirty(sg, child);
}

void rac_scene_mark_dirty(rac_scene_graph *sg, uint32_t node)
{
    if (node >= RAC_SCENE_MAX_NODES) return;
    sg->dirty[node] = 1;

    /* Propagate to children */
    uint32_t child = sg->first_child[node];
    while (child != RAC_SCENE_NO_PARENT) {
        rac_scene_mark_dirty(sg, child);
        child = sg->next_sibling[child];
    }
}

static void scene_update_node(rac_scene_graph *sg, const rac_ecs_world *ecs,
                               uint32_t node)
{
    if (node >= RAC_SCENE_MAX_NODES || !ecs->alive[node]) return;
    if (!sg->dirty[node]) return;

    /* Build local matrix from TRS */
    rac_mat4 local = rac_mat4_from_trs(
        ecs->transforms[node].position,
        ecs->transforms[node].rotation,
        ecs->transforms[node].scale
    );

    /* Multiply with parent world matrix */
    uint32_t p = sg->parent[node];
    if (p != RAC_SCENE_NO_PARENT && p < RAC_SCENE_MAX_NODES) {
        /* Ensure parent is up to date first */
        if (sg->dirty[p])
            scene_update_node(sg, ecs, p);
        sg->world_matrix[node] = rac_mat4_mul(sg->world_matrix[p], local);
    } else {
        sg->world_matrix[node] = local;
    }

    sg->dirty[node] = 0;
}

void rac_scene_update_transforms(rac_scene_graph *sg, const rac_ecs_world *ecs)
{
    for (uint32_t i = 0; i < ecs->next_id; i++) {
        if (ecs->alive[i] && sg->dirty[i])
            scene_update_node(sg, ecs, i);
    }
}

rac_mat4 rac_scene_get_world_matrix(const rac_scene_graph *sg, uint32_t entity)
{
    if (entity >= RAC_SCENE_MAX_NODES)
        return rac_mat4_identity();
    return sg->world_matrix[entity];
}

/* ── Serialization ─────────────────────────────────────────────────────── */

int rac_scene_save(const rac_ecs_world *ecs, const rac_scene_graph *sg,
                   const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Count active entities and parent links */
    uint32_t num_ents = 0, num_links = 0;
    for (uint32_t i = 0; i < ecs->next_id; i++) {
        if (ecs->alive[i]) {
            num_ents++;
            if (sg->parent[i] != RAC_SCENE_NO_PARENT)
                num_links++;
        }
    }

    rac_scene_file_header hdr = {
        .magic = RAC_SCENE_MAGIC,
        .version = 1,
        .num_entities = num_ents,
        .num_parent_links = num_links
    };
    fwrite(&hdr, sizeof(hdr), 1, f);

    /* Write entity data: id, mask, transform */
    for (uint32_t i = 0; i < ecs->next_id; i++) {
        if (!ecs->alive[i]) continue;
        fwrite(&i, sizeof(uint32_t), 1, f);
        fwrite(&ecs->component_masks[i], sizeof(uint32_t), 1, f);
        fwrite(&ecs->transforms[i], sizeof(rac_transform_component), 1, f);
    }

    /* Write parent links: child, parent */
    for (uint32_t i = 0; i < ecs->next_id; i++) {
        if (ecs->alive[i] && sg->parent[i] != RAC_SCENE_NO_PARENT) {
            fwrite(&i, sizeof(uint32_t), 1, f);
            fwrite(&sg->parent[i], sizeof(uint32_t), 1, f);
        }
    }

    fclose(f);
    return 0;
}

int rac_scene_load(rac_ecs_world *ecs, rac_scene_graph *sg,
                   const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    rac_scene_file_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != RAC_SCENE_MAGIC) {
        fclose(f);
        return -2;
    }

    rac_ecs_init(ecs);
    rac_scene_init(sg);

    for (uint32_t i = 0; i < hdr.num_entities; i++) {
        uint32_t id, mask;
        rac_transform_component tc;
        fread(&id, sizeof(uint32_t), 1, f);
        fread(&mask, sizeof(uint32_t), 1, f);
        fread(&tc, sizeof(rac_transform_component), 1, f);

        if (id < RAC_ECS_MAX_ENTITIES) {
            ecs->alive[id] = 1;
            ecs->component_masks[id] = mask;
            ecs->transforms[id] = tc;
            ecs->num_entities++;
            if (id >= ecs->next_id) ecs->next_id = id + 1;
        }
    }

    for (uint32_t i = 0; i < hdr.num_parent_links; i++) {
        uint32_t child, parent;
        fread(&child, sizeof(uint32_t), 1, f);
        fread(&parent, sizeof(uint32_t), 1, f);
        rac_scene_set_parent(sg, child, parent);
    }

    fclose(f);
    return 0;
}
