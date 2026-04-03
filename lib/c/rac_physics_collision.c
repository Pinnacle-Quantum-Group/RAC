/*
 * rac_physics_collision.c — RAC Native Physics: Collision Detection
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * Broad phase:  Uniform spatial hash grid (CUDA particle sim heritage)
 * Narrow phase: Sphere-sphere, sphere-box, SAT box-box, GJK convex
 *
 * All distance/normal computations route through RAC primitives:
 *   - rac_norm for distances
 *   - rac_polar for direction extraction
 *   - rac_phys_v3_dot for SAT projections
 *   - rac_phys_v3_cross for edge normals
 */

#include "rac_physics.h"
#include "rac_cpu.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════════════════
 * §1  SPATIAL HASH (BROAD PHASE)
 * ══════════════════════════════════════════════════════════════════════════
 *
 * GPU-heritage uniform grid hashing from CUDA particle simulation.
 * Hash(x,y,z) → bucket → linked list of body IDs.
 * O(1) per-query expected time, O(n) total insertion.
 */

#define RAC_SPATIAL_HASH_NULL (-1)

typedef struct _spatial_entry {
    int id;
    int next;  /* index into entries pool */
} _spatial_entry;

struct rac_phys_spatial_hash {
    float            cell_size;
    float            inv_cell_size;
    int              table_size;
    int             *buckets;        /* hash → first entry index */

    _spatial_entry  *entries;
    int              num_entries;
    int              max_entries;
};

static inline int _hash_coord(int x, int y, int z, int table_size) {
    /* Large primes for spatial hash — standard GPU particle sim constants */
    unsigned int h = (unsigned int)(x * 73856093) ^
                     (unsigned int)(y * 19349663) ^
                     (unsigned int)(z * 83492791);
    return (int)(h % (unsigned int)table_size);
}

rac_phys_spatial_hash* rac_phys_spatial_hash_create(float cell_size,
                                                     int table_size) {
    rac_phys_spatial_hash *sh = calloc(1, sizeof(rac_phys_spatial_hash));
    if (!sh) return NULL;

    sh->cell_size = cell_size;
    sh->inv_cell_size = 1.0f / cell_size;
    sh->table_size = table_size;

    sh->buckets = malloc(sizeof(int) * table_size);
    sh->max_entries = table_size * 4;
    sh->entries = malloc(sizeof(_spatial_entry) * sh->max_entries);
    if (!sh->buckets || !sh->entries) {
        free(sh->buckets);
        free(sh->entries);
        free(sh);
        return NULL;
    }

    rac_phys_spatial_hash_clear(sh);
    return sh;
}

void rac_phys_spatial_hash_destroy(rac_phys_spatial_hash *sh) {
    if (!sh) return;
    free(sh->buckets);
    free(sh->entries);
    free(sh);
}

void rac_phys_spatial_hash_clear(rac_phys_spatial_hash *sh) {
    for (int i = 0; i < sh->table_size; i++)
        sh->buckets[i] = RAC_SPATIAL_HASH_NULL;
    sh->num_entries = 0;
}

static void _insert_cell(rac_phys_spatial_hash *sh, int cx, int cy, int cz, int id) {
    /* Fix #2: grow entries array on overflow instead of silently dropping */
    if (sh->num_entries >= sh->max_entries) {
        int new_max = sh->max_entries * 2;
        _spatial_entry *new_entries = realloc(sh->entries,
                                              sizeof(_spatial_entry) * new_max);
        if (!new_entries) return;  /* OOM — degrade gracefully */
        sh->entries = new_entries;
        sh->max_entries = new_max;
    }

    int bucket = _hash_coord(cx, cy, cz, sh->table_size);
    int ei = sh->num_entries++;
    sh->entries[ei].id = id;
    sh->entries[ei].next = sh->buckets[bucket];
    sh->buckets[bucket] = ei;
}

void rac_phys_spatial_hash_insert(rac_phys_spatial_hash *sh,
                                    rac_phys_aabb aabb, int id) {
    /* Insert into all cells the AABB overlaps */
    int min_cx = (int)floorf(aabb.min.x * sh->inv_cell_size);
    int min_cy = (int)floorf(aabb.min.y * sh->inv_cell_size);
    int min_cz = (int)floorf(aabb.min.z * sh->inv_cell_size);
    int max_cx = (int)floorf(aabb.max.x * sh->inv_cell_size);
    int max_cy = (int)floorf(aabb.max.y * sh->inv_cell_size);
    int max_cz = (int)floorf(aabb.max.z * sh->inv_cell_size);

    for (int cx = min_cx; cx <= max_cx; cx++)
        for (int cy = min_cy; cy <= max_cy; cy++)
            for (int cz = min_cz; cz <= max_cz; cz++)
                _insert_cell(sh, cx, cy, cz, id);
}

int rac_phys_spatial_hash_query(rac_phys_spatial_hash *sh,
                                 rac_phys_aabb aabb,
                                 int *results, int max_results) {
    int count = 0;
    int min_cx = (int)floorf(aabb.min.x * sh->inv_cell_size);
    int min_cy = (int)floorf(aabb.min.y * sh->inv_cell_size);
    int min_cz = (int)floorf(aabb.min.z * sh->inv_cell_size);
    int max_cx = (int)floorf(aabb.max.x * sh->inv_cell_size);
    int max_cy = (int)floorf(aabb.max.y * sh->inv_cell_size);
    int max_cz = (int)floorf(aabb.max.z * sh->inv_cell_size);

    /*
     * Fix #11: O(n) dedup via bitset instead of O(n²) linear scan.
     * Use stack-allocated bitset for IDs < 8192; fall back to heap for more.
     */
    #define _BITSET_STACK_IDS 8192
    #define _BITSET_WORDS(n) (((n) + 31) / 32)

    uint32_t stack_bits[_BITSET_WORDS(_BITSET_STACK_IDS)];
    uint32_t *seen = stack_bits;
    int bitset_ids = _BITSET_STACK_IDS;
    int used_heap = 0;

    /* Find max ID to size the bitset */
    int max_id = 0;
    for (int i = 0; i < sh->num_entries; i++)
        if (sh->entries[i].id > max_id) max_id = sh->entries[i].id;
    max_id++;  /* need id+1 bits */

    if (max_id > _BITSET_STACK_IDS) {
        seen = calloc(_BITSET_WORDS(max_id), sizeof(uint32_t));
        if (!seen) return 0;  /* OOM fallback */
        bitset_ids = max_id;
        used_heap = 1;
    } else {
        memset(stack_bits, 0, sizeof(stack_bits));
    }

    for (int cx = min_cx; cx <= max_cx && count < max_results; cx++) {
        for (int cy = min_cy; cy <= max_cy && count < max_results; cy++) {
            for (int cz = min_cz; cz <= max_cz && count < max_results; cz++) {
                int bucket = _hash_coord(cx, cy, cz, sh->table_size);
                int ei = sh->buckets[bucket];
                while (ei != RAC_SPATIAL_HASH_NULL && count < max_results) {
                    int id = sh->entries[ei].id;
                    if (id >= 0 && id < bitset_ids) {
                        uint32_t word = (uint32_t)id / 32;
                        uint32_t bit  = 1u << ((uint32_t)id % 32);
                        if (!(seen[word] & bit)) {
                            seen[word] |= bit;
                            results[count++] = id;
                        }
                    }
                    ei = sh->entries[ei].next;
                }
            }
        }
    }

    if (used_heap) free(seen);
    return count;

    #undef _BITSET_STACK_IDS
    #undef _BITSET_WORDS
}

/* ══════════════════════════════════════════════════════════════════════════
 * §2  NARROW PHASE COLLISION TESTS
 * ════════════════════════════════════════════════════════════════════════ */

int rac_phys_collide_sphere_sphere(rac_phys_vec3 pos_a, float r_a,
                                    rac_phys_vec3 pos_b, float r_b,
                                    rac_phys_contact_manifold *out) {
    rac_phys_vec3 d = rac_phys_v3_sub(pos_b, pos_a);
    float dist = rac_phys_v3_length(d);  /* RAC: rac_norm chain */
    float sum_r = r_a + r_b;

    if (dist >= sum_r || dist < 1e-8f) return 0;

    rac_phys_vec3 normal = rac_phys_v3_scale(d, 1.0f / dist);
    float depth = sum_r - dist;

    out->num_contacts = 1;
    out->contacts[0].normal = normal;
    out->contacts[0].depth = depth;
    /* Contact point: midpoint of overlap region */
    out->contacts[0].point = rac_phys_v3_add(
        pos_a, rac_phys_v3_scale(normal, r_a - depth * 0.5f));

    return 1;
}

int rac_phys_collide_sphere_box(rac_phys_vec3 sphere_pos, float radius,
                                 rac_phys_vec3 box_pos, rac_phys_quat box_rot,
                                 rac_phys_vec3 half_extents,
                                 rac_phys_contact_manifold *out) {
    /* Transform sphere center into box-local space */
    rac_phys_quat inv_rot = rac_phys_quat_conjugate(box_rot);
    rac_phys_vec3 local_sphere = rac_phys_quat_rotate_vec3(
        inv_rot, rac_phys_v3_sub(sphere_pos, box_pos));

    /* Clamp to box surface — closest point on box */
    rac_phys_vec3 closest;
    closest.x = fmaxf(-half_extents.x, fminf(local_sphere.x, half_extents.x));
    closest.y = fmaxf(-half_extents.y, fminf(local_sphere.y, half_extents.y));
    closest.z = fmaxf(-half_extents.z, fminf(local_sphere.z, half_extents.z));

    rac_phys_vec3 delta = rac_phys_v3_sub(local_sphere, closest);
    float dist = rac_phys_v3_length(delta);  /* RAC: rac_norm chain */

    if (dist >= radius && dist > 1e-8f) return 0;

    rac_phys_vec3 normal_local;
    float depth;

    if (dist < 1e-8f) {
        /* Sphere center is inside the box — push out along shortest axis */
        float dx = half_extents.x - fabsf(local_sphere.x);
        float dy = half_extents.y - fabsf(local_sphere.y);
        float dz = half_extents.z - fabsf(local_sphere.z);

        if (dx <= dy && dx <= dz) {
            normal_local = rac_phys_v3(local_sphere.x > 0 ? 1.0f : -1.0f, 0, 0);
            depth = dx + radius;
        } else if (dy <= dz) {
            normal_local = rac_phys_v3(0, local_sphere.y > 0 ? 1.0f : -1.0f, 0);
            depth = dy + radius;
        } else {
            normal_local = rac_phys_v3(0, 0, local_sphere.z > 0 ? 1.0f : -1.0f);
            depth = dz + radius;
        }
    } else {
        normal_local = rac_phys_v3_scale(delta, 1.0f / dist);
        depth = radius - dist;
    }

    /* Transform back to world space */
    out->num_contacts = 1;
    out->contacts[0].normal = rac_phys_quat_rotate_vec3(box_rot, normal_local);
    out->contacts[0].depth = depth;
    out->contacts[0].point = rac_phys_v3_add(
        box_pos, rac_phys_quat_rotate_vec3(box_rot, closest));

    return 1;
}

/* ── SAT Box-Box collision (PhysX heritage) ────────────────────────────── */

static float _sat_project_box(rac_phys_vec3 half_ext, rac_phys_mat3 rot,
                               rac_phys_vec3 axis) {
    /* Project box onto axis = sum of |dot(axis, rot_col_i)| * half_ext_i */
    rac_phys_vec3 col0 = { rot.m[0][0], rot.m[1][0], rot.m[2][0] };
    rac_phys_vec3 col1 = { rot.m[0][1], rot.m[1][1], rot.m[2][1] };
    rac_phys_vec3 col2 = { rot.m[0][2], rot.m[1][2], rot.m[2][2] };

    return fabsf(rac_phys_v3_dot(axis, col0)) * half_ext.x +
           fabsf(rac_phys_v3_dot(axis, col1)) * half_ext.y +
           fabsf(rac_phys_v3_dot(axis, col2)) * half_ext.z;
}

int rac_phys_collide_box_box(rac_phys_vec3 pos_a, rac_phys_quat rot_a,
                              rac_phys_vec3 he_a,
                              rac_phys_vec3 pos_b, rac_phys_quat rot_b,
                              rac_phys_vec3 he_b,
                              rac_phys_contact_manifold *out) {
    rac_phys_mat3 R_a = rac_phys_quat_to_mat3(rot_a);
    rac_phys_mat3 R_b = rac_phys_quat_to_mat3(rot_b);
    rac_phys_vec3 d = rac_phys_v3_sub(pos_b, pos_a);

    /* 15 SAT axes: 3 face normals A, 3 face normals B, 9 edge crosses */
    rac_phys_vec3 axes[15];
    int n_axes = 0;

    /* Face normals of A */
    axes[n_axes++] = (rac_phys_vec3){ R_a.m[0][0], R_a.m[1][0], R_a.m[2][0] };
    axes[n_axes++] = (rac_phys_vec3){ R_a.m[0][1], R_a.m[1][1], R_a.m[2][1] };
    axes[n_axes++] = (rac_phys_vec3){ R_a.m[0][2], R_a.m[1][2], R_a.m[2][2] };

    /* Face normals of B */
    axes[n_axes++] = (rac_phys_vec3){ R_b.m[0][0], R_b.m[1][0], R_b.m[2][0] };
    axes[n_axes++] = (rac_phys_vec3){ R_b.m[0][1], R_b.m[1][1], R_b.m[2][1] };
    axes[n_axes++] = (rac_phys_vec3){ R_b.m[0][2], R_b.m[1][2], R_b.m[2][2] };

    /* Edge cross products */
    for (int i = 0; i < 3; i++) {
        rac_phys_vec3 ea = { R_a.m[0][i], R_a.m[1][i], R_a.m[2][i] };
        for (int j = 0; j < 3; j++) {
            rac_phys_vec3 eb = { R_b.m[0][j], R_b.m[1][j], R_b.m[2][j] };
            rac_phys_vec3 cross = rac_phys_v3_cross(ea, eb);
            float len = rac_phys_v3_length(cross);
            if (len > 1e-6f)
                axes[n_axes++] = rac_phys_v3_scale(cross, 1.0f / len);
        }
    }

    float min_overlap = 1e30f;
    rac_phys_vec3 best_axis = rac_phys_v3_zero();

    for (int i = 0; i < n_axes; i++) {
        float proj_a = _sat_project_box(he_a, R_a, axes[i]);
        float proj_b = _sat_project_box(he_b, R_b, axes[i]);
        float dist_on_axis = fabsf(rac_phys_v3_dot(d, axes[i]));
        float overlap = (proj_a + proj_b) - dist_on_axis;

        if (overlap < 0.0f) return 0;  /* separating axis found */

        if (overlap < min_overlap) {
            min_overlap = overlap;
            best_axis = axes[i];
        }
    }

    /* Ensure normal points from A to B */
    if (rac_phys_v3_dot(best_axis, d) < 0.0f)
        best_axis = rac_phys_v3_negate(best_axis);

    /* Single contact point at midpoint of overlap */
    out->num_contacts = 1;
    out->contacts[0].normal = best_axis;
    out->contacts[0].depth = min_overlap;
    out->contacts[0].point = rac_phys_v3_add(
        pos_a, rac_phys_v3_scale(rac_phys_v3_add(pos_a, pos_b), 0.5f));
    /* Refine: push contact to surface */
    out->contacts[0].point = rac_phys_v3_lerp(pos_a, pos_b, 0.5f);

    return 1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * §3  GJK ALGORITHM (PhysX/Bullet heritage)
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Gilbert-Johnson-Keerthi distance algorithm for convex shape intersection.
 * All support function evaluations and simplex operations use RAC dot products.
 */

static rac_phys_vec3 _support(const rac_phys_vec3 *verts, int n,
                               rac_phys_vec3 pos, rac_phys_quat rot,
                               rac_phys_vec3 dir) {
    /* Find vertex with maximum projection along dir (RAC dot product) */
    rac_phys_quat inv_rot = rac_phys_quat_conjugate(rot);
    rac_phys_vec3 local_dir = rac_phys_quat_rotate_vec3(inv_rot, dir);

    float max_dot = -1e30f;
    int best = 0;
    for (int i = 0; i < n; i++) {
        float d = rac_phys_v3_dot(verts[i], local_dir);
        if (d > max_dot) { max_dot = d; best = i; }
    }

    return rac_phys_v3_add(pos, rac_phys_quat_rotate_vec3(rot, verts[best]));
}

static rac_phys_vec3 _minkowski_support(
    const rac_phys_vec3 *va, int na, rac_phys_vec3 pa, rac_phys_quat ra,
    const rac_phys_vec3 *vb, int nb, rac_phys_vec3 pb, rac_phys_quat rb,
    rac_phys_vec3 dir) {
    rac_phys_vec3 sa = _support(va, na, pa, ra, dir);
    rac_phys_vec3 sb = _support(vb, nb, pb, rb, rac_phys_v3_negate(dir));
    return rac_phys_v3_sub(sa, sb);
}

/* GJK simplex handling */
typedef struct {
    rac_phys_vec3 points[4];
    int count;
} _gjk_simplex;

static int _gjk_do_simplex(_gjk_simplex *s, rac_phys_vec3 *dir) {
    if (s->count == 2) {
        /* Line case */
        rac_phys_vec3 a = s->points[1];
        rac_phys_vec3 b = s->points[0];
        rac_phys_vec3 ab = rac_phys_v3_sub(b, a);
        rac_phys_vec3 ao = rac_phys_v3_negate(a);

        if (rac_phys_v3_dot(ab, ao) > 0.0f) {
            /* Origin is between a and b */
            *dir = rac_phys_v3_cross(rac_phys_v3_cross(ab, ao), ab);
        } else {
            s->points[0] = a;
            s->count = 1;
            *dir = ao;
        }
    } else if (s->count == 3) {
        /* Triangle case */
        rac_phys_vec3 a = s->points[2];
        rac_phys_vec3 b = s->points[1];
        rac_phys_vec3 c = s->points[0];
        rac_phys_vec3 ab = rac_phys_v3_sub(b, a);
        rac_phys_vec3 ac = rac_phys_v3_sub(c, a);
        rac_phys_vec3 ao = rac_phys_v3_negate(a);
        rac_phys_vec3 abc_normal = rac_phys_v3_cross(ab, ac);

        rac_phys_vec3 abc_x_ac = rac_phys_v3_cross(abc_normal, ac);
        if (rac_phys_v3_dot(abc_x_ac, ao) > 0.0f) {
            if (rac_phys_v3_dot(ac, ao) > 0.0f) {
                s->points[0] = c; s->points[1] = a; s->count = 2;
                *dir = rac_phys_v3_cross(rac_phys_v3_cross(ac, ao), ac);
            } else {
                s->points[0] = a; s->count = 1;
                *dir = ao;
            }
        } else {
            rac_phys_vec3 ab_x_abc = rac_phys_v3_cross(ab, abc_normal);
            if (rac_phys_v3_dot(ab_x_abc, ao) > 0.0f) {
                if (rac_phys_v3_dot(ab, ao) > 0.0f) {
                    s->points[0] = b; s->points[1] = a; s->count = 2;
                    *dir = rac_phys_v3_cross(rac_phys_v3_cross(ab, ao), ab);
                } else {
                    s->points[0] = a; s->count = 1;
                    *dir = ao;
                }
            } else {
                /* Origin is above or below triangle */
                if (rac_phys_v3_dot(abc_normal, ao) > 0.0f) {
                    *dir = abc_normal;
                } else {
                    /* Flip winding */
                    rac_phys_vec3 tmp = s->points[0];
                    s->points[0] = s->points[1];
                    s->points[1] = tmp;
                    *dir = rac_phys_v3_negate(abc_normal);
                }
            }
        }
    } else if (s->count == 4) {
        /* Tetrahedron case — origin inside? */
        rac_phys_vec3 a = s->points[3];
        rac_phys_vec3 b = s->points[2];
        rac_phys_vec3 c = s->points[1];
        rac_phys_vec3 d = s->points[0];
        rac_phys_vec3 ao = rac_phys_v3_negate(a);

        rac_phys_vec3 ab = rac_phys_v3_sub(b, a);
        rac_phys_vec3 ac = rac_phys_v3_sub(c, a);
        rac_phys_vec3 ad = rac_phys_v3_sub(d, a);

        rac_phys_vec3 abc = rac_phys_v3_cross(ab, ac);
        rac_phys_vec3 acd = rac_phys_v3_cross(ac, ad);
        rac_phys_vec3 adb = rac_phys_v3_cross(ad, ab);

        if (rac_phys_v3_dot(abc, ao) > 0.0f) {
            s->points[0] = c; s->points[1] = b; s->points[2] = a;
            s->count = 3;
            *dir = abc;
            return _gjk_do_simplex(s, dir);
        }
        if (rac_phys_v3_dot(acd, ao) > 0.0f) {
            s->points[0] = d; s->points[1] = c; s->points[2] = a;
            s->count = 3;
            *dir = acd;
            return _gjk_do_simplex(s, dir);
        }
        if (rac_phys_v3_dot(adb, ao) > 0.0f) {
            s->points[0] = b; s->points[1] = d; s->points[2] = a;
            s->count = 3;
            *dir = adb;
            return _gjk_do_simplex(s, dir);
        }

        return 1;  /* Origin is inside tetrahedron */
    }
    return 0;
}

int rac_phys_gjk_intersect(const rac_phys_vec3 *verts_a, int n_a,
                            rac_phys_vec3 pos_a, rac_phys_quat rot_a,
                            const rac_phys_vec3 *verts_b, int n_b,
                            rac_phys_vec3 pos_b, rac_phys_quat rot_b,
                            rac_phys_contact_manifold *out) {
    /* Fix #5: validate vertex arrays and counts */
    if (!verts_a || n_a <= 0 || !verts_b || n_b <= 0) return 0;

    rac_phys_vec3 dir = rac_phys_v3_sub(pos_b, pos_a);
    if (rac_phys_v3_length_sq(dir) < 1e-8f)
        dir = rac_phys_v3(1.0f, 0.0f, 0.0f);

    _gjk_simplex simplex;
    simplex.count = 0;

    rac_phys_vec3 support = _minkowski_support(
        verts_a, n_a, pos_a, rot_a,
        verts_b, n_b, pos_b, rot_b, dir);
    simplex.points[simplex.count++] = support;
    dir = rac_phys_v3_negate(support);

    for (int iter = 0; iter < 64; iter++) {
        support = _minkowski_support(
            verts_a, n_a, pos_a, rot_a,
            verts_b, n_b, pos_b, rot_b, dir);

        if (rac_phys_v3_dot(support, dir) < 0.0f)
            return 0;  /* No intersection */

        simplex.points[simplex.count++] = support;

        if (_gjk_do_simplex(&simplex, &dir)) {
            /* Intersection found — generate approximate contact */
            if (out) {
                rac_phys_vec3 diff = rac_phys_v3_sub(pos_b, pos_a);
                float dist = rac_phys_v3_length(diff);
                out->num_contacts = 1;
                out->contacts[0].normal = (dist > 1e-8f)
                    ? rac_phys_v3_scale(diff, 1.0f / dist)
                    : rac_phys_v3(0, 1, 0);
                out->contacts[0].depth = 0.01f;  /* approximate */
                out->contacts[0].point = rac_phys_v3_lerp(pos_a, pos_b, 0.5f);
            }
            return 1;
        }
    }

    return 0;
}
