/*
 * rac_physics_world.c — RAC Native Physics: World / Scene Management
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * Physics world ties together rigid bodies, collision detection,
 * and constraint solving into a unified simulation step.
 *
 * Pipeline per step:
 *   1. Apply gravity to all dynamic bodies
 *   2. Broad phase: build spatial hash, find overlapping pairs
 *   3. Narrow phase: generate contact manifolds
 *   4. Solve constraints (PGS: contacts + joints)
 *   5. Integrate bodies (Euler/Verlet/RK4)
 *   6. Sleep management
 */

#include "rac_physics.h"
#include "rac_cpu.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RAC_WORLD_MAX_BODIES       4096
#define RAC_WORLD_MAX_SHAPES       4096
#define RAC_WORLD_MAX_CONSTRAINTS  2048
#define RAC_WORLD_MAX_CONTACTS     8192

struct rac_phys_world {
    rac_phys_world_config config;

    /* Bodies */
    rac_phys_rigid_body  bodies[RAC_WORLD_MAX_BODIES];
    rac_phys_shape       shapes[RAC_WORLD_MAX_SHAPES];
    int                  num_bodies;

    /* Constraints */
    rac_phys_constraint  constraints[RAC_WORLD_MAX_CONSTRAINTS];
    int                  num_constraints;

    /* Contact manifolds (rebuilt each step) */
    rac_phys_contact_manifold contacts[RAC_WORLD_MAX_CONTACTS];
    int                       num_contacts;

    /* Broad phase */
    rac_phys_spatial_hash *spatial_hash;

    /* Time accumulator for fixed timestep */
    float time_accumulator;

    /* Sleeping islands (union-find) */
    int island_parent[RAC_WORLD_MAX_BODIES];
    int island_rank[RAC_WORLD_MAX_BODIES];
};

/* ══════════════════════════════════════════════════════════════════════════
 * SLEEPING ISLANDS (Union-Find / Disjoint Set)
 *
 * PhysX/Bullet heritage: bodies connected by contacts or joints form
 * "islands." If all bodies in an island are below sleep threshold,
 * the entire island sleeps. If any body is disturbed, the whole island
 * wakes. This is O(n*α(n)) ≈ O(n) via path compression + union by rank.
 * ════════════════════════════════════════════════════════════════════════ */

static int _island_find(int *parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  /* path compression */
        x = parent[x];
    }
    return x;
}

static void _island_union(int *parent, int *rank, int a, int b) {
    int ra = _island_find(parent, a);
    int rb = _island_find(parent, b);
    if (ra == rb) return;
    if (rank[ra] < rank[rb]) { int t = ra; ra = rb; rb = t; }
    parent[rb] = ra;
    if (rank[ra] == rank[rb]) rank[ra]++;
}

/* ══════════════════════════════════════════════════════════════════════════
 * CONTINUOUS COLLISION DETECTION (CCD) — Sphere Sweep
 *
 * PhysX heritage: prevents fast-moving small objects from tunneling
 * through walls. Uses conservative advancement (sphere sweep against
 * spheres and planes).
 * ════════════════════════════════════════════════════════════════════════ */

/* Sweep sphere A (moving from p0 to p1) against static sphere B.
 * Returns time-of-impact t in [0,1], or -1 if no hit. */
static float _ccd_sphere_sweep(rac_phys_vec3 p0, rac_phys_vec3 p1, float r_a,
                                 rac_phys_vec3 b_pos, float r_b) {
    rac_phys_vec3 d = rac_phys_v3_sub(p1, p0);        /* motion vector */
    rac_phys_vec3 m = rac_phys_v3_sub(p0, b_pos);     /* relative position */
    float r = r_a + r_b;

    float dd = rac_phys_v3_dot(d, d);
    float md = rac_phys_v3_dot(m, d);
    float mm = rac_phys_v3_dot(m, m);

    float a = dd;
    float b = 2.0f * md;
    float c = mm - r * r;

    if (c < 0.0f) return 0.0f;  /* already overlapping */

    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f || a < 1e-8f) return -1.0f;

    float t = (-b - sqrtf(disc)) / (2.0f * a);
    if (t >= 0.0f && t <= 1.0f) return t;
    return -1.0f;
}

/* ── Config defaults ───────────────────────────────────────────────────── */

rac_phys_world_config rac_phys_world_default_config(void) {
    return (rac_phys_world_config){
        .gravity = { 0.0f, -9.81f, 0.0f },
        .integrator = RAC_INTEGRATE_EULER,
        .fixed_dt = 1.0f / 60.0f,
        .max_substeps = 4,
        .sleep_threshold = 0.01f,
        .sleep_time = 2.0f,
        .pgs_config = {
            .iterations = 8,
            .slop = 0.005f,
            .baumgarte = 0.2f,
            .warm_start_factor = 0.85f
        },
        .broad_phase_cell_size = 2.0f,
        .broad_phase_table_size = 4096
    };
}

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

rac_phys_world* rac_phys_world_create(const rac_phys_world_config *cfg) {
    rac_phys_world *w = calloc(1, sizeof(rac_phys_world));
    if (!w) return NULL;

    w->config = cfg ? *cfg : rac_phys_world_default_config();

    w->spatial_hash = rac_phys_spatial_hash_create(
        w->config.broad_phase_cell_size,
        w->config.broad_phase_table_size);

    if (!w->spatial_hash) {
        free(w);
        return NULL;
    }

    return w;
}

void rac_phys_world_destroy(rac_phys_world *world) {
    if (!world) return;
    rac_phys_spatial_hash_destroy(world->spatial_hash);
    free(world);
}

/* ── Body management ───────────────────────────────────────────────────── */

int rac_phys_world_add_body(rac_phys_world *world,
                              rac_phys_rigid_body body, rac_phys_shape shape) {
    if (world->num_bodies >= RAC_WORLD_MAX_BODIES) return -1;
    int idx = world->num_bodies++;
    body.id = (uint32_t)idx;
    body.shape_index = idx;
    world->bodies[idx] = body;
    world->shapes[idx] = shape;
    return idx;
}

rac_phys_rigid_body* rac_phys_world_get_body(rac_phys_world *world, int index) {
    if (index < 0 || index >= world->num_bodies) return NULL;
    return &world->bodies[index];
}

int rac_phys_world_add_constraint(rac_phys_world *world,
                                    rac_phys_constraint constraint) {
    if (world->num_constraints >= RAC_WORLD_MAX_CONSTRAINTS) return -1;
    int idx = world->num_constraints++;
    world->constraints[idx] = constraint;
    return idx;
}

/* ── Compute AABB for a body+shape ─────────────────────────────────────── */

static rac_phys_aabb _body_aabb(const rac_phys_rigid_body *body,
                                  const rac_phys_shape *shape) {
    rac_phys_vec3 pos = body->position;

    switch (shape->type) {
        case RAC_SHAPE_SPHERE: {
            float r = shape->sphere.radius;
            rac_phys_vec3 rv = rac_phys_v3(r, r, r);
            return rac_phys_aabb_from_center_half(pos, rv);
        }
        case RAC_SHAPE_BOX: {
            /* Rotated box AABB: project each axis onto world axes */
            rac_phys_mat3 R = rac_phys_quat_to_mat3(body->orientation);
            rac_phys_vec3 he = shape->box.half_extents;
            float ex = fabsf(R.m[0][0]) * he.x + fabsf(R.m[0][1]) * he.y +
                       fabsf(R.m[0][2]) * he.z;
            float ey = fabsf(R.m[1][0]) * he.x + fabsf(R.m[1][1]) * he.y +
                       fabsf(R.m[1][2]) * he.z;
            float ez = fabsf(R.m[2][0]) * he.x + fabsf(R.m[2][1]) * he.y +
                       fabsf(R.m[2][2]) * he.z;
            return rac_phys_aabb_from_center_half(pos, rac_phys_v3(ex, ey, ez));
        }
        case RAC_SHAPE_CAPSULE: {
            float r = shape->capsule.radius + shape->capsule.half_height;
            rac_phys_vec3 rv = rac_phys_v3(r, r, r);
            return rac_phys_aabb_from_center_half(pos, rv);
        }
        default: {
            /* Conservative fallback */
            rac_phys_vec3 rv = rac_phys_v3(10.0f, 10.0f, 10.0f);
            return rac_phys_aabb_from_center_half(pos, rv);
        }
    }
}

/* ── Narrow phase dispatch ─────────────────────────────────────────────── */

static int _narrow_phase(const rac_phys_rigid_body *a, const rac_phys_shape *sa,
                           const rac_phys_rigid_body *b, const rac_phys_shape *sb,
                           rac_phys_contact_manifold *out) {
    out->body_a = (int)a->id;
    out->body_b = (int)b->id;
    out->num_contacts = 0;

    if (sa->type == RAC_SHAPE_SPHERE && sb->type == RAC_SHAPE_SPHERE) {
        return rac_phys_collide_sphere_sphere(
            a->position, sa->sphere.radius,
            b->position, sb->sphere.radius, out);
    }
    if (sa->type == RAC_SHAPE_SPHERE && sb->type == RAC_SHAPE_BOX) {
        return rac_phys_collide_sphere_box(
            a->position, sa->sphere.radius,
            b->position, b->orientation, sb->box.half_extents, out);
    }
    if (sa->type == RAC_SHAPE_BOX && sb->type == RAC_SHAPE_SPHERE) {
        int r = rac_phys_collide_sphere_box(
            b->position, sb->sphere.radius,
            a->position, a->orientation, sa->box.half_extents, out);
        if (r) {
            /* Flip normal and body indices */
            out->body_a = (int)a->id;
            out->body_b = (int)b->id;
            for (int i = 0; i < out->num_contacts; i++)
                out->contacts[i].normal = rac_phys_v3_negate(out->contacts[i].normal);
        }
        return r;
    }
    if (sa->type == RAC_SHAPE_BOX && sb->type == RAC_SHAPE_BOX) {
        return rac_phys_collide_box_box(
            a->position, a->orientation, sa->box.half_extents,
            b->position, b->orientation, sb->box.half_extents, out);
    }

    return 0;
}

/* ── World step ────────────────────────────────────────────────────────── */

static void _world_substep(rac_phys_world *w, float dt) {
    int nb = w->num_bodies;

    /* 1. Apply gravity */
    for (int i = 0; i < nb; i++) {
        if (w->bodies[i].type != RAC_BODY_DYNAMIC) continue;
        if (w->bodies[i].is_sleeping) continue;
        rac_phys_vec3 grav_force = rac_phys_v3_scale(
            w->config.gravity, w->bodies[i].mass);
        rac_phys_body_apply_force(&w->bodies[i], grav_force);
    }

    /* 2. Broad phase — rebuild spatial hash */
    rac_phys_spatial_hash_clear(w->spatial_hash);
    for (int i = 0; i < nb; i++) {
        rac_phys_aabb aabb = _body_aabb(&w->bodies[i], &w->shapes[i]);
        rac_phys_spatial_hash_insert(w->spatial_hash, aabb, i);
    }

    /* 3. Narrow phase — generate contacts */
    w->num_contacts = 0;
    for (int i = 0; i < nb && w->num_contacts < RAC_WORLD_MAX_CONTACTS; i++) {
        if (w->bodies[i].type == RAC_BODY_STATIC && w->bodies[i].is_sleeping)
            continue;

        rac_phys_aabb aabb_i = _body_aabb(&w->bodies[i], &w->shapes[i]);
        /* Expand slightly for broad phase margin */
        rac_phys_vec3 margin = rac_phys_v3(0.05f, 0.05f, 0.05f);
        rac_phys_aabb query = rac_phys_aabb_expand(aabb_i, margin);

        int candidates[128];
        int n_cand = rac_phys_spatial_hash_query(w->spatial_hash, query,
                                                  candidates, 128);

        for (int ci = 0; ci < n_cand && w->num_contacts < RAC_WORLD_MAX_CONTACTS; ci++) {
            int j = candidates[ci];
            if (j <= i) continue;  /* avoid duplicate pairs */
            if (j >= nb) continue; /* Fix #7: bounds check candidate index */

            /* Fix #7: validate shape indices before narrow phase */
            if (w->bodies[i].shape_index < 0 || w->bodies[i].shape_index >= nb ||
                w->bodies[j].shape_index < 0 || w->bodies[j].shape_index >= nb)
                continue;

            /* Skip static-static pairs */
            if (w->bodies[i].type == RAC_BODY_STATIC &&
                w->bodies[j].type == RAC_BODY_STATIC) continue;

            /* AABB overlap check */
            rac_phys_aabb aabb_j = _body_aabb(&w->bodies[j], &w->shapes[j]);
            if (!rac_phys_aabb_overlap(aabb_i, aabb_j)) continue;

            /* Narrow phase */
            rac_phys_contact_manifold manifold;
            if (_narrow_phase(&w->bodies[i], &w->shapes[i],
                              &w->bodies[j], &w->shapes[j], &manifold)) {
                w->contacts[w->num_contacts++] = manifold;
            }
        }
    }

    /* 4. Solve constraints */
    rac_phys_pgs_solve(w->bodies, nb,
                        w->constraints, w->num_constraints,
                        w->contacts, w->num_contacts,
                        &w->config.pgs_config, dt);

    /* 4b. Build sleeping islands + wake propagation BEFORE integration,
     *     so woken bodies actually get integrated this step. */
    for (int i = 0; i < nb; i++) {
        w->island_parent[i] = i;
        w->island_rank[i] = 0;
    }
    for (int ci = 0; ci < w->num_contacts; ci++) {
        rac_phys_contact_manifold *m = &w->contacts[ci];
        if (m->body_a >= 0 && m->body_a < nb &&
            m->body_b >= 0 && m->body_b < nb)
            _island_union(w->island_parent, w->island_rank,
                          m->body_a, m->body_b);
    }
    for (int ci = 0; ci < w->num_constraints; ci++) {
        rac_phys_constraint *c = &w->constraints[ci];
        if (c->body_a >= 0 && c->body_a < nb &&
            c->body_b >= 0 && c->body_b < nb)
            _island_union(w->island_parent, w->island_rank,
                          c->body_a, c->body_b);
    }
    /* Wake-on-contact: awake body in island wakes entire island */
    {
        int island_awake[RAC_WORLD_MAX_BODIES];
        memset(island_awake, 0, sizeof(int) * nb);
        for (int i = 0; i < nb; i++) {
            if (w->bodies[i].type == RAC_BODY_DYNAMIC && !w->bodies[i].is_sleeping) {
                int root = _island_find(w->island_parent, i);
                island_awake[root] = 1;
            }
        }
        for (int i = 0; i < nb; i++) {
            if (w->bodies[i].type != RAC_BODY_DYNAMIC) continue;
            int root = _island_find(w->island_parent, i);
            if (island_awake[root] && w->bodies[i].is_sleeping) {
                w->bodies[i].is_sleeping = 0;
                /* Grace period: don't re-sleep for at least sleep_time */
                w->bodies[i].sleep_timer = -w->config.sleep_time;
            }
        }
    }

    /* 5. Integrate (now includes freshly woken bodies) */
    for (int i = 0; i < nb; i++) {
        rac_phys_body_integrate(&w->bodies[i], dt, w->config.integrator);
    }

    /* 5b. CCD — sweep fast-moving spheres to prevent tunneling */
    for (int i = 0; i < nb; i++) {
        if (w->bodies[i].type != RAC_BODY_DYNAMIC) continue;
        if (w->shapes[i].type != RAC_SHAPE_SPHERE) continue;

        float speed = rac_phys_v3_length(w->bodies[i].linear_velocity);
        float r = w->shapes[i].sphere.radius;
        if (speed * dt < r * 0.5f) continue;  /* skip slow bodies */

        rac_phys_vec3 p0 = rac_phys_v3_sub(w->bodies[i].position,
            rac_phys_v3_scale(w->bodies[i].linear_velocity, dt));
        rac_phys_vec3 p1 = w->bodies[i].position;

        for (int j = 0; j < nb; j++) {
            if (i == j) continue;
            if (w->shapes[j].type != RAC_SHAPE_SPHERE) continue;

            float toi = _ccd_sphere_sweep(p0, p1, r,
                w->bodies[j].position, w->shapes[j].sphere.radius);
            if (toi >= 0.0f && toi < 1.0f) {
                /* Rewind to time of impact */
                w->bodies[i].position = rac_phys_v3_add(p0,
                    rac_phys_v3_scale(rac_phys_v3_sub(p1, p0), toi));
                /* Reflect velocity along contact normal */
                rac_phys_vec3 n = rac_phys_v3_normalize(
                    rac_phys_v3_sub(w->bodies[i].position,
                                     w->bodies[j].position));
                float vn = rac_phys_v3_dot(w->bodies[i].linear_velocity, n);
                if (vn < 0.0f) {
                    float e = fminf(w->bodies[i].restitution,
                                    w->bodies[j].restitution);
                    w->bodies[i].linear_velocity = rac_phys_v3_sub(
                        w->bodies[i].linear_velocity,
                        rac_phys_v3_scale(n, (1.0f + e) * vn));
                }
                break;  /* handle one CCD event per body per step */
            }
        }
    }

    /* 6. Per-island sleep management: island sleeps only if ALL members idle */
    /* First: update per-body sleep timers */
    for (int i = 0; i < nb; i++) {
        if (w->bodies[i].type != RAC_BODY_DYNAMIC) continue;

        float speed = rac_phys_v3_length(w->bodies[i].linear_velocity) +
                      rac_phys_v3_length(w->bodies[i].angular_velocity);

        if (speed < w->config.sleep_threshold) {
            w->bodies[i].sleep_timer += dt;
        } else {
            w->bodies[i].sleep_timer = 0.0f;
            w->bodies[i].is_sleeping = 0;
        }
    }
    /* Check if entire island can sleep */
    float island_min_timer[RAC_WORLD_MAX_BODIES];
    for (int i = 0; i < nb; i++) island_min_timer[i] = 1e30f;
    for (int i = 0; i < nb; i++) {
        if (w->bodies[i].type != RAC_BODY_DYNAMIC) continue;
        int root = _island_find(w->island_parent, i);
        if (w->bodies[i].sleep_timer < island_min_timer[root])
            island_min_timer[root] = w->bodies[i].sleep_timer;
    }
    for (int i = 0; i < nb; i++) {
        if (w->bodies[i].type != RAC_BODY_DYNAMIC) continue;
        int root = _island_find(w->island_parent, i);
        if (island_min_timer[root] >= w->config.sleep_time)
            w->bodies[i].is_sleeping = 1;
    }
}

void rac_phys_world_step(rac_phys_world *world, float dt) {
    float fixed_dt = world->config.fixed_dt;
    world->time_accumulator += dt;

    int steps = 0;
    while (world->time_accumulator >= fixed_dt &&
           steps < world->config.max_substeps) {
        _world_substep(world, fixed_dt);
        world->time_accumulator -= fixed_dt;
        steps++;
    }

    /* Clamp accumulator to prevent spiral of death */
    if (world->time_accumulator > fixed_dt * 2.0f)
        world->time_accumulator = 0.0f;
}

/* ── Queries ───────────────────────────────────────────────────────────── */

int rac_phys_world_num_bodies(const rac_phys_world *world) {
    return world->num_bodies;
}

int rac_phys_world_num_contacts(const rac_phys_world *world) {
    return world->num_contacts;
}

const rac_phys_contact_manifold* rac_phys_world_get_contacts(
    const rac_phys_world *world) {
    return world->contacts;
}

/* ── Ray cast ──────────────────────────────────────────────────────────── */

rac_phys_ray_hit rac_phys_world_raycast(const rac_phys_world *world,
                                          rac_phys_vec3 origin,
                                          rac_phys_vec3 direction,
                                          float max_distance) {
    rac_phys_ray_hit result;
    memset(&result, 0, sizeof(result));
    result.distance = max_distance;

    rac_phys_vec3 dir = rac_phys_v3_normalize(direction);

    for (int i = 0; i < world->num_bodies; i++) {
        const rac_phys_shape *shape = &world->shapes[i];
        const rac_phys_rigid_body *body = &world->bodies[i];

        if (shape->type == RAC_SHAPE_SPHERE) {
            /* Ray-sphere intersection */
            rac_phys_vec3 oc = rac_phys_v3_sub(origin, body->position);
            float b = rac_phys_v3_dot(oc, dir);
            float c = rac_phys_v3_dot(oc, oc) -
                      shape->sphere.radius * shape->sphere.radius;
            float discriminant = b * b - c;

            if (discriminant >= 0.0f) {
                float t = -b - sqrtf(discriminant);
                if (t > 0.0f && t < result.distance) {
                    result.hit = 1;
                    result.body_index = i;
                    result.distance = t;
                    result.point = rac_phys_v3_add(origin,
                        rac_phys_v3_scale(dir, t));
                    result.normal = rac_phys_v3_normalize(
                        rac_phys_v3_sub(result.point, body->position));
                }
            }
        }
        else if (shape->type == RAC_SHAPE_BOX) {
            /* Ray-OBB intersection via slab method.
             * Transform ray into box-local space, test against AABB. */
            rac_phys_quat inv_rot = rac_phys_quat_conjugate(body->orientation);
            rac_phys_vec3 local_origin = rac_phys_quat_rotate_vec3(
                inv_rot, rac_phys_v3_sub(origin, body->position));
            rac_phys_vec3 local_dir = rac_phys_quat_rotate_vec3(inv_rot, dir);

            rac_phys_vec3 he = shape->box.half_extents;
            float tmin = -1e30f, tmax = 1e30f;
            float lo[3] = { local_origin.x, local_origin.y, local_origin.z };
            float ld[3] = { local_dir.x, local_dir.y, local_dir.z };
            float hea[3] = { he.x, he.y, he.z };
            int valid = 1;

            for (int ax = 0; ax < 3; ax++) {
                if (fabsf(ld[ax]) < 1e-8f) {
                    if (lo[ax] < -hea[ax] || lo[ax] > hea[ax]) { valid = 0; break; }
                } else {
                    float inv_d = 1.0f / ld[ax];
                    float t1 = (-hea[ax] - lo[ax]) * inv_d;
                    float t2 = ( hea[ax] - lo[ax]) * inv_d;
                    if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                    if (t1 > tmin) tmin = t1;
                    if (t2 < tmax) tmax = t2;
                    if (tmin > tmax) { valid = 0; break; }
                }
            }

            if (valid && tmin > 0.0f && tmin < result.distance) {
                result.hit = 1;
                result.body_index = i;
                result.distance = tmin;
                result.point = rac_phys_v3_add(origin,
                    rac_phys_v3_scale(dir, tmin));
                /* Normal: find which slab was hit */
                rac_phys_vec3 local_hit = rac_phys_v3_add(local_origin,
                    rac_phys_v3_scale(local_dir, tmin));
                rac_phys_vec3 local_n = rac_phys_v3_zero();
                float min_dist = 1e30f;
                float lh[3] = { local_hit.x, local_hit.y, local_hit.z };
                for (int ax = 0; ax < 3; ax++) {
                    float d_pos = fabsf(lh[ax] - hea[ax]);
                    float d_neg = fabsf(lh[ax] + hea[ax]);
                    float d = (d_pos < d_neg) ? d_pos : d_neg;
                    if (d < min_dist) {
                        min_dist = d;
                        float *n_arr = &local_n.x;
                        n_arr[0] = n_arr[1] = n_arr[2] = 0;
                        n_arr[ax] = (lh[ax] > 0) ? 1.0f : -1.0f;
                    }
                }
                result.normal = rac_phys_quat_rotate_vec3(
                    body->orientation, local_n);
            }
        }
    }

    return result;
}
