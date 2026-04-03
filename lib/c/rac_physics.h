/*
 * rac_physics.h — RAC Native Physics Library
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * GPU-heritage physics built on RAC rotation-accumulate primitives.
 * Amalgamates the best of CUDA (PhysX/Flex) and AMD (FEMFX/ROCm) physics
 * into a unified, vendor-neutral library that replaces multiply-accumulate
 * with CORDIC rotation at every level of the physics pipeline.
 *
 * Subsystems:
 *   1. Vec3/Quat math           — all via rac_rotate/rac_project/rac_polar
 *   2. Rigid body dynamics      — Euler/Verlet/RK4 integration
 *   3. Collision detection      — spatial hash broad phase + GJK narrow phase
 *   4. Constraint solving       — PGS (PhysX-style) + PBD (Flex/FEMFX-style)
 *   5. Particle systems         — SPH fluid, cloth (PBD constraints)
 *   6. Soft body FEM            — tetrahedral deformation + fracture
 *   7. World management         — scene graph, island solver, sub-stepping
 *
 * All math funnels through RAC primitives. Zero standalone multiplies in
 * the physics compute path — rotations, projections, and polar conversions
 * replace traditional MAC operations.
 *
 * Thread safety: rac_phys_world functions are NOT thread-safe.
 *                Individual math functions (vec3/quat) are pure and re-entrant.
 */

#ifndef RAC_PHYSICS_H
#define RAC_PHYSICS_H

#include "rac_cpu.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * §1  VECTOR & QUATERNION TYPES
 * ════════════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; }       rac_phys_vec3;
typedef struct { float w, x, y, z; }    rac_phys_quat;   /* w + xi + yj + zk */
typedef struct { float m[3][3]; }        rac_phys_mat3;
typedef struct { rac_phys_vec3 min, max; } rac_phys_aabb;

/* ── Vec3 operations (RAC-native) ──────────────────────────────────────── */

rac_phys_vec3 rac_phys_v3_zero(void);
rac_phys_vec3 rac_phys_v3(float x, float y, float z);
rac_phys_vec3 rac_phys_v3_add(rac_phys_vec3 a, rac_phys_vec3 b);
rac_phys_vec3 rac_phys_v3_sub(rac_phys_vec3 a, rac_phys_vec3 b);
rac_phys_vec3 rac_phys_v3_scale(rac_phys_vec3 v, float s);
rac_phys_vec3 rac_phys_v3_negate(rac_phys_vec3 v);
float         rac_phys_v3_dot(rac_phys_vec3 a, rac_phys_vec3 b);
rac_phys_vec3 rac_phys_v3_cross(rac_phys_vec3 a, rac_phys_vec3 b);
float         rac_phys_v3_length(rac_phys_vec3 v);
float         rac_phys_v3_length_sq(rac_phys_vec3 v);
rac_phys_vec3 rac_phys_v3_normalize(rac_phys_vec3 v);
rac_phys_vec3 rac_phys_v3_lerp(rac_phys_vec3 a, rac_phys_vec3 b, float t);

/* ── Quaternion operations (RAC-native) ────────────────────────────────── */

rac_phys_quat rac_phys_quat_identity(void);
rac_phys_quat rac_phys_quat_from_axis_angle(rac_phys_vec3 axis, float angle);
rac_phys_quat rac_phys_quat_mul(rac_phys_quat a, rac_phys_quat b);
rac_phys_quat rac_phys_quat_conjugate(rac_phys_quat q);
rac_phys_quat rac_phys_quat_normalize(rac_phys_quat q);
rac_phys_vec3 rac_phys_quat_rotate_vec3(rac_phys_quat q, rac_phys_vec3 v);
rac_phys_mat3 rac_phys_quat_to_mat3(rac_phys_quat q);
rac_phys_quat rac_phys_quat_slerp(rac_phys_quat a, rac_phys_quat b, float t);

/* ── Mat3 operations ───────────────────────────────────────────────────── */

rac_phys_mat3 rac_phys_mat3_identity(void);
rac_phys_vec3 rac_phys_mat3_mul_vec3(rac_phys_mat3 m, rac_phys_vec3 v);
rac_phys_mat3 rac_phys_mat3_mul(rac_phys_mat3 a, rac_phys_mat3 b);
rac_phys_mat3 rac_phys_mat3_transpose(rac_phys_mat3 m);
rac_phys_mat3 rac_phys_mat3_scale(rac_phys_mat3 m, float s);

/* ── AABB operations ───────────────────────────────────────────────────── */

rac_phys_aabb rac_phys_aabb_from_center_half(rac_phys_vec3 center,
                                              rac_phys_vec3 half_extents);
int           rac_phys_aabb_overlap(rac_phys_aabb a, rac_phys_aabb b);
rac_phys_aabb rac_phys_aabb_merge(rac_phys_aabb a, rac_phys_aabb b);
rac_phys_aabb rac_phys_aabb_expand(rac_phys_aabb box, rac_phys_vec3 margin);

/* ══════════════════════════════════════════════════════════════════════════
 * §2  RIGID BODY DYNAMICS
 * ════════════════════════════════════════════════════════════════════════ */

typedef enum {
    RAC_BODY_STATIC    = 0,   /* infinite mass, never moves                */
    RAC_BODY_DYNAMIC   = 1,   /* fully simulated                           */
    RAC_BODY_KINEMATIC = 2,   /* user-driven, generates contacts but no response */
} rac_phys_body_type;

typedef enum {
    RAC_INTEGRATE_EULER   = 0,  /* symplectic Euler (fast)                 */
    RAC_INTEGRATE_VERLET  = 1,  /* velocity Verlet (energy-stable)         */
    RAC_INTEGRATE_RK4     = 2,  /* Runge-Kutta 4th order (precision)       */
} rac_phys_integrator;

typedef struct {
    uint32_t          id;
    rac_phys_body_type type;

    /* State */
    rac_phys_vec3     position;
    rac_phys_quat     orientation;
    rac_phys_vec3     linear_velocity;
    rac_phys_vec3     angular_velocity;

    /* Mass properties */
    float             mass;
    float             inv_mass;        /* 0 for static bodies */
    rac_phys_mat3     inertia_tensor;  /* body-space inertia */
    rac_phys_mat3     inv_inertia;     /* body-space inverse inertia */

    /* Damping */
    float             linear_damping;   /* [0,1] per step */
    float             angular_damping;

    /* Material */
    float             restitution;      /* bounciness [0,1] */
    float             friction;         /* Coulomb friction [0,1] */

    /* Accumulated forces (cleared each step) */
    rac_phys_vec3     force;
    rac_phys_vec3     torque;

    /* Collision shape index (-1 = no shape) */
    int32_t           shape_index;

    /* Sleeping */
    float             sleep_timer;
    int               is_sleeping;

    /* User data */
    void             *user_data;
} rac_phys_rigid_body;

rac_phys_rigid_body rac_phys_body_create(rac_phys_body_type type, float mass);
void rac_phys_body_set_inertia_box(rac_phys_rigid_body *body,
                                    float hx, float hy, float hz);
void rac_phys_body_set_inertia_sphere(rac_phys_rigid_body *body, float radius);
void rac_phys_body_apply_force(rac_phys_rigid_body *body, rac_phys_vec3 force);
void rac_phys_body_apply_force_at(rac_phys_rigid_body *body,
                                   rac_phys_vec3 force, rac_phys_vec3 point);
void rac_phys_body_apply_torque(rac_phys_rigid_body *body, rac_phys_vec3 torque);
void rac_phys_body_apply_impulse(rac_phys_rigid_body *body, rac_phys_vec3 impulse);
void rac_phys_body_apply_impulse_at(rac_phys_rigid_body *body,
                                     rac_phys_vec3 impulse, rac_phys_vec3 point);
void rac_phys_body_integrate(rac_phys_rigid_body *body, float dt,
                              rac_phys_integrator integrator);

/* ══════════════════════════════════════════════════════════════════════════
 * §3  COLLISION SHAPES & DETECTION
 * ════════════════════════════════════════════════════════════════════════ */

typedef enum {
    RAC_SHAPE_SPHERE   = 0,
    RAC_SHAPE_BOX      = 1,
    RAC_SHAPE_CAPSULE  = 2,
    RAC_SHAPE_CONVEX   = 3,   /* convex hull — GJK/EPA */
    RAC_SHAPE_MESH     = 4,   /* triangle mesh (static only) */
} rac_phys_shape_type;

typedef struct {
    rac_phys_shape_type type;
    union {
        struct { float radius; }                                    sphere;
        struct { rac_phys_vec3 half_extents; }                      box;
        struct { float radius; float half_height; }                 capsule;
        struct { rac_phys_vec3 *vertices; int num_vertices; }       convex;
        struct { rac_phys_vec3 *vertices; int *indices;
                 int num_vertices; int num_triangles; }             mesh;
    };
    rac_phys_vec3 local_offset;   /* shape-space offset from body center */
} rac_phys_shape;

/* Contact manifold — generated by narrow phase */
typedef struct {
    rac_phys_vec3 point;       /* world-space contact point */
    rac_phys_vec3 normal;      /* contact normal (body_a → body_b) */
    float         depth;       /* penetration depth (positive = overlap) */
} rac_phys_contact_point;

#define RAC_MAX_CONTACTS 4

typedef struct {
    int                    body_a;
    int                    body_b;
    int                    num_contacts;
    rac_phys_contact_point contacts[RAC_MAX_CONTACTS];
} rac_phys_contact_manifold;

/* Broad phase — spatial hash grid (GPU-heritage: uniform grid hashing) */

typedef struct rac_phys_spatial_hash rac_phys_spatial_hash;

rac_phys_spatial_hash* rac_phys_spatial_hash_create(float cell_size,
                                                     int table_size);
void  rac_phys_spatial_hash_destroy(rac_phys_spatial_hash *sh);
void  rac_phys_spatial_hash_clear(rac_phys_spatial_hash *sh);
void  rac_phys_spatial_hash_insert(rac_phys_spatial_hash *sh,
                                    rac_phys_aabb aabb, int id);
int   rac_phys_spatial_hash_query(rac_phys_spatial_hash *sh,
                                   rac_phys_aabb aabb,
                                   int *results, int max_results);

/* Narrow phase — collision tests */

int rac_phys_collide_sphere_sphere(rac_phys_vec3 pos_a, float r_a,
                                    rac_phys_vec3 pos_b, float r_b,
                                    rac_phys_contact_manifold *out);

int rac_phys_collide_sphere_box(rac_phys_vec3 sphere_pos, float radius,
                                 rac_phys_vec3 box_pos, rac_phys_quat box_rot,
                                 rac_phys_vec3 half_extents,
                                 rac_phys_contact_manifold *out);

int rac_phys_collide_box_box(rac_phys_vec3 pos_a, rac_phys_quat rot_a,
                              rac_phys_vec3 he_a,
                              rac_phys_vec3 pos_b, rac_phys_quat rot_b,
                              rac_phys_vec3 he_b,
                              rac_phys_contact_manifold *out);

/* GJK support for convex shapes (PhysX/Bullet heritage) */

int rac_phys_gjk_intersect(const rac_phys_vec3 *verts_a, int n_a,
                            rac_phys_vec3 pos_a, rac_phys_quat rot_a,
                            const rac_phys_vec3 *verts_b, int n_b,
                            rac_phys_vec3 pos_b, rac_phys_quat rot_b,
                            rac_phys_contact_manifold *out);

/* ══════════════════════════════════════════════════════════════════════════
 * §4  CONSTRAINT SOLVERS
 * ════════════════════════════════════════════════════════════════════════ */

typedef enum {
    RAC_CONSTRAINT_DISTANCE  = 0,  /* maintain fixed distance             */
    RAC_CONSTRAINT_CONTACT   = 1,  /* non-penetration + friction           */
    RAC_CONSTRAINT_HINGE     = 2,  /* rotation around single axis          */
    RAC_CONSTRAINT_BALL      = 3,  /* ball-and-socket (3 DOF rotation)     */
    RAC_CONSTRAINT_SLIDER    = 4,  /* translation along single axis        */
    RAC_CONSTRAINT_D6        = 5,  /* configurable 6-DOF (PhysX D6-style)  */
} rac_phys_constraint_type;

typedef struct {
    rac_phys_constraint_type type;
    int body_a;
    int body_b;     /* -1 = world anchor */

    /* Anchors in body-local space */
    rac_phys_vec3 anchor_a;
    rac_phys_vec3 anchor_b;
    rac_phys_vec3 axis_a;     /* for hinge/slider */
    rac_phys_vec3 axis_b;

    /* Parameters */
    float rest_length;        /* for distance constraints */
    float stiffness;          /* PBD compliance (0 = rigid) */
    float damping;
    float max_force;          /* force limit (0 = unlimited) */

    /* D6 limits */
    float limit_lower[6];     /* [tx, ty, tz, rx, ry, rz] */
    float limit_upper[6];

    /* Solver state (internal) */
    float lambda;             /* accumulated impulse (PGS warm-start) */
} rac_phys_constraint;

/* PGS solver — PhysX/Bullet heritage (Sequential Impulse) */

typedef struct {
    int   iterations;         /* solver iterations (default 8) */
    float slop;               /* allowed penetration (default 0.005) */
    float baumgarte;          /* position correction bias (default 0.2) */
    float warm_start_factor;  /* impulse warm-starting (default 0.85) */
} rac_phys_pgs_config;

rac_phys_pgs_config rac_phys_pgs_default_config(void);

void rac_phys_pgs_solve(rac_phys_rigid_body *bodies, int num_bodies,
                         rac_phys_constraint *constraints, int num_constraints,
                         rac_phys_contact_manifold *contacts, int num_contacts,
                         const rac_phys_pgs_config *cfg, float dt);

/* PBD solver — Flex/FEMFX heritage (Position-Based Dynamics) */

typedef struct {
    int   substeps;           /* PBD sub-steps (default 4) */
    int   iterations;         /* constraint iterations per substep (default 2) */
    float damping;            /* global velocity damping (default 0.99) */
} rac_phys_pbd_config;

rac_phys_pbd_config rac_phys_pbd_default_config(void);

void rac_phys_pbd_solve_distance(rac_phys_vec3 *positions,
                                  float *inv_masses,
                                  const int *constraint_pairs, /* [a0,b0,a1,b1,...] */
                                  const float *rest_lengths,
                                  int num_constraints,
                                  const rac_phys_pbd_config *cfg);

/* ══════════════════════════════════════════════════════════════════════════
 * §5  PARTICLE SYSTEMS & FLUIDS
 * ════════════════════════════════════════════════════════════════════════ */

/* ── Particle system ───────────────────────────────────────────────────── */

typedef struct {
    int              num_particles;
    int              max_particles;
    rac_phys_vec3   *positions;
    rac_phys_vec3   *velocities;
    rac_phys_vec3   *forces;
    float           *masses;
    float           *inv_masses;
    float           *densities;        /* for SPH */
    float           *pressures;        /* for SPH */
    int             *alive;            /* 1 = active, 0 = dead */

    /* Per-particle phase (Flex-style: fluid vs rigid vs cloth) */
    int             *phase;
} rac_phys_particle_system;

rac_phys_particle_system* rac_phys_particles_create(int max_particles);
void  rac_phys_particles_destroy(rac_phys_particle_system *ps);
int   rac_phys_particles_emit(rac_phys_particle_system *ps,
                               rac_phys_vec3 position,
                               rac_phys_vec3 velocity, float mass);
void  rac_phys_particles_integrate(rac_phys_particle_system *ps,
                                    rac_phys_vec3 gravity, float dt);

/* ── SPH fluid simulation (GPU-heritage: CUDA particle sim) ────────────── */

typedef struct {
    float rest_density;       /* target fluid density (default 1000 kg/m³) */
    float gas_constant;       /* pressure-density constant (default 2000)  */
    float viscosity;          /* dynamic viscosity (default 0.01)          */
    float smoothing_radius;   /* kernel support radius (default 0.1)       */
    float particle_mass;      /* mass per particle                         */
    float surface_tension;    /* surface tension coefficient               */
    float boundary_damping;   /* velocity damping at boundaries            */
} rac_phys_sph_config;

rac_phys_sph_config rac_phys_sph_default_config(void);

void rac_phys_sph_compute_density_pressure(rac_phys_particle_system *ps,
                                            rac_phys_spatial_hash *grid,
                                            const rac_phys_sph_config *cfg);

void rac_phys_sph_compute_forces(rac_phys_particle_system *ps,
                                  rac_phys_spatial_hash *grid,
                                  const rac_phys_sph_config *cfg);

void rac_phys_sph_step(rac_phys_particle_system *ps,
                        rac_phys_spatial_hash *grid,
                        rac_phys_vec3 gravity,
                        const rac_phys_sph_config *cfg, float dt);

/* ── Cloth simulation (PBD — Flex heritage) ────────────────────────────── */

typedef struct {
    rac_phys_particle_system *particles;

    /* Stretch constraints (edges) */
    int   *stretch_pairs;      /* [a0,b0,a1,b1,...] */
    float *stretch_rest;       /* rest lengths */
    int    num_stretch;

    /* Bend constraints (across edges) */
    int   *bend_pairs;
    float *bend_rest;
    int    num_bend;

    /* Simulation parameters */
    float stretch_stiffness;   /* [0,1] */
    float bend_stiffness;      /* [0,1] */
    int   solver_iterations;
} rac_phys_cloth;

rac_phys_cloth* rac_phys_cloth_create_grid(int width, int height,
                                            float spacing, float mass);
void  rac_phys_cloth_destroy(rac_phys_cloth *cloth);
void  rac_phys_cloth_pin(rac_phys_cloth *cloth, int particle_index);
void  rac_phys_cloth_step(rac_phys_cloth *cloth,
                           rac_phys_vec3 gravity, float dt);

/* ══════════════════════════════════════════════════════════════════════════
 * §6  SOFT BODY FEM (AMD FEMFX + PhysX heritage)
 * ════════════════════════════════════════════════════════════════════════ */

/* Tetrahedral element */
typedef struct {
    int indices[4];            /* vertex indices */
    float rest_volume;
    rac_phys_mat3 Dm_inv;     /* inverse reference shape matrix */
    float youngs_modulus;      /* material stiffness */
    float poisson_ratio;       /* lateral contraction ratio */
    float plastic_strain;      /* accumulated plastic deformation (FEMFX-style) */
    float fracture_threshold;  /* stress threshold for fracture (0 = unbreakable) */
} rac_phys_tet_element;

typedef struct {
    int                   num_vertices;
    int                   num_elements;
    rac_phys_vec3        *positions;
    rac_phys_vec3        *velocities;
    float                *inv_masses;
    rac_phys_tet_element *elements;

    /* Surface for collision (subset of tet faces) */
    int                  *surface_triangles;  /* [v0,v1,v2,...] */
    int                   num_surface_tris;

    /* Simulation */
    float                 damping;
    int                   solver_iterations;
} rac_phys_soft_body;

rac_phys_soft_body* rac_phys_softbody_create(int num_vertices, int num_elements);
void  rac_phys_softbody_destroy(rac_phys_soft_body *sb);
void  rac_phys_softbody_compute_rest_state(rac_phys_soft_body *sb);
void  rac_phys_softbody_step(rac_phys_soft_body *sb,
                              rac_phys_vec3 gravity, float dt);

/* Generate a simple beam soft body for testing */
rac_phys_soft_body* rac_phys_softbody_create_beam(float length, float width,
                                                    float height, int segments,
                                                    float density,
                                                    float youngs_modulus);

/* ══════════════════════════════════════════════════════════════════════════
 * §7  PHYSICS WORLD
 * ════════════════════════════════════════════════════════════════════════ */

typedef struct {
    rac_phys_vec3         gravity;
    rac_phys_integrator   integrator;
    float                 fixed_dt;           /* fixed timestep (default 1/60) */
    int                   max_substeps;       /* max sub-steps per frame */
    float                 sleep_threshold;    /* velocity threshold for sleeping */
    float                 sleep_time;         /* seconds idle before sleeping */

    /* Solver config */
    rac_phys_pgs_config   pgs_config;

    /* Broad phase */
    float                 broad_phase_cell_size;
    int                   broad_phase_table_size;
} rac_phys_world_config;

rac_phys_world_config rac_phys_world_default_config(void);

typedef struct rac_phys_world rac_phys_world;

rac_phys_world* rac_phys_world_create(const rac_phys_world_config *cfg);
void  rac_phys_world_destroy(rac_phys_world *world);

/* Body management */
int   rac_phys_world_add_body(rac_phys_world *world,
                               rac_phys_rigid_body body, rac_phys_shape shape);
rac_phys_rigid_body* rac_phys_world_get_body(rac_phys_world *world, int index);

/* Constraint management */
int   rac_phys_world_add_constraint(rac_phys_world *world,
                                     rac_phys_constraint constraint);

/* Step the world */
void  rac_phys_world_step(rac_phys_world *world, float dt);

/* Query */
int   rac_phys_world_num_bodies(const rac_phys_world *world);
int   rac_phys_world_num_contacts(const rac_phys_world *world);
const rac_phys_contact_manifold* rac_phys_world_get_contacts(
    const rac_phys_world *world);

/* Ray cast */
typedef struct {
    int           hit;
    int           body_index;
    rac_phys_vec3 point;
    rac_phys_vec3 normal;
    float         distance;
} rac_phys_ray_hit;

rac_phys_ray_hit rac_phys_world_raycast(const rac_phys_world *world,
                                         rac_phys_vec3 origin,
                                         rac_phys_vec3 direction,
                                         float max_distance);

#ifdef __cplusplus
}
#endif

#endif /* RAC_PHYSICS_H */
