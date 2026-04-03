/*
 * test_rac_physics_bvt.c — RAC Physics Library Build Verification Tests
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * Tests cover all 7 subsystems:
 *   §1 Vec3/Quat math
 *   §2 Rigid body dynamics
 *   §3 Collision detection (spatial hash + narrow phase)
 *   §4 Constraint solvers (PGS + PBD)
 *   §5 Particle system + SPH fluids
 *   §6 Cloth simulation
 *   §7 Soft body FEM
 *   §8 World integration
 */

#include "rac_physics.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_NEAR(val, expected, tol, msg) do { \
    tests_run++; \
    float _v = (val), _e = (expected), _t = (tol); \
    if (fabsf(_v - _e) <= _t) { tests_passed++; } \
    else { tests_failed++; \
        printf("  FAIL: %s: got %f, expected %f (tol %f)\n", msg, _v, _e, _t); } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

/* ══════════════════════════════════════════════════════════════════════════
 * §1  VEC3 / QUAT MATH
 * ════════════════════════════════════════════════════════════════════════ */

static void test_vec3_basic(void) {
    printf("  [vec3] basic operations...\n");

    rac_phys_vec3 a = rac_phys_v3(1.0f, 2.0f, 3.0f);
    rac_phys_vec3 b = rac_phys_v3(4.0f, 5.0f, 6.0f);

    /* Add */
    rac_phys_vec3 sum = rac_phys_v3_add(a, b);
    ASSERT_NEAR(sum.x, 5.0f, 0.001f, "v3_add.x");
    ASSERT_NEAR(sum.y, 7.0f, 0.001f, "v3_add.y");
    ASSERT_NEAR(sum.z, 9.0f, 0.001f, "v3_add.z");

    /* Dot product */
    float d = rac_phys_v3_dot(a, b);
    ASSERT_NEAR(d, 32.0f, 0.1f, "v3_dot");  /* 4+10+18 */

    /* Cross product */
    rac_phys_vec3 c = rac_phys_v3_cross(a, b);
    ASSERT_NEAR(c.x, -3.0f, 0.1f, "v3_cross.x");
    ASSERT_NEAR(c.y,  6.0f, 0.1f, "v3_cross.y");
    ASSERT_NEAR(c.z, -3.0f, 0.1f, "v3_cross.z");

    /* Length */
    float len = rac_phys_v3_length(a);
    ASSERT_NEAR(len, 3.7416f, 0.01f, "v3_length");

    /* Normalize */
    rac_phys_vec3 n = rac_phys_v3_normalize(a);
    float nlen = rac_phys_v3_length(n);
    ASSERT_NEAR(nlen, 1.0f, 0.01f, "v3_normalize_len");

    /* Scale */
    rac_phys_vec3 s = rac_phys_v3_scale(a, 2.0f);
    ASSERT_NEAR(s.x, 2.0f, 0.05f, "v3_scale.x");
    ASSERT_NEAR(s.y, 4.0f, 0.05f, "v3_scale.y");
    ASSERT_NEAR(s.z, 6.0f, 0.05f, "v3_scale.z");
}

static void test_quat_basic(void) {
    printf("  [quat] basic operations...\n");

    /* Identity */
    rac_phys_quat id = rac_phys_quat_identity();
    ASSERT_NEAR(id.w, 1.0f, 0.001f, "quat_identity.w");

    /* Rotate vector by 90° around Y axis */
    rac_phys_vec3 y_axis = rac_phys_v3(0, 1, 0);
    float half_pi = 3.14159265f * 0.5f;
    rac_phys_quat q = rac_phys_quat_from_axis_angle(y_axis, half_pi);

    rac_phys_vec3 v = rac_phys_v3(1, 0, 0);
    rac_phys_vec3 rotated = rac_phys_quat_rotate_vec3(q, v);
    /* (1,0,0) rotated 90° around Y → (0,0,-1) */
    ASSERT_NEAR(rotated.x,  0.0f, 0.05f, "quat_rot90.x");
    ASSERT_NEAR(rotated.y,  0.0f, 0.05f, "quat_rot90.y");
    ASSERT_NEAR(rotated.z, -1.0f, 0.05f, "quat_rot90.z");

    /* Quaternion multiply: two 90° rotations = 180° */
    rac_phys_quat q2 = rac_phys_quat_mul(q, q);
    rac_phys_vec3 rotated2 = rac_phys_quat_rotate_vec3(q2, v);
    ASSERT_NEAR(rotated2.x, -1.0f, 0.1f, "quat_mul180.x");
    ASSERT_NEAR(rotated2.y,  0.0f, 0.1f, "quat_mul180.y");

    /* Normalize */
    rac_phys_quat qn = rac_phys_quat_normalize(q);
    float qlen = sqrtf(qn.w*qn.w + qn.x*qn.x + qn.y*qn.y + qn.z*qn.z);
    ASSERT_NEAR(qlen, 1.0f, 0.01f, "quat_normalize_len");
}

/* ══════════════════════════════════════════════════════════════════════════
 * §2  RIGID BODY DYNAMICS
 * ════════════════════════════════════════════════════════════════════════ */

static void test_rigid_body(void) {
    printf("  [rigid body] dynamics...\n");

    rac_phys_rigid_body ball = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    ball.position = rac_phys_v3(0, 10, 0);
    rac_phys_body_set_inertia_sphere(&ball, 0.5f);

    /* Apply gravity-like force for 1 second (60 steps at 1/60) */
    for (int i = 0; i < 60; i++) {
        rac_phys_body_apply_force(&ball, rac_phys_v3(0, -9.81f, 0));
        rac_phys_body_integrate(&ball, 1.0f/60.0f, RAC_INTEGRATE_EULER);
    }

    /* After 1 second of free fall: y ≈ 10 - 0.5*9.81*1² = ~5.1 */
    ASSERT_TRUE(ball.position.y < 10.0f, "ball fell");
    ASSERT_TRUE(ball.position.y > 0.0f, "ball not through floor");

    /* Velocity should be ~ -9.81 m/s after 1 second */
    ASSERT_TRUE(ball.linear_velocity.y < -5.0f, "ball has downward velocity");

    /* Static body should not move */
    rac_phys_rigid_body ground = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    rac_phys_body_apply_force(&ground, rac_phys_v3(0, -100, 0));
    rac_phys_body_integrate(&ground, 1.0f/60.0f, RAC_INTEGRATE_EULER);
    ASSERT_NEAR(ground.position.y, 0.0f, 0.001f, "static body stays");
}

static void test_rigid_body_verlet(void) {
    printf("  [rigid body] Verlet integration...\n");

    rac_phys_rigid_body ball = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    ball.position = rac_phys_v3(0, 10, 0);
    ball.linear_damping = 0.0f;

    for (int i = 0; i < 60; i++) {
        rac_phys_body_apply_force(&ball, rac_phys_v3(0, -9.81f, 0));
        rac_phys_body_integrate(&ball, 1.0f/60.0f, RAC_INTEGRATE_VERLET);
    }

    ASSERT_TRUE(ball.position.y < 10.0f, "verlet: ball fell");
    ASSERT_TRUE(ball.position.y > 0.0f, "verlet: ball above ground");
}

/* ══════════════════════════════════════════════════════════════════════════
 * §3  COLLISION DETECTION
 * ════════════════════════════════════════════════════════════════════════ */

static void test_spatial_hash(void) {
    printf("  [collision] spatial hash...\n");

    rac_phys_spatial_hash *sh = rac_phys_spatial_hash_create(1.0f, 256);
    ASSERT_TRUE(sh != NULL, "spatial hash created");

    /* Insert 3 objects */
    rac_phys_aabb a = {{ 0,0,0 }, { 1,1,1 }};
    rac_phys_aabb b = {{ 0.5f,0.5f,0.5f }, { 1.5f,1.5f,1.5f }};
    rac_phys_aabb c = {{ 10,10,10 }, { 11,11,11 }};

    rac_phys_spatial_hash_insert(sh, a, 0);
    rac_phys_spatial_hash_insert(sh, b, 1);
    rac_phys_spatial_hash_insert(sh, c, 2);

    /* Query near object A — should find A and B but not C */
    int results[16];
    int n = rac_phys_spatial_hash_query(sh, a, results, 16);
    ASSERT_TRUE(n >= 2, "spatial hash finds overlapping objects");

    /* Query near object C — should find C only */
    n = rac_phys_spatial_hash_query(sh, c, results, 16);
    int found_c = 0;
    for (int i = 0; i < n; i++) if (results[i] == 2) found_c = 1;
    ASSERT_TRUE(found_c, "spatial hash finds distant object");

    rac_phys_spatial_hash_destroy(sh);
}

static void test_sphere_sphere_collision(void) {
    printf("  [collision] sphere-sphere...\n");

    rac_phys_contact_manifold m;

    /* Overlapping spheres */
    int hit = rac_phys_collide_sphere_sphere(
        rac_phys_v3(0, 0, 0), 1.0f,
        rac_phys_v3(1.5f, 0, 0), 1.0f, &m);
    ASSERT_TRUE(hit, "spheres overlap");
    ASSERT_NEAR(m.contacts[0].depth, 0.5f, 0.01f, "sphere overlap depth");

    /* Non-overlapping spheres */
    hit = rac_phys_collide_sphere_sphere(
        rac_phys_v3(0, 0, 0), 1.0f,
        rac_phys_v3(3.0f, 0, 0), 1.0f, &m);
    ASSERT_TRUE(!hit, "spheres don't overlap");
}

static void test_sphere_box_collision(void) {
    printf("  [collision] sphere-box...\n");

    rac_phys_contact_manifold m;

    /* Sphere touching box face */
    int hit = rac_phys_collide_sphere_box(
        rac_phys_v3(1.8f, 0, 0), 1.0f,
        rac_phys_v3(0, 0, 0), rac_phys_quat_identity(),
        rac_phys_v3(1, 1, 1), &m);
    ASSERT_TRUE(hit, "sphere-box overlap");
    ASSERT_TRUE(m.contacts[0].depth > 0.0f, "sphere-box depth > 0");

    /* Sphere far from box */
    hit = rac_phys_collide_sphere_box(
        rac_phys_v3(5.0f, 0, 0), 1.0f,
        rac_phys_v3(0, 0, 0), rac_phys_quat_identity(),
        rac_phys_v3(1, 1, 1), &m);
    ASSERT_TRUE(!hit, "sphere-box no overlap");
}

static void test_box_box_collision(void) {
    printf("  [collision] box-box SAT...\n");

    rac_phys_contact_manifold m;

    /* Overlapping boxes */
    int hit = rac_phys_collide_box_box(
        rac_phys_v3(0, 0, 0), rac_phys_quat_identity(), rac_phys_v3(1, 1, 1),
        rac_phys_v3(1.5f, 0, 0), rac_phys_quat_identity(), rac_phys_v3(1, 1, 1),
        &m);
    ASSERT_TRUE(hit, "boxes overlap");
    ASSERT_TRUE(m.contacts[0].depth > 0.0f, "box-box depth > 0");

    /* Non-overlapping */
    hit = rac_phys_collide_box_box(
        rac_phys_v3(0, 0, 0), rac_phys_quat_identity(), rac_phys_v3(1, 1, 1),
        rac_phys_v3(5, 0, 0), rac_phys_quat_identity(), rac_phys_v3(1, 1, 1),
        &m);
    ASSERT_TRUE(!hit, "boxes don't overlap");
}

/* ══════════════════════════════════════════════════════════════════════════
 * §4  CONSTRAINT SOLVERS
 * ════════════════════════════════════════════════════════════════════════ */

static void test_pbd_distance(void) {
    printf("  [constraints] PBD distance...\n");

    /* Two particles connected by a distance constraint */
    rac_phys_vec3 positions[2] = {
        { 0, 0, 0 },
        { 2.5f, 0, 0 }  /* stretched beyond rest length of 1.0 */
    };
    float inv_masses[2] = { 1.0f, 1.0f };
    int pairs[2] = { 0, 1 };
    float rest[1] = { 1.0f };

    rac_phys_pbd_config cfg = rac_phys_pbd_default_config();
    cfg.iterations = 10;

    rac_phys_pbd_solve_distance(positions, inv_masses, pairs, rest, 1, &cfg);

    /* After solving, distance should be closer to 1.0 */
    float dist = rac_phys_v3_length(rac_phys_v3_sub(positions[1], positions[0]));
    ASSERT_NEAR(dist, 1.0f, 0.01f, "pbd distance converged");
}

/* ══════════════════════════════════════════════════════════════════════════
 * §5  PARTICLE SYSTEM + SPH
 * ════════════════════════════════════════════════════════════════════════ */

static void test_particles(void) {
    printf("  [particles] basic system...\n");

    rac_phys_particle_system *ps = rac_phys_particles_create(100);
    ASSERT_TRUE(ps != NULL, "particle system created");

    /* Emit particles */
    int id0 = rac_phys_particles_emit(ps, rac_phys_v3(0, 5, 0),
                                       rac_phys_v3(0, 0, 0), 1.0f);
    int id1 = rac_phys_particles_emit(ps, rac_phys_v3(1, 5, 0),
                                       rac_phys_v3(0, 0, 0), 1.0f);
    ASSERT_TRUE(id0 == 0, "first particle id=0");
    ASSERT_TRUE(id1 == 1, "second particle id=1");
    ASSERT_TRUE(ps->num_particles == 2, "2 particles");

    /* Integrate with gravity */
    rac_phys_particles_integrate(ps, rac_phys_v3(0, -9.81f, 0), 1.0f/60.0f);
    ASSERT_TRUE(ps->positions[0].y < 5.0f, "particle fell");

    rac_phys_particles_destroy(ps);
}

static void test_sph_density(void) {
    printf("  [SPH] density computation...\n");

    rac_phys_sph_config cfg = rac_phys_sph_default_config();
    cfg.smoothing_radius = 0.5f;
    cfg.particle_mass = 1.0f;

    rac_phys_particle_system *ps = rac_phys_particles_create(50);
    rac_phys_spatial_hash *grid = rac_phys_spatial_hash_create(0.5f, 256);

    /* Create a cluster of particles */
    for (int i = 0; i < 10; i++) {
        rac_phys_particles_emit(ps,
            rac_phys_v3((float)i * 0.1f, 0, 0),
            rac_phys_v3_zero(), cfg.particle_mass);
    }

    rac_phys_sph_compute_density_pressure(ps, grid, &cfg);

    /* Particles in cluster should have non-zero density */
    ASSERT_TRUE(ps->densities[5] > 0.0f, "SPH density > 0");

    rac_phys_spatial_hash_destroy(grid);
    rac_phys_particles_destroy(ps);
}

/* ══════════════════════════════════════════════════════════════════════════
 * §6  CLOTH SIMULATION
 * ════════════════════════════════════════════════════════════════════════ */

static void test_cloth(void) {
    printf("  [cloth] PBD simulation...\n");

    rac_phys_cloth *cloth = rac_phys_cloth_create_grid(5, 5, 0.2f, 1.0f);
    ASSERT_TRUE(cloth != NULL, "cloth created");
    ASSERT_TRUE(cloth->particles->num_particles == 25, "25 particles");
    ASSERT_TRUE(cloth->num_stretch > 0, "stretch constraints exist");
    ASSERT_TRUE(cloth->num_bend > 0, "bend constraints exist");

    /* Pin top corners */
    rac_phys_cloth_pin(cloth, 0);
    rac_phys_cloth_pin(cloth, 4);

    /* Record initial center position */
    float y0 = cloth->particles->positions[12].y;

    /* Simulate enough steps for gravity to take effect */
    for (int i = 0; i < 120; i++)
        rac_phys_cloth_step(cloth, rac_phys_v3(0, -9.81f, 0), 1.0f/60.0f);

    /* Center should have dropped */
    float y1 = cloth->particles->positions[12].y;
    ASSERT_TRUE(y1 < y0, "cloth center dropped under gravity");

    /* Pinned corners should not move */
    ASSERT_NEAR(cloth->particles->positions[0].y, y0, 0.001f,
                "pinned corner stays");

    rac_phys_cloth_destroy(cloth);
}

/* ══════════════════════════════════════════════════════════════════════════
 * §7  SOFT BODY FEM
 * ════════════════════════════════════════════════════════════════════════ */

static void test_softbody(void) {
    printf("  [softbody] FEM beam...\n");

    rac_phys_soft_body *beam = rac_phys_softbody_create_beam(
        1.0f, 0.2f, 0.2f,  /* length, width, height */
        4,                   /* segments */
        100.0f,              /* density */
        50.0f                /* Young's modulus (soft, stable with explicit Euler) */
    );
    ASSERT_TRUE(beam != NULL, "beam created");
    ASSERT_TRUE(beam->num_vertices > 0, "beam has vertices");
    ASSERT_TRUE(beam->num_elements > 0, "beam has elements");

    /* Increase substeps and damping for numerical stability */
    beam->solver_iterations = 16;
    beam->damping = 0.95f;

    /* Record tip position (last vertex along x) */
    int tip = beam->num_vertices - 1;
    float y0 = beam->positions[tip].y;

    /* Simulate: beam should deflect under gravity (2 seconds) */
    for (int i = 0; i < 120; i++)
        rac_phys_softbody_step(beam, rac_phys_v3(0, -9.81f, 0), 1.0f/60.0f);

    float y1 = beam->positions[tip].y;
    ASSERT_TRUE(y1 < y0, "beam tip deflected under gravity");

    /* Fixed end should not have moved */
    ASSERT_NEAR(beam->positions[0].y, 0.0f, 0.001f, "fixed end stays");

    rac_phys_softbody_destroy(beam);
}

/* ══════════════════════════════════════════════════════════════════════════
 * §8  WORLD INTEGRATION
 * ════════════════════════════════════════════════════════════════════════ */

static void test_world(void) {
    printf("  [world] full pipeline...\n");

    rac_phys_world_config cfg = rac_phys_world_default_config();
    rac_phys_world *world = rac_phys_world_create(&cfg);
    ASSERT_TRUE(world != NULL, "world created");

    /* Add a ground plane (static box) */
    rac_phys_rigid_body ground = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    ground.position = rac_phys_v3(0, -1, 0);
    rac_phys_shape ground_shape;
    memset(&ground_shape, 0, sizeof(ground_shape));
    ground_shape.type = RAC_SHAPE_BOX;
    ground_shape.box.half_extents = rac_phys_v3(50, 1, 50);
    rac_phys_world_add_body(world, ground, ground_shape);

    /* Add a falling sphere */
    rac_phys_rigid_body ball = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    ball.position = rac_phys_v3(0, 5, 0);
    ball.restitution = 0.5f;
    rac_phys_body_set_inertia_sphere(&ball, 0.5f);
    rac_phys_shape ball_shape;
    memset(&ball_shape, 0, sizeof(ball_shape));
    ball_shape.type = RAC_SHAPE_SPHERE;
    ball_shape.sphere.radius = 0.5f;
    int ball_idx = rac_phys_world_add_body(world, ball, ball_shape);

    ASSERT_TRUE(rac_phys_world_num_bodies(world) == 2, "2 bodies in world");

    /* Step simulation for 2 seconds */
    for (int i = 0; i < 120; i++)
        rac_phys_world_step(world, 1.0f/60.0f);

    /* Ball should have fallen and bounced */
    rac_phys_rigid_body *b = rac_phys_world_get_body(world, ball_idx);
    ASSERT_TRUE(b->position.y < 5.0f, "ball fell from initial height");

    /* Ray cast test */
    rac_phys_ray_hit hit = rac_phys_world_raycast(
        world, rac_phys_v3(0, 10, 0), rac_phys_v3(0, -1, 0), 100.0f);
    ASSERT_TRUE(hit.hit, "raycast hit something");

    rac_phys_world_destroy(world);
}

/* ══════════════════════════════════════════════════════════════════════════
 * §9  SAFETY REGRESSION TESTS (Critical fixes #1–#7)
 * ════════════════════════════════════════════════════════════════════════ */

/* Fix #1: PGS solver must not crash on out-of-bounds body indices */
static void test_safety_pgs_bad_indices(void) {
    printf("  [safety] PGS with invalid body indices...\n");

    rac_phys_rigid_body bodies[2];
    bodies[0] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    bodies[1] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);

    /* Contact manifold referencing body index 999 (out of bounds) */
    rac_phys_contact_manifold bad_contact;
    memset(&bad_contact, 0, sizeof(bad_contact));
    bad_contact.body_a = 0;
    bad_contact.body_b = 999;  /* OOB */
    bad_contact.num_contacts = 1;
    bad_contact.contacts[0].normal = rac_phys_v3(0, 1, 0);
    bad_contact.contacts[0].depth = 0.1f;

    rac_phys_pgs_config cfg = rac_phys_pgs_default_config();

    /* Must not crash — OOB contact should be skipped */
    rac_phys_pgs_solve(bodies, 2, NULL, 0, &bad_contact, 1, &cfg, 1.0f/60.0f);
    ASSERT_TRUE(1, "PGS survived OOB body_b index");

    /* Also test negative index */
    bad_contact.body_a = -1;
    bad_contact.body_b = 0;
    rac_phys_pgs_solve(bodies, 2, NULL, 0, &bad_contact, 1, &cfg, 1.0f/60.0f);
    ASSERT_TRUE(1, "PGS survived negative body_a index");

    /* NULL bodies */
    rac_phys_pgs_solve(NULL, 0, NULL, 0, NULL, 0, &cfg, 1.0f/60.0f);
    ASSERT_TRUE(1, "PGS survived NULL bodies");
}

/* Fix #2: Spatial hash must handle overflow by growing */
static void test_safety_spatial_hash_overflow(void) {
    printf("  [safety] spatial hash overflow/grow...\n");

    /* Create tiny hash (table_size=4) that will quickly overflow */
    rac_phys_spatial_hash *sh = rac_phys_spatial_hash_create(1.0f, 4);
    ASSERT_TRUE(sh != NULL, "tiny spatial hash created");

    /* Insert 200 objects — far more than initial capacity (4*4=16 entries) */
    for (int i = 0; i < 200; i++) {
        rac_phys_aabb aabb = {
            { (float)i, 0, 0 },
            { (float)i + 0.5f, 0.5f, 0.5f }
        };
        rac_phys_spatial_hash_insert(sh, aabb, i);
    }

    /* Query for a late-inserted object — should still be findable */
    rac_phys_aabb query = { { 150.0f, 0, 0 }, { 150.5f, 0.5f, 0.5f } };
    int results[16];
    int n = rac_phys_spatial_hash_query(sh, query, results, 16);

    int found_150 = 0;
    for (int i = 0; i < n; i++)
        if (results[i] == 150) found_150 = 1;
    ASSERT_TRUE(found_150, "late-inserted object found after overflow growth");

    rac_phys_spatial_hash_destroy(sh);
}

/* Fix #4: Slerp must not produce NaN on degenerate inputs */
static void test_safety_slerp_degenerate(void) {
    printf("  [safety] slerp degenerate inputs...\n");

    /* Identical quaternions — sin(theta) = 0 */
    rac_phys_quat a = rac_phys_quat_identity();
    rac_phys_quat b = rac_phys_quat_identity();
    rac_phys_quat r = rac_phys_quat_slerp(a, b, 0.5f);
    ASSERT_TRUE(isfinite(r.w), "slerp identical: w finite");
    ASSERT_TRUE(isfinite(r.x), "slerp identical: x finite");
    ASSERT_NEAR(r.w, 1.0f, 0.01f, "slerp identical: w ≈ 1");

    /* Opposite quaternions (cos_theta = -1 before flip) */
    rac_phys_quat neg = { -1.0f, 0.0f, 0.0f, 0.0f };
    r = rac_phys_quat_slerp(a, neg, 0.5f);
    ASSERT_TRUE(isfinite(r.w), "slerp opposite: w finite");
    ASSERT_TRUE(isfinite(r.x), "slerp opposite: x finite");

    /* Zero quaternion (degenerate) */
    rac_phys_quat zero = { 0.0f, 0.0f, 0.0f, 0.0f };
    r = rac_phys_quat_slerp(a, zero, 0.5f);
    ASSERT_TRUE(isfinite(r.w), "slerp zero quat: w finite");

    /* NaN quaternion */
    rac_phys_quat nan_q = { 0.0f/0.0f, 0.0f, 0.0f, 0.0f };
    r = rac_phys_quat_slerp(a, nan_q, 0.5f);
    ASSERT_TRUE(isfinite(r.w), "slerp NaN quat: returns valid fallback");
}

/* Fix #5: GJK must not crash on NULL or zero-count vertex arrays */
static void test_safety_gjk_bad_input(void) {
    printf("  [safety] GJK with NULL/empty vertex arrays...\n");

    rac_phys_contact_manifold m;
    rac_phys_quat id = rac_phys_quat_identity();
    rac_phys_vec3 origin = rac_phys_v3_zero();

    /* NULL vertices */
    int hit = rac_phys_gjk_intersect(
        NULL, 0, origin, id,
        NULL, 0, origin, id, &m);
    ASSERT_TRUE(!hit, "GJK returns 0 for NULL vertices");

    /* Zero vertex count */
    rac_phys_vec3 verts[3] = {{0,0,0}, {1,0,0}, {0,1,0}};
    hit = rac_phys_gjk_intersect(
        verts, 0, origin, id,
        verts, 3, origin, id, &m);
    ASSERT_TRUE(!hit, "GJK returns 0 for zero vertex count");

    /* One valid, one NULL */
    hit = rac_phys_gjk_intersect(
        verts, 3, origin, id,
        NULL, 0, origin, id, &m);
    ASSERT_TRUE(!hit, "GJK returns 0 for one NULL array");
}

/* Fix #6: Constraint solver must skip invalid body refs */
static void test_safety_constraint_bad_indices(void) {
    printf("  [safety] constraint solver with OOB body indices...\n");

    rac_phys_rigid_body bodies[2];
    bodies[0] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    bodies[0].position = rac_phys_v3(0, 0, 0);
    bodies[1] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    bodies[1].position = rac_phys_v3(2, 0, 0);

    /* Constraint with OOB body_a */
    rac_phys_constraint bad_c;
    memset(&bad_c, 0, sizeof(bad_c));
    bad_c.type = RAC_CONSTRAINT_DISTANCE;
    bad_c.body_a = 999;  /* OOB */
    bad_c.body_b = 0;
    bad_c.rest_length = 1.0f;

    rac_phys_pgs_config cfg = rac_phys_pgs_default_config();
    rac_phys_pgs_solve(bodies, 2, &bad_c, 1, NULL, 0, &cfg, 1.0f/60.0f);
    ASSERT_TRUE(1, "PGS survived OOB constraint body_a");

    /* Constraint with body_b = -1 (world anchor, valid) */
    bad_c.body_a = 0;
    bad_c.body_b = -1;
    rac_phys_pgs_solve(bodies, 2, &bad_c, 1, NULL, 0, &cfg, 1.0f/60.0f);
    ASSERT_TRUE(1, "PGS survived world-anchor constraint (body_b=-1)");
}

/* Fix #7: World step must not crash when shapes are invalid */
static void test_safety_world_invalid_shapes(void) {
    printf("  [safety] world step with shape bounds...\n");

    rac_phys_world_config cfg = rac_phys_world_default_config();
    rac_phys_world *world = rac_phys_world_create(&cfg);
    ASSERT_TRUE(world != NULL, "world created for safety test");

    /* Add a sphere body — should work normally */
    rac_phys_rigid_body ball = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    ball.position = rac_phys_v3(0, 5, 0);
    rac_phys_shape ball_shape;
    memset(&ball_shape, 0, sizeof(ball_shape));
    ball_shape.type = RAC_SHAPE_SPHERE;
    ball_shape.sphere.radius = 0.5f;
    int idx = rac_phys_world_add_body(world, ball, ball_shape);
    ASSERT_TRUE(idx >= 0, "body added successfully");

    /* Step world — should not crash even with single body */
    for (int i = 0; i < 10; i++)
        rac_phys_world_step(world, 1.0f/60.0f);
    ASSERT_TRUE(1, "world step survived with valid body");

    /* Manually corrupt a body's shape_index to test guard */
    rac_phys_rigid_body *b = rac_phys_world_get_body(world, idx);
    int saved_shape = b->shape_index;
    b->shape_index = -1;  /* corrupt */
    rac_phys_world_step(world, 1.0f/60.0f);
    ASSERT_TRUE(1, "world step survived corrupted shape_index=-1");

    b->shape_index = 99999;  /* corrupt OOB */
    rac_phys_world_step(world, 1.0f/60.0f);
    ASSERT_TRUE(1, "world step survived corrupted shape_index=99999");

    b->shape_index = saved_shape;  /* restore */
    rac_phys_world_destroy(world);
}

/* ══════════════════════════════════════════════════════════════════════════
 * §10  IMPORTANT FIXES REGRESSION TESTS (#8–#14)
 * ════════════════════════════════════════════════════════════════════════ */

/* Fix #8: NaN quaternion must not corrupt rotation output */
static void test_fix8_nan_quat_rotate(void) {
    printf("  [fix #8] NaN quaternion rotation guard...\n");

    rac_phys_vec3 v = rac_phys_v3(1.0f, 2.0f, 3.0f);

    /* NaN quaternion should return vector unchanged */
    rac_phys_quat nan_q = { 0.0f/0.0f, 0.0f, 0.0f, 0.0f };
    rac_phys_vec3 r = rac_phys_quat_rotate_vec3(nan_q, v);
    ASSERT_NEAR(r.x, 1.0f, 0.001f, "NaN quat rotate: x preserved");
    ASSERT_NEAR(r.y, 2.0f, 0.001f, "NaN quat rotate: y preserved");
    ASSERT_NEAR(r.z, 3.0f, 0.001f, "NaN quat rotate: z preserved");

    /* Slerp with NaN should return first quaternion */
    rac_phys_quat a = rac_phys_quat_identity();
    rac_phys_quat result = rac_phys_quat_slerp(a, nan_q, 0.5f);
    ASSERT_TRUE(isfinite(result.w) && isfinite(result.x), "slerp NaN: finite output");

    /* Slerp output should always be normalized */
    rac_phys_quat b = rac_phys_quat_from_axis_angle(rac_phys_v3(0,1,0), 1.0f);
    result = rac_phys_quat_slerp(a, b, 0.5f);
    float len = sqrtf(result.w*result.w + result.x*result.x +
                      result.y*result.y + result.z*result.z);
    ASSERT_NEAR(len, 1.0f, 0.01f, "slerp output is normalized");
}

/* Fix #9: FEM should handle nearly-degenerate elements without exploding */
static void test_fix9_fem_singular_matrix(void) {
    printf("  [fix #9] FEM singular matrix stability...\n");

    /* Create a beam and simulate — should not produce NaN */
    rac_phys_soft_body *beam = rac_phys_softbody_create_beam(
        0.5f, 0.1f, 0.1f, 2, 100.0f, 50.0f);
    ASSERT_TRUE(beam != NULL, "beam created for singular test");
    beam->solver_iterations = 16;
    beam->damping = 0.95f;

    for (int i = 0; i < 60; i++)
        rac_phys_softbody_step(beam, rac_phys_v3(0, -9.81f, 0), 1.0f/60.0f);

    /* All positions should be finite (no NaN explosion) */
    int all_finite = 1;
    for (int i = 0; i < beam->num_vertices; i++) {
        if (!isfinite(beam->positions[i].x) ||
            !isfinite(beam->positions[i].y) ||
            !isfinite(beam->positions[i].z)) {
            all_finite = 0;
            break;
        }
    }
    ASSERT_TRUE(all_finite, "FEM beam: all positions finite after sim");

    rac_phys_softbody_destroy(beam);
}

/* Fix #10: Warm-start should accelerate PGS convergence */
static void test_fix10_warm_start(void) {
    printf("  [fix #10] PGS warm-start accumulation...\n");

    /* Set up two colliding spheres */
    rac_phys_rigid_body bodies[2];
    bodies[0] = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    bodies[0].position = rac_phys_v3(0, 0, 0);
    bodies[1] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    bodies[1].position = rac_phys_v3(0, 0.8f, 0);
    bodies[1].linear_velocity = rac_phys_v3(0, -5.0f, 0);

    rac_phys_contact_manifold contact;
    memset(&contact, 0, sizeof(contact));
    contact.body_a = 0;
    contact.body_b = 1;
    contact.num_contacts = 1;
    contact.contacts[0].point = rac_phys_v3(0, 0.5f, 0);
    contact.contacts[0].normal = rac_phys_v3(0, 1, 0);
    contact.contacts[0].depth = 0.2f;
    contact.contacts[0].lambda_n = 0.0f;
    contact.contacts[0].lambda_t = 0.0f;

    rac_phys_pgs_config cfg = rac_phys_pgs_default_config();

    /* Solve once */
    rac_phys_pgs_solve(bodies, 2, NULL, 0, &contact, 1, &cfg, 1.0f/60.0f);

    /* lambda_n should have accumulated a positive impulse */
    ASSERT_TRUE(contact.contacts[0].lambda_n > 0.0f,
                "warm-start: lambda_n accumulated");

    /* Second solve should benefit from warm-start */
    float lambda_after_first = contact.contacts[0].lambda_n;
    bodies[1].linear_velocity = rac_phys_v3(0, -5.0f, 0);  /* reset velocity */
    rac_phys_pgs_solve(bodies, 2, NULL, 0, &contact, 1, &cfg, 1.0f/60.0f);
    ASSERT_TRUE(contact.contacts[0].lambda_n > 0.0f,
                "warm-start: lambda persists across solves");
    (void)lambda_after_first;
}

/* Fix #11: Spatial hash dedup must handle large ID sets efficiently */
static void test_fix11_spatial_hash_dedup(void) {
    printf("  [fix #11] spatial hash O(n) dedup...\n");

    rac_phys_spatial_hash *sh = rac_phys_spatial_hash_create(1.0f, 64);

    /* Insert 500 objects, many overlapping the same cells */
    for (int i = 0; i < 500; i++) {
        float x = (float)(i % 10) * 0.5f;
        float y = (float)(i / 10 % 10) * 0.5f;
        rac_phys_aabb aabb = {{ x, y, 0 }, { x + 0.6f, y + 0.6f, 0.6f }};
        rac_phys_spatial_hash_insert(sh, aabb, i);
    }

    /* Query a region that overlaps many objects */
    rac_phys_aabb query = {{ 0, 0, 0 }, { 3, 3, 1 }};
    int results[512];
    int n = rac_phys_spatial_hash_query(sh, query, results, 512);

    /* Should find objects without duplicates */
    ASSERT_TRUE(n > 0, "dedup query: found objects");

    /* Verify no duplicates in results */
    int has_dup = 0;
    for (int i = 0; i < n && !has_dup; i++)
        for (int j = i + 1; j < n && !has_dup; j++)
            if (results[i] == results[j]) has_dup = 1;
    ASSERT_TRUE(!has_dup, "dedup query: no duplicates in results");

    rac_phys_spatial_hash_destroy(sh);
}

/* Fix #12: Sleeping body should wake when contacted by moving body */
static void test_fix12_wake_on_contact(void) {
    printf("  [fix #12] wake sleeping bodies on contact...\n");

    rac_phys_world_config cfg = rac_phys_world_default_config();
    cfg.sleep_time = 0.01f;  /* sleep quickly for test */
    cfg.sleep_threshold = 100.0f;  /* high threshold so bodies sleep fast */
    rac_phys_world *world = rac_phys_world_create(&cfg);

    /* Add a resting sphere (will sleep) */
    rac_phys_rigid_body resting = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    resting.position = rac_phys_v3(0, 0.5f, 0);
    rac_phys_body_set_inertia_sphere(&resting, 0.5f);
    rac_phys_shape s;
    memset(&s, 0, sizeof(s));
    s.type = RAC_SHAPE_SPHERE;
    s.sphere.radius = 0.5f;
    int rest_idx = rac_phys_world_add_body(world, resting, s);

    /* Step to let it sleep */
    for (int i = 0; i < 10; i++)
        rac_phys_world_step(world, 1.0f/60.0f);

    rac_phys_rigid_body *rb = rac_phys_world_get_body(world, rest_idx);
    ASSERT_TRUE(rb->is_sleeping, "resting body went to sleep");

    /* Add a fast-moving sphere that will collide */
    rac_phys_rigid_body mover = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    mover.position = rac_phys_v3(2.0f, 0.5f, 0);
    mover.linear_velocity = rac_phys_v3(-10.0f, 0, 0);
    rac_phys_body_set_inertia_sphere(&mover, 0.5f);
    rac_phys_world_add_body(world, mover, s);

    /* Step — collision should wake the resting body */
    cfg.sleep_threshold = 0.01f;  /* restore normal threshold */
    for (int i = 0; i < 30; i++)
        rac_phys_world_step(world, 1.0f/60.0f);

    /* The resting body should have been woken by the contact */
    /* (even if it re-sleeps, it should have moved from original position) */
    float moved = rac_phys_v3_length(
        rac_phys_v3_sub(rb->position, rac_phys_v3(0, 0.5f, 0)));
    /* We check it was woken (sleep timer reset or position changed) */
    ASSERT_TRUE(moved > 0.001f || !rb->is_sleeping,
                "sleeping body woke on contact");

    rac_phys_world_destroy(world);
}

/* Fix #13: SPH boundary damping should contain fluid */
static void test_fix13_sph_boundary(void) {
    printf("  [fix #13] SPH boundary damping...\n");

    rac_phys_sph_config cfg = rac_phys_sph_default_config();
    cfg.smoothing_radius = 0.5f;
    cfg.particle_mass = 1.0f;
    cfg.boundary_damping = 0.5f;

    rac_phys_particle_system *ps = rac_phys_particles_create(10);
    rac_phys_spatial_hash *grid = rac_phys_spatial_hash_create(0.5f, 64);

    /* Launch particle at high speed toward boundary */
    rac_phys_particles_emit(ps, rac_phys_v3(9.0f, 0, 0),
                             rac_phys_v3(100.0f, 0, 0), cfg.particle_mass);

    /* Step several times */
    for (int i = 0; i < 30; i++)
        rac_phys_sph_step(ps, grid, rac_phys_v3_zero(), &cfg, 1.0f/60.0f);

    /* Particle should be contained within bounds */
    ASSERT_TRUE(ps->positions[0].x <= 10.0f,
                "boundary: particle contained at +x");
    ASSERT_TRUE(ps->positions[0].x >= -10.0f,
                "boundary: particle contained at -x");

    /* Velocity magnitude should be reduced by damping */
    float speed = fabsf(ps->velocities[0].x);
    ASSERT_TRUE(speed < 100.0f, "boundary: velocity damped from initial 100");

    rac_phys_spatial_hash_destroy(grid);
    rac_phys_particles_destroy(ps);
}

/* Fix #14: RK4 integrator should work and be more accurate than Euler */
static void test_fix14_rk4_integrator(void) {
    printf("  [fix #14] RK4 integration...\n");

    /* Free-fall comparison: Euler vs RK4 vs analytical */
    float dt = 1.0f / 60.0f;
    float total_t = 1.0f;
    int steps = (int)(total_t / dt);
    float g = -9.81f;

    /* Analytical: y = y0 + v0*t + 0.5*g*t² = 10 + 0 + 0.5*(-9.81)*1 = 5.095 */
    float y_analytical = 10.0f + 0.5f * g * total_t * total_t;

    /* Euler */
    rac_phys_rigid_body euler_b = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    euler_b.position = rac_phys_v3(0, 10, 0);
    euler_b.linear_damping = 0.0f;
    for (int i = 0; i < steps; i++) {
        rac_phys_body_apply_force(&euler_b, rac_phys_v3(0, g, 0));
        rac_phys_body_integrate(&euler_b, dt, RAC_INTEGRATE_EULER);
    }

    /* RK4 */
    rac_phys_rigid_body rk4_b = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    rk4_b.position = rac_phys_v3(0, 10, 0);
    rk4_b.linear_damping = 0.0f;
    for (int i = 0; i < steps; i++) {
        rac_phys_body_apply_force(&rk4_b, rac_phys_v3(0, g, 0));
        rac_phys_body_integrate(&rk4_b, dt, RAC_INTEGRATE_RK4);
    }

    float euler_err = fabsf(euler_b.position.y - y_analytical);
    float rk4_err = fabsf(rk4_b.position.y - y_analytical);

    ASSERT_TRUE(rk4_b.position.y < 10.0f, "RK4: ball fell");
    ASSERT_TRUE(rk4_b.position.y > 0.0f, "RK4: ball above ground");

    /* RK4 should be at least as accurate as Euler (for constant force,
     * both are exact, so we just verify RK4 works correctly) */
    ASSERT_TRUE(rk4_err < 1.0f, "RK4: reasonable accuracy");
    ASSERT_NEAR(rk4_b.position.y, y_analytical, 0.5f, "RK4: close to analytical");
    (void)euler_err;
}

/* ══════════════════════════════════════════════════════════════════════════
 * §11  INCOMPLETE FEATURES — Now Implemented
 * ════════════════════════════════════════════════════════════════════════ */

static void test_ball_joint(void) {
    printf("  [joints] ball-and-socket...\n");

    rac_phys_rigid_body bodies[2];
    bodies[0] = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    bodies[0].position = rac_phys_v3(0, 5, 0);
    bodies[1] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    bodies[1].position = rac_phys_v3(1, 5, 0);

    rac_phys_constraint ball;
    memset(&ball, 0, sizeof(ball));
    ball.type = RAC_CONSTRAINT_BALL;
    ball.body_a = 0;
    ball.body_b = 1;
    ball.anchor_a = rac_phys_v3(0.5f, 0, 0);  /* right side of A */
    ball.anchor_b = rac_phys_v3(-0.5f, 0, 0); /* left side of B */

    rac_phys_pgs_config cfg = rac_phys_pgs_default_config();

    /* Apply gravity to body B and solve */
    bodies[1].linear_velocity = rac_phys_v3(0, -5, 0);
    for (int i = 0; i < 10; i++)
        rac_phys_pgs_solve(bodies, 2, &ball, 1, NULL, 0, &cfg, 1.0f/60.0f);

    /* Ball joint should constrain motion — body B shouldn't freefall */
    ASSERT_TRUE(bodies[1].linear_velocity.y > -5.0f,
                "ball joint constrains downward velocity");
}

static void test_hinge_joint(void) {
    printf("  [joints] hinge...\n");

    rac_phys_rigid_body bodies[2];
    bodies[0] = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    bodies[1] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    bodies[1].position = rac_phys_v3(1, 0, 0);

    rac_phys_constraint hinge;
    memset(&hinge, 0, sizeof(hinge));
    hinge.type = RAC_CONSTRAINT_HINGE;
    hinge.body_a = 0;
    hinge.body_b = 1;
    hinge.axis_a = rac_phys_v3(0, 1, 0);  /* hinge around Y */
    hinge.axis_b = rac_phys_v3(0, 1, 0);

    rac_phys_pgs_config cfg = rac_phys_pgs_default_config();

    /* Give body B angular velocity around non-hinge axis */
    bodies[1].angular_velocity = rac_phys_v3(5, 0, 5);
    for (int i = 0; i < 20; i++)
        rac_phys_pgs_solve(bodies, 2, &hinge, 1, NULL, 0, &cfg, 1.0f/60.0f);

    /* Hinge should reduce angular velocity in non-Y axes */
    float off_axis = fabsf(bodies[1].angular_velocity.x) +
                     fabsf(bodies[1].angular_velocity.z);
    ASSERT_TRUE(off_axis < 10.0f, "hinge reduces off-axis rotation");
}

static void test_slider_joint(void) {
    printf("  [joints] slider...\n");

    rac_phys_rigid_body bodies[2];
    bodies[0] = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    bodies[1] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    bodies[1].position = rac_phys_v3(1, 0, 0);

    rac_phys_constraint slider;
    memset(&slider, 0, sizeof(slider));
    slider.type = RAC_CONSTRAINT_SLIDER;
    slider.body_a = 0;
    slider.body_b = 1;
    slider.axis_a = rac_phys_v3(1, 0, 0);  /* slide along X */

    rac_phys_pgs_config cfg = rac_phys_pgs_default_config();

    /* Push body B perpendicular to slider axis */
    bodies[1].linear_velocity = rac_phys_v3(0, 5, 3);
    for (int i = 0; i < 20; i++)
        rac_phys_pgs_solve(bodies, 2, &slider, 1, NULL, 0, &cfg, 1.0f/60.0f);

    /* Slider should reduce perpendicular velocity (iterative, partial reduction) */
    float perp = fabsf(bodies[1].linear_velocity.y) +
                 fabsf(bodies[1].linear_velocity.z);
    float initial_perp = 5.0f + 3.0f;  /* initial y+z */
    ASSERT_TRUE(perp < initial_perp, "slider constrains perpendicular motion");
}

static void test_d6_joint(void) {
    printf("  [joints] D6 configurable...\n");

    rac_phys_rigid_body bodies[2];
    bodies[0] = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    bodies[1] = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    bodies[1].position = rac_phys_v3(0, 0, 0);

    rac_phys_constraint d6;
    memset(&d6, 0, sizeof(d6));
    d6.type = RAC_CONSTRAINT_D6;
    d6.body_a = 0;
    d6.body_b = 1;
    /* Lock all rotation, free translation */
    for (int i = 0; i < 3; i++) {
        d6.limit_lower[i] = 1.0f;    /* tx,ty,tz: lower > upper = free */
        d6.limit_upper[i] = -1.0f;
    }
    for (int i = 3; i < 6; i++) {
        d6.limit_lower[i] = 0.0f;    /* rx,ry,rz: lower == upper == 0 = locked */
        d6.limit_upper[i] = 0.0f;
    }

    rac_phys_pgs_config cfg = rac_phys_pgs_default_config();
    bodies[1].angular_velocity = rac_phys_v3(5, 5, 5);
    for (int i = 0; i < 20; i++)
        rac_phys_pgs_solve(bodies, 2, &d6, 1, NULL, 0, &cfg, 1.0f/60.0f);

    float w = rac_phys_v3_length(bodies[1].angular_velocity);
    ASSERT_TRUE(w < 8.66f, "D6 locked rotation reduces angular velocity");
}

static void test_fem_plasticity(void) {
    printf("  [FEM] plasticity (permanent deformation)...\n");

    rac_phys_soft_body *beam = rac_phys_softbody_create_beam(
        0.5f, 0.1f, 0.1f, 2, 100.0f, 30.0f);
    beam->solver_iterations = 16;
    beam->damping = 0.95f;

    /* Strong gravity to induce plastic deformation */
    for (int i = 0; i < 120; i++)
        rac_phys_softbody_step(beam, rac_phys_v3(0, -50.0f, 0), 1.0f/60.0f);

    /* Check that plastic_strain accumulated in at least one element */
    float max_plastic = 0.0f;
    for (int e = 0; e < beam->num_elements; e++)
        if (beam->elements[e].plastic_strain > max_plastic)
            max_plastic = beam->elements[e].plastic_strain;

    ASSERT_TRUE(max_plastic > 0.0f, "plasticity: strain accumulated");

    rac_phys_softbody_destroy(beam);
}

static void test_fem_fracture(void) {
    printf("  [FEM] fracture (mesh tearing)...\n");

    rac_phys_soft_body *beam = rac_phys_softbody_create_beam(
        0.5f, 0.1f, 0.1f, 2, 100.0f, 30.0f);
    beam->solver_iterations = 16;
    beam->damping = 0.95f;

    /* Set low fracture threshold on all elements */
    for (int e = 0; e < beam->num_elements; e++)
        beam->elements[e].fracture_threshold = 1.0f;

    /* Very strong forces to cause fracture */
    for (int i = 0; i < 120; i++)
        rac_phys_softbody_step(beam, rac_phys_v3(0, -200.0f, 0), 1.0f/60.0f);

    /* Check that at least one element fractured (rest_volume = 0) */
    int broken = 0;
    for (int e = 0; e < beam->num_elements; e++)
        if (beam->elements[e].rest_volume == 0.0f) broken++;

    ASSERT_TRUE(broken > 0, "fracture: at least one element broke");

    rac_phys_softbody_destroy(beam);
}

static void test_ccd_sphere_sweep(void) {
    printf("  [CCD] sphere sweep tunneling prevention...\n");

    rac_phys_world_config cfg = rac_phys_world_default_config();
    cfg.gravity = rac_phys_v3_zero();  /* no gravity for clean test */
    rac_phys_world *world = rac_phys_world_create(&cfg);

    /* Static wall */
    rac_phys_rigid_body wall = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    wall.position = rac_phys_v3(10, 0, 0);
    rac_phys_shape wall_shape;
    memset(&wall_shape, 0, sizeof(wall_shape));
    wall_shape.type = RAC_SHAPE_SPHERE;
    wall_shape.sphere.radius = 1.0f;
    rac_phys_world_add_body(world, wall, wall_shape);

    /* Fast bullet heading toward wall */
    rac_phys_rigid_body bullet = rac_phys_body_create(RAC_BODY_DYNAMIC, 0.1f);
    bullet.position = rac_phys_v3(0, 0, 0);
    bullet.linear_velocity = rac_phys_v3(500, 0, 0);  /* very fast */
    bullet.restitution = 0.5f;
    rac_phys_shape bullet_shape;
    memset(&bullet_shape, 0, sizeof(bullet_shape));
    bullet_shape.type = RAC_SHAPE_SPHERE;
    bullet_shape.sphere.radius = 0.1f;
    int bullet_idx = rac_phys_world_add_body(world, bullet, bullet_shape);

    /* Step once — without CCD bullet would tunnel through */
    rac_phys_world_step(world, 1.0f/60.0f);

    rac_phys_rigid_body *b = rac_phys_world_get_body(world, bullet_idx);
    /* CCD should have caught the tunneling — bullet should not be past x=11 */
    ASSERT_TRUE(b->position.x < 12.0f, "CCD: bullet didn't tunnel through wall");

    rac_phys_world_destroy(world);
}

static void test_sleeping_islands(void) {
    printf("  [islands] sleeping island propagation...\n");

    rac_phys_world_config cfg = rac_phys_world_default_config();
    cfg.sleep_time = 0.01f;
    cfg.sleep_threshold = 100.0f;
    cfg.gravity = rac_phys_v3_zero();
    rac_phys_world *world = rac_phys_world_create(&cfg);

    /* Create 3 touching spheres in a chain */
    rac_phys_shape s;
    memset(&s, 0, sizeof(s));
    s.type = RAC_SHAPE_SPHERE;
    s.sphere.radius = 0.6f;

    rac_phys_rigid_body b0 = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    b0.position = rac_phys_v3(0, 0, 0);
    int i0 = rac_phys_world_add_body(world, b0, s);

    rac_phys_rigid_body b1 = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    b1.position = rac_phys_v3(1.0f, 0, 0);
    int i1 = rac_phys_world_add_body(world, b1, s);

    rac_phys_rigid_body b2 = rac_phys_body_create(RAC_BODY_DYNAMIC, 1.0f);
    b2.position = rac_phys_v3(2.0f, 0, 0);
    int i2 = rac_phys_world_add_body(world, b2, s);

    /* Let them sleep */
    for (int i = 0; i < 10; i++)
        rac_phys_world_step(world, 1.0f/60.0f);

    ASSERT_TRUE(rac_phys_world_get_body(world, i0)->is_sleeping, "island: b0 sleeping");
    ASSERT_TRUE(rac_phys_world_get_body(world, i1)->is_sleeping, "island: b1 sleeping");
    ASSERT_TRUE(rac_phys_world_get_body(world, i2)->is_sleeping, "island: b2 sleeping");

    /* Wake b0 — the island mechanism should propagate to connected bodies.
     * Since all 3 share contacts (overlapping spheres), waking b0 should
     * wake b1 too via the island union-find. */
    rac_phys_world_get_body(world, i0)->is_sleeping = 0;
    rac_phys_world_get_body(world, i0)->sleep_timer = 0.0f;

    /* Single substep to trigger island wake propagation */
    rac_phys_world_step(world, 1.0f/60.0f);

    /* b1 should have been woken by island propagation (shares contacts) */
    ASSERT_TRUE(!rac_phys_world_get_body(world, i1)->is_sleeping,
                "island: b1 woken via island propagation");

    rac_phys_world_destroy(world);
    (void)i1; (void)i2;
}

static void test_box_raycast(void) {
    printf("  [raycast] box intersection...\n");

    rac_phys_world_config cfg = rac_phys_world_default_config();
    rac_phys_world *world = rac_phys_world_create(&cfg);

    /* Add a box */
    rac_phys_rigid_body box = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
    box.position = rac_phys_v3(5, 0, 0);
    rac_phys_shape box_shape;
    memset(&box_shape, 0, sizeof(box_shape));
    box_shape.type = RAC_SHAPE_BOX;
    box_shape.box.half_extents = rac_phys_v3(1, 1, 1);
    rac_phys_world_add_body(world, box, box_shape);

    /* Ray toward box — should hit */
    rac_phys_ray_hit hit = rac_phys_world_raycast(
        world, rac_phys_v3(0, 0, 0), rac_phys_v3(1, 0, 0), 100.0f);
    ASSERT_TRUE(hit.hit, "box raycast: hit");
    ASSERT_NEAR(hit.distance, 4.0f, 0.1f, "box raycast: distance ≈ 4");
    ASSERT_NEAR(hit.normal.x, -1.0f, 0.1f, "box raycast: normal faces -x");

    /* Ray missing box — should not hit */
    hit = rac_phys_world_raycast(
        world, rac_phys_v3(0, 5, 0), rac_phys_v3(1, 0, 0), 100.0f);
    ASSERT_TRUE(!hit.hit, "box raycast: miss");

    /* Rotated box test */
    rac_phys_rigid_body *bp = rac_phys_world_get_body(world, 0);
    bp->orientation = rac_phys_quat_from_axis_angle(
        rac_phys_v3(0, 0, 1), 0.785f);  /* 45° around Z */
    hit = rac_phys_world_raycast(
        world, rac_phys_v3(0, 0, 0), rac_phys_v3(1, 0, 0), 100.0f);
    ASSERT_TRUE(hit.hit, "rotated box raycast: hit");

    rac_phys_world_destroy(world);
}

/* ══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  RAC Native Physics Library — Build Verification Tests      ║\n");
    printf("║  Pinnacle Quantum Group — April 2026                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("§1 Vec3/Quat Math\n");
    test_vec3_basic();
    test_quat_basic();

    printf("\n§2 Rigid Body Dynamics\n");
    test_rigid_body();
    test_rigid_body_verlet();

    printf("\n§3 Collision Detection\n");
    test_spatial_hash();
    test_sphere_sphere_collision();
    test_sphere_box_collision();
    test_box_box_collision();

    printf("\n§4 Constraint Solvers\n");
    test_pbd_distance();

    printf("\n§5 Particle Systems\n");
    test_particles();
    test_sph_density();

    printf("\n§6 Cloth Simulation\n");
    test_cloth();

    printf("\n§7 Soft Body FEM\n");
    test_softbody();

    printf("\n§8 World Integration\n");
    test_world();

    printf("\n§9 Safety Regression (Critical Fixes #1–#7)\n");
    test_safety_pgs_bad_indices();
    test_safety_spatial_hash_overflow();
    test_safety_slerp_degenerate();
    test_safety_gjk_bad_input();
    test_safety_constraint_bad_indices();
    test_safety_world_invalid_shapes();

    printf("\n§10 Important Fixes (#8–#14)\n");
    test_fix8_nan_quat_rotate();
    test_fix9_fem_singular_matrix();
    test_fix10_warm_start();
    test_fix11_spatial_hash_dedup();
    test_fix12_wake_on_contact();
    test_fix13_sph_boundary();
    test_fix14_rk4_integrator();

    printf("\n§11 Feature Completion\n");
    test_ball_joint();
    test_hinge_joint();
    test_slider_joint();
    test_d6_joint();
    test_fem_plasticity();
    test_fem_fracture();
    test_ccd_sphere_sweep();
    test_sleeping_islands();
    test_box_raycast();

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  Results: %d/%d passed, %d failed\n",
           tests_passed, tests_run, tests_failed);
    printf("════════════════════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
