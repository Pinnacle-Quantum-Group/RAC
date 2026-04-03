/*
 * rac_physics_constraints.c — RAC Native Physics: Constraint Solvers
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * Two solver paradigms, both GPU-heritage:
 *   PGS  — Projected Gauss-Seidel / Sequential Impulse (PhysX/Bullet)
 *   PBD  — Position-Based Dynamics (NVIDIA Flex / AMD FEMFX)
 *
 * All impulse/correction math routes through RAC primitives:
 *   - Normal impulse via rac_phys_v3_dot
 *   - Velocity corrections via rac_phys_v3_scale
 *   - Constraint jacobians via rac_phys_v3_cross
 */

#include "rac_physics.h"
#include "rac_cpu.h"
#include <math.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════════════════
 * §1  PGS / SEQUENTIAL IMPULSE SOLVER
 * ══════════════════════════════════════════════════════════════════════════
 *
 * PhysX/Bullet-heritage iterative constraint solver.
 * Each iteration solves one constraint at a time, projecting velocities
 * to satisfy non-penetration, friction, and joint limits.
 *
 * Key formula (per contact):
 *   J = [-n, -(r_a × n), n, (r_b × n)]
 *   effective_mass = 1 / (J * M⁻¹ * Jᵀ)
 *   lambda = -effective_mass * (J*v + bias)
 *   clamp lambda to [0, ∞) for contacts
 *   apply velocity corrections
 */

rac_phys_pgs_config rac_phys_pgs_default_config(void) {
    return (rac_phys_pgs_config){
        .iterations = 8,
        .slop = 0.005f,
        .baumgarte = 0.2f,
        .warm_start_factor = 0.85f
    };
}

/* Compute effective mass for a contact constraint */
static float _contact_effective_mass(
    const rac_phys_rigid_body *a, const rac_phys_rigid_body *b,
    rac_phys_vec3 r_a, rac_phys_vec3 r_b, rac_phys_vec3 normal) {

    float inv_mass = a->inv_mass + b->inv_mass;

    /* Angular contribution: (I⁻¹ * (r × n)) × r · n */
    rac_phys_vec3 rn_a = rac_phys_v3_cross(r_a, normal);
    rac_phys_vec3 rn_b = rac_phys_v3_cross(r_b, normal);

    /* Transform through world-space inverse inertia */
    rac_phys_mat3 R_a = rac_phys_quat_to_mat3(a->orientation);
    rac_phys_mat3 Rt_a = rac_phys_mat3_transpose(R_a);
    rac_phys_mat3 world_inv_I_a = rac_phys_mat3_mul(
        rac_phys_mat3_mul(R_a, a->inv_inertia), Rt_a);

    rac_phys_mat3 R_b = rac_phys_quat_to_mat3(b->orientation);
    rac_phys_mat3 Rt_b = rac_phys_mat3_transpose(R_b);
    rac_phys_mat3 world_inv_I_b = rac_phys_mat3_mul(
        rac_phys_mat3_mul(R_b, b->inv_inertia), Rt_b);

    rac_phys_vec3 ang_a = rac_phys_v3_cross(
        rac_phys_mat3_mul_vec3(world_inv_I_a, rn_a), r_a);
    rac_phys_vec3 ang_b = rac_phys_v3_cross(
        rac_phys_mat3_mul_vec3(world_inv_I_b, rn_b), r_b);

    inv_mass += rac_phys_v3_dot(ang_a, normal);
    inv_mass += rac_phys_v3_dot(ang_b, normal);

    return (inv_mass > 1e-8f) ? 1.0f / inv_mass : 0.0f;
}

/* Apply impulse at contact point */
static void _apply_contact_impulse(
    rac_phys_rigid_body *a, rac_phys_rigid_body *b,
    rac_phys_vec3 r_a, rac_phys_vec3 r_b,
    rac_phys_vec3 impulse) {

    /* Linear */
    a->linear_velocity = rac_phys_v3_sub(
        a->linear_velocity, rac_phys_v3_scale(impulse, a->inv_mass));
    b->linear_velocity = rac_phys_v3_add(
        b->linear_velocity, rac_phys_v3_scale(impulse, b->inv_mass));

    /* Angular */
    rac_phys_mat3 R_a = rac_phys_quat_to_mat3(a->orientation);
    rac_phys_mat3 Rt_a = rac_phys_mat3_transpose(R_a);
    rac_phys_mat3 world_inv_I_a = rac_phys_mat3_mul(
        rac_phys_mat3_mul(R_a, a->inv_inertia), Rt_a);

    rac_phys_mat3 R_b = rac_phys_quat_to_mat3(b->orientation);
    rac_phys_mat3 Rt_b = rac_phys_mat3_transpose(R_b);
    rac_phys_mat3 world_inv_I_b = rac_phys_mat3_mul(
        rac_phys_mat3_mul(R_b, b->inv_inertia), Rt_b);

    a->angular_velocity = rac_phys_v3_sub(
        a->angular_velocity,
        rac_phys_mat3_mul_vec3(world_inv_I_a, rac_phys_v3_cross(r_a, impulse)));
    b->angular_velocity = rac_phys_v3_add(
        b->angular_velocity,
        rac_phys_mat3_mul_vec3(world_inv_I_b, rac_phys_v3_cross(r_b, impulse)));
}

void rac_phys_pgs_solve(rac_phys_rigid_body *bodies, int num_bodies,
                         rac_phys_constraint *constraints, int num_constraints,
                         rac_phys_contact_manifold *contacts, int num_contacts,
                         const rac_phys_pgs_config *cfg, float dt) {
    if (!bodies || num_bodies <= 0) return;
    if (!cfg) return;
    float inv_dt = (dt > 0.0f) ? 1.0f / dt : 0.0f;

    /* Fix #10: warm-start — apply previous frame's impulses scaled down.
     * This gives the solver a head start from last frame's solution. */
    float ws = cfg->warm_start_factor;
    for (int ci = 0; ci < num_contacts; ci++) {
        rac_phys_contact_manifold *m = &contacts[ci];
        if (m->body_a < 0 || m->body_a >= num_bodies ||
            m->body_b < 0 || m->body_b >= num_bodies) continue;

        rac_phys_rigid_body *a = &bodies[m->body_a];
        rac_phys_rigid_body *b = &bodies[m->body_b];
        for (int pi = 0; pi < m->num_contacts; pi++) {
            rac_phys_contact_point *cp = &m->contacts[pi];
            cp->lambda_n *= ws;
            cp->lambda_t *= ws;
            /* Apply warm-start impulse */
            if (fabsf(cp->lambda_n) > 1e-8f) {
                rac_phys_vec3 r_a = rac_phys_v3_sub(cp->point, a->position);
                rac_phys_vec3 r_b = rac_phys_v3_sub(cp->point, b->position);
                rac_phys_vec3 ws_impulse = rac_phys_v3_scale(cp->normal, cp->lambda_n);
                _apply_contact_impulse(a, b, r_a, r_b, ws_impulse);
            }
        }
    }

    for (int iter = 0; iter < cfg->iterations; iter++) {
        /* ── Solve contact constraints ──────────────────────────────── */
        for (int ci = 0; ci < num_contacts; ci++) {
            rac_phys_contact_manifold *m = &contacts[ci];

            /* Fix #1: bounds-check body indices */
            if (m->body_a < 0 || m->body_a >= num_bodies ||
                m->body_b < 0 || m->body_b >= num_bodies) continue;

            rac_phys_rigid_body *a = &bodies[m->body_a];
            rac_phys_rigid_body *b = &bodies[m->body_b];

            for (int pi = 0; pi < m->num_contacts; pi++) {
                rac_phys_contact_point *cp = &m->contacts[pi];

                rac_phys_vec3 r_a = rac_phys_v3_sub(cp->point, a->position);
                rac_phys_vec3 r_b = rac_phys_v3_sub(cp->point, b->position);

                /* Relative velocity at contact */
                rac_phys_vec3 v_a = rac_phys_v3_add(
                    a->linear_velocity,
                    rac_phys_v3_cross(a->angular_velocity, r_a));
                rac_phys_vec3 v_b = rac_phys_v3_add(
                    b->linear_velocity,
                    rac_phys_v3_cross(b->angular_velocity, r_b));
                rac_phys_vec3 v_rel = rac_phys_v3_sub(v_b, v_a);

                float v_n = rac_phys_v3_dot(v_rel, cp->normal);

                /* Baumgarte position correction bias */
                float bias = 0.0f;
                if (cp->depth > cfg->slop)
                    bias = cfg->baumgarte * inv_dt * (cp->depth - cfg->slop);

                /* Restitution bias (bounce) */
                float e = fminf(a->restitution, b->restitution);
                if (v_n < -1.0f)
                    bias += e * v_n;

                /* Normal impulse */
                float eff_mass = _contact_effective_mass(a, b, r_a, r_b, cp->normal);
                float lambda = eff_mass * (-v_n + bias);

                /*
                 * Fix #10: warm-start — accumulate impulses across iterations.
                 * On first iteration, warm-start from previous frame's lambda.
                 * Clamp accumulated impulse >= 0 (non-pulling constraint).
                 */
                float old_lambda = cp->lambda_n;
                float new_lambda = fmaxf(old_lambda + lambda, 0.0f);
                lambda = new_lambda - old_lambda;
                cp->lambda_n = new_lambda;

                rac_phys_vec3 impulse = rac_phys_v3_scale(cp->normal, lambda);
                _apply_contact_impulse(a, b, r_a, r_b, impulse);

                /* ── Friction impulse (Coulomb) ─────────────────── */
                float mu = (a->friction + b->friction) * 0.5f;
                if (mu > 0.0f && fabsf(new_lambda) > 1e-8f) {
                    /* Recompute relative velocity */
                    v_a = rac_phys_v3_add(a->linear_velocity,
                        rac_phys_v3_cross(a->angular_velocity, r_a));
                    v_b = rac_phys_v3_add(b->linear_velocity,
                        rac_phys_v3_cross(b->angular_velocity, r_b));
                    v_rel = rac_phys_v3_sub(v_b, v_a);

                    /* Tangent direction */
                    float vn2 = rac_phys_v3_dot(v_rel, cp->normal);
                    rac_phys_vec3 v_tan = rac_phys_v3_sub(
                        v_rel, rac_phys_v3_scale(cp->normal, vn2));
                    float v_tan_len = rac_phys_v3_length(v_tan);

                    if (v_tan_len > 1e-6f) {
                        rac_phys_vec3 tangent = rac_phys_v3_scale(v_tan,
                            1.0f / v_tan_len);
                        float eff_mass_t = _contact_effective_mass(
                            a, b, r_a, r_b, tangent);
                        float friction_lambda = eff_mass_t * (-v_tan_len);

                        /* Coulomb clamp with warm-start accumulation */
                        float max_friction = mu * new_lambda;
                        float old_ft = cp->lambda_t;
                        float new_ft = fmaxf(-max_friction,
                            fminf(old_ft + friction_lambda, max_friction));
                        friction_lambda = new_ft - old_ft;
                        cp->lambda_t = new_ft;

                        rac_phys_vec3 friction_impulse = rac_phys_v3_scale(
                            tangent, friction_lambda);
                        _apply_contact_impulse(a, b, r_a, r_b, friction_impulse);
                    }
                }
            }
        }

        /* ── Solve joint/distance constraints ──────────────────────── */
        for (int ci = 0; ci < num_constraints; ci++) {
            rac_phys_constraint *c = &constraints[ci];

            /* Fix #6: bounds-check constraint body indices */
            if (c->body_a < 0 || c->body_a >= num_bodies) continue;
            if (c->body_b >= num_bodies) continue;  /* -1 = world anchor (valid) */

            rac_phys_rigid_body *a = &bodies[c->body_a];
            rac_phys_rigid_body *b = (c->body_b >= 0) ? &bodies[c->body_b] : NULL;

            /* Helper: compute world-space anchors and lever arms */
            rac_phys_vec3 wa = rac_phys_v3_add(a->position,
                rac_phys_quat_rotate_vec3(a->orientation, c->anchor_a));
            rac_phys_vec3 wb = b ?
                rac_phys_v3_add(b->position,
                    rac_phys_quat_rotate_vec3(b->orientation, c->anchor_b))
                : c->anchor_b;
            rac_phys_vec3 r_a = rac_phys_v3_sub(wa, a->position);
            rac_phys_vec3 r_b_vec = b ? rac_phys_v3_sub(wb, b->position)
                                      : rac_phys_v3_zero();

            if (c->type == RAC_CONSTRAINT_DISTANCE) {
                /* Distance constraint: maintain rest_length between anchors */
                rac_phys_vec3 diff = rac_phys_v3_sub(wb, wa);
                float dist = rac_phys_v3_length(diff);
                if (dist < 1e-8f) continue;

                rac_phys_vec3 n = rac_phys_v3_scale(diff, 1.0f / dist);
                float error = dist - c->rest_length;

                rac_phys_vec3 v_a = rac_phys_v3_add(a->linear_velocity,
                    rac_phys_v3_cross(a->angular_velocity, r_a));
                float v_rel_n;
                if (b) {
                    rac_phys_vec3 v_b = rac_phys_v3_add(b->linear_velocity,
                        rac_phys_v3_cross(b->angular_velocity, r_b_vec));
                    v_rel_n = rac_phys_v3_dot(rac_phys_v3_sub(v_b, v_a), n);
                } else {
                    v_rel_n = rac_phys_v3_dot(rac_phys_v3_negate(v_a), n);
                }

                float bias = cfg->baumgarte * inv_dt * error;
                float inv_mass_sum = a->inv_mass + (b ? b->inv_mass : 0.0f);
                if (inv_mass_sum < 1e-8f) continue;

                float lambda_c = (1.0f / inv_mass_sum) * (-v_rel_n + bias);

                a->linear_velocity = rac_phys_v3_sub(a->linear_velocity,
                    rac_phys_v3_scale(n, lambda_c * a->inv_mass));
                if (b) {
                    b->linear_velocity = rac_phys_v3_add(b->linear_velocity,
                        rac_phys_v3_scale(n, lambda_c * b->inv_mass));
                }
            }

            else if (c->type == RAC_CONSTRAINT_BALL) {
                /*
                 * Ball-and-socket: anchor points must coincide.
                 * 3 linear constraints (x, y, z positional error).
                 * PhysX/Bullet heritage — sequential impulse per axis.
                 */
                rac_phys_vec3 error = rac_phys_v3_sub(wb, wa);
                float inv_mass_sum = a->inv_mass + (b ? b->inv_mass : 0.0f);
                if (inv_mass_sum < 1e-8f) continue;

                rac_phys_vec3 correction = rac_phys_v3_scale(
                    error, cfg->baumgarte * inv_dt / inv_mass_sum);

                /* Relative velocity at anchors */
                rac_phys_vec3 v_a = rac_phys_v3_add(a->linear_velocity,
                    rac_phys_v3_cross(a->angular_velocity, r_a));
                rac_phys_vec3 v_b_val = b ?
                    rac_phys_v3_add(b->linear_velocity,
                        rac_phys_v3_cross(b->angular_velocity, r_b_vec))
                    : rac_phys_v3_zero();
                rac_phys_vec3 v_rel = rac_phys_v3_sub(v_b_val, v_a);

                rac_phys_vec3 impulse = rac_phys_v3_add(
                    rac_phys_v3_scale(v_rel, -1.0f / inv_mass_sum),
                    correction);

                _apply_contact_impulse(a, b ? b : a, r_a, r_b_vec, impulse);
            }

            else if (c->type == RAC_CONSTRAINT_HINGE) {
                /*
                 * Hinge joint: anchors coincide (ball constraint) +
                 * rotation restricted to single axis.
                 * Heritage: PhysX revolute joint, Bullet hinge constraint.
                 */

                /* 1. Positional: same as ball joint */
                rac_phys_vec3 pos_err = rac_phys_v3_sub(wb, wa);
                float inv_mass_sum = a->inv_mass + (b ? b->inv_mass : 0.0f);
                if (inv_mass_sum < 1e-8f) continue;

                rac_phys_vec3 pos_correction = rac_phys_v3_scale(
                    pos_err, cfg->baumgarte * inv_dt / inv_mass_sum);
                rac_phys_vec3 v_a = rac_phys_v3_add(a->linear_velocity,
                    rac_phys_v3_cross(a->angular_velocity, r_a));
                rac_phys_vec3 v_b_val = b ?
                    rac_phys_v3_add(b->linear_velocity,
                        rac_phys_v3_cross(b->angular_velocity, r_b_vec))
                    : rac_phys_v3_zero();
                rac_phys_vec3 v_rel = rac_phys_v3_sub(v_b_val, v_a);
                rac_phys_vec3 lin_impulse = rac_phys_v3_add(
                    rac_phys_v3_scale(v_rel, -1.0f / inv_mass_sum),
                    pos_correction);
                _apply_contact_impulse(a, b ? b : a, r_a, r_b_vec, lin_impulse);

                /* 2. Angular: constrain relative rotation to hinge axis.
                 * The two axes (world-space) should be aligned. */
                rac_phys_vec3 axis_a_world = rac_phys_quat_rotate_vec3(
                    a->orientation, c->axis_a);
                rac_phys_vec3 axis_b_world = b ?
                    rac_phys_quat_rotate_vec3(b->orientation, c->axis_b)
                    : c->axis_b;

                /* Error: cross product of axes (zero when aligned) */
                rac_phys_vec3 axis_err = rac_phys_v3_cross(axis_a_world,
                                                             axis_b_world);
                rac_phys_vec3 ang_correction = rac_phys_v3_scale(
                    axis_err, cfg->baumgarte * inv_dt * 0.5f);

                /* Relative angular velocity perpendicular to hinge axis */
                rac_phys_vec3 w_rel = b ?
                    rac_phys_v3_sub(b->angular_velocity, a->angular_velocity)
                    : rac_phys_v3_negate(a->angular_velocity);
                /* Remove component along hinge axis */
                float w_along = rac_phys_v3_dot(w_rel, axis_a_world);
                rac_phys_vec3 w_perp = rac_phys_v3_sub(
                    w_rel, rac_phys_v3_scale(axis_a_world, w_along));

                rac_phys_vec3 ang_impulse = rac_phys_v3_add(
                    rac_phys_v3_scale(w_perp, -0.5f), ang_correction);

                /* Apply angular impulse directly */
                a->angular_velocity = rac_phys_v3_sub(
                    a->angular_velocity, ang_impulse);
                if (b) b->angular_velocity = rac_phys_v3_add(
                    b->angular_velocity, ang_impulse);
            }

            else if (c->type == RAC_CONSTRAINT_SLIDER) {
                /*
                 * Slider/prismatic: bodies can only translate along axis.
                 * Constrains 2 linear DOFs (perpendicular to axis) +
                 * 3 angular DOFs (no relative rotation).
                 */
                rac_phys_vec3 axis_world = rac_phys_quat_rotate_vec3(
                    a->orientation, c->axis_a);
                rac_phys_vec3 diff = rac_phys_v3_sub(wb, wa);

                /* Remove on-axis component to get perpendicular error */
                float on_axis = rac_phys_v3_dot(diff, axis_world);
                rac_phys_vec3 perp_err = rac_phys_v3_sub(
                    diff, rac_phys_v3_scale(axis_world, on_axis));

                float inv_mass_sum = a->inv_mass + (b ? b->inv_mass : 0.0f);
                if (inv_mass_sum < 1e-8f) continue;

                rac_phys_vec3 correction = rac_phys_v3_scale(
                    perp_err, cfg->baumgarte * inv_dt / inv_mass_sum);

                /* Also remove perpendicular velocity component */
                rac_phys_vec3 v_a_s = rac_phys_v3_add(a->linear_velocity,
                    rac_phys_v3_cross(a->angular_velocity, r_a));
                rac_phys_vec3 v_b_s = b ?
                    rac_phys_v3_add(b->linear_velocity,
                        rac_phys_v3_cross(b->angular_velocity, r_b_vec))
                    : rac_phys_v3_zero();
                rac_phys_vec3 v_rel_s = rac_phys_v3_sub(v_b_s, v_a_s);
                float v_along = rac_phys_v3_dot(v_rel_s, axis_world);
                rac_phys_vec3 v_perp = rac_phys_v3_sub(
                    v_rel_s, rac_phys_v3_scale(axis_world, v_along));
                rac_phys_vec3 vel_correction = rac_phys_v3_scale(
                    v_perp, -1.0f / inv_mass_sum);

                rac_phys_vec3 total_corr = rac_phys_v3_add(correction, vel_correction);

                a->linear_velocity = rac_phys_v3_sub(a->linear_velocity,
                    rac_phys_v3_scale(total_corr, a->inv_mass));
                if (b) b->linear_velocity = rac_phys_v3_add(b->linear_velocity,
                    rac_phys_v3_scale(total_corr, b->inv_mass));

                /* Lock relative rotation (same as rigid attachment) */
                rac_phys_vec3 w_rel = b ?
                    rac_phys_v3_sub(b->angular_velocity, a->angular_velocity)
                    : rac_phys_v3_negate(a->angular_velocity);
                rac_phys_vec3 ang_impulse = rac_phys_v3_scale(w_rel, -0.5f);
                a->angular_velocity = rac_phys_v3_sub(
                    a->angular_velocity, ang_impulse);
                if (b) b->angular_velocity = rac_phys_v3_add(
                    b->angular_velocity, ang_impulse);
            }

            else if (c->type == RAC_CONSTRAINT_D6) {
                /*
                 * D6 joint: configurable 6-DOF constraint (PhysX heritage).
                 * Each DOF can be free, locked, or limited.
                 * limits: [tx, ty, tz, rx, ry, rz]
                 * lower == upper == 0 → locked
                 * lower < upper → limited range
                 * lower > upper → free
                 */
                rac_phys_vec3 pos_err = rac_phys_v3_sub(wb, wa);
                float inv_mass_sum = a->inv_mass + (b ? b->inv_mass : 0.0f);
                if (inv_mass_sum < 1e-8f) continue;

                /* Transform error into body-A local frame */
                rac_phys_quat inv_a = rac_phys_quat_conjugate(a->orientation);
                rac_phys_vec3 local_err = rac_phys_quat_rotate_vec3(inv_a, pos_err);

                /* Apply per-axis constraints for translation */
                float err_arr[3] = { local_err.x, local_err.y, local_err.z };
                float corr[3] = { 0, 0, 0 };
                for (int ax = 0; ax < 3; ax++) {
                    float lo = c->limit_lower[ax];
                    float hi = c->limit_upper[ax];
                    if (lo > hi) continue;  /* free */
                    float clamped = err_arr[ax];
                    if (clamped < lo) clamped = lo;
                    if (clamped > hi) clamped = hi;
                    corr[ax] = (clamped - err_arr[ax]) * cfg->baumgarte * inv_dt
                               / inv_mass_sum;
                }
                rac_phys_vec3 local_corr = rac_phys_v3(corr[0], corr[1], corr[2]);
                rac_phys_vec3 world_corr = rac_phys_quat_rotate_vec3(
                    a->orientation, local_corr);

                a->linear_velocity = rac_phys_v3_sub(a->linear_velocity,
                    rac_phys_v3_scale(world_corr, a->inv_mass));
                if (b) b->linear_velocity = rac_phys_v3_add(b->linear_velocity,
                    rac_phys_v3_scale(world_corr, b->inv_mass));

                /* Angular D6 limits */
                rac_phys_vec3 w_rel = b ?
                    rac_phys_v3_sub(b->angular_velocity, a->angular_velocity)
                    : rac_phys_v3_negate(a->angular_velocity);
                rac_phys_vec3 local_w = rac_phys_quat_rotate_vec3(inv_a, w_rel);
                float w_arr[3] = { local_w.x, local_w.y, local_w.z };
                float ang_corr[3] = { 0, 0, 0 };
                for (int ax = 0; ax < 3; ax++) {
                    float lo = c->limit_lower[3 + ax];
                    float hi = c->limit_upper[3 + ax];
                    if (lo > hi) continue;  /* free */
                    if (lo == 0.0f && hi == 0.0f) {
                        /* Locked: kill relative angular velocity */
                        ang_corr[ax] = -w_arr[ax] * 0.5f;
                    }
                    /* Limited angular: simplified — just damp if near limit */
                }
                rac_phys_vec3 local_ang = rac_phys_v3(
                    ang_corr[0], ang_corr[1], ang_corr[2]);
                rac_phys_vec3 world_ang = rac_phys_quat_rotate_vec3(
                    a->orientation, local_ang);
                a->angular_velocity = rac_phys_v3_sub(
                    a->angular_velocity, world_ang);
                if (b) b->angular_velocity = rac_phys_v3_add(
                    b->angular_velocity, world_ang);
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * §2  PBD SOLVER (POSITION-BASED DYNAMICS)
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Flex/FEMFX-heritage. Works directly on positions instead of velocities.
 * Each constraint projects particle positions to satisfy the constraint.
 *
 * Key: Δx = -w_i / (w_1 + w_2) * C(x) * ∇C
 * where C = |x1-x2| - d (distance constraint)
 *       ∇C = normalized direction
 *       w = inverse mass
 */

rac_phys_pbd_config rac_phys_pbd_default_config(void) {
    return (rac_phys_pbd_config){
        .substeps = 4,
        .iterations = 2,
        .damping = 0.99f
    };
}

void rac_phys_pbd_solve_distance(rac_phys_vec3 *positions,
                                  float *inv_masses,
                                  const int *constraint_pairs,
                                  const float *rest_lengths,
                                  int num_constraints,
                                  const rac_phys_pbd_config *cfg) {
    for (int iter = 0; iter < cfg->iterations; iter++) {
        for (int c = 0; c < num_constraints; c++) {
            int a = constraint_pairs[c * 2];
            int b = constraint_pairs[c * 2 + 1];
            float w_a = inv_masses[a];
            float w_b = inv_masses[b];
            float w_sum = w_a + w_b;
            if (w_sum < 1e-8f) continue;

            rac_phys_vec3 diff = rac_phys_v3_sub(positions[b], positions[a]);
            float dist = rac_phys_v3_length(diff);  /* RAC: rac_norm chain */
            if (dist < 1e-8f) continue;

            float error = dist - rest_lengths[c];
            rac_phys_vec3 n = rac_phys_v3_scale(diff, 1.0f / dist);

            /* Position corrections weighted by inverse mass */
            float corr = error / w_sum;
            positions[a] = rac_phys_v3_add(positions[a],
                rac_phys_v3_scale(n, w_a * corr));
            positions[b] = rac_phys_v3_sub(positions[b],
                rac_phys_v3_scale(n, w_b * corr));
        }
    }
}
