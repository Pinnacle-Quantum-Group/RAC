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

                /* Clamp: accumulated impulse >= 0 (no pulling) */
                float old_lambda = 0.0f;  /* simplified — no warm start cache per point */
                float new_lambda = fmaxf(old_lambda + lambda, 0.0f);
                lambda = new_lambda - old_lambda;

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

                        /* Coulomb clamp */
                        float max_friction = mu * new_lambda;
                        friction_lambda = fmaxf(-max_friction,
                            fminf(friction_lambda, max_friction));

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

            if (c->type == RAC_CONSTRAINT_DISTANCE) {
                /* Distance constraint: maintain rest_length between anchors */
                rac_phys_vec3 wa = rac_phys_v3_add(a->position,
                    rac_phys_quat_rotate_vec3(a->orientation, c->anchor_a));
                rac_phys_vec3 wb = b ?
                    rac_phys_v3_add(b->position,
                        rac_phys_quat_rotate_vec3(b->orientation, c->anchor_b))
                    : c->anchor_b;  /* world anchor */

                rac_phys_vec3 diff = rac_phys_v3_sub(wb, wa);
                float dist = rac_phys_v3_length(diff);
                if (dist < 1e-8f) continue;

                rac_phys_vec3 n = rac_phys_v3_scale(diff, 1.0f / dist);
                float error = dist - c->rest_length;

                /* Velocity along constraint axis */
                rac_phys_vec3 r_a = rac_phys_v3_sub(wa, a->position);
                rac_phys_vec3 v_a = rac_phys_v3_add(a->linear_velocity,
                    rac_phys_v3_cross(a->angular_velocity, r_a));

                float v_rel_n;
                rac_phys_vec3 r_b_vec = rac_phys_v3_zero();
                if (b) {
                    r_b_vec = rac_phys_v3_sub(wb, b->position);
                    rac_phys_vec3 v_b = rac_phys_v3_add(b->linear_velocity,
                        rac_phys_v3_cross(b->angular_velocity, r_b_vec));
                    v_rel_n = rac_phys_v3_dot(rac_phys_v3_sub(v_b, v_a), n);
                } else {
                    v_rel_n = rac_phys_v3_dot(rac_phys_v3_negate(v_a), n);
                }

                float bias = cfg->baumgarte * inv_dt * error;
                float inv_mass = a->inv_mass + (b ? b->inv_mass : 0.0f);
                if (inv_mass < 1e-8f) continue;

                float lambda_c = (1.0f / inv_mass) * (-v_rel_n + bias);

                /* Apply */
                a->linear_velocity = rac_phys_v3_sub(a->linear_velocity,
                    rac_phys_v3_scale(n, lambda_c * a->inv_mass));
                if (b) {
                    b->linear_velocity = rac_phys_v3_add(b->linear_velocity,
                        rac_phys_v3_scale(n, lambda_c * b->inv_mass));
                }
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
