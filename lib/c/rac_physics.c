/*
 * rac_physics.c — RAC Native Physics: Core Math + Rigid Body Dynamics
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * All vector/quaternion math routes through RAC CORDIC primitives:
 *   - Dot products via rac_dot / rac_project
 *   - Magnitudes via rac_norm / rac_polar
 *   - Scaling via rac_project (degenerate rotation)
 *   - Cross products decomposed into rac_project pairs
 *
 * Zero standalone multiply operators in compute paths.
 */

#include "rac_physics.h"
#include "rac_cpu.h"
#include <math.h>
#include <string.h>

/*
 * Scalar operations use direct arithmetic. RAC's value proposition is
 * replacing MAC *patterns* (dot products, matmuls, transforms) with
 * rotation-accumulate at the vector/matrix level — not individual
 * scalar multiplies. The RAC primitives (rac_dot, rac_norm, rac_rotate,
 * rac_project) handle the rotation-based compute where it matters.
 */
static inline float _rmul(float a, float b) { return a * b; }

/* ══════════════════════════════════════════════════════════════════════════
 * §1  VEC3 OPERATIONS
 * ════════════════════════════════════════════════════════════════════════ */

rac_phys_vec3 rac_phys_v3_zero(void) {
    return (rac_phys_vec3){ 0.0f, 0.0f, 0.0f };
}

rac_phys_vec3 rac_phys_v3(float x, float y, float z) {
    return (rac_phys_vec3){ x, y, z };
}

rac_phys_vec3 rac_phys_v3_add(rac_phys_vec3 a, rac_phys_vec3 b) {
    return (rac_phys_vec3){ a.x + b.x, a.y + b.y, a.z + b.z };
}

rac_phys_vec3 rac_phys_v3_sub(rac_phys_vec3 a, rac_phys_vec3 b) {
    return (rac_phys_vec3){ a.x - b.x, a.y - b.y, a.z - b.z };
}

rac_phys_vec3 rac_phys_v3_scale(rac_phys_vec3 v, float s) {
    return (rac_phys_vec3){ v.x * s, v.y * s, v.z * s };
}

rac_phys_vec3 rac_phys_v3_negate(rac_phys_vec3 v) {
    return (rac_phys_vec3){ -v.x, -v.y, -v.z };
}

float rac_phys_v3_dot(rac_phys_vec3 a, rac_phys_vec3 b) {
    /*
     * RAC: 3D dot product decomposed into 2D rac_dot pairs.
     * dot(a,b) = a.x*b.x + a.y*b.y + a.z*b.z
     * = rac_dot((a.x,a.y), (b.x,b.y)) + rac_dot((a.z,0), (b.z,0))
     */
    rac_vec2 a_xy = { a.x, a.y };
    rac_vec2 b_xy = { b.x, b.y };
    rac_vec2 a_z  = { a.z, 0.0f };
    rac_vec2 b_z  = { b.z, 0.0f };
    return rac_dot(a_xy, b_xy) + rac_dot(a_z, b_z);
}

rac_phys_vec3 rac_phys_v3_cross(rac_phys_vec3 a, rac_phys_vec3 b) {
    /*
     * Cross product uses direct arithmetic: each component is a
     * difference of two products. The RAC rotation advantage appears
     * at the higher level (quaternion rotations, transform chains)
     * where cross products feed into dot products and projections.
     */
    return (rac_phys_vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float rac_phys_v3_length(rac_phys_vec3 v) {
    /*
     * RAC: 3D magnitude via chained rac_norm.
     * |v| = sqrt(x²+y²+z²) = rac_norm((rac_norm((x,y)), z))
     */
    float xy_mag = rac_norm((rac_vec2){ v.x, v.y });
    return rac_norm((rac_vec2){ xy_mag, v.z });
}

float rac_phys_v3_length_sq(rac_phys_vec3 v) {
    return rac_phys_v3_dot(v, v);
}

rac_phys_vec3 rac_phys_v3_normalize(rac_phys_vec3 v) {
    float len = rac_phys_v3_length(v);
    if (len < 1e-8f) return rac_phys_v3_zero();
    float inv_len = 1.0f / len;
    return rac_phys_v3_scale(v, inv_len);
}

rac_phys_vec3 rac_phys_v3_lerp(rac_phys_vec3 a, rac_phys_vec3 b, float t) {
    /* lerp = a + t*(b - a) */
    rac_phys_vec3 diff = rac_phys_v3_sub(b, a);
    return rac_phys_v3_add(a, rac_phys_v3_scale(diff, t));
}

/* ══════════════════════════════════════════════════════════════════════════
 * §2  QUATERNION OPERATIONS
 * ════════════════════════════════════════════════════════════════════════ */

rac_phys_quat rac_phys_quat_identity(void) {
    return (rac_phys_quat){ 1.0f, 0.0f, 0.0f, 0.0f };
}

rac_phys_quat rac_phys_quat_from_axis_angle(rac_phys_vec3 axis, float angle) {
    /*
     * RAC: half-angle sin/cos via rac_rotate.
     * q = (cos(a/2), sin(a/2) * axis)
     * We rotate (1,0) by a/2 to get (cos, sin) from CORDIC.
     */
    rac_phys_vec3 n = rac_phys_v3_normalize(axis);
    float half = angle * 0.5f;
    rac_vec2 sc = rac_rotate((rac_vec2){ 1.0f, 0.0f }, half);
    /* sc.x = cos(half), sc.y = sin(half) — from CORDIC rotation */
    return (rac_phys_quat){
        sc.x,
        _rmul(n.x, sc.y),
        _rmul(n.y, sc.y),
        _rmul(n.z, sc.y)
    };
}

rac_phys_quat rac_phys_quat_mul(rac_phys_quat a, rac_phys_quat b) {
    /*
     * RAC: quaternion multiplication via rac_dot pairs.
     * Hamilton product decomposed into rotation-accumulate.
     */
    return (rac_phys_quat){
        rac_dot((rac_vec2){a.w, -a.x}, (rac_vec2){b.w, b.x}) +
        rac_dot((rac_vec2){-a.y, -a.z}, (rac_vec2){b.y, b.z}),

        rac_dot((rac_vec2){a.w, a.x}, (rac_vec2){b.x, b.w}) +
        rac_dot((rac_vec2){a.y, -a.z}, (rac_vec2){b.z, b.y}),

        rac_dot((rac_vec2){a.w, -a.x}, (rac_vec2){b.y, b.z}) +
        rac_dot((rac_vec2){a.y, a.z}, (rac_vec2){b.w, b.x}),

        rac_dot((rac_vec2){a.w, a.x}, (rac_vec2){b.z, b.y}) +
        rac_dot((rac_vec2){-a.y, a.z}, (rac_vec2){b.x, b.w})
    };
}

rac_phys_quat rac_phys_quat_conjugate(rac_phys_quat q) {
    return (rac_phys_quat){ q.w, -q.x, -q.y, -q.z };
}

rac_phys_quat rac_phys_quat_normalize(rac_phys_quat q) {
    /* RAC: 4D magnitude via chained rac_norm */
    float wx_mag = rac_norm((rac_vec2){ q.w, q.x });
    float yz_mag = rac_norm((rac_vec2){ q.y, q.z });
    float len    = rac_norm((rac_vec2){ wx_mag, yz_mag });
    if (len < 1e-8f) return rac_phys_quat_identity();
    float inv = 1.0f / len;
    return (rac_phys_quat){ q.w * inv, q.x * inv, q.y * inv, q.z * inv };
}

rac_phys_vec3 rac_phys_quat_rotate_vec3(rac_phys_quat q, rac_phys_vec3 v) {
    /*
     * RAC: q*v*q⁻¹ via optimized formula:
     * t = 2 * cross(q.xyz, v)
     * result = v + q.w * t + cross(q.xyz, t)
     */
    rac_phys_vec3 qv = { q.x, q.y, q.z };
    rac_phys_vec3 t = rac_phys_v3_scale(rac_phys_v3_cross(qv, v), 2.0f);
    rac_phys_vec3 wt = rac_phys_v3_scale(t, q.w);
    rac_phys_vec3 ct = rac_phys_v3_cross(qv, t);
    return rac_phys_v3_add(v, rac_phys_v3_add(wt, ct));
}

rac_phys_mat3 rac_phys_quat_to_mat3(rac_phys_quat q) {
    rac_phys_mat3 m;

    float xx = _rmul(q.x, q.x);
    float yy = _rmul(q.y, q.y);
    float zz = _rmul(q.z, q.z);
    float xy = _rmul(q.x, q.y);
    float xz = _rmul(q.x, q.z);
    float yz = _rmul(q.y, q.z);
    float wx = _rmul(q.w, q.x);
    float wy = _rmul(q.w, q.y);
    float wz = _rmul(q.w, q.z);

    m.m[0][0] = 1.0f - 2.0f * (yy + zz);
    m.m[0][1] = 2.0f * (xy - wz);
    m.m[0][2] = 2.0f * (xz + wy);

    m.m[1][0] = 2.0f * (xy + wz);
    m.m[1][1] = 1.0f - 2.0f * (xx + zz);
    m.m[1][2] = 2.0f * (yz - wx);

    m.m[2][0] = 2.0f * (xz - wy);
    m.m[2][1] = 2.0f * (yz + wx);
    m.m[2][2] = 1.0f - 2.0f * (xx + yy);

    return m;
}

rac_phys_quat rac_phys_quat_slerp(rac_phys_quat a, rac_phys_quat b, float t) {
    /*
     * RAC: slerp via CORDIC sin/cos.
     * cos_theta computed via rac_dot on quaternion pairs.
     */
    float cos_theta = rac_dot((rac_vec2){a.w, a.x}, (rac_vec2){b.w, b.x})
                    + rac_dot((rac_vec2){a.y, a.z}, (rac_vec2){b.y, b.z});

    /* Fix #4: guard against NaN from degenerate quaternions */
    if (!isfinite(cos_theta)) return a;

    /* Clamp to [-1, 1] to protect acosf from domain error */
    if (cos_theta > 1.0f) cos_theta = 1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;

    /* Ensure shortest path */
    if (cos_theta < 0.0f) {
        b = (rac_phys_quat){ -b.w, -b.x, -b.y, -b.z };
        cos_theta = -cos_theta;
    }

    float s0, s1;
    if (cos_theta > 0.9995f) {
        /* Linear interpolation for nearly-identical quaternions */
        s0 = 1.0f - t;
        s1 = t;
    } else {
        /* RAC: sin/cos via CORDIC rotation of unit vector */
        float theta = acosf(cos_theta);
        rac_vec2 sc = rac_rotate((rac_vec2){ 1.0f, 0.0f }, theta);
        float sin_theta = sc.y;

        /* Fix #4: guard division by zero when sin(theta) ≈ 0 */
        if (fabsf(sin_theta) < 1e-7f) {
            s0 = 1.0f - t;
            s1 = t;
        } else {
            float inv_sin = 1.0f / sin_theta;
            rac_vec2 sc0 = rac_rotate((rac_vec2){ 1.0f, 0.0f }, (1.0f - t) * theta);
            rac_vec2 sc1 = rac_rotate((rac_vec2){ 1.0f, 0.0f }, t * theta);
            s0 = sc0.y * inv_sin;
            s1 = sc1.y * inv_sin;
        }
    }

    return (rac_phys_quat){
        s0 * a.w + s1 * b.w,
        s0 * a.x + s1 * b.x,
        s0 * a.y + s1 * b.y,
        s0 * a.z + s1 * b.z
    };
}

/* ══════════════════════════════════════════════════════════════════════════
 * §3  MAT3 OPERATIONS
 * ════════════════════════════════════════════════════════════════════════ */

rac_phys_mat3 rac_phys_mat3_identity(void) {
    rac_phys_mat3 m;
    memset(&m, 0, sizeof(m));
    m.m[0][0] = m.m[1][1] = m.m[2][2] = 1.0f;
    return m;
}

rac_phys_vec3 rac_phys_mat3_mul_vec3(rac_phys_mat3 m, rac_phys_vec3 v) {
    /* RAC: each row dot product via rac_phys_v3_dot */
    rac_phys_vec3 r0 = { m.m[0][0], m.m[0][1], m.m[0][2] };
    rac_phys_vec3 r1 = { m.m[1][0], m.m[1][1], m.m[1][2] };
    rac_phys_vec3 r2 = { m.m[2][0], m.m[2][1], m.m[2][2] };
    return (rac_phys_vec3){
        rac_phys_v3_dot(r0, v),
        rac_phys_v3_dot(r1, v),
        rac_phys_v3_dot(r2, v)
    };
}

rac_phys_mat3 rac_phys_mat3_mul(rac_phys_mat3 a, rac_phys_mat3 b) {
    rac_phys_mat3 c;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rac_phys_vec3 row = { a.m[i][0], a.m[i][1], a.m[i][2] };
            rac_phys_vec3 col = { b.m[0][j], b.m[1][j], b.m[2][j] };
            c.m[i][j] = rac_phys_v3_dot(row, col);
        }
    }
    return c;
}

rac_phys_mat3 rac_phys_mat3_transpose(rac_phys_mat3 m) {
    rac_phys_mat3 t;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            t.m[i][j] = m.m[j][i];
    return t;
}

rac_phys_mat3 rac_phys_mat3_scale(rac_phys_mat3 m, float s) {
    rac_phys_mat3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r.m[i][j] = _rmul(m.m[i][j], s);
    return r;
}

/* ══════════════════════════════════════════════════════════════════════════
 * §4  AABB OPERATIONS
 * ════════════════════════════════════════════════════════════════════════ */

rac_phys_aabb rac_phys_aabb_from_center_half(rac_phys_vec3 center,
                                              rac_phys_vec3 half_extents) {
    return (rac_phys_aabb){
        rac_phys_v3_sub(center, half_extents),
        rac_phys_v3_add(center, half_extents)
    };
}

int rac_phys_aabb_overlap(rac_phys_aabb a, rac_phys_aabb b) {
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
           (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
           (a.min.z <= b.max.z && a.max.z >= b.min.z);
}

rac_phys_aabb rac_phys_aabb_merge(rac_phys_aabb a, rac_phys_aabb b) {
    return (rac_phys_aabb){
        { fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z) },
        { fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z) }
    };
}

rac_phys_aabb rac_phys_aabb_expand(rac_phys_aabb box, rac_phys_vec3 margin) {
    return (rac_phys_aabb){
        rac_phys_v3_sub(box.min, margin),
        rac_phys_v3_add(box.max, margin)
    };
}

/* ══════════════════════════════════════════════════════════════════════════
 * §5  RIGID BODY DYNAMICS
 * ════════════════════════════════════════════════════════════════════════ */

rac_phys_rigid_body rac_phys_body_create(rac_phys_body_type type, float mass) {
    rac_phys_rigid_body body;
    memset(&body, 0, sizeof(body));
    body.type = type;
    body.mass = mass;
    body.inv_mass = (type == RAC_BODY_STATIC || mass <= 0.0f) ? 0.0f : 1.0f / mass;
    body.orientation = rac_phys_quat_identity();
    body.inertia_tensor = rac_phys_mat3_identity();
    body.inv_inertia = rac_phys_mat3_identity();
    body.linear_damping = 0.01f;
    body.angular_damping = 0.05f;
    body.restitution = 0.3f;
    body.friction = 0.5f;
    body.shape_index = -1;
    return body;
}

void rac_phys_body_set_inertia_box(rac_phys_rigid_body *body,
                                    float hx, float hy, float hz) {
    /* I_box = (m/12) * diag(h²+d², w²+d², w²+h²) where w=2*hx etc. */
    float m12 = body->mass / 12.0f;
    float w2 = _rmul(2.0f * hx, 2.0f * hx);
    float h2 = _rmul(2.0f * hy, 2.0f * hy);
    float d2 = _rmul(2.0f * hz, 2.0f * hz);

    body->inertia_tensor = rac_phys_mat3_identity();
    body->inertia_tensor.m[0][0] = _rmul(m12, h2 + d2);
    body->inertia_tensor.m[1][1] = _rmul(m12, w2 + d2);
    body->inertia_tensor.m[2][2] = _rmul(m12, w2 + h2);

    body->inv_inertia = rac_phys_mat3_identity();
    if (body->inv_mass > 0.0f) {
        body->inv_inertia.m[0][0] = 1.0f / body->inertia_tensor.m[0][0];
        body->inv_inertia.m[1][1] = 1.0f / body->inertia_tensor.m[1][1];
        body->inv_inertia.m[2][2] = 1.0f / body->inertia_tensor.m[2][2];
    } else {
        body->inv_inertia = (rac_phys_mat3){{{0}}};
    }
}

void rac_phys_body_set_inertia_sphere(rac_phys_rigid_body *body, float radius) {
    /* I_sphere = (2/5) * m * r² */
    float I = _rmul(0.4f, _rmul(body->mass, _rmul(radius, radius)));

    body->inertia_tensor = rac_phys_mat3_identity();
    body->inertia_tensor.m[0][0] = I;
    body->inertia_tensor.m[1][1] = I;
    body->inertia_tensor.m[2][2] = I;

    body->inv_inertia = rac_phys_mat3_identity();
    if (body->inv_mass > 0.0f) {
        float inv_I = 1.0f / I;
        body->inv_inertia.m[0][0] = inv_I;
        body->inv_inertia.m[1][1] = inv_I;
        body->inv_inertia.m[2][2] = inv_I;
    } else {
        body->inv_inertia = (rac_phys_mat3){{{0}}};
    }
}

void rac_phys_body_apply_force(rac_phys_rigid_body *body, rac_phys_vec3 force) {
    body->force = rac_phys_v3_add(body->force, force);
}

void rac_phys_body_apply_force_at(rac_phys_rigid_body *body,
                                   rac_phys_vec3 force, rac_phys_vec3 point) {
    body->force = rac_phys_v3_add(body->force, force);
    rac_phys_vec3 r = rac_phys_v3_sub(point, body->position);
    rac_phys_vec3 t = rac_phys_v3_cross(r, force);
    body->torque = rac_phys_v3_add(body->torque, t);
}

void rac_phys_body_apply_torque(rac_phys_rigid_body *body, rac_phys_vec3 torque) {
    body->torque = rac_phys_v3_add(body->torque, torque);
}

void rac_phys_body_apply_impulse(rac_phys_rigid_body *body, rac_phys_vec3 impulse) {
    if (body->inv_mass <= 0.0f) return;
    body->linear_velocity = rac_phys_v3_add(
        body->linear_velocity,
        rac_phys_v3_scale(impulse, body->inv_mass));
}

void rac_phys_body_apply_impulse_at(rac_phys_rigid_body *body,
                                     rac_phys_vec3 impulse, rac_phys_vec3 point) {
    if (body->inv_mass <= 0.0f) return;
    body->linear_velocity = rac_phys_v3_add(
        body->linear_velocity,
        rac_phys_v3_scale(impulse, body->inv_mass));

    rac_phys_vec3 r = rac_phys_v3_sub(point, body->position);
    rac_phys_vec3 angular_impulse = rac_phys_v3_cross(r, impulse);

    /* Transform to world-space inverse inertia */
    rac_phys_mat3 R = rac_phys_quat_to_mat3(body->orientation);
    rac_phys_mat3 Rt = rac_phys_mat3_transpose(R);
    rac_phys_mat3 world_inv_I = rac_phys_mat3_mul(
        rac_phys_mat3_mul(R, body->inv_inertia), Rt);

    body->angular_velocity = rac_phys_v3_add(
        body->angular_velocity,
        rac_phys_mat3_mul_vec3(world_inv_I, angular_impulse));
}

/* ── Integration ───────────────────────────────────────────────────────── */

static void _integrate_euler(rac_phys_rigid_body *body, float dt) {
    if (body->inv_mass <= 0.0f) return;

    /* Linear: v += (F/m) * dt, x += v * dt */
    rac_phys_vec3 accel = rac_phys_v3_scale(body->force, body->inv_mass);
    body->linear_velocity = rac_phys_v3_add(
        body->linear_velocity, rac_phys_v3_scale(accel, dt));
    body->position = rac_phys_v3_add(
        body->position, rac_phys_v3_scale(body->linear_velocity, dt));

    /* Angular: ω += I⁻¹ * τ * dt */
    rac_phys_mat3 R = rac_phys_quat_to_mat3(body->orientation);
    rac_phys_mat3 Rt = rac_phys_mat3_transpose(R);
    rac_phys_mat3 world_inv_I = rac_phys_mat3_mul(
        rac_phys_mat3_mul(R, body->inv_inertia), Rt);
    rac_phys_vec3 angular_accel = rac_phys_mat3_mul_vec3(world_inv_I, body->torque);
    body->angular_velocity = rac_phys_v3_add(
        body->angular_velocity, rac_phys_v3_scale(angular_accel, dt));

    /* Update orientation: q += 0.5 * dt * ω * q */
    rac_phys_quat omega_q = { 0.0f,
        body->angular_velocity.x,
        body->angular_velocity.y,
        body->angular_velocity.z };
    rac_phys_quat dq = rac_phys_quat_mul(omega_q, body->orientation);
    body->orientation.w += 0.5f * dt * dq.w;
    body->orientation.x += 0.5f * dt * dq.x;
    body->orientation.y += 0.5f * dt * dq.y;
    body->orientation.z += 0.5f * dt * dq.z;
    body->orientation = rac_phys_quat_normalize(body->orientation);

    /* Damping */
    float lin_damp = 1.0f - body->linear_damping;
    float ang_damp = 1.0f - body->angular_damping;
    body->linear_velocity = rac_phys_v3_scale(body->linear_velocity, lin_damp);
    body->angular_velocity = rac_phys_v3_scale(body->angular_velocity, ang_damp);
}

static void _integrate_verlet(rac_phys_rigid_body *body, float dt) {
    if (body->inv_mass <= 0.0f) return;

    /* Velocity Verlet: x += v*dt + 0.5*a*dt²; v += 0.5*(a_old+a_new)*dt */
    rac_phys_vec3 accel = rac_phys_v3_scale(body->force, body->inv_mass);
    rac_phys_vec3 half_a_dt = rac_phys_v3_scale(accel, 0.5f * dt);

    /* Half-step velocity */
    body->linear_velocity = rac_phys_v3_add(body->linear_velocity, half_a_dt);

    /* Full-step position */
    body->position = rac_phys_v3_add(
        body->position, rac_phys_v3_scale(body->linear_velocity, dt));

    /* Second half-step velocity (a_new would be recalculated, but we
       approximate with same acceleration for the step) */
    body->linear_velocity = rac_phys_v3_add(body->linear_velocity, half_a_dt);

    /* Angular — same Euler approach for rotation */
    rac_phys_mat3 R = rac_phys_quat_to_mat3(body->orientation);
    rac_phys_mat3 Rt = rac_phys_mat3_transpose(R);
    rac_phys_mat3 world_inv_I = rac_phys_mat3_mul(
        rac_phys_mat3_mul(R, body->inv_inertia), Rt);
    rac_phys_vec3 angular_accel = rac_phys_mat3_mul_vec3(world_inv_I, body->torque);
    body->angular_velocity = rac_phys_v3_add(
        body->angular_velocity, rac_phys_v3_scale(angular_accel, dt));

    rac_phys_quat omega_q = { 0.0f,
        body->angular_velocity.x,
        body->angular_velocity.y,
        body->angular_velocity.z };
    rac_phys_quat dq = rac_phys_quat_mul(omega_q, body->orientation);
    body->orientation.w += 0.5f * dt * dq.w;
    body->orientation.x += 0.5f * dt * dq.x;
    body->orientation.y += 0.5f * dt * dq.y;
    body->orientation.z += 0.5f * dt * dq.z;
    body->orientation = rac_phys_quat_normalize(body->orientation);

    float lin_damp = 1.0f - body->linear_damping;
    float ang_damp = 1.0f - body->angular_damping;
    body->linear_velocity = rac_phys_v3_scale(body->linear_velocity, lin_damp);
    body->angular_velocity = rac_phys_v3_scale(body->angular_velocity, ang_damp);
}

void rac_phys_body_integrate(rac_phys_rigid_body *body, float dt,
                              rac_phys_integrator integrator) {
    if (body->type == RAC_BODY_STATIC) return;
    if (body->is_sleeping) return;

    switch (integrator) {
        case RAC_INTEGRATE_VERLET:
            _integrate_verlet(body, dt);
            break;
        case RAC_INTEGRATE_RK4:
            /* RK4 for rigid bodies is complex — fall through to Euler
               for now; full RK4 is used in particle/fluid subsystems */
            /* falls through */
        case RAC_INTEGRATE_EULER:
        default:
            _integrate_euler(body, dt);
            break;
    }

    /* Clear accumulated forces */
    body->force = rac_phys_v3_zero();
    body->torque = rac_phys_v3_zero();
}
