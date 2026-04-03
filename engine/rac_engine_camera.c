/*
 * rac_engine_camera.c — Camera System Implementation
 * All trigonometry via rac_rotate, all dot products via rac_dot/rac_phys_v3_dot.
 */

#include "rac_engine_camera.h"
#include <string.h>
#include <math.h>

void rac_camera_registry_init(rac_camera_registry *reg)
{
    memset(reg, 0, sizeof(*reg));
    reg->active_camera = -1;
}

int rac_camera_create(rac_camera_registry *reg)
{
    if (reg->num_cameras >= RAC_MAX_CAMERAS) return -1;
    int id = reg->num_cameras++;
    rac_camera *cam = &reg->cameras[id];
    memset(cam, 0, sizeof(*cam));

    cam->position = rac_phys_v3_zero();
    cam->orientation = rac_phys_quat_identity();
    cam->yaw = 0.0f;
    cam->pitch = 0.0f;
    cam->proj_mode = RAC_PROJ_PERSPECTIVE;
    cam->fov_y = RAC_PI / 3.0f;  /* 60 degrees */
    cam->aspect = 640.0f / 480.0f;
    cam->near_plane = 0.1f;
    cam->far_plane = 1000.0f;
    cam->ortho_size = 10.0f;
    cam->follow_entity = -1;
    cam->active = 1;

    if (reg->active_camera < 0)
        reg->active_camera = id;

    return id;
}

void rac_camera_set_perspective(rac_camera *cam, float fov_y, float aspect,
                                float near_p, float far_p)
{
    cam->proj_mode = RAC_PROJ_PERSPECTIVE;
    cam->fov_y = fov_y;
    cam->aspect = aspect;
    cam->near_plane = near_p;
    cam->far_plane = far_p;
}

void rac_camera_set_orthographic(rac_camera *cam, float size, float aspect,
                                 float near_p, float far_p)
{
    cam->proj_mode = RAC_PROJ_ORTHOGRAPHIC;
    cam->ortho_size = size;
    cam->aspect = aspect;
    cam->near_plane = near_p;
    cam->far_plane = far_p;
}

void rac_camera_fps_look(rac_camera *cam, float delta_yaw, float delta_pitch)
{
    cam->yaw += delta_yaw;
    cam->pitch += delta_pitch;

    /* Clamp pitch to avoid gimbal lock */
    float limit = RAC_PI * 0.5f - 0.01f;
    if (cam->pitch > limit) cam->pitch = limit;
    if (cam->pitch < -limit) cam->pitch = -limit;

    /* Rebuild orientation from yaw/pitch via RAC rotation */
    rac_phys_quat qyaw = rac_phys_quat_from_axis_angle(
        rac_phys_v3(0.0f, 1.0f, 0.0f), cam->yaw);
    rac_phys_quat qpitch = rac_phys_quat_from_axis_angle(
        rac_phys_v3(1.0f, 0.0f, 0.0f), cam->pitch);

    cam->orientation = rac_phys_quat_mul(qyaw, qpitch);
    cam->orientation = rac_phys_quat_normalize(cam->orientation);
}

rac_mat4 rac_camera_build_perspective(float fov_y, float aspect, float near_p, float far_p)
{
    rac_mat4 m;
    memset(&m, 0, sizeof(m));

    /* tan(fov_y/2) via rac_rotate: rotate (1,0) by fov_y/2, then sin/cos */
    float half_fov = fov_y * 0.5f;
    rac_vec2 sc = rac_rotate((rac_vec2){1.0f, 0.0f}, half_fov);
    /* sc.x = cos(half_fov), sc.y = sin(half_fov) */
    /* tan = sin/cos — use rac_project to get the ratio */
    float tan_half = (sc.x > 1e-6f) ? (sc.y / sc.x) : 1000.0f;

    float f = 1.0f / tan_half;
    float range = far_p - near_p;

    m.m[0][0] = f / aspect;
    m.m[1][1] = f;
    m.m[2][2] = -(far_p + near_p) / range;
    m.m[2][3] = -(2.0f * far_p * near_p) / range;
    m.m[3][2] = -1.0f;

    return m;
}

rac_mat4 rac_camera_build_orthographic(float size, float aspect, float near_p, float far_p)
{
    rac_mat4 m;
    memset(&m, 0, sizeof(m));

    float range = far_p - near_p;
    m.m[0][0] = 1.0f / (size * aspect);
    m.m[1][1] = 1.0f / size;
    m.m[2][2] = -2.0f / range;
    m.m[2][3] = -(far_p + near_p) / range;
    m.m[3][3] = 1.0f;

    return m;
}

rac_mat4 rac_camera_build_view(rac_phys_vec3 pos, rac_phys_quat orient)
{
    /* View matrix = inverse of camera transform
     * For a rotation quaternion, inverse = conjugate
     * Then apply negative translation rotated into camera space */
    rac_phys_quat inv_q = rac_phys_quat_conjugate(orient);
    rac_phys_mat3 rot = rac_phys_quat_to_mat3(inv_q);

    /* Rotate the negated position */
    rac_phys_vec3 neg_pos = rac_phys_v3_negate(pos);
    rac_phys_vec3 t = rac_phys_mat3_mul_vec3(rot, neg_pos);

    rac_mat4 v;
    memset(&v, 0, sizeof(v));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v.m[i][j] = rot.m[i][j];
    v.m[0][3] = t.x;
    v.m[1][3] = t.y;
    v.m[2][3] = t.z;
    v.m[3][3] = 1.0f;

    return v;
}

static void extract_frustum_planes(rac_camera *cam)
{
    /* Extract frustum planes from view-projection matrix */
    rac_mat4 *vp = &cam->view_proj;

    /* Left: row3 + row0 */
    cam->frustum[0].normal = rac_phys_v3(
        vp->m[3][0] + vp->m[0][0],
        vp->m[3][1] + vp->m[0][1],
        vp->m[3][2] + vp->m[0][2]);
    cam->frustum[0].distance = vp->m[3][3] + vp->m[0][3];

    /* Right: row3 - row0 */
    cam->frustum[1].normal = rac_phys_v3(
        vp->m[3][0] - vp->m[0][0],
        vp->m[3][1] - vp->m[0][1],
        vp->m[3][2] - vp->m[0][2]);
    cam->frustum[1].distance = vp->m[3][3] - vp->m[0][3];

    /* Bottom: row3 + row1 */
    cam->frustum[2].normal = rac_phys_v3(
        vp->m[3][0] + vp->m[1][0],
        vp->m[3][1] + vp->m[1][1],
        vp->m[3][2] + vp->m[1][2]);
    cam->frustum[2].distance = vp->m[3][3] + vp->m[1][3];

    /* Top: row3 - row1 */
    cam->frustum[3].normal = rac_phys_v3(
        vp->m[3][0] - vp->m[1][0],
        vp->m[3][1] - vp->m[1][1],
        vp->m[3][2] - vp->m[1][2]);
    cam->frustum[3].distance = vp->m[3][3] - vp->m[1][3];

    /* Near: row3 + row2 */
    cam->frustum[4].normal = rac_phys_v3(
        vp->m[3][0] + vp->m[2][0],
        vp->m[3][1] + vp->m[2][1],
        vp->m[3][2] + vp->m[2][2]);
    cam->frustum[4].distance = vp->m[3][3] + vp->m[2][3];

    /* Far: row3 - row2 */
    cam->frustum[5].normal = rac_phys_v3(
        vp->m[3][0] - vp->m[2][0],
        vp->m[3][1] - vp->m[2][1],
        vp->m[3][2] - vp->m[2][2]);
    cam->frustum[5].distance = vp->m[3][3] - vp->m[2][3];

    /* Normalize planes using rac_phys_v3_length */
    for (int i = 0; i < 6; i++) {
        float len = rac_phys_v3_length(cam->frustum[i].normal);
        if (len > 1e-6f) {
            cam->frustum[i].normal = rac_phys_v3_scale(cam->frustum[i].normal, 1.0f / len);
            cam->frustum[i].distance /= len;
        }
    }
}

void rac_camera_update(rac_camera *cam)
{
    cam->view_matrix = rac_camera_build_view(cam->position, cam->orientation);

    if (cam->proj_mode == RAC_PROJ_PERSPECTIVE)
        cam->proj_matrix = rac_camera_build_perspective(
            cam->fov_y, cam->aspect, cam->near_plane, cam->far_plane);
    else
        cam->proj_matrix = rac_camera_build_orthographic(
            cam->ortho_size, cam->aspect, cam->near_plane, cam->far_plane);

    cam->view_proj = rac_mat4_mul(cam->proj_matrix, cam->view_matrix);
    extract_frustum_planes(cam);
}

int rac_camera_frustum_test_aabb(const rac_camera *cam, rac_phys_aabb aabb)
{
    /* Test AABB against all 6 frustum planes using rac_phys_v3_dot */
    for (int i = 0; i < 6; i++) {
        rac_phys_vec3 n = cam->frustum[i].normal;
        float d = cam->frustum[i].distance;

        /* Find the positive vertex (furthest along plane normal) */
        rac_phys_vec3 p;
        p.x = (n.x >= 0.0f) ? aabb.max.x : aabb.min.x;
        p.y = (n.y >= 0.0f) ? aabb.max.y : aabb.min.y;
        p.z = (n.z >= 0.0f) ? aabb.max.z : aabb.min.z;

        if (rac_phys_v3_dot(n, p) + d < 0.0f)
            return 0;  /* outside */
    }
    return 1;  /* inside or intersecting */
}

void rac_camera_follow_update(rac_camera *cam, rac_phys_vec3 target_pos, float dt)
{
    /* Desired position: behind and above target */
    rac_phys_vec3 forward = rac_phys_quat_rotate_vec3(cam->orientation,
        rac_phys_v3(0.0f, 0.0f, -1.0f));
    rac_phys_vec3 desired = rac_phys_v3_sub(target_pos,
        rac_phys_v3_scale(forward, cam->follow_distance));
    desired.y += cam->follow_height;

    /* Smooth lerp */
    float t = 1.0f - rac_exp(-cam->follow_smoothing * dt);
    cam->position = rac_phys_v3_lerp(cam->position, desired, t);
}
