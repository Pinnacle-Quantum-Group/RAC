/*
 * rac_engine_camera.h — Camera System
 * Perspective/orthographic projection, frustum culling, FPS look.
 * All math via RAC primitives.
 */

#ifndef RAC_ENGINE_CAMERA_H
#define RAC_ENGINE_CAMERA_H

#include "rac_engine_scene.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Projection mode ───────────────────────────────────────────────────── */

typedef enum {
    RAC_PROJ_PERSPECTIVE  = 0,
    RAC_PROJ_ORTHOGRAPHIC = 1,
} rac_projection_mode;

/* ── Frustum plane ─────────────────────────────────────────────────────── */

typedef struct {
    rac_phys_vec3 normal;
    float         distance;
} rac_frustum_plane;

/* ── Camera ────────────────────────────────────────────────────────────── */

#define RAC_MAX_CAMERAS 8

typedef struct {
    /* Transform */
    rac_phys_vec3 position;
    rac_phys_quat orientation;
    float         yaw;
    float         pitch;

    /* Projection */
    rac_projection_mode proj_mode;
    float fov_y;            /* radians, perspective only */
    float aspect;           /* width / height */
    float near_plane;
    float far_plane;
    float ortho_size;       /* half-height for ortho */

    /* Computed matrices */
    rac_mat4 view_matrix;
    rac_mat4 proj_matrix;
    rac_mat4 view_proj;

    /* Frustum planes (left, right, bottom, top, near, far) */
    rac_frustum_plane frustum[6];

    /* Follow target */
    int           follow_entity;   /* -1 = no follow */
    float         follow_distance;
    float         follow_height;
    float         follow_smoothing; /* lerp factor per second */

    int active;
} rac_camera;

typedef struct {
    rac_camera cameras[RAC_MAX_CAMERAS];
    int        num_cameras;
    int        active_camera;
} rac_camera_registry;

/* ── API ───────────────────────────────────────────────────────────────── */

void rac_camera_registry_init(rac_camera_registry *reg);
int  rac_camera_create(rac_camera_registry *reg);

/* Set up perspective */
void rac_camera_set_perspective(rac_camera *cam, float fov_y, float aspect,
                                float near_p, float far_p);
void rac_camera_set_orthographic(rac_camera *cam, float size, float aspect,
                                 float near_p, float far_p);

/* FPS-style look: apply yaw/pitch deltas */
void rac_camera_fps_look(rac_camera *cam, float delta_yaw, float delta_pitch);

/* Update view/proj matrices and frustum from current position + orientation */
void rac_camera_update(rac_camera *cam);

/* Build projection matrix (RAC-native) */
rac_mat4 rac_camera_build_perspective(float fov_y, float aspect, float near_p, float far_p);
rac_mat4 rac_camera_build_orthographic(float size, float aspect, float near_p, float far_p);

/* Build view matrix from position + orientation */
rac_mat4 rac_camera_build_view(rac_phys_vec3 pos, rac_phys_quat orient);

/* Frustum culling: test AABB against frustum */
int rac_camera_frustum_test_aabb(const rac_camera *cam, rac_phys_aabb aabb);

/* Smooth follow update */
void rac_camera_follow_update(rac_camera *cam, rac_phys_vec3 target_pos, float dt);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_CAMERA_H */
