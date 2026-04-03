/*
 * rac_engine_core.h — Game Loop & Engine Lifecycle
 * Fixed-timestep physics, variable render, frame profiling.
 */

#ifndef RAC_ENGINE_CORE_H
#define RAC_ENGINE_CORE_H

#include "rac_physics.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct rac_ecs_world rac_ecs_world_fwd;

/* ── Engine configuration ──────────────────────────────────────────────── */

typedef struct {
    int   window_width;
    int   window_height;
    float physics_dt;          /* fixed timestep (default 1/60) */
    int   max_substeps;        /* max physics sub-steps per frame */
    int   target_fps;          /* 0 = unlimited */
    int   headless;            /* 1 = no SDL, PPM output only */
    const char *window_title;
} rac_engine_config;

rac_engine_config rac_engine_default_config(void);

/* ── Frame timing ──────────────────────────────────────────────────────── */

typedef struct {
    double  frame_time;        /* seconds for last frame */
    double  physics_time;      /* accumulated physics time */
    double  render_time;       /* render time this frame */
    double  total_time;        /* time since engine start */
    int     frame_count;
    float   fps;               /* smoothed FPS */
    float   fps_accumulator;
    int     fps_frame_count;
} rac_engine_timing;

/* ── Engine state ──────────────────────────────────────────────────────── */

typedef struct rac_engine rac_engine;

/* ── Callback types ────────────────────────────────────────────────────── */

typedef void (*rac_engine_update_fn)(rac_engine *engine, float dt);
typedef void (*rac_engine_render_fn)(rac_engine *engine);
typedef void (*rac_engine_init_fn)(rac_engine *engine);
typedef void (*rac_engine_cleanup_fn)(rac_engine *engine);

/* ── Engine struct ─────────────────────────────────────────────────────── */

struct rac_engine {
    rac_engine_config  config;
    rac_engine_timing  timing;

    /* Subsystem pointers (set during init) */
    void *ecs;           /* rac_ecs_world* */
    void *scene;         /* rac_scene_graph* */
    void *physics;       /* rac_phys_world* */
    void *framebuffer;   /* rac_framebuffer* */
    void *camera_reg;    /* rac_camera_registry* */
    void *light_reg;     /* rac_light_registry* */
    void *mesh_reg;      /* rac_mesh_registry* */
    void *audio;         /* rac_audio_engine* */
    void *input;         /* rac_input_system* */
    void *render_state;  /* rac_render_state* */

    /* User callbacks */
    rac_engine_init_fn    on_init;
    rac_engine_cleanup_fn on_cleanup;
    rac_engine_update_fn  on_update;
    rac_engine_render_fn  on_render;
    void                 *user_data;

    /* State */
    int running;
    int frame_output;    /* 1 = write PPM each frame */
};

/* ── API ───────────────────────────────────────────────────────────────── */

/* Create and initialize engine with all subsystems */
rac_engine *rac_engine_create(const rac_engine_config *cfg);

/* Set user callbacks */
void rac_engine_set_callbacks(rac_engine *engine,
                              rac_engine_init_fn init,
                              rac_engine_update_fn update,
                              rac_engine_render_fn render,
                              rac_engine_cleanup_fn cleanup);

/* Initialize and enter main loop */
int  rac_engine_init(rac_engine *engine);
void rac_engine_run(rac_engine *engine);
void rac_engine_run_frames(rac_engine *engine, int num_frames);
void rac_engine_shutdown(rac_engine *engine);

/* Request engine stop */
void rac_engine_quit(rac_engine *engine);

/* Get current time in seconds (high-resolution) */
double rac_engine_get_time(void);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_CORE_H */
