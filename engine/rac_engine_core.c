/*
 * rac_engine_core.c — Game Loop & Engine Lifecycle Implementation
 * Fixed-timestep physics (1/60), variable render, signal-safe shutdown.
 */

#include "rac_engine_core.h"
#include "rac_engine.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>
#include <time.h>

/* ── Timing ────────────────────────────────────────────────────────────── */

double rac_engine_get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Signal handling ───────────────────────────────────────────────────── */

static volatile int g_engine_quit = 0;

static void signal_handler(int sig)
{
    (void)sig;
    g_engine_quit = 1;
}

/* ── Default config ────────────────────────────────────────────────────── */

rac_engine_config rac_engine_default_config(void)
{
    rac_engine_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.window_width = 640;
    cfg.window_height = 480;
    cfg.physics_dt = 1.0f / 60.0f;
    cfg.max_substeps = 4;
    cfg.target_fps = 0;
    cfg.headless = 1;
    cfg.window_title = "RAC Engine";
    return cfg;
}

/* ── Engine creation ───────────────────────────────────────────────────── */

rac_engine *rac_engine_create(const rac_engine_config *cfg)
{
    rac_engine *e = (rac_engine *)calloc(1, sizeof(rac_engine));
    if (!e) return NULL;

    e->config = cfg ? *cfg : rac_engine_default_config();
    return e;
}

void rac_engine_set_callbacks(rac_engine *engine,
                              rac_engine_init_fn init_fn,
                              rac_engine_update_fn update_fn,
                              rac_engine_render_fn render_fn,
                              rac_engine_cleanup_fn cleanup_fn)
{
    engine->on_init = init_fn;
    engine->on_update = update_fn;
    engine->on_render = render_fn;
    engine->on_cleanup = cleanup_fn;
}

int rac_engine_init(rac_engine *engine)
{
    /* Set up signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Allocate subsystems */
    rac_ecs_world *ecs = (rac_ecs_world *)calloc(1, sizeof(rac_ecs_world));
    rac_scene_graph *scene = (rac_scene_graph *)calloc(1, sizeof(rac_scene_graph));
    rac_camera_registry *cam_reg = (rac_camera_registry *)calloc(1, sizeof(rac_camera_registry));
    rac_light_registry *light_reg = (rac_light_registry *)calloc(1, sizeof(rac_light_registry));
    rac_mesh_registry *mesh_reg = (rac_mesh_registry *)calloc(1, sizeof(rac_mesh_registry));
    rac_input_system *input = (rac_input_system *)calloc(1, sizeof(rac_input_system));
    rac_render_state *rs = (rac_render_state *)calloc(1, sizeof(rac_render_state));

    if (!ecs || !scene || !cam_reg || !light_reg || !mesh_reg || !input || !rs) {
        free(ecs); free(scene); free(cam_reg); free(light_reg);
        free(mesh_reg); free(input); free(rs);
        return -1;
    }

    /* Initialize subsystems */
    rac_ecs_init(ecs);
    rac_scene_init(scene);
    rac_camera_registry_init(cam_reg);
    rac_light_registry_init(light_reg);
    rac_mesh_registry_init(mesh_reg);
    rac_input_init(input);

    /* Physics world */
    rac_phys_world_config wcfg = rac_phys_world_default_config();
    rac_phys_world *phys = rac_phys_world_create(&wcfg);

    /* Framebuffer */
    rac_framebuffer *fb = rac_framebuffer_create(
        engine->config.window_width, engine->config.window_height);

    /* Audio */
    rac_audio_engine *audio = rac_audio_create();

    /* Store in engine */
    engine->ecs = ecs;
    engine->scene = scene;
    engine->physics = phys;
    engine->framebuffer = fb;
    engine->camera_reg = cam_reg;
    engine->light_reg = light_reg;
    engine->mesh_reg = mesh_reg;
    engine->audio = audio;
    engine->input = input;
    engine->render_state = rs;

    /* Initialize render state (camera created later by user) */
    engine->running = 1;
    g_engine_quit = 0;

    /* Call user init */
    if (engine->on_init)
        engine->on_init(engine);

    /* Now set up render state with active camera */
    if (cam_reg->active_camera >= 0) {
        rac_render_init(rs, fb, &cam_reg->cameras[cam_reg->active_camera], light_reg);
    }

    printf("[RAC Engine] Initialized %dx%d (%s)\n",
           engine->config.window_width, engine->config.window_height,
           engine->config.headless ? "headless" : "windowed");
    return 0;
}

/* ── Main loop ─────────────────────────────────────────────────────────── */

static void engine_frame(rac_engine *engine, float dt)
{
    rac_ecs_world *ecs = (rac_ecs_world *)engine->ecs;
    rac_scene_graph *scene = (rac_scene_graph *)engine->scene;
    rac_phys_world *phys = (rac_phys_world *)engine->physics;
    rac_input_system *input = (rac_input_system *)engine->input;
    rac_audio_engine *audio = (rac_audio_engine *)engine->audio;
    rac_camera_registry *cam_reg = (rac_camera_registry *)engine->camera_reg;
    rac_render_state *rs = (rac_render_state *)engine->render_state;
    rac_framebuffer *fb = (rac_framebuffer *)engine->framebuffer;

    /* 1. Input */
    rac_input_poll(input);

    /* 2. User update (game logic) */
    if (engine->on_update)
        engine->on_update(engine, dt);

    /* 3. Physics (fixed timestep with accumulator) */
    engine->timing.physics_time += dt;
    int steps = 0;
    while (engine->timing.physics_time >= engine->config.physics_dt &&
           steps < engine->config.max_substeps) {
        rac_phys_world_step(phys, engine->config.physics_dt);
        engine->timing.physics_time -= engine->config.physics_dt;
        steps++;
    }

    /* 4. Sync physics bodies to ECS transforms */
    uint32_t rb_entities[RAC_ECS_MAX_ENTITIES];
    int rb_count = rac_ecs_query(ecs, RAC_COMP_TRANSFORM | RAC_COMP_RIGIDBODY,
                                 rb_entities, RAC_ECS_MAX_ENTITIES);
    for (int i = 0; i < rb_count; i++) {
        uint32_t e = rb_entities[i];
        rac_rigidbody_component *rbc = &ecs->rigidbodies[e];
        if (rbc->sync_to_transform && rbc->physics_body_index >= 0) {
            rac_phys_rigid_body *body = rac_phys_world_get_body(phys, rbc->physics_body_index);
            if (body) {
                ecs->transforms[e].position = body->position;
                ecs->transforms[e].rotation = body->orientation;
                rac_scene_mark_dirty(scene, e);
            }
        }
    }

    /* 5. Update scene graph transforms */
    rac_scene_update_transforms(scene, ecs);

    /* 6. Update camera */
    if (cam_reg->active_camera >= 0)
        rac_camera_update(&cam_reg->cameras[cam_reg->active_camera]);

    /* 7. Audio spatial update */
    if (cam_reg->active_camera >= 0) {
        rac_camera *cam = &cam_reg->cameras[cam_reg->active_camera];
        rac_audio_set_listener(audio, cam->position, cam->orientation);
    }
    rac_audio_update_spatial(audio);
    rac_audio_mix(audio);

    /* 8. Render */
    double render_start = rac_engine_get_time();
    rac_framebuffer_clear(fb, 20, 20, 30);
    rs->triangles_submitted = 0;
    rs->triangles_drawn = 0;
    rs->pixels_drawn = 0;

    if (engine->on_render)
        engine->on_render(engine);

    engine->timing.render_time = rac_engine_get_time() - render_start;

    /* 9. Input state transition */
    rac_input_update(input);
}

void rac_engine_run(rac_engine *engine)
{
    double last_time = rac_engine_get_time();
    engine->timing.total_time = 0.0;
    engine->timing.fps_accumulator = 0.0f;

    while (engine->running && !g_engine_quit) {
        double now = rac_engine_get_time();
        float dt = (float)(now - last_time);
        last_time = now;

        /* Clamp dt to avoid spiral of death */
        if (dt > 0.25f) dt = 0.25f;

        engine->timing.frame_time = dt;
        engine->timing.total_time += dt;
        engine->timing.frame_count++;

        engine_frame(engine, dt);

        /* FPS tracking */
        engine->timing.fps_accumulator += dt;
        engine->timing.fps_frame_count++;
        if (engine->timing.fps_accumulator >= 1.0f) {
            engine->timing.fps = (float)engine->timing.fps_frame_count /
                                 engine->timing.fps_accumulator;
            engine->timing.fps_accumulator = 0.0f;
            engine->timing.fps_frame_count = 0;
        }

        /* PPM frame output */
        if (engine->frame_output) {
            char path[128];
            snprintf(path, sizeof(path), "frame_%04d.ppm", engine->timing.frame_count);
            rac_framebuffer_write_ppm((rac_framebuffer *)engine->framebuffer, path);
        }
    }
}

void rac_engine_run_frames(rac_engine *engine, int num_frames)
{
    float dt = engine->config.physics_dt;
    for (int i = 0; i < num_frames && engine->running && !g_engine_quit; i++) {
        engine->timing.frame_time = dt;
        engine->timing.total_time += dt;
        engine->timing.frame_count++;
        engine_frame(engine, dt);
    }
}

void rac_engine_shutdown(rac_engine *engine)
{
    if (engine->on_cleanup)
        engine->on_cleanup(engine);

    rac_input_shutdown((rac_input_system *)engine->input);
    rac_audio_destroy((rac_audio_engine *)engine->audio);
    rac_framebuffer_destroy((rac_framebuffer *)engine->framebuffer);
    rac_phys_world_destroy((rac_phys_world *)engine->physics);
    rac_mesh_registry_cleanup((rac_mesh_registry *)engine->mesh_reg);

    free(engine->ecs);
    free(engine->scene);
    free(engine->camera_reg);
    free(engine->light_reg);
    free(engine->mesh_reg);
    free(engine->input);
    free(engine->render_state);
    free(engine);

    printf("[RAC Engine] Shutdown complete\n");
}

void rac_engine_quit(rac_engine *engine)
{
    engine->running = 0;
}
