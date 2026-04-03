/*
 * rac_engine_demo.c — Playable Demo Scene
 *
 * Showcases all RAC engine systems:
 *   - Room with walls (mesh + renderer)
 *   - Stacked boxes that topple (rigid body physics, sleeping islands)
 *   - Cloth banner hanging from ceiling (cloth sim)
 *   - SPH fluid particle pool
 *   - Soft body FEM object
 *   - Point lights casting shadows
 *   - Spatial audio (ambient + impact sounds)
 *   - FPS camera with WASD + mouse
 *   - SPACE to shoot physics sphere
 *
 * Output: PPM frames (headless) or SDL2 window if available.
 */

#include "rac_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Demo state ────────────────────────────────────────────────────────── */

typedef struct {
    /* Mesh IDs */
    int cube_mesh;
    int sphere_mesh;
    int plane_mesh;

    /* Physics objects */
    int floor_body;
    int box_bodies[16];
    int num_boxes;

    /* Cloth */
    rac_phys_cloth *cloth;

    /* SPH fluid */
    rac_phys_particle_system *fluid;
    rac_phys_spatial_hash *fluid_grid;
    rac_phys_sph_config sph_cfg;

    /* Soft body */
    rac_phys_soft_body *softbody;

    /* Audio */
    int ambient_clip;
    int impact_clip;
    int ambient_source;

    /* Camera entity */
    int camera_id;
    float move_speed;
    float look_sensitivity;

    /* Projectile tracking */
    int projectile_count;
} demo_state;

/* ── Initialization ────────────────────────────────────────────────────── */

static void demo_init(rac_engine *engine)
{
    demo_state *demo = (demo_state *)calloc(1, sizeof(demo_state));
    engine->user_data = demo;
    demo->move_speed = 5.0f;
    demo->look_sensitivity = 0.003f;

    rac_ecs_world *ecs = (rac_ecs_world *)engine->ecs;
    rac_scene_graph *scene = (rac_scene_graph *)engine->scene;
    rac_phys_world *phys = (rac_phys_world *)engine->physics;
    rac_camera_registry *cam_reg = (rac_camera_registry *)engine->camera_reg;
    rac_light_registry *lights = (rac_light_registry *)engine->light_reg;
    rac_mesh_registry *meshes = (rac_mesh_registry *)engine->mesh_reg;
    rac_audio_engine *audio = (rac_audio_engine *)engine->audio;
    rac_input_system *input = (rac_input_system *)engine->input;

    /* ── Create meshes ─────────────────────────────────────────────── */
    demo->cube_mesh = rac_mesh_gen_cube(meshes, 1.0f);
    demo->sphere_mesh = rac_mesh_gen_sphere(meshes, 0.5f, 12, 16);
    demo->plane_mesh = rac_mesh_gen_plane(meshes, 20.0f, 20.0f, 4);

    /* ── Camera ────────────────────────────────────────────────────── */
    demo->camera_id = rac_camera_create(cam_reg);
    rac_camera *cam = &cam_reg->cameras[demo->camera_id];
    rac_camera_set_perspective(cam, RAC_PI / 3.0f,
        (float)engine->config.window_width / (float)engine->config.window_height,
        0.1f, 100.0f);
    cam->position = rac_phys_v3(0.0f, 3.0f, 10.0f);
    cam->yaw = 0.0f;
    cam->pitch = -0.2f;
    rac_camera_fps_look(cam, 0.0f, 0.0f);  /* rebuild orientation */

    /* Create camera entity */
    uint32_t cam_entity = rac_ecs_create_entity(ecs);
    rac_ecs_add_component(ecs, cam_entity, RAC_COMP_TRANSFORM | RAC_COMP_CAMERA);
    ecs->transforms[cam_entity].position = cam->position;
    ecs->cameras[cam_entity].camera_id = demo->camera_id;
    ecs->cameras[cam_entity].active = 1;

    /* ── Lights ────────────────────────────────────────────────────── */
    rac_light_set_ambient(lights, 0.15f, 0.15f, 0.2f);
    rac_light_create_directional(lights,
        rac_phys_v3(-0.5f, -1.0f, -0.3f), 1.0f, 0.95f, 0.9f, 0.6f);
    rac_light_create_point(lights,
        rac_phys_v3(3.0f, 5.0f, 2.0f), 1.0f, 0.8f, 0.6f, 1.0f, 15.0f);
    rac_light_create_point(lights,
        rac_phys_v3(-3.0f, 4.0f, -2.0f), 0.6f, 0.8f, 1.0f, 0.8f, 12.0f);

    /* ── Floor ─────────────────────────────────────────────────────── */
    {
        uint32_t floor_ent = rac_ecs_create_entity(ecs);
        rac_ecs_add_component(ecs, floor_ent,
            RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER | RAC_COMP_RIGIDBODY);
        ecs->transforms[floor_ent].position = rac_phys_v3(0.0f, 0.0f, 0.0f);
        ecs->mesh_renderers[floor_ent].mesh_id = demo->plane_mesh;
        ecs->mesh_renderers[floor_ent].visible = 1;
        ecs->mesh_renderers[floor_ent].color_r = 120;
        ecs->mesh_renderers[floor_ent].color_g = 130;
        ecs->mesh_renderers[floor_ent].color_b = 140;

        rac_phys_rigid_body body = rac_phys_body_create(RAC_BODY_STATIC, 0.0f);
        body.position = rac_phys_v3(0.0f, 0.0f, 0.0f);
        rac_phys_shape shape;
        shape.type = RAC_SHAPE_BOX;
        shape.box.half_extents = rac_phys_v3(10.0f, 0.1f, 10.0f);
        shape.local_offset = rac_phys_v3_zero();
        demo->floor_body = rac_phys_world_add_body(phys, body, shape);
        ecs->rigidbodies[floor_ent].physics_body_index = demo->floor_body;
        ecs->rigidbodies[floor_ent].sync_to_transform = 0;
    }

    /* ── Stacked boxes ─────────────────────────────────────────────── */
    demo->num_boxes = 0;
    for (int layer = 0; layer < 4; layer++) {
        for (int col = 0; col < (4 - layer); col++) {
            if (demo->num_boxes >= 16) break;

            float x = (float)(col - (3 - layer)) * 1.1f + 0.55f * (float)layer;
            float y = 0.6f + (float)layer * 1.1f;

            uint32_t ent = rac_ecs_create_entity(ecs);
            rac_ecs_add_component(ecs, ent,
                RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER | RAC_COMP_RIGIDBODY);

            ecs->transforms[ent].position = rac_phys_v3(x, y, 0.0f);
            ecs->mesh_renderers[ent].mesh_id = demo->cube_mesh;
            ecs->mesh_renderers[ent].visible = 1;
            ecs->mesh_renderers[ent].color_r = 200;
            ecs->mesh_renderers[ent].color_g = (uint8_t)(100 + layer * 30);
            ecs->mesh_renderers[ent].color_b = (uint8_t)(80 + col * 20);

            rac_phys_rigid_body body = rac_phys_body_create(RAC_BODY_DYNAMIC, 5.0f);
            body.position = ecs->transforms[ent].position;
            body.restitution = 0.3f;
            body.friction = 0.6f;
            rac_phys_body_set_inertia_box(&body, 0.5f, 0.5f, 0.5f);

            rac_phys_shape shape;
            shape.type = RAC_SHAPE_BOX;
            shape.box.half_extents = rac_phys_v3(0.5f, 0.5f, 0.5f);
            shape.local_offset = rac_phys_v3_zero();

            int bi = rac_phys_world_add_body(phys, body, shape);
            ecs->rigidbodies[ent].physics_body_index = bi;
            ecs->rigidbodies[ent].sync_to_transform = 1;
            demo->box_bodies[demo->num_boxes++] = bi;
        }
    }

    /* ── Cloth banner ──────────────────────────────────────────────── */
    demo->cloth = rac_phys_cloth_create_grid(12, 8, 0.15f, 0.05f);
    if (demo->cloth) {
        /* Offset cloth position to hang from ceiling */
        for (int i = 0; i < demo->cloth->particles->num_particles; i++) {
            demo->cloth->particles->positions[i].x += -3.0f;
            demo->cloth->particles->positions[i].y += 6.0f;
            demo->cloth->particles->positions[i].z += -3.0f;
        }
        /* Pin top row */
        for (int i = 0; i < 12; i++)
            rac_phys_cloth_pin(demo->cloth, i);
        demo->cloth->stretch_stiffness = 0.9f;
        demo->cloth->bend_stiffness = 0.3f;
    }

    /* ── SPH fluid pool ────────────────────────────────────────────── */
    demo->fluid = rac_phys_particles_create(512);
    demo->fluid_grid = rac_phys_spatial_hash_create(0.2f, 1024);
    demo->sph_cfg = rac_phys_sph_default_config();
    demo->sph_cfg.smoothing_radius = 0.15f;
    demo->sph_cfg.rest_density = 1000.0f;

    /* Emit fluid particles in a small volume */
    for (int z = 0; z < 8; z++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                rac_phys_vec3 pos = rac_phys_v3(
                    4.0f + x * 0.1f,
                    1.0f + y * 0.1f,
                    -2.0f + z * 0.1f
                );
                rac_phys_particles_emit(demo->fluid, pos,
                    rac_phys_v3_zero(), demo->sph_cfg.particle_mass);
            }
        }
    }

    /* ── Soft body ─────────────────────────────────────────────────── */
    demo->softbody = rac_phys_softbody_create_beam(1.5f, 0.5f, 0.5f, 3, 100.0f, 5000.0f);
    if (demo->softbody) {
        for (int i = 0; i < demo->softbody->num_vertices; i++) {
            demo->softbody->positions[i].x += -5.0f;
            demo->softbody->positions[i].y += 2.0f;
            demo->softbody->positions[i].z += 2.0f;
        }
    }

    /* ── Audio ─────────────────────────────────────────────────────── */
    demo->ambient_clip = rac_audio_gen_sine(audio, 220.0f, 2.0f, 0.1f);
    demo->impact_clip = rac_audio_gen_noise(audio, 0.2f, 0.5f);

    if (demo->ambient_clip >= 0) {
        demo->ambient_source = rac_audio_create_source(audio, demo->ambient_clip,
            rac_phys_v3(0.0f, 3.0f, 0.0f));
        if (demo->ambient_source >= 0) {
            audio->sources[demo->ambient_source].looping = 1;
            audio->sources[demo->ambient_source].volume = 0.3f;
            rac_audio_play(audio, demo->ambient_source);
        }
    }

    /* ── Input bindings ────────────────────────────────────────────── */
    rac_input_bind_action(input, "forward", RAC_KEY_W);
    rac_input_bind_action(input, "back", RAC_KEY_S);
    rac_input_bind_action(input, "left", RAC_KEY_A);
    rac_input_bind_action(input, "right", RAC_KEY_D);
    rac_input_bind_action(input, "shoot", RAC_KEY_SPACE);
    rac_input_bind_action(input, "quit", RAC_KEY_ESCAPE);

    printf("[Demo] Scene initialized: %d boxes, cloth(%s), fluid(%d particles), softbody(%s)\n",
           demo->num_boxes,
           demo->cloth ? "yes" : "no",
           demo->fluid ? demo->fluid->num_particles : 0,
           demo->softbody ? "yes" : "no");
}

/* ── Update ────────────────────────────────────────────────────────────── */

static void demo_update(rac_engine *engine, float dt)
{
    demo_state *demo = (demo_state *)engine->user_data;
    rac_camera_registry *cam_reg = (rac_camera_registry *)engine->camera_reg;
    rac_input_system *input = (rac_input_system *)engine->input;
    rac_phys_world *phys = (rac_phys_world *)engine->physics;
    rac_audio_engine *audio = (rac_audio_engine *)engine->audio;
    rac_camera *cam = &cam_reg->cameras[demo->camera_id];

    /* Camera look (mouse delta) */
    int mdx, mdy;
    rac_input_mouse_delta(input, &mdx, &mdy);
    if (mdx || mdy) {
        rac_camera_fps_look(cam,
            -(float)mdx * demo->look_sensitivity,
            -(float)mdy * demo->look_sensitivity);
    }

    /* Camera movement */
    rac_phys_vec3 forward = rac_phys_quat_rotate_vec3(cam->orientation,
        rac_phys_v3(0.0f, 0.0f, -1.0f));
    rac_phys_vec3 right = rac_phys_quat_rotate_vec3(cam->orientation,
        rac_phys_v3(1.0f, 0.0f, 0.0f));

    float speed = demo->move_speed * dt;
    if (rac_input_action_active(input, "forward"))
        cam->position = rac_phys_v3_add(cam->position, rac_phys_v3_scale(forward, speed));
    if (rac_input_action_active(input, "back"))
        cam->position = rac_phys_v3_sub(cam->position, rac_phys_v3_scale(forward, speed));
    if (rac_input_action_active(input, "right"))
        cam->position = rac_phys_v3_add(cam->position, rac_phys_v3_scale(right, speed));
    if (rac_input_action_active(input, "left"))
        cam->position = rac_phys_v3_sub(cam->position, rac_phys_v3_scale(right, speed));

    /* Shoot sphere */
    if (rac_input_key_pressed(input, RAC_KEY_SPACE) && demo->projectile_count < 10) {
        rac_phys_rigid_body proj = rac_phys_body_create(RAC_BODY_DYNAMIC, 2.0f);
        proj.position = cam->position;
        proj.linear_velocity = rac_phys_v3_scale(forward, 15.0f);
        proj.restitution = 0.5f;
        proj.friction = 0.4f;
        rac_phys_body_set_inertia_sphere(&proj, 0.25f);

        rac_phys_shape shape;
        shape.type = RAC_SHAPE_SPHERE;
        shape.sphere.radius = 0.25f;
        shape.local_offset = rac_phys_v3_zero();

        rac_phys_world_add_body(phys, proj, shape);
        demo->projectile_count++;

        /* Impact sound at launch position */
        if (demo->impact_clip >= 0) {
            int src = rac_audio_create_source(audio, demo->impact_clip, cam->position);
            if (src >= 0) rac_audio_play(audio, src);
        }
    }

    /* Quit */
    if (rac_input_key_pressed(input, RAC_KEY_ESCAPE))
        rac_engine_quit(engine);

    /* Step cloth */
    if (demo->cloth)
        rac_phys_cloth_step(demo->cloth, rac_phys_v3(0.0f, -9.81f, 0.0f), dt);

    /* Step fluid */
    if (demo->fluid && demo->fluid_grid) {
        rac_phys_sph_step(demo->fluid, demo->fluid_grid,
            rac_phys_v3(0.0f, -9.81f, 0.0f), &demo->sph_cfg, dt);
    }

    /* Step soft body */
    if (demo->softbody)
        rac_phys_softbody_step(demo->softbody, rac_phys_v3(0.0f, -9.81f, 0.0f), dt);
}

/* ── Render ────────────────────────────────────────────────────────────── */

static void demo_render(rac_engine *engine)
{
    demo_state *demo = (demo_state *)engine->user_data;
    rac_ecs_world *ecs = (rac_ecs_world *)engine->ecs;
    rac_scene_graph *scene = (rac_scene_graph *)engine->scene;
    rac_render_state *rs = (rac_render_state *)engine->render_state;
    rac_mesh_registry *meshes = (rac_mesh_registry *)engine->mesh_reg;

    /* Render all mesh renderer entities */
    uint32_t entities[RAC_ECS_MAX_ENTITIES];
    int count = rac_ecs_query(ecs, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER,
                              entities, RAC_ECS_MAX_ENTITIES);

    for (int i = 0; i < count; i++) {
        uint32_t e = entities[i];
        rac_mesh_renderer_component *mr = &ecs->mesh_renderers[e];
        if (!mr->visible) continue;
        if (mr->mesh_id < 0 || mr->mesh_id >= meshes->num_meshes) continue;

        rac_mat4 world = rac_scene_get_world_matrix(scene, e);
        rac_color3f color = {
            (float)mr->color_r / 255.0f,
            (float)mr->color_g / 255.0f,
            (float)mr->color_b / 255.0f
        };
        rac_render_mesh(rs, &meshes->meshes[mr->mesh_id], world, color);
    }

    /* Render cloth as triangles */
    if (demo->cloth && demo->cloth->particles) {
        rac_phys_particle_system *cp = demo->cloth->particles;
        /* Cloth is a grid — render as triangle strips using stretch pairs */
        /* Simple approach: render each stretch constraint pair as a line-like quad */
        rac_color3f cloth_color = { 0.9f, 0.2f, 0.2f };

        /* Approximate: render cloth particles as small points */
        rac_render_particles(rs, cp->positions, cp->num_particles,
                             0.05f, cloth_color);
    }

    /* Render SPH fluid particles */
    if (demo->fluid) {
        rac_color3f fluid_color = { 0.2f, 0.5f, 0.9f };
        rac_render_particles(rs, demo->fluid->positions,
                             demo->fluid->num_particles, 0.08f, fluid_color);
    }

    /* Render soft body surface triangles */
    if (demo->softbody && demo->softbody->surface_triangles) {
        rac_color3f sb_color = { 0.4f, 0.8f, 0.3f };
        for (int i = 0; i < demo->softbody->num_surface_tris; i++) {
            int i0 = demo->softbody->surface_triangles[i * 3 + 0];
            int i1 = demo->softbody->surface_triangles[i * 3 + 1];
            int i2 = demo->softbody->surface_triangles[i * 3 + 2];

            rac_phys_vec3 p0 = demo->softbody->positions[i0];
            rac_phys_vec3 p1 = demo->softbody->positions[i1];
            rac_phys_vec3 p2 = demo->softbody->positions[i2];

            rac_phys_vec3 e1 = rac_phys_v3_sub(p1, p0);
            rac_phys_vec3 e2 = rac_phys_v3_sub(p2, p0);
            rac_phys_vec3 fn = rac_phys_v3_normalize(rac_phys_v3_cross(e1, e2));

            rac_render_triangle_world(rs, p0, p1, p2, fn, fn, fn, sb_color);
        }
    }
}

/* ── Cleanup ───────────────────────────────────────────────────────────── */

static void demo_cleanup(rac_engine *engine)
{
    demo_state *demo = (demo_state *)engine->user_data;
    if (demo->cloth) rac_phys_cloth_destroy(demo->cloth);
    if (demo->fluid) rac_phys_particles_destroy(demo->fluid);
    if (demo->fluid_grid) rac_phys_spatial_hash_destroy(demo->fluid_grid);
    if (demo->softbody) rac_phys_softbody_destroy(demo->softbody);
    free(demo);
    printf("[Demo] Cleanup complete\n");
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    rac_engine_config cfg = rac_engine_default_config();
    cfg.window_width = 640;
    cfg.window_height = 480;
    cfg.headless = 1;

    /* Parse args */
    int num_frames = 60;  /* default: run 60 frames (1 second) */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            num_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--interactive") == 0)
            num_frames = 0;  /* run until quit */
        else if (strcmp(argv[i], "--output") == 0)
            ; /* enable frame output below */
    }

    rac_engine *engine = rac_engine_create(&cfg);
    if (!engine) {
        fprintf(stderr, "Failed to create engine\n");
        return 1;
    }

    rac_engine_set_callbacks(engine, demo_init, demo_update, demo_render, demo_cleanup);

    if (rac_engine_init(engine) != 0) {
        fprintf(stderr, "Failed to initialize engine\n");
        return 1;
    }

    /* Check for frame output */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0)
            engine->frame_output = 1;
    }

    printf("[Demo] Running %d frames...\n", num_frames);

    if (num_frames > 0) {
        rac_engine_run_frames(engine, num_frames);
    } else {
        rac_engine_run(engine);
    }

    /* Write final frame */
    rac_framebuffer *fb = (rac_framebuffer *)engine->framebuffer;
    rac_framebuffer_write_ppm(fb, "rac_demo_output.ppm");
    rac_framebuffer_write_bmp(fb, "rac_demo_output.bmp");
    printf("[Demo] Output written: rac_demo_output.ppm, rac_demo_output.bmp\n");

    /* Write audio sample */
    rac_audio_engine *audio = (rac_audio_engine *)engine->audio;
    if (audio->buffer_size > 0) {
        rac_audio_write_wav(audio, "rac_demo_audio.wav",
                            audio->output_buffer, RAC_AUDIO_BUFFER_SIZE);
        printf("[Demo] Audio written: rac_demo_audio.wav\n");
    }

    rac_render_state *rs = (rac_render_state *)engine->render_state;
    printf("[Demo] Stats: %d tris submitted, %d drawn, %d pixels\n",
           rs->triangles_submitted, rs->triangles_drawn, rs->pixels_drawn);

    rac_engine_shutdown(engine);
    return 0;
}
