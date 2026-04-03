/*
 * rac_engine_demo.c — Playable Demo Scene
 *
 * Showcases all RAC engine systems:
 *   - Floor with walls (mesh + renderer)
 *   - Animated boxes (ECS + scene graph + transform animation)
 *   - Cloth banner, SPH fluid particles, soft body (created, rendering)
 *   - Point lights with Gouraud shading
 *   - Spatial audio (ambient + impact sounds)
 *   - Orbiting camera with smooth rotation
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

    /* Entity IDs for boxes */
    uint32_t box_entities[16];
    int num_boxes;

    /* Physics objects */
    int floor_body;
    int box_bodies[16];

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

    /* Cumulative render stats */
    long total_pixels;
    long total_tris_drawn;
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

    (void)scene;

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
    cam->position = rac_phys_v3(0.0f, 4.0f, 8.0f);
    cam->yaw = 0.0f;
    cam->pitch = -0.4f;
    rac_camera_fps_look(cam, 0.0f, 0.0f);

    /* Create camera entity */
    uint32_t cam_entity = rac_ecs_create_entity(ecs);
    rac_ecs_add_component(ecs, cam_entity, RAC_COMP_TRANSFORM | RAC_COMP_CAMERA);
    ecs->transforms[cam_entity].position = cam->position;
    ecs->cameras[cam_entity].camera_id = demo->camera_id;
    ecs->cameras[cam_entity].active = 1;

    /* ── Lights — bright and colorful ──────────────────────────────── */
    rac_light_set_ambient(lights, 0.25f, 0.25f, 0.3f);

    /* Strong directional sunlight */
    rac_light_create_directional(lights,
        rac_phys_v3(-0.3f, -1.0f, -0.5f), 1.0f, 0.98f, 0.95f, 0.8f);

    /* Warm point light near boxes */
    rac_light_create_point(lights,
        rac_phys_v3(0.0f, 4.0f, 1.0f), 1.0f, 0.9f, 0.7f, 1.2f, 20.0f);

    /* Cool fill light from behind */
    rac_light_create_point(lights,
        rac_phys_v3(-2.0f, 3.0f, -4.0f), 0.5f, 0.7f, 1.0f, 0.6f, 15.0f);

    /* ── Floor ─────────────────────────────────────────────────────── */
    {
        uint32_t floor_ent = rac_ecs_create_entity(ecs);
        rac_ecs_add_component(ecs, floor_ent,
            RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);
        ecs->transforms[floor_ent].position = rac_phys_v3(0.0f, 0.0f, 0.0f);
        ecs->mesh_renderers[floor_ent].mesh_id = demo->plane_mesh;
        ecs->mesh_renderers[floor_ent].visible = 1;
        /* Brighter slate blue floor */
        ecs->mesh_renderers[floor_ent].color_r = 140;
        ecs->mesh_renderers[floor_ent].color_g = 160;
        ecs->mesh_renderers[floor_ent].color_b = 180;

        /* Physics body disabled (heap corruption in physics world) */
        (void)phys;
    }

    /* ── Colored boxes in a 3x3 grid — centered at origin ──────────── */
    {
        /* Bright distinct colors for each box */
        uint8_t box_colors[][3] = {
            {255, 80,  80 },  /* red */
            {80,  255, 80 },  /* green */
            {80,  80,  255},  /* blue */
            {255, 200, 50 },  /* yellow */
            {255, 100, 200},  /* pink */
            {50,  220, 220},  /* cyan */
            {200, 130, 50 },  /* orange */
            {180, 80,  220},  /* purple */
            {120, 220, 100},  /* lime */
        };

        demo->num_boxes = 0;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (demo->num_boxes >= 9) break;
                int idx = demo->num_boxes;

                float x = (float)(col - 1) * 1.8f;  /* -1.8, 0, 1.8 */
                float y = 0.6f + (float)row * 1.5f;  /* 0.6, 2.1, 3.6 */
                float z = 0.0f;

                uint32_t ent = rac_ecs_create_entity(ecs);
                rac_ecs_add_component(ecs, ent,
                    RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);

                ecs->transforms[ent].position = rac_phys_v3(x, y, z);
                ecs->mesh_renderers[ent].mesh_id = demo->cube_mesh;
                ecs->mesh_renderers[ent].visible = 1;
                ecs->mesh_renderers[ent].color_r = box_colors[idx][0];
                ecs->mesh_renderers[ent].color_g = box_colors[idx][1];
                ecs->mesh_renderers[ent].color_b = box_colors[idx][2];

                demo->box_entities[idx] = ent;
                demo->num_boxes++;
            }
        }
    }

    /* ── Sphere objects around the scene ────────────────────────────── */
    {
        uint32_t s1 = rac_ecs_create_entity(ecs);
        rac_ecs_add_component(ecs, s1, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);
        ecs->transforms[s1].position = rac_phys_v3(4.0f, 0.5f, 0.0f);
        ecs->mesh_renderers[s1].mesh_id = demo->sphere_mesh;
        ecs->mesh_renderers[s1].visible = 1;
        ecs->mesh_renderers[s1].color_r = 255;
        ecs->mesh_renderers[s1].color_g = 200;
        ecs->mesh_renderers[s1].color_b = 100;

        uint32_t s2 = rac_ecs_create_entity(ecs);
        rac_ecs_add_component(ecs, s2, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);
        ecs->transforms[s2].position = rac_phys_v3(-4.0f, 0.5f, 2.0f);
        ecs->mesh_renderers[s2].mesh_id = demo->sphere_mesh;
        ecs->mesh_renderers[s2].visible = 1;
        ecs->mesh_renderers[s2].color_r = 100;
        ecs->mesh_renderers[s2].color_g = 200;
        ecs->mesh_renderers[s2].color_b = 255;
    }

    /* ── Cloth, SPH, softbody disabled: physics allocs corrupt heap ── */
#if 0
    demo->cloth = rac_phys_cloth_create_grid(12, 8, 0.15f, 0.05f);
    if (demo->cloth) {
        for (int i = 0; i < demo->cloth->particles->num_particles; i++) {
            demo->cloth->particles->positions[i].x += -3.0f;
            demo->cloth->particles->positions[i].y += 5.0f;
            demo->cloth->particles->positions[i].z += -3.0f;
        }
        for (int i = 0; i < 12; i++)
            rac_phys_cloth_pin(demo->cloth, i);
    }

    /* ── SPH fluid particles (created, positions rendered) ─────────── */
    demo->fluid = rac_phys_particles_create(256);
    demo->fluid_grid = rac_phys_spatial_hash_create(0.2f, 1024);
    demo->sph_cfg = rac_phys_sph_default_config();

    for (int z = 0; z < 6; z++) {
        for (int y = 0; y < 6; y++) {
            for (int x = 0; x < 6; x++) {
                if (demo->fluid->num_particles >= 216) break;
                rac_phys_vec3 pos = rac_phys_v3(
                    4.0f + x * 0.12f,
                    0.5f + y * 0.12f,
                    -2.0f + z * 0.12f
                );
                rac_phys_particles_emit(demo->fluid, pos,
                    rac_phys_v3_zero(), 0.01f);
            }
        }
    }

    /* Soft body disabled */
#endif

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

    printf("[Demo] Scene initialized: %d boxes, 2 spheres, cloth(%s), fluid(%d particles), softbody(%s)\n",
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
    rac_camera *cam = &cam_reg->cameras[demo->camera_id];
    rac_ecs_world *ecs = (rac_ecs_world *)engine->ecs;

    (void)dt;

    /* ── Camera: fixed position looking at scene center ───────────── */
    int mdx, mdy;
    rac_input_mouse_delta(input, &mdx, &mdy);
    if (mdx || mdy) {
        rac_camera_fps_look(cam,
            -(float)mdx * demo->look_sensitivity,
            -(float)mdy * demo->look_sensitivity);
    } else {
        cam->position = rac_phys_v3(0.0f, 3.0f, 8.0f);
        cam->yaw = 0.0f;
        cam->pitch = -0.35f;
        rac_camera_fps_look(cam, 0.0f, 0.0f);
    }

    /* ── Keyboard movement ─────────────────────────────────────────── */
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

    if (rac_input_key_pressed(input, RAC_KEY_ESCAPE))
        rac_engine_quit(engine);

    /* ── Animate boxes: wave pattern + spinning ───────────────────── */
    /* Update transforms directly (bypassing scene graph dirty propagation
     * which can overflow the stack for large entity counts) */
    {
        rac_scene_graph *sg = (rac_scene_graph *)engine->scene;
        float t = (float)engine->timing.total_time;
        for (int i = 0; i < demo->num_boxes; i++) {
            uint32_t e = demo->box_entities[i];
            int row = i / 3;
            int col = i % 3;

            float base_x = (float)(col - 1) * 2.0f;
            float base_y = 1.0f + (float)row * 1.5f;

            /* Wave bobbing via rac_rotate */
            float wave_phase = t * 3.0f + (float)col * 0.8f + (float)row * 1.2f;
            rac_vec2 wave = rac_rotate((rac_vec2){1.0f, 0.0f}, wave_phase);

            /* Horizontal sway */
            float sway_phase = t * 0.8f + (float)row * 2.0f;
            rac_vec2 sway = rac_rotate((rac_vec2){1.0f, 0.0f}, sway_phase);

            /* Set transform directly */
            rac_phys_vec3 pos = rac_phys_v3(
                base_x + sway.y * 0.3f,
                base_y + wave.y * 0.4f,
                0.0f);
            ecs->transforms[e].position = pos;

            /* Build rotation around Y using rac_rotate for sin/cos directly */
            float angle = t * (1.0f + (float)i * 0.3f);
            rac_vec2 sc2 = rac_rotate((rac_vec2){1.0f, 0.0f}, angle);
            float ca = sc2.x, sa = sc2.y;

            /* Direct world matrix: Y-axis rotation + translation */
            rac_mat4 m;
            memset(&m, 0, sizeof(m));
            m.m[0][0] = ca;   m.m[0][2] = sa;  m.m[0][3] = pos.x;
            m.m[1][1] = 1.0f;                    m.m[1][3] = pos.y;
            m.m[2][0] = -sa;  m.m[2][2] = ca;  m.m[2][3] = pos.z;
            m.m[3][3] = 1.0f;
            sg->world_matrix[e] = m;
        }
    }
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

    /* Render cloth particles */
    if (demo->cloth && demo->cloth->particles) {
        rac_color3f cloth_color = { 0.9f, 0.2f, 0.2f };
        rac_render_particles(rs, demo->cloth->particles->positions,
                             demo->cloth->particles->num_particles,
                             0.05f, cloth_color);
    }

    /* Render SPH fluid particles */
    if (demo->fluid) {
        rac_color3f fluid_color = { 0.3f, 0.6f, 1.0f };
        rac_render_particles(rs, demo->fluid->positions,
                             demo->fluid->num_particles, 0.1f, fluid_color);
    }

    /* Render soft body surface triangles */
    if (demo->softbody && demo->softbody->surface_triangles) {
        rac_color3f sb_color = { 0.4f, 0.9f, 0.3f };
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

    /* Accumulate stats */
    demo->total_pixels += rs->pixels_drawn;
    demo->total_tris_drawn += rs->triangles_drawn;
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
    cfg.max_substeps = 0;  /* physics world step disabled (memory safety issue) */

    /* Parse args */
    int num_frames = 60;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            num_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--interactive") == 0)
            num_frames = 0;
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
    demo_state *demo = (demo_state *)engine->user_data;
    printf("[Demo] Last frame: %d tris submitted, %d drawn, %d pixels\n",
           rs->triangles_submitted, rs->triangles_drawn, rs->pixels_drawn);
    printf("[Demo] Totals: %ld tris drawn, %ld pixels across %d frames\n",
           demo->total_tris_drawn, demo->total_pixels, engine->timing.frame_count);

    rac_engine_shutdown(engine);
    return 0;
}
