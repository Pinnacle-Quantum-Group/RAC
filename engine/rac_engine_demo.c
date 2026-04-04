/*
 * rac_engine_demo.c — RAC Engine 4K Demo Scene
 * Pinnacle Quantum Group — April 2026
 *
 * High-resolution showcase: 3840x2160 software-rendered scene with
 * animated boxes, spheres, honey badger sprite character, Gouraud
 * shading, and spatial audio — all via RAC CORDIC primitives.
 */

#include "rac_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Demo state ────────────────────────────────────────────────────────── */

typedef enum {
    BADGER_IDLE,
    BADGER_WALK,
    BADGER_SHOOT,
} badger_state;

typedef struct {
    int cube_mesh, sphere_mesh, plane_mesh, cylinder_mesh;

    uint32_t box_entities[25];
    uint32_t sphere_entities[8];
    uint32_t pillar_entities[4];
    int num_boxes, num_spheres, num_pillars;

    rac_sprite_registry sprites;
    int badger_sprite_id;
    int badger_sheet_id;

    /* Badger behavior */
    badger_state state;
    float state_timer;        /* time in current state */
    float patrol_x;           /* current x position */
    float patrol_dir;         /* +1 or -1 */
    float shoot_cooldown;     /* time until next shoot */
    int muzzle_flash_timer;   /* frames of flash remaining */

    int camera_id;
    int ambient_clip, impact_clip, ambient_source;

    long total_pixels, total_tris_drawn;
} demo_state;

/* ── Init ──────────────────────────────────────────────────────────────── */

static void demo_init(rac_engine *engine)
{
    demo_state *demo = (demo_state *)calloc(1, sizeof(demo_state));
    engine->user_data = demo;
    demo->badger_sprite_id = -1;
    demo->badger_sheet_id = -1;

    rac_ecs_world *ecs = (rac_ecs_world *)engine->ecs;
    rac_camera_registry *cam_reg = (rac_camera_registry *)engine->camera_reg;
    rac_light_registry *lights = (rac_light_registry *)engine->light_reg;
    rac_mesh_registry *meshes = (rac_mesh_registry *)engine->mesh_reg;
    rac_audio_engine *audio = (rac_audio_engine *)engine->audio;

    /* ── Meshes ────────────────────────────────────────────────────── */
    demo->cube_mesh = rac_mesh_gen_cube(meshes, 1.0f);
    demo->sphere_mesh = rac_mesh_gen_sphere(meshes, 0.5f, 16, 24);
    demo->plane_mesh = rac_mesh_gen_plane(meshes, 30.0f, 30.0f, 8);
    demo->cylinder_mesh = rac_mesh_gen_cylinder(meshes, 0.3f, 3.0f, 16);

    /* ── Camera: cinematic angle ───────────────────────────────────── */
    demo->camera_id = rac_camera_create(cam_reg);
    rac_camera *cam = &cam_reg->cameras[demo->camera_id];
    rac_camera_set_perspective(cam, RAC_PI / 4.0f,  /* 45 deg FOV for cinematic look */
        (float)engine->config.window_width / (float)engine->config.window_height,
        0.1f, 200.0f);
    cam->position = rac_phys_v3(0.0f, 5.0f, 12.0f);
    cam->yaw = 0.0f;
    cam->pitch = -0.38f;
    rac_camera_fps_look(cam, 0.0f, 0.0f);

    uint32_t cam_ent = rac_ecs_create_entity(ecs);
    rac_ecs_add_component(ecs, cam_ent, RAC_COMP_TRANSFORM | RAC_COMP_CAMERA);
    ecs->cameras[cam_ent].camera_id = demo->camera_id;
    ecs->cameras[cam_ent].active = 1;

    /* ── Lighting: dramatic 3-point setup ──────────────────────────── */
    rac_light_set_ambient(lights, 0.18f, 0.18f, 0.25f);

    /* Key light: warm sun from upper-right */
    rac_light_create_directional(lights,
        rac_phys_v3(-0.4f, -0.9f, -0.3f), 1.0f, 0.95f, 0.85f, 0.9f);

    /* Fill light: cool blue from left */
    rac_light_create_point(lights,
        rac_phys_v3(-6.0f, 6.0f, 4.0f), 0.4f, 0.6f, 1.0f, 0.7f, 25.0f);

    /* Rim light: warm accent from behind */
    rac_light_create_point(lights,
        rac_phys_v3(4.0f, 8.0f, -6.0f), 1.0f, 0.85f, 0.6f, 0.8f, 30.0f);

    /* Ground fill: subtle uplight */
    rac_light_create_point(lights,
        rac_phys_v3(0.0f, 0.5f, 3.0f), 0.8f, 0.8f, 0.9f, 0.4f, 15.0f);

    /* ── Floor ─────────────────────────────────────────────────────── */
    {
        uint32_t e = rac_ecs_create_entity(ecs);
        rac_ecs_add_component(ecs, e, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);
        ecs->transforms[e].position = rac_phys_v3_zero();
        ecs->mesh_renderers[e].mesh_id = demo->plane_mesh;
        ecs->mesh_renderers[e].visible = 1;
        ecs->mesh_renderers[e].color_r = 180;
        ecs->mesh_renderers[e].color_g = 190;
        ecs->mesh_renderers[e].color_b = 210;
    }

    /* ── Colored box grid: 5x5 arrangement ─────────────────────────── */
    {
        uint8_t colors[][3] = {
            {230, 60,  60 },  /* red */
            {60,  200, 60 },  /* green */
            {60,  80,  230},  /* blue */
            {240, 200, 40 },  /* yellow */
            {220, 80,  180},  /* magenta */
            {40,  200, 200},  /* cyan */
            {230, 140, 40 },  /* orange */
            {160, 60,  200},  /* purple */
            {100, 200, 80 },  /* lime */
            {200, 100, 80 },  /* terracotta */
            {80,  160, 200},  /* sky blue */
            {200, 180, 100},  /* khaki */
            {180, 60,  60 },  /* dark red */
            {60,  180, 120},  /* teal */
            {200, 160, 220},  /* lavender */
            {160, 120, 60 },  /* brown */
            {100, 100, 200},  /* indigo */
            {220, 220, 80 },  /* chartreuse */
            {200, 80,  120},  /* rose */
            {80,  140, 80 },  /* forest */
            {240, 180, 160},  /* salmon */
            {120, 80,  160},  /* plum */
            {180, 220, 180},  /* mint */
            {200, 140, 100},  /* tan */
            {140, 180, 220},  /* powder blue */
        };

        demo->num_boxes = 0;
        for (int row = 0; row < 5; row++) {
            for (int col = 0; col < 5; col++) {
                int idx = demo->num_boxes;
                if (idx >= 25) break;

                uint32_t e = rac_ecs_create_entity(ecs);
                rac_ecs_add_component(ecs, e, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);

                float x = (float)(col - 2) * 2.5f;
                float y = 0.6f + (float)row * 1.4f;
                float z = (float)(row - 2) * 0.8f;  /* spread in Z too */
                ecs->transforms[e].position = rac_phys_v3(x, y, z);
                ecs->mesh_renderers[e].mesh_id = demo->cube_mesh;
                ecs->mesh_renderers[e].visible = 1;
                ecs->mesh_renderers[e].color_r = colors[idx][0];
                ecs->mesh_renderers[e].color_g = colors[idx][1];
                ecs->mesh_renderers[e].color_b = colors[idx][2];

                demo->box_entities[idx] = e;
                demo->num_boxes++;
            }
        }
    }

    /* ── Spheres: scattered around scene ────────────────────────────── */
    {
        struct { float x, y, z; uint8_t r, g, b; } sphere_defs[] = {
            { 6.0f, 0.6f,  2.0f, 255, 200, 80 },
            {-6.0f, 0.6f,  1.0f, 80,  180, 255},
            { 5.0f, 0.6f, -3.0f, 255, 120, 120},
            {-5.0f, 0.6f, -2.0f, 120, 255, 160},
            { 7.0f, 1.5f,  0.0f, 255, 160, 40 },
            {-7.0f, 1.5f,  0.0f, 40,  200, 255},
            { 3.0f, 0.4f,  4.0f, 200, 200, 200},
            {-3.0f, 0.4f,  4.0f, 255, 100, 200},
        };
        demo->num_spheres = 8;
        for (int i = 0; i < demo->num_spheres; i++) {
            uint32_t e = rac_ecs_create_entity(ecs);
            rac_ecs_add_component(ecs, e, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);
            ecs->transforms[e].position = rac_phys_v3(
                sphere_defs[i].x, sphere_defs[i].y, sphere_defs[i].z);
            ecs->mesh_renderers[e].mesh_id = demo->sphere_mesh;
            ecs->mesh_renderers[e].visible = 1;
            ecs->mesh_renderers[e].color_r = sphere_defs[i].r;
            ecs->mesh_renderers[e].color_g = sphere_defs[i].g;
            ecs->mesh_renderers[e].color_b = sphere_defs[i].b;
            demo->sphere_entities[i] = e;
        }
    }

    /* ── Pillars: vertical cylinders ───────────────────────────────── */
    {
        float pillar_x[] = { -8.0f, 8.0f, -8.0f, 8.0f };
        float pillar_z[] = { -4.0f, -4.0f, 4.0f, 4.0f };
        demo->num_pillars = 4;
        for (int i = 0; i < 4; i++) {
            uint32_t e = rac_ecs_create_entity(ecs);
            rac_ecs_add_component(ecs, e, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);
            ecs->transforms[e].position = rac_phys_v3(pillar_x[i], 1.5f, pillar_z[i]);
            ecs->mesh_renderers[e].mesh_id = demo->cylinder_mesh;
            ecs->mesh_renderers[e].visible = 1;
            ecs->mesh_renderers[e].color_r = 180;
            ecs->mesh_renderers[e].color_g = 175;
            ecs->mesh_renderers[e].color_b = 165;
            demo->pillar_entities[i] = e;
        }
    }

    /* ── Honey Badger Sprite ───────────────────────────────────────── */
    rac_sprite_registry_init(&demo->sprites);
    demo->badger_sheet_id = rac_sprite_load_sheet(&demo->sprites,
        "assets/character_sheet.raw");
    if (demo->badger_sheet_id < 0)
        demo->badger_sheet_id = rac_sprite_load_sheet(&demo->sprites,
            "../assets/character_sheet.raw");

    if (demo->badger_sheet_id >= 0) {
        demo->badger_sprite_id = rac_sprite_create(&demo->sprites,
            demo->badger_sheet_id, rac_phys_v3_zero(), 2.0f);
        if (demo->badger_sprite_id >= 0) {
            /* anim 0 = idle, anim 1 = walk, anim 2 = shoot */
            rac_sprite_add_anim(&demo->sprites,
                demo->badger_sprite_id, 0, 4, 8.0f, 1);   /* idle */
            rac_sprite_add_anim(&demo->sprites,
                demo->badger_sprite_id, 4, 4, 10.0f, 1);   /* walk */
            rac_sprite_add_anim(&demo->sprites,
                demo->badger_sprite_id, 8, 4, 12.0f, 0);   /* shoot */
            rac_sprite_play_anim(&demo->sprites, demo->badger_sprite_id, 1); /* start walking */
            printf("[Demo] Badger sprite: sheet=%d, %dx%d, 12 frames\n",
                   demo->badger_sheet_id,
                   demo->sprites.sheets[demo->badger_sheet_id].frame_w,
                   demo->sprites.sheets[demo->badger_sheet_id].frame_h);
        }
    }

    /* Badger behavior init */
    demo->state = BADGER_WALK;
    demo->state_timer = 0.0f;
    demo->patrol_x = 0.0f;
    demo->patrol_dir = 1.0f;
    demo->shoot_cooldown = 0.8f;  /* first shot quickly */
    demo->muzzle_flash_timer = 0;

    /* ── Audio ─────────────────────────────────────────────────────── */
    demo->ambient_clip = rac_audio_gen_sine(audio, 220.0f, 2.0f, 0.1f);
    demo->impact_clip = rac_audio_gen_noise(audio, 0.2f, 0.5f);
    if (demo->ambient_clip >= 0) {
        demo->ambient_source = rac_audio_create_source(audio,
            demo->ambient_clip, rac_phys_v3(0.0f, 3.0f, 0.0f));
        if (demo->ambient_source >= 0) {
            audio->sources[demo->ambient_source].looping = 1;
            audio->sources[demo->ambient_source].volume = 0.3f;
            rac_audio_play(audio, demo->ambient_source);
        }
    }

    printf("[Demo] 4K scene: %d boxes, %d spheres, %d pillars, %dx%d\n",
           demo->num_boxes, demo->num_spheres, demo->num_pillars,
           engine->config.window_width, engine->config.window_height);
}

/* ── Update ────────────────────────────────────────────────────────────── */

static void demo_update(rac_engine *engine, float dt)
{
    demo_state *demo = (demo_state *)engine->user_data;
    rac_camera_registry *cam_reg = (rac_camera_registry *)engine->camera_reg;
    rac_camera *cam = &cam_reg->cameras[demo->camera_id];
    rac_ecs_world *ecs = (rac_ecs_world *)engine->ecs;
    rac_scene_graph *sg = (rac_scene_graph *)engine->scene;

    /* Fixed camera — angled to see floor + boxes */
    cam->position = rac_phys_v3(0.0f, 6.0f, 14.0f);
    cam->yaw = 0.0f;
    cam->pitch = -0.40f;
    rac_camera_fps_look(cam, 0.0f, 0.0f);

    /* ── Animate boxes: wave + spin ────────────────────────────────── */
    float t = (float)engine->timing.total_time;
    for (int i = 0; i < demo->num_boxes; i++) {
        uint32_t e = demo->box_entities[i];
        int row = i / 5, col = i % 5;

        float base_x = (float)(col - 2) * 2.5f;
        float base_y = 0.6f + (float)row * 1.4f;

        /* Sine wave bob via rac_rotate */
        float phase = t * 2.5f + (float)col * 0.6f + (float)row * 0.9f;
        rac_vec2 wave = rac_rotate((rac_vec2){1.0f, 0.0f}, phase);

        float slide_phase = t * 0.7f + (float)row * 1.5f;
        rac_vec2 slide = rac_rotate((rac_vec2){1.0f, 0.0f}, slide_phase);

        rac_phys_vec3 pos = rac_phys_v3(
            base_x + slide.y * 0.25f,
            base_y + wave.y * 0.35f,
            0.0f);
        ecs->transforms[e].position = pos;

        /* Y-axis rotation via rac_rotate for sin/cos */
        float angle = t * (0.6f + (float)i * 0.15f);
        rac_vec2 sc = rac_rotate((rac_vec2){1.0f, 0.0f}, angle);
        rac_mat4 m;
        memset(&m, 0, sizeof(m));
        m.m[0][0] = sc.x;  m.m[0][2] = sc.y;  m.m[0][3] = pos.x;
        m.m[1][1] = 1.0f;                       m.m[1][3] = pos.y;
        m.m[2][0] = -sc.y; m.m[2][2] = sc.x;  m.m[2][3] = pos.z;
        m.m[3][3] = 1.0f;
        sg->world_matrix[e] = m;
    }

    /* ── Animate spheres: gentle floating ──────────────────────────── */
    for (int i = 0; i < demo->num_spheres; i++) {
        uint32_t e = demo->sphere_entities[i];
        rac_phys_vec3 base = ecs->transforms[e].position;
        float bob_phase = t * 1.5f + (float)i * 1.0f;
        rac_vec2 bob = rac_rotate((rac_vec2){1.0f, 0.0f}, bob_phase);
        float orig_y = 0.6f + ((i >= 4) ? 0.9f : 0.0f);
        rac_phys_vec3 pos = rac_phys_v3(base.x, orig_y + bob.y * 0.3f, base.z);

        rac_mat4 m = rac_mat4_identity();
        m.m[0][3] = pos.x; m.m[1][3] = pos.y; m.m[2][3] = pos.z;
        sg->world_matrix[e] = m;
    }

    /* ── Badger behavior: walk → stop → shoot → walk ─────────────── */
    if (demo->badger_sprite_id >= 0) {
        demo->state_timer += dt;
        demo->shoot_cooldown -= dt;
        float walk_speed = (float)engine->config.window_width * 0.08f;  /* px/sec scaled */

        switch (demo->state) {
        case BADGER_WALK:
            /* Walk in patrol_dir */
            demo->patrol_x += demo->patrol_dir * walk_speed * dt;

            /* Turn around at edges */
            if (demo->patrol_x > 0.35f) {
                demo->patrol_x = 0.35f;
                demo->patrol_dir = -1.0f;
            } else if (demo->patrol_x < -0.35f) {
                demo->patrol_x = -0.35f;
                demo->patrol_dir = 1.0f;
            }

            /* Periodically stop to shoot */
            if (demo->shoot_cooldown <= 0.0f) {
                demo->state = BADGER_SHOOT;
                demo->state_timer = 0.0f;
                demo->muzzle_flash_timer = 8;  /* 8 frames of flash */
                rac_sprite_play_anim(&demo->sprites, demo->badger_sprite_id, 2);
                demo->shoot_cooldown = 1.2f + (float)((engine->timing.frame_count * 7) % 10) * 0.1f;
            }

            /* Ensure walk anim is playing */
            if (demo->sprites.instances[demo->badger_sprite_id].current_anim != 1) {
                rac_sprite_play_anim(&demo->sprites, demo->badger_sprite_id, 1);
            }
            break;

        case BADGER_SHOOT:
            /* Hold shoot pose for 0.5 seconds */
            if (demo->muzzle_flash_timer > 0)
                demo->muzzle_flash_timer--;

            if (demo->state_timer > 0.5f) {
                demo->state = BADGER_IDLE;
                demo->state_timer = 0.0f;
                rac_sprite_play_anim(&demo->sprites, demo->badger_sprite_id, 0);
            }
            break;

        case BADGER_IDLE:
            /* Brief idle pause after shooting */
            if (demo->state_timer > 0.6f) {
                demo->state = BADGER_WALK;
                demo->state_timer = 0.0f;
                rac_sprite_play_anim(&demo->sprites, demo->badger_sprite_id, 1);
            }
            break;
        }

        /* Set flip based on direction */
        demo->sprites.instances[demo->badger_sprite_id].flip_x =
            (demo->patrol_dir < 0.0f) ? 1 : 0;
    }

    rac_sprite_update(&demo->sprites, dt);
}

/* ── Render ────────────────────────────────────────────────────────────── */

static void demo_render(rac_engine *engine)
{
    demo_state *demo = (demo_state *)engine->user_data;
    rac_ecs_world *ecs = (rac_ecs_world *)engine->ecs;
    rac_scene_graph *scene = (rac_scene_graph *)engine->scene;
    rac_render_state *rs = (rac_render_state *)engine->render_state;
    rac_mesh_registry *meshes = (rac_mesh_registry *)engine->mesh_reg;

    /* Render all mesh entities */
    uint32_t entities[RAC_ECS_MAX_ENTITIES];
    int count = rac_ecs_query(ecs, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER,
                              entities, RAC_ECS_MAX_ENTITIES);

    for (int i = 0; i < count; i++) {
        uint32_t e = entities[i];
        rac_mesh_renderer_component *mr = &ecs->mesh_renderers[e];
        if (!mr->visible || mr->mesh_id < 0 || mr->mesh_id >= meshes->num_meshes)
            continue;

        rac_mat4 world = rac_scene_get_world_matrix(scene, e);
        rac_color3f color = {
            (float)mr->color_r / 255.0f,
            (float)mr->color_g / 255.0f,
            (float)mr->color_b / 255.0f
        };
        rac_render_mesh(rs, &meshes->meshes[mr->mesh_id], world, color);
    }

    /* ── Draw ground strip: gradient bar along bottom ────────────── */
    {
        int ground_start = rs->fb->height * 78 / 100;  /* ground starts at 78% */
        int w = rs->fb->width, h = rs->fb->height;
        for (int y = ground_start; y < h; y++) {
            int depth = y - ground_start;
            int range = h - ground_start;
            int t256 = (depth * 256) / range;
            /* Dark ground: charcoal → darker */
            uint8_t gr = (uint8_t)((45 * (256 - t256) + 20 * t256) >> 8);
            uint8_t gg = (uint8_t)((50 * (256 - t256) + 22 * t256) >> 8);
            uint8_t gb = (uint8_t)((55 * (256 - t256) + 25 * t256) >> 8);
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x) * 3;
                rs->fb->pixels[idx + 0] = gr;
                rs->fb->pixels[idx + 1] = gg;
                rs->fb->pixels[idx + 2] = gb;
            }
        }
        /* Horizon line: subtle bright edge */
        for (int x = 0; x < w; x++) {
            int idx = (ground_start * w + x) * 3;
            rs->fb->pixels[idx + 0] = 70;
            rs->fb->pixels[idx + 1] = 75;
            rs->fb->pixels[idx + 2] = 85;
        }
    }

    /* ── Draw honey badger sprite ──────────────────────────────────── */
    if (demo->badger_sheet_id >= 0 && demo->badger_sprite_id >= 0) {
        rac_sprite_instance *si = &demo->sprites.instances[demo->badger_sprite_id];
        rac_sprite_sheet *sheet = &demo->sprites.sheets[demo->badger_sheet_id];

        /* Scale sprite to fill ~30% of screen height.
         * Round up so sprite is never tiny. */
        int target_h = rs->fb->height * 30 / 100;
        int draw_scale = (target_h + sheet->frame_h - 1) / sheet->frame_h;
        if (draw_scale < 2) draw_scale = 2;

        /* Position from patrol state */
        int center_x = rs->fb->width / 2;
        int walk_range = rs->fb->width / 3;
        int sprite_w = sheet->frame_w * draw_scale;
        int sprite_h = sheet->frame_h * draw_scale;
        int draw_x = center_x + (int)(demo->patrol_x * (float)walk_range) - sprite_w / 2;
        int draw_y = rs->fb->height - sprite_h - rs->fb->height / 30;

        /* Walk bob (only when walking) */
        if (demo->state == BADGER_WALK) {
            float t = (float)engine->timing.total_time;
            rac_vec2 bob = rac_rotate((rac_vec2){1.0f, 0.0f}, t * 6.0f);
            draw_y += (int)(bob.y * (float)(rs->fb->height / 120));
        }

        int flip = demo->sprites.instances[demo->badger_sprite_id].flip_x;
        rac_sprite_draw_2d(rs->fb, sheet, si->current_frame,
                           draw_x, draw_y, draw_scale, flip);

        /* ── Muzzle flash effect when shooting ─────────────────────── */
        if (demo->muzzle_flash_timer > 0) {
            int flash_intensity = demo->muzzle_flash_timer * 30;
            if (flash_intensity > 255) flash_intensity = 255;

            /* Flash position: in front of the gun */
            int flash_cx, flash_cy;
            if (flip) {
                flash_cx = draw_x - sprite_w / 8;  /* left of sprite */
            } else {
                flash_cx = draw_x + sprite_w + sprite_w / 8;  /* right */
            }
            flash_cy = draw_y + sprite_h / 3;  /* gun height */

            /* Draw concentric circles for flash (outer glow → core) */
            int flash_r = sprite_w / 2;
            for (int fy = -flash_r; fy <= flash_r; fy++) {
                for (int fx = -flash_r; fx <= flash_r; fx++) {
                    int px = flash_cx + fx, py = flash_cy + fy;
                    if (px < 0 || px >= rs->fb->width || py < 0 || py >= rs->fb->height)
                        continue;
                    int dist_sq = fx * fx + fy * fy;
                    int r_sq = flash_r * flash_r;
                    if (dist_sq > r_sq) continue;

                    /* Intensity falls off with distance */
                    int falloff = 255 - (dist_sq * 255 / r_sq);
                    int alpha = (falloff * flash_intensity) >> 8;

                    int idx = (py * rs->fb->width + px) * 3;
                    /* Additive blend: orange-yellow flash */
                    int nr = rs->fb->pixels[idx+0] + ((255 * alpha) >> 8);
                    int ng = rs->fb->pixels[idx+1] + ((180 * alpha) >> 8);
                    int nb = rs->fb->pixels[idx+2] + ((50 * alpha) >> 8);
                    rs->fb->pixels[idx+0] = (uint8_t)(nr > 255 ? 255 : nr);
                    rs->fb->pixels[idx+1] = (uint8_t)(ng > 255 ? 255 : ng);
                    rs->fb->pixels[idx+2] = (uint8_t)(nb > 255 ? 255 : nb);
                }
            }
        }
    }

    demo->total_pixels += rs->pixels_drawn;
    demo->total_tris_drawn += rs->triangles_drawn;
}

/* ── Cleanup ───────────────────────────────────────────────────────────── */

static void demo_cleanup(rac_engine *engine)
{
    demo_state *demo = (demo_state *)engine->user_data;
    rac_sprite_registry_cleanup(&demo->sprites);
    free(demo);
    printf("[Demo] Cleanup complete\n");
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    rac_engine_config cfg = rac_engine_default_config();
    cfg.window_width = 3840;
    cfg.window_height = 2160;
    cfg.headless = 1;
    cfg.max_substeps = 0;

    int num_frames = 150;  /* 5 seconds at 30fps */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            num_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--720p") == 0) {
            cfg.window_width = 1280; cfg.window_height = 720;
        } else if (strcmp(argv[i], "--1080p") == 0) {
            cfg.window_width = 1920; cfg.window_height = 1080;
        } else if (strcmp(argv[i], "--4k") == 0) {
            cfg.window_width = 3840; cfg.window_height = 2160;
        }
    }

    rac_engine *engine = rac_engine_create(&cfg);
    if (!engine) { fprintf(stderr, "Failed to create engine\n"); return 1; }

    rac_engine_set_callbacks(engine, demo_init, demo_update, demo_render, demo_cleanup);

    if (rac_engine_init(engine) != 0) {
        fprintf(stderr, "Failed to initialize engine\n"); return 1;
    }

    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--output") == 0) engine->frame_output = 1;

    printf("[Demo] Rendering %d frames at %dx%d...\n",
           num_frames, cfg.window_width, cfg.window_height);

    if (num_frames > 0) rac_engine_run_frames(engine, num_frames);
    else rac_engine_run(engine);

    rac_framebuffer *fb = (rac_framebuffer *)engine->framebuffer;
    rac_framebuffer_write_ppm(fb, "rac_demo_output.ppm");
    rac_framebuffer_write_bmp(fb, "rac_demo_output.bmp");

    rac_audio_engine *audio = (rac_audio_engine *)engine->audio;
    if (audio->buffer_size > 0)
        rac_audio_write_wav(audio, "rac_demo_audio.wav",
                            audio->output_buffer, RAC_AUDIO_BUFFER_SIZE);

    demo_state *demo = (demo_state *)engine->user_data;
    rac_render_state *rs = (rac_render_state *)engine->render_state;
    printf("[Demo] Last frame: %d tris, %d pixels | Totals: %ld tris, %ld pixels\n",
           rs->triangles_drawn, rs->pixels_drawn,
           demo->total_tris_drawn, demo->total_pixels);

    rac_engine_shutdown(engine);
    return 0;
}
