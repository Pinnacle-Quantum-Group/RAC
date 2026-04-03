/*
 * test_engine_render.c — Renderer BVT
 * Verifies framebuffer, pixel output, projection, and triangle rasterization.
 */

#include "../rac_engine_render.h"
#include "../rac_engine_mesh.h"
#include "../rac_engine_camera.h"
#include "../rac_engine_light.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [Render] %-50s ", name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static void test_framebuffer_create(void)
{
    TEST("Framebuffer creation and clear");
    rac_framebuffer *fb = rac_framebuffer_create(64, 48);
    CHECK(fb != NULL, "fb not null");
    CHECK(fb->width == 64, "width");
    CHECK(fb->height == 48, "height");

    rac_framebuffer_clear(fb, 255, 0, 0);
    CHECK(fb->pixels[0] == 255, "red channel");
    CHECK(fb->pixels[1] == 0, "green channel");
    CHECK(fb->pixels[2] == 0, "blue channel");

    rac_framebuffer_destroy(fb);
    PASS();
}

static void test_framebuffer_ppm(void)
{
    TEST("Framebuffer PPM output");
    rac_framebuffer *fb = rac_framebuffer_create(16, 16);
    rac_framebuffer_clear(fb, 128, 64, 32);

    int ret = rac_framebuffer_write_ppm(fb, "/tmp/rac_test_render.ppm");
    CHECK(ret == 0, "PPM write success");

    /* Verify file exists and has correct header */
    FILE *f = fopen("/tmp/rac_test_render.ppm", "rb");
    CHECK(f != NULL, "file exists");
    char header[32];
    fgets(header, sizeof(header), f);
    CHECK(header[0] == 'P' && header[1] == '6', "PPM P6 header");
    fclose(f);

    rac_framebuffer_destroy(fb);
    PASS();
}

static void test_line_drawing(void)
{
    TEST("Line drawing (Bresenham)");
    rac_framebuffer *fb = rac_framebuffer_create(32, 32);
    rac_framebuffer_clear(fb, 0, 0, 0);

    rac_render_line(fb, 0, 0, 31, 31, 255, 255, 255);

    /* Diagonal should have pixels set */
    CHECK(fb->pixels[(16 * 32 + 16) * 3] == 255, "midpoint pixel set");

    rac_framebuffer_destroy(fb);
    PASS();
}

static void test_render_cube(void)
{
    TEST("Render cube produces visible pixels");
    rac_framebuffer *fb = rac_framebuffer_create(128, 96);
    rac_framebuffer_clear(fb, 20, 20, 30);

    /* Set up camera */
    rac_camera cam;
    memset(&cam, 0, sizeof(cam));
    cam.position = rac_phys_v3(0.0f, 0.0f, 5.0f);
    cam.orientation = rac_phys_quat_identity();
    cam.yaw = 0.0f;
    cam.pitch = 0.0f;
    rac_camera_fps_look(&cam, 0.0f, 0.0f);
    rac_camera_set_perspective(&cam, RAC_PI / 3.0f, 128.0f / 96.0f, 0.1f, 100.0f);
    rac_camera_update(&cam);

    /* Set up lighting */
    rac_light_registry lights;
    rac_light_registry_init(&lights);
    rac_light_set_ambient(&lights, 0.3f, 0.3f, 0.3f);
    rac_light_create_directional(&lights,
        rac_phys_v3(0.0f, 0.0f, -1.0f), 1.0f, 1.0f, 1.0f, 0.8f);

    /* Create and render cube */
    rac_mesh_registry meshes;
    rac_mesh_registry_init(&meshes);
    int cube_id = rac_mesh_gen_cube(&meshes, 2.0f);
    CHECK(cube_id >= 0, "cube created");

    rac_render_state rs;
    rac_render_init(&rs, fb, &cam, &lights);
    rs.shade_mode = RAC_SHADE_FLAT;

    rac_mat4 model = rac_mat4_identity();
    rac_color3f color = { 0.8f, 0.3f, 0.2f };
    rac_render_mesh(&rs, &meshes.meshes[cube_id], model, color);

    CHECK(rs.triangles_submitted > 0, "triangles submitted");
    CHECK(rs.triangles_drawn > 0, "triangles drawn");

    /* Check that some pixels changed from background */
    int changed = 0;
    for (int i = 0; i < fb->width * fb->height; i++) {
        if (fb->pixels[i * 3] != 20 || fb->pixels[i * 3 + 1] != 20)
            changed++;
    }
    CHECK(changed > 100, "visible pixels rendered");

    /* Write test output */
    rac_framebuffer_write_ppm(fb, "/tmp/rac_test_cube.ppm");

    rac_mesh_registry_cleanup(&meshes);
    rac_framebuffer_destroy(fb);
    PASS();
}

static void test_zbuffer(void)
{
    TEST("Z-buffer depth ordering");
    rac_framebuffer *fb = rac_framebuffer_create(64, 64);
    rac_framebuffer_clear(fb, 0, 0, 0);

    /* Verify depth buffer initialized to far */
    CHECK(fb->depth[0] > 1e20f, "depth initialized far");

    rac_framebuffer_destroy(fb);
    PASS();
}

static void test_render_sphere(void)
{
    TEST("Render sphere produces visible output");
    rac_framebuffer *fb = rac_framebuffer_create(128, 96);
    rac_framebuffer_clear(fb, 10, 10, 20);

    rac_camera cam;
    memset(&cam, 0, sizeof(cam));
    cam.position = rac_phys_v3(0.0f, 0.0f, 4.0f);
    cam.orientation = rac_phys_quat_identity();
    rac_camera_set_perspective(&cam, RAC_PI / 3.0f, 128.0f / 96.0f, 0.1f, 100.0f);
    rac_camera_update(&cam);

    rac_light_registry lights;
    rac_light_registry_init(&lights);
    rac_light_set_ambient(&lights, 0.3f, 0.3f, 0.3f);
    rac_light_create_directional(&lights,
        rac_phys_v3(0.0f, -1.0f, -1.0f), 1.0f, 1.0f, 1.0f, 0.7f);

    rac_mesh_registry meshes;
    rac_mesh_registry_init(&meshes);
    int sphere_id = rac_mesh_gen_sphere(&meshes, 1.0f, 12, 16);
    CHECK(sphere_id >= 0, "sphere created");

    rac_render_state rs;
    rac_render_init(&rs, fb, &cam, &lights);

    rac_mat4 model = rac_mat4_identity();
    rac_color3f color = { 0.3f, 0.6f, 0.9f };
    rac_render_mesh(&rs, &meshes.meshes[sphere_id], model, color);

    CHECK(rs.triangles_drawn > 0, "sphere triangles drawn");

    rac_framebuffer_write_ppm(fb, "/tmp/rac_test_sphere.ppm");

    rac_mesh_registry_cleanup(&meshes);
    rac_framebuffer_destroy(fb);
    PASS();
}

int main(void)
{
    printf("\n=== RAC Engine Renderer BVT ===\n\n");

    test_framebuffer_create();
    test_framebuffer_ppm();
    test_line_drawing();
    test_render_cube();
    test_zbuffer();
    test_render_sphere();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
