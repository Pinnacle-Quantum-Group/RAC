/*
 * test_engine_mesh.c — Mesh System BVT
 * Verifies procedural generation, OBJ loading, AABB, normals.
 */

#include "../rac_engine_mesh.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [Mesh] %-50s ", name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static void test_cube_gen(void)
{
    TEST("Procedural cube generation");
    rac_mesh_registry reg;
    rac_mesh_registry_init(&reg);

    int id = rac_mesh_gen_cube(&reg, 2.0f);
    CHECK(id >= 0, "cube created");

    rac_mesh *m = &reg.meshes[id];
    CHECK(m->valid, "mesh valid");
    CHECK(m->num_vertices == 24, "24 vertices");
    CHECK(m->num_indices == 36, "36 indices");

    /* Check AABB */
    CHECK(m->aabb.min.x >= -1.01f && m->aabb.min.x <= -0.99f, "aabb min x");
    CHECK(m->aabb.max.x >= 0.99f && m->aabb.max.x <= 1.01f, "aabb max x");

    rac_mesh_registry_cleanup(&reg);
    PASS();
}

static void test_sphere_gen(void)
{
    TEST("Procedural sphere generation");
    rac_mesh_registry reg;
    rac_mesh_registry_init(&reg);

    int id = rac_mesh_gen_sphere(&reg, 1.0f, 8, 12);
    CHECK(id >= 0, "sphere created");

    rac_mesh *m = &reg.meshes[id];
    CHECK(m->valid, "mesh valid");
    CHECK(m->num_vertices > 0, "has vertices");
    CHECK(m->num_indices > 0, "has indices");

    /* All vertices should be approximately radius 1.0 from origin */
    for (int i = 0; i < m->num_vertices; i++) {
        float len = rac_phys_v3_length(m->vertices[i].position);
        CHECK(len > 0.95f && len < 1.05f, "vertex on sphere surface");
    }

    /* Normals should be approximately unit length */
    for (int i = 0; i < m->num_vertices; i++) {
        float nlen = rac_phys_v3_length(m->vertices[i].normal);
        if (nlen > 0.01f) {
            CHECK(nlen > 0.9f && nlen < 1.1f, "unit normal");
        }
    }

    rac_mesh_registry_cleanup(&reg);
    PASS();
}

static void test_plane_gen(void)
{
    TEST("Procedural plane generation");
    rac_mesh_registry reg;
    rac_mesh_registry_init(&reg);

    int id = rac_mesh_gen_plane(&reg, 10.0f, 10.0f, 4);
    CHECK(id >= 0, "plane created");

    rac_mesh *m = &reg.meshes[id];
    CHECK(m->valid, "mesh valid");
    CHECK(m->num_vertices == 25, "5x5 = 25 vertices");
    CHECK(m->num_indices == 4 * 4 * 6, "4x4 quads * 6 indices");

    /* All Y values should be 0 */
    for (int i = 0; i < m->num_vertices; i++)
        CHECK(fabsf(m->vertices[i].position.y) < 0.001f, "flat plane");

    /* All normals should be (0, 1, 0) */
    for (int i = 0; i < m->num_vertices; i++)
        CHECK(m->vertices[i].normal.y > 0.99f, "upward normal");

    rac_mesh_registry_cleanup(&reg);
    PASS();
}

static void test_cylinder_gen(void)
{
    TEST("Procedural cylinder generation");
    rac_mesh_registry reg;
    rac_mesh_registry_init(&reg);

    int id = rac_mesh_gen_cylinder(&reg, 0.5f, 2.0f, 12);
    CHECK(id >= 0, "cylinder created");

    rac_mesh *m = &reg.meshes[id];
    CHECK(m->valid, "mesh valid");
    CHECK(m->num_vertices > 0, "has vertices");
    CHECK(m->num_indices > 0, "has indices");

    rac_mesh_registry_cleanup(&reg);
    PASS();
}

static void test_obj_loader(void)
{
    TEST("OBJ file loading");

    /* Create a simple OBJ file */
    FILE *f = fopen("/tmp/rac_test.obj", "w");
    CHECK(f != NULL, "write test obj");
    fprintf(f, "# Test OBJ\n");
    fprintf(f, "v 0.0 0.0 0.0\n");
    fprintf(f, "v 1.0 0.0 0.0\n");
    fprintf(f, "v 0.0 1.0 0.0\n");
    fprintf(f, "v 1.0 1.0 0.0\n");
    fprintf(f, "vn 0.0 0.0 1.0\n");
    fprintf(f, "vt 0.0 0.0\n");
    fprintf(f, "vt 1.0 0.0\n");
    fprintf(f, "vt 0.0 1.0\n");
    fprintf(f, "vt 1.0 1.0\n");
    fprintf(f, "f 1/1/1 2/2/1 3/3/1\n");
    fprintf(f, "f 2/2/1 4/4/1 3/3/1\n");
    fclose(f);

    rac_mesh_registry reg;
    rac_mesh_registry_init(&reg);

    int id = rac_mesh_load_obj(&reg, "/tmp/rac_test.obj");
    CHECK(id >= 0, "OBJ loaded");

    rac_mesh *m = &reg.meshes[id];
    CHECK(m->valid, "mesh valid");
    CHECK(m->num_vertices == 6, "6 vertices (2 triangles)");
    CHECK(m->num_indices == 6, "6 indices");

    rac_mesh_registry_cleanup(&reg);
    PASS();
}

static void test_compute_normals(void)
{
    TEST("Normal recomputation");
    rac_mesh_registry reg;
    rac_mesh_registry_init(&reg);

    int id = rac_mesh_create(&reg, 4, 6);
    CHECK(id >= 0, "mesh created");

    rac_mesh *m = &reg.meshes[id];
    /* Single quad in XY plane */
    m->vertices[0].position = rac_phys_v3(0, 0, 0);
    m->vertices[1].position = rac_phys_v3(1, 0, 0);
    m->vertices[2].position = rac_phys_v3(1, 1, 0);
    m->vertices[3].position = rac_phys_v3(0, 1, 0);
    m->num_vertices = 4;

    m->indices[0] = 0; m->indices[1] = 1; m->indices[2] = 2;
    m->indices[3] = 0; m->indices[4] = 2; m->indices[5] = 3;
    m->num_indices = 6;

    rac_mesh_compute_normals(m);

    /* All normals should point in +Z */
    for (int i = 0; i < 4; i++) {
        float nz = m->vertices[i].normal.z;
        CHECK(nz > 0.9f, "normal points +Z");
    }

    rac_mesh_registry_cleanup(&reg);
    PASS();
}

int main(void)
{
    printf("\n=== RAC Engine Mesh BVT ===\n\n");

    test_cube_gen();
    test_sphere_gen();
    test_plane_gen();
    test_cylinder_gen();
    test_obj_loader();
    test_compute_normals();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
