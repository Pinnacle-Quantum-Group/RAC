/*
 * test_engine_ecs.c — ECS BVT
 */

#include "../rac_engine_ecs.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [ECS] %-50s ", name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static void test_create_destroy(void)
{
    TEST("Create and destroy entities");
    rac_ecs_world ecs;
    rac_ecs_init(&ecs);

    uint32_t e0 = rac_ecs_create_entity(&ecs);
    uint32_t e1 = rac_ecs_create_entity(&ecs);
    uint32_t e2 = rac_ecs_create_entity(&ecs);

    CHECK(e0 != RAC_ECS_INVALID_ENTITY, "e0 valid");
    CHECK(e1 != RAC_ECS_INVALID_ENTITY, "e1 valid");
    CHECK(e2 != RAC_ECS_INVALID_ENTITY, "e2 valid");
    CHECK(ecs.num_entities == 3, "count = 3");
    CHECK(rac_ecs_is_alive(&ecs, e0), "e0 alive");

    rac_ecs_destroy_entity(&ecs, e1);
    CHECK(!rac_ecs_is_alive(&ecs, e1), "e1 dead");
    CHECK(ecs.num_entities == 2, "count = 2");

    /* Reuse slot */
    uint32_t e3 = rac_ecs_create_entity(&ecs);
    CHECK(e3 == e1, "recycled slot");
    CHECK(ecs.num_entities == 3, "count = 3 again");
    PASS();
}

static void test_components(void)
{
    TEST("Add/remove/query components");
    rac_ecs_world ecs;
    rac_ecs_init(&ecs);

    uint32_t e = rac_ecs_create_entity(&ecs);
    CHECK(!rac_ecs_has_component(&ecs, e, RAC_COMP_TRANSFORM), "no transform yet");

    rac_ecs_add_component(&ecs, e, RAC_COMP_TRANSFORM);
    CHECK(rac_ecs_has_component(&ecs, e, RAC_COMP_TRANSFORM), "has transform");

    rac_ecs_add_component(&ecs, e, RAC_COMP_RIGIDBODY);
    CHECK(rac_ecs_has_component(&ecs, e, RAC_COMP_RIGIDBODY), "has rigidbody");
    CHECK(rac_ecs_has_component(&ecs, e, RAC_COMP_TRANSFORM), "still has transform");

    rac_ecs_remove_component(&ecs, e, RAC_COMP_TRANSFORM);
    CHECK(!rac_ecs_has_component(&ecs, e, RAC_COMP_TRANSFORM), "transform removed");
    CHECK(rac_ecs_has_component(&ecs, e, RAC_COMP_RIGIDBODY), "rigidbody remains");
    PASS();
}

static void test_query(void)
{
    TEST("Query entities by component mask");
    rac_ecs_world ecs;
    rac_ecs_init(&ecs);

    uint32_t e0 = rac_ecs_create_entity(&ecs);
    uint32_t e1 = rac_ecs_create_entity(&ecs);
    uint32_t e2 = rac_ecs_create_entity(&ecs);

    rac_ecs_add_component(&ecs, e0, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER);
    rac_ecs_add_component(&ecs, e1, RAC_COMP_TRANSFORM | RAC_COMP_RIGIDBODY);
    rac_ecs_add_component(&ecs, e2, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER | RAC_COMP_RIGIDBODY);

    uint32_t results[16];
    int count;

    /* All with TRANSFORM */
    count = rac_ecs_query(&ecs, RAC_COMP_TRANSFORM, results, 16);
    CHECK(count == 3, "3 entities with transform");

    /* TRANSFORM + MESH_RENDERER */
    count = rac_ecs_query(&ecs, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER, results, 16);
    CHECK(count == 2, "2 entities with transform+mesh");

    /* All three components */
    count = rac_ecs_query(&ecs, RAC_COMP_TRANSFORM | RAC_COMP_MESH_RENDERER | RAC_COMP_RIGIDBODY, results, 16);
    CHECK(count == 1, "1 entity with all three");
    CHECK(results[0] == e2, "correct entity");
    PASS();
}

static void test_max_entities(void)
{
    TEST("Max entity limit");
    rac_ecs_world *ecs = (rac_ecs_world *)calloc(1, sizeof(rac_ecs_world));
    rac_ecs_init(ecs);

    /* Create up to limit */
    int created = 0;
    for (int i = 0; i < RAC_ECS_MAX_ENTITIES + 10; i++) {
        uint32_t e = rac_ecs_create_entity(ecs);
        if (e != RAC_ECS_INVALID_ENTITY) created++;
    }
    CHECK(created == RAC_ECS_MAX_ENTITIES, "exactly max entities");
    free(ecs);
    PASS();
}

static void test_transform_data(void)
{
    TEST("Transform component data");
    rac_ecs_world ecs;
    rac_ecs_init(&ecs);

    uint32_t e = rac_ecs_create_entity(&ecs);
    rac_ecs_add_component(&ecs, e, RAC_COMP_TRANSFORM);

    ecs.transforms[e].position = rac_phys_v3(1.0f, 2.0f, 3.0f);
    ecs.transforms[e].rotation = rac_phys_quat_from_axis_angle(
        rac_phys_v3(0.0f, 1.0f, 0.0f), RAC_PI * 0.5f);
    ecs.transforms[e].scale = rac_phys_v3(2.0f, 2.0f, 2.0f);

    CHECK(ecs.transforms[e].position.x == 1.0f, "pos.x");
    CHECK(ecs.transforms[e].position.y == 2.0f, "pos.y");
    CHECK(ecs.transforms[e].position.z == 3.0f, "pos.z");
    CHECK(ecs.transforms[e].scale.x == 2.0f, "scale.x");
    PASS();
}

int main(void)
{
    printf("\n=== RAC Engine ECS BVT ===\n\n");

    test_create_destroy();
    test_components();
    test_query();
    test_max_entities();
    test_transform_data();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
