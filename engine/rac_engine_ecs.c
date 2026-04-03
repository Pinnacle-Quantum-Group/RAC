/*
 * rac_engine_ecs.c — Entity-Component System Implementation
 */

#include "rac_engine_ecs.h"
#include <string.h>

void rac_ecs_init(rac_ecs_world *ecs)
{
    memset(ecs, 0, sizeof(*ecs));
    ecs->next_id = 0;
    ecs->num_entities = 0;
}

uint32_t rac_ecs_create_entity(rac_ecs_world *ecs)
{
    /* Find first dead slot or use next_id */
    for (uint32_t i = 0; i < ecs->next_id && i < RAC_ECS_MAX_ENTITIES; i++) {
        if (!ecs->alive[i]) {
            ecs->alive[i] = 1;
            ecs->component_masks[i] = RAC_COMP_NONE;
            ecs->num_entities++;
            return i;
        }
    }

    if (ecs->next_id >= RAC_ECS_MAX_ENTITIES)
        return RAC_ECS_INVALID_ENTITY;

    uint32_t id = ecs->next_id++;
    ecs->alive[id] = 1;
    ecs->component_masks[id] = RAC_COMP_NONE;
    ecs->num_entities++;

    /* Initialize default transform */
    ecs->transforms[id].position = rac_phys_v3_zero();
    ecs->transforms[id].rotation = rac_phys_quat_identity();
    ecs->transforms[id].scale = rac_phys_v3(1.0f, 1.0f, 1.0f);

    return id;
}

void rac_ecs_destroy_entity(rac_ecs_world *ecs, uint32_t entity)
{
    if (entity >= RAC_ECS_MAX_ENTITIES || !ecs->alive[entity])
        return;
    ecs->alive[entity] = 0;
    ecs->component_masks[entity] = RAC_COMP_NONE;
    if (ecs->num_entities > 0)
        ecs->num_entities--;
}

int rac_ecs_is_alive(const rac_ecs_world *ecs, uint32_t entity)
{
    if (entity >= RAC_ECS_MAX_ENTITIES) return 0;
    return ecs->alive[entity];
}

void rac_ecs_add_component(rac_ecs_world *ecs, uint32_t entity,
                           rac_component_flag comp)
{
    if (entity >= RAC_ECS_MAX_ENTITIES || !ecs->alive[entity])
        return;
    ecs->component_masks[entity] |= (uint32_t)comp;
}

void rac_ecs_remove_component(rac_ecs_world *ecs, uint32_t entity,
                              rac_component_flag comp)
{
    if (entity >= RAC_ECS_MAX_ENTITIES || !ecs->alive[entity])
        return;
    ecs->component_masks[entity] &= ~(uint32_t)comp;
}

int rac_ecs_has_component(const rac_ecs_world *ecs, uint32_t entity,
                          rac_component_flag comp)
{
    if (entity >= RAC_ECS_MAX_ENTITIES || !ecs->alive[entity])
        return 0;
    return (ecs->component_masks[entity] & (uint32_t)comp) == (uint32_t)comp;
}

int rac_ecs_query(const rac_ecs_world *ecs, uint32_t required_mask,
                  uint32_t *out_entities, int max_results)
{
    int count = 0;
    for (uint32_t i = 0; i < ecs->next_id && count < max_results; i++) {
        if (ecs->alive[i] &&
            (ecs->component_masks[i] & required_mask) == required_mask) {
            out_entities[count++] = i;
        }
    }
    return count;
}
