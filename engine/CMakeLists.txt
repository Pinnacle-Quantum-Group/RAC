# RAC Game Engine — Built on RAC rotation-accumulate primitives
# Pinnacle Quantum Group — April 2026
#
# Build:
#   cd RAC/lib && mkdir -p build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release -DRAC_ENGINE=ON
#   make -j$(nproc)
#
# Standalone:
#   cd RAC/engine && mkdir -p build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)

cmake_minimum_required(VERSION 3.16)

# Allow standalone build or inclusion from parent
if(NOT TARGET rac_physics_static)
    project(rac_engine VERSION 0.1.0 LANGUAGES C)
    set(CMAKE_C_STANDARD 99)

    # Build physics library from source
    set(RAC_PHYSICS_SOURCES
        ../lib/c/rac_cpu.c
        ../lib/c/rac_physics.c
        ../lib/c/rac_physics_collision.c
        ../lib/c/rac_physics_constraints.c
        ../lib/c/rac_physics_particles.c
        ../lib/c/rac_physics_softbody.c
        ../lib/c/rac_physics_world.c
    )
    add_library(rac_physics_static STATIC ${RAC_PHYSICS_SOURCES})
    target_include_directories(rac_physics_static PUBLIC ../lib/c/)
    target_link_libraries(rac_physics_static m)
    target_compile_options(rac_physics_static PRIVATE -O3 -march=native -Wall -Wextra)

    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(rac_physics_static OpenMP::OpenMP_C)
    endif()
endif()

# ── Engine library ──────────────────────────────────────────────────────

set(RAC_ENGINE_SOURCES
    rac_engine_core.c
    rac_engine_ecs.c
    rac_engine_scene.c
    rac_engine_mesh.c
    rac_engine_camera.c
    rac_engine_light.c
    rac_engine_render.c
    rac_engine_audio.c
    rac_engine_input.c
    rac_engine_sprite.c
)

add_library(rac_engine STATIC ${RAC_ENGINE_SOURCES})
target_include_directories(rac_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../lib/c/)
target_link_libraries(rac_engine rac_physics_static m)
target_compile_options(rac_engine PRIVATE -O3 -march=native -Wall -Wextra)

if(OpenMP_C_FOUND)
    target_link_libraries(rac_engine OpenMP::OpenMP_C)
    message(STATUS "RAC Engine: OpenMP enabled for parallel rasterization")
endif()

# ── Demo executable ─────────────────────────────────────────────────────

add_executable(rac_engine_demo rac_engine_demo.c)
target_link_libraries(rac_engine_demo rac_engine)

# Optional SDL2 for windowed output
find_package(SDL2 QUIET)
if(SDL2_FOUND)
    target_link_libraries(rac_engine_demo SDL2::SDL2)
    target_compile_definitions(rac_engine_demo PRIVATE RAC_HAS_SDL2)
    message(STATUS "RAC Engine: SDL2 found — windowed output available")
else()
    message(STATUS "RAC Engine: SDL2 not found — headless PPM output only")
endif()

# ── Tests ───────────────────────────────────────────────────────────────

enable_testing()

add_executable(test_engine_ecs tests/test_engine_ecs.c)
target_link_libraries(test_engine_ecs rac_engine)
add_test(NAME engine_ecs_bvt COMMAND test_engine_ecs)

add_executable(test_engine_render tests/test_engine_render.c)
target_link_libraries(test_engine_render rac_engine)
add_test(NAME engine_render_bvt COMMAND test_engine_render)

add_executable(test_engine_audio tests/test_engine_audio.c)
target_link_libraries(test_engine_audio rac_engine)
add_test(NAME engine_audio_bvt COMMAND test_engine_audio)

add_executable(test_engine_mesh tests/test_engine_mesh.c)
target_link_libraries(test_engine_mesh rac_engine)
add_test(NAME engine_mesh_bvt COMMAND test_engine_mesh)

message(STATUS "RAC Engine v0.1.0 configured")

# Copy assets to build directory
file(GLOB RAC_ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/assets/*")
file(COPY ${RAC_ASSETS} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/assets/")
message(STATUS "  Modules: ECS, Scene, Mesh, Camera, Light, Render, Audio, Input")
message(STATUS "  Demo: rac_engine_demo")
message(STATUS "  Tests: engine_ecs_bvt, engine_render_bvt, engine_audio_bvt, engine_mesh_bvt")
