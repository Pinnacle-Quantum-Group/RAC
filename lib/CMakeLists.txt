# RAC Library Build — C, C++, AVX2, OpenMP, MPI
# Pinnacle Quantum Group — March 2026
#
# Build:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#
# Options:
#   -DRAC_AVX2=ON    (default: auto-detect)
#   -DRAC_OPENMP=ON  (default: ON)
#   -DRAC_MPI=ON     (default: OFF)

cmake_minimum_required(VERSION 3.16)
project(rac VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# ── Options ──────────────────────────────────────────────────────────────

option(RAC_OPENMP "Enable OpenMP parallelism" ON)
option(RAC_MPI    "Enable MPI distributed matmul" OFF)
option(RAC_AVX2   "Enable AVX2/FMA intrinsics" ON)

# ── Core C library ──────────────────────────────────────────────────────

add_library(rac SHARED c/rac_cpu.c c/rac_alu.c)
target_include_directories(rac PUBLIC c/)
target_link_libraries(rac m)
target_compile_options(rac PRIVATE -O3 -march=native -Wall -Wextra)

add_library(rac_static STATIC c/rac_cpu.c c/rac_alu.c)
target_include_directories(rac_static PUBLIC c/)
target_link_libraries(rac_static m)
target_compile_options(rac_static PRIVATE -O3 -march=native -Wall -Wextra)

# ── OpenMP ──────────────────────────────────────────────────────────────

if(RAC_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(rac OpenMP::OpenMP_C)
        target_link_libraries(rac_static OpenMP::OpenMP_C)
        message(STATUS "RAC: OpenMP enabled")
    endif()
endif()

# ── AVX2 ────────────────────────────────────────────────────────────────

if(RAC_AVX2)
    include(CheckCCompilerFlag)
    check_c_compiler_flag("-mavx2" HAS_AVX2)
    check_c_compiler_flag("-mfma" HAS_FMA)

    if(HAS_AVX2 AND HAS_FMA)
        enable_language(ASM)
        add_library(rac_avx2 SHARED c/rac_cpu.c c/rac_avx2.c c/rac_hal.c c/rac_zen3_kern.S c/rac_avx512_kern.S)
        target_include_directories(rac_avx2 PUBLIC c/)
        target_compile_options(rac_avx2 PRIVATE -O3 -mavx2 -mfma -march=native -Wall)
        target_link_libraries(rac_avx2 m)
        if(OpenMP_C_FOUND)
            target_link_libraries(rac_avx2 OpenMP::OpenMP_C)
        endif()
        message(STATUS "RAC: AVX2+FMA+HAL enabled → librac_avx2")
    else()
        message(WARNING "RAC: AVX2/FMA not supported by compiler")
    endif()
endif()

# ── MPI ─────────────────────────────────────────────────────────────────

if(RAC_MPI)
    find_package(MPI REQUIRED)
    add_executable(rac_mpi c/rac_mpi.c c/rac_cpu.c c/rac_avx2.c c/rac_hal.c c/rac_zen3_kern.S c/rac_avx512_kern.S)
    target_include_directories(rac_mpi PRIVATE c/)
    target_compile_options(rac_mpi PRIVATE -O3 -mavx2 -mfma -march=native)
    target_link_libraries(rac_mpi MPI::MPI_C m)
    if(OpenMP_C_FOUND)
        target_link_libraries(rac_mpi OpenMP::OpenMP_C)
    endif()
    message(STATUS "RAC: MPI distributed SGEMM → rac_mpi")
endif()

# ── Tests ───────────────────────────────────────────────────────────────

enable_testing()

# Library BVT
if(TARGET rac_avx2)
    add_executable(test_rac_lib_bvt c/test_rac_lib_bvt.c)
    target_include_directories(test_rac_lib_bvt PRIVATE c/)
    target_link_libraries(test_rac_lib_bvt rac_avx2 m)
    add_test(NAME lib_bvt COMMAND test_rac_lib_bvt)

    # Library DVT
    add_executable(test_rac_lib_dvt c/test_rac_lib_dvt.c)
    target_include_directories(test_rac_lib_dvt PRIVATE c/)
    target_link_libraries(test_rac_lib_dvt rac_avx2 m)
    add_test(NAME lib_dvt COMMAND test_rac_lib_dvt)

    # Library E2E + Benchmarks
    add_executable(test_rac_lib_e2e c/test_rac_lib_e2e.c)
    target_include_directories(test_rac_lib_e2e PRIVATE c/)
    target_link_libraries(test_rac_lib_e2e rac_avx2 m)
    add_test(NAME lib_e2e COMMAND test_rac_lib_e2e)

    # Long-run stress test
    add_executable(test_rac_longrun c/test_rac_longrun.c)
    target_include_directories(test_rac_longrun PRIVATE c/)
    target_link_libraries(test_rac_longrun rac_avx2 m)

    # Thread scaling benchmark
    add_executable(test_rac_threads c/test_rac_thread_scaling.c)
    target_include_directories(test_rac_threads PRIVATE c/)
    target_link_libraries(test_rac_threads rac_avx2 m)
endif()

# RAC Adder + ALU BVT. rac_alu.c contains an AVX2+FMA fast path that
# is compiled in automatically when -mavx2 -mfma are enabled via
# -march=native (which rac_static uses).
add_executable(test_rac_alu c/test_rac_alu.c)
target_include_directories(test_rac_alu PRIVATE c/)
target_link_libraries(test_rac_alu rac_static m)
add_test(NAME rac_alu_bvt COMMAND test_rac_alu)

# RAC ALU microbenchmark (not added to ctest — interactive perf tool)
add_executable(bench_rac_alu c/bench_rac_alu.c)
target_include_directories(bench_rac_alu PRIVATE c/)
target_link_libraries(bench_rac_alu rac_static m)

# ── Physics library ─────────────────────────────────────────────────────
#
# RAC Native Physics: amalgamates CUDA (PhysX/Flex) + AMD (FEMFX/ROCm)
# physics patterns, built entirely on RAC CORDIC primitives.
#

set(RAC_PHYSICS_SOURCES
    c/rac_physics.c
    c/rac_physics_collision.c
    c/rac_physics_constraints.c
    c/rac_physics_particles.c
    c/rac_physics_softbody.c
    c/rac_physics_world.c
)

add_library(rac_physics SHARED ${RAC_PHYSICS_SOURCES} c/rac_cpu.c)
target_include_directories(rac_physics PUBLIC c/)
target_link_libraries(rac_physics m)
target_compile_options(rac_physics PRIVATE -O3 -march=native -Wall -Wextra)

add_library(rac_physics_static STATIC ${RAC_PHYSICS_SOURCES} c/rac_cpu.c)
target_include_directories(rac_physics_static PUBLIC c/)
target_link_libraries(rac_physics_static m)
target_compile_options(rac_physics_static PRIVATE -O3 -march=native -Wall -Wextra)

if(OpenMP_C_FOUND)
    target_link_libraries(rac_physics OpenMP::OpenMP_C)
    target_link_libraries(rac_physics_static OpenMP::OpenMP_C)
    message(STATUS "RAC Physics: OpenMP enabled for parallel SPH/particles")
endif()

# Physics BVT
add_executable(test_rac_physics_bvt c/test_rac_physics_bvt.c)
target_include_directories(test_rac_physics_bvt PRIVATE c/)
target_link_libraries(test_rac_physics_bvt rac_physics_static m)
add_test(NAME physics_bvt COMMAND test_rac_physics_bvt)

# ── Install ─────────────────────────────────────────────────────────────

install(TARGETS rac rac_static DESTINATION lib)
install(TARGETS rac_physics rac_physics_static DESTINATION lib)
install(FILES c/rac_cpu.h c/rac_alu.h c/rac_avx2.h c/rac_physics.h DESTINATION include/rac)
if(TARGET rac_avx2)
    install(TARGETS rac_avx2 DESTINATION lib)
endif()

# ── Game Engine ──────────────────────────────────────────────────────────
#
# RAC Native Game Engine: software rasterizer, spatial audio, ECS,
# scene graph — all built on RAC CORDIC primitives.
#

option(RAC_ENGINE "Build RAC game engine" ON)

if(RAC_ENGINE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/../engine engine_build)
    message(STATUS "RAC Engine: enabled")
endif()

message(STATUS "RAC library v${PROJECT_VERSION} configured")
message(STATUS "RAC Physics library configured (rigid body, collision, constraints, SPH, cloth, FEM)")
