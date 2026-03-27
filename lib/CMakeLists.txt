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

add_library(rac SHARED c/rac_cpu.c)
target_include_directories(rac PUBLIC c/)
target_link_libraries(rac m)
target_compile_options(rac PRIVATE -O3 -march=native -Wall -Wextra)

add_library(rac_static STATIC c/rac_cpu.c)
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
        add_library(rac_avx2 SHARED c/rac_cpu.c c/rac_avx2.c c/rac_hal.c)
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
    add_executable(rac_mpi c/rac_mpi.c c/rac_cpu.c c/rac_avx2.c)
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

# C primitives test (from root)
add_executable(test_rac_primitives ../test_rac_primitives.cu)
set_source_files_properties(../test_rac_primitives.cu PROPERTIES LANGUAGE CXX)
target_include_directories(test_rac_primitives PRIVATE ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_rac_primitives rac_static m)
add_test(NAME rac_primitives COMMAND test_rac_primitives)

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
endif()

# ── Install ─────────────────────────────────────────────────────────────

install(TARGETS rac rac_static DESTINATION lib)
install(FILES c/rac_cpu.h c/rac_avx2.h DESTINATION include/rac)
if(TARGET rac_avx2)
    install(TARGETS rac_avx2 DESTINATION lib)
endif()

message(STATUS "RAC library v${PROJECT_VERSION} configured")
