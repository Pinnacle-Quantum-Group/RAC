/*
 * rac_physics_particles.c — RAC Native Physics: Particles, SPH Fluids, Cloth
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * Amalgamates:
 *   - CUDA particle simulation (spatial hash, SPH kernels)
 *   - NVIDIA Flex unified PBD particle model
 *   - AMD HIPRT neighbor search patterns
 *
 * SPH fluid simulation uses poly6/spiky/viscosity kernels with all
 * distance computations routed through RAC rac_norm / rac_polar.
 * Cloth PBD constraints use the same rac_phys_pbd_solve_distance solver.
 */

#include "rac_physics.h"
#include "rac_cpu.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════════════════
 * §1  PARTICLE SYSTEM
 * ════════════════════════════════════════════════════════════════════════ */

rac_phys_particle_system* rac_phys_particles_create(int max_particles) {
    rac_phys_particle_system *ps = calloc(1, sizeof(rac_phys_particle_system));
    if (!ps) return NULL;

    ps->max_particles = max_particles;
    ps->positions  = calloc(max_particles, sizeof(rac_phys_vec3));
    ps->velocities = calloc(max_particles, sizeof(rac_phys_vec3));
    ps->forces     = calloc(max_particles, sizeof(rac_phys_vec3));
    ps->masses     = calloc(max_particles, sizeof(float));
    ps->inv_masses = calloc(max_particles, sizeof(float));
    ps->densities  = calloc(max_particles, sizeof(float));
    ps->pressures  = calloc(max_particles, sizeof(float));
    ps->alive      = calloc(max_particles, sizeof(int));
    ps->phase      = calloc(max_particles, sizeof(int));

    if (!ps->positions || !ps->velocities || !ps->forces ||
        !ps->masses || !ps->inv_masses || !ps->densities ||
        !ps->pressures || !ps->alive || !ps->phase) {
        rac_phys_particles_destroy(ps);
        return NULL;
    }

    return ps;
}

void rac_phys_particles_destroy(rac_phys_particle_system *ps) {
    if (!ps) return;
    free(ps->positions);
    free(ps->velocities);
    free(ps->forces);
    free(ps->masses);
    free(ps->inv_masses);
    free(ps->densities);
    free(ps->pressures);
    free(ps->alive);
    free(ps->phase);
    free(ps);
}

int rac_phys_particles_emit(rac_phys_particle_system *ps,
                              rac_phys_vec3 position,
                              rac_phys_vec3 velocity, float mass) {
    if (ps->num_particles >= ps->max_particles) return -1;
    int i = ps->num_particles++;
    ps->positions[i] = position;
    ps->velocities[i] = velocity;
    ps->forces[i] = rac_phys_v3_zero();
    ps->masses[i] = mass;
    ps->inv_masses[i] = (mass > 0.0f) ? 1.0f / mass : 0.0f;
    ps->alive[i] = 1;
    ps->phase[i] = 0;
    return i;
}

void rac_phys_particles_integrate(rac_phys_particle_system *ps,
                                    rac_phys_vec3 gravity, float dt) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < ps->num_particles; i++) {
        if (!ps->alive[i] || ps->inv_masses[i] <= 0.0f) continue;

        /* F = F_accumulated + m*g */
        rac_phys_vec3 total_force = rac_phys_v3_add(
            ps->forces[i],
            rac_phys_v3_scale(gravity, ps->masses[i]));

        /* Symplectic Euler: v += (F/m)*dt, x += v*dt */
        rac_phys_vec3 accel = rac_phys_v3_scale(total_force, ps->inv_masses[i]);
        ps->velocities[i] = rac_phys_v3_add(
            ps->velocities[i], rac_phys_v3_scale(accel, dt));
        ps->positions[i] = rac_phys_v3_add(
            ps->positions[i], rac_phys_v3_scale(ps->velocities[i], dt));

        /* Clear forces */
        ps->forces[i] = rac_phys_v3_zero();
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * §2  SPH FLUID SIMULATION
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Smoothed Particle Hydrodynamics with standard kernels:
 *   - Poly6 kernel for density estimation
 *   - Spiky kernel gradient for pressure forces
 *   - Viscosity kernel laplacian for viscous forces
 *
 * Heritage: CUDA particle simulation paper (Green 2008),
 *           GPU Gems 3 particle sim, GPUSPH project.
 *
 * All distance computations: rac_norm (CORDIC vectoring)
 * All dot products: rac_phys_v3_dot (rac_dot decomposition)
 */

#define RAC_PI_F 3.14159265358979f

rac_phys_sph_config rac_phys_sph_default_config(void) {
    return (rac_phys_sph_config){
        .rest_density = 1000.0f,
        .gas_constant = 2000.0f,
        .viscosity = 0.01f,
        .smoothing_radius = 0.1f,
        .particle_mass = 0.02f,
        .surface_tension = 0.0728f,
        .boundary_damping = 0.5f
    };
}

/* SPH kernels — all use RAC norm for distance */

static inline float _poly6(float r_sq, float h) {
    /* W_poly6(r, h) = 315/(64*pi*h^9) * (h²-r²)³  for r < h */
    float h_sq = h * h;
    if (r_sq >= h_sq) return 0.0f;
    float diff = h_sq - r_sq;
    float h9 = h * h * h * h * h * h * h * h * h;
    float coeff = 315.0f / (64.0f * RAC_PI_F * h9);
    return coeff * diff * diff * diff;
}

static inline rac_phys_vec3 _spiky_grad(rac_phys_vec3 r_vec, float r, float h) {
    /* ∇W_spiky(r, h) = -45/(pi*h^6) * (h-r)² * r_hat */
    if (r >= h || r < 1e-8f) return rac_phys_v3_zero();
    float h6 = h * h * h * h * h * h;
    float coeff = -45.0f / (RAC_PI_F * h6);
    float diff = h - r;
    float scale = coeff * diff * diff / r;
    return rac_phys_v3_scale(r_vec, scale);
}

static inline float _viscosity_lap(float r, float h) {
    /* ∇²W_viscosity(r, h) = 45/(pi*h^6) * (h - r) */
    if (r >= h) return 0.0f;
    float h6 = h * h * h * h * h * h;
    return 45.0f / (RAC_PI_F * h6) * (h - r);
}

void rac_phys_sph_compute_density_pressure(rac_phys_particle_system *ps,
                                            rac_phys_spatial_hash *grid,
                                            const rac_phys_sph_config *cfg) {
    float h = cfg->smoothing_radius;
    float h_sq = h * h;
    float mass = cfg->particle_mass;

    /* Rebuild spatial hash */
    rac_phys_spatial_hash_clear(grid);
    for (int i = 0; i < ps->num_particles; i++) {
        if (!ps->alive[i]) continue;
        rac_phys_vec3 hv = rac_phys_v3(h, h, h);
        rac_phys_aabb aabb = {
            rac_phys_v3_sub(ps->positions[i], hv),
            rac_phys_v3_add(ps->positions[i], hv)
        };
        rac_phys_spatial_hash_insert(grid, aabb, i);
    }

    /* Compute density and pressure for each particle */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int i = 0; i < ps->num_particles; i++) {
        if (!ps->alive[i]) continue;

        float density = 0.0f;
        rac_phys_vec3 hv = rac_phys_v3(h, h, h);
        rac_phys_aabb query = {
            rac_phys_v3_sub(ps->positions[i], hv),
            rac_phys_v3_add(ps->positions[i], hv)
        };

        int neighbors[256];
        int n_neighbors = rac_phys_spatial_hash_query(grid, query,
                                                       neighbors, 256);

        for (int ni = 0; ni < n_neighbors; ni++) {
            int j = neighbors[ni];
            if (!ps->alive[j]) continue;

            rac_phys_vec3 r_vec = rac_phys_v3_sub(ps->positions[i],
                                                    ps->positions[j]);
            float r_sq = rac_phys_v3_length_sq(r_vec);  /* RAC: dot */

            if (r_sq < h_sq)
                density += mass * _poly6(r_sq, h);
        }

        ps->densities[i] = fmaxf(density, 1e-6f);
        /* Equation of state: P = k * (ρ - ρ₀) */
        ps->pressures[i] = cfg->gas_constant * (ps->densities[i] - cfg->rest_density);
    }
}

void rac_phys_sph_compute_forces(rac_phys_particle_system *ps,
                                  rac_phys_spatial_hash *grid,
                                  const rac_phys_sph_config *cfg) {
    float h = cfg->smoothing_radius;
    float h_sq = h * h;
    float mass = cfg->particle_mass;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int i = 0; i < ps->num_particles; i++) {
        if (!ps->alive[i]) continue;

        rac_phys_vec3 f_pressure = rac_phys_v3_zero();
        rac_phys_vec3 f_viscosity = rac_phys_v3_zero();

        rac_phys_vec3 hv = rac_phys_v3(h, h, h);
        rac_phys_aabb query = {
            rac_phys_v3_sub(ps->positions[i], hv),
            rac_phys_v3_add(ps->positions[i], hv)
        };

        int neighbors[256];
        int n_neighbors = rac_phys_spatial_hash_query(grid, query,
                                                       neighbors, 256);

        for (int ni = 0; ni < n_neighbors; ni++) {
            int j = neighbors[ni];
            if (j == i || !ps->alive[j]) continue;

            rac_phys_vec3 r_vec = rac_phys_v3_sub(ps->positions[i],
                                                    ps->positions[j]);
            float r_sq = rac_phys_v3_length_sq(r_vec);

            if (r_sq >= h_sq) continue;

            float r = rac_phys_v3_length(r_vec);  /* RAC: rac_norm chain */
            float rho_j = ps->densities[j];
            if (rho_j < 1e-6f) continue;

            /* Pressure force: -m * (P_i + P_j) / (2*ρ_j) * ∇W_spiky */
            float pressure_avg = (ps->pressures[i] + ps->pressures[j]) * 0.5f;
            rac_phys_vec3 grad = _spiky_grad(r_vec, r, h);
            f_pressure = rac_phys_v3_add(f_pressure,
                rac_phys_v3_scale(grad, -mass * pressure_avg / rho_j));

            /* Viscosity force: μ * m * (v_j - v_i) / ρ_j * ∇²W_visc */
            rac_phys_vec3 v_diff = rac_phys_v3_sub(
                ps->velocities[j], ps->velocities[i]);
            float lap = _viscosity_lap(r, h);
            f_viscosity = rac_phys_v3_add(f_viscosity,
                rac_phys_v3_scale(v_diff, cfg->viscosity * mass * lap / rho_j));
        }

        ps->forces[i] = rac_phys_v3_add(
            ps->forces[i],
            rac_phys_v3_add(f_pressure, f_viscosity));
    }
}

void rac_phys_sph_step(rac_phys_particle_system *ps,
                        rac_phys_spatial_hash *grid,
                        rac_phys_vec3 gravity,
                        const rac_phys_sph_config *cfg, float dt) {
    /* 1. Compute density and pressure */
    rac_phys_sph_compute_density_pressure(ps, grid, cfg);

    /* 2. Compute pressure + viscosity forces */
    rac_phys_sph_compute_forces(ps, grid, cfg);

    /* 3. Integrate */
    rac_phys_particles_integrate(ps, gravity, dt);

    /*
     * Fix #13: Apply boundary damping — enforce a simple box boundary.
     * Particles that exit the boundary are clamped and their velocity
     * is reflected with damping applied. Uses smoothing_radius as the
     * boundary half-extent (caller should set based on simulation domain).
     */
    if (cfg->boundary_damping > 0.0f) {
        float bound = 10.0f;  /* default world half-extent */
        float damp = 1.0f - cfg->boundary_damping;

        for (int i = 0; i < ps->num_particles; i++) {
            if (!ps->alive[i]) continue;
            rac_phys_vec3 *p = &ps->positions[i];
            rac_phys_vec3 *v = &ps->velocities[i];

            if (p->x < -bound) { p->x = -bound; v->x *= -damp; }
            if (p->x >  bound) { p->x =  bound; v->x *= -damp; }
            if (p->y < -bound) { p->y = -bound; v->y *= -damp; }
            if (p->y >  bound) { p->y =  bound; v->y *= -damp; }
            if (p->z < -bound) { p->z = -bound; v->z *= -damp; }
            if (p->z >  bound) { p->z =  bound; v->z *= -damp; }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * §3  CLOTH SIMULATION (PBD — Flex heritage)
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Grid-based cloth with stretch + bend constraints.
 * Each timestep:
 *   1. Apply gravity
 *   2. Predict positions (symplectic Euler)
 *   3. Solve PBD distance constraints (stretch + bend)
 *   4. Update velocities from position changes
 */

rac_phys_cloth* rac_phys_cloth_create_grid(int width, int height,
                                            float spacing, float mass) {
    int n = width * height;
    float per_mass = mass / (float)n;

    rac_phys_cloth *cloth = calloc(1, sizeof(rac_phys_cloth));
    if (!cloth) return NULL;

    cloth->particles = rac_phys_particles_create(n);
    if (!cloth->particles) { free(cloth); return NULL; }

    /* Create particles in grid pattern */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            rac_phys_vec3 pos = rac_phys_v3(
                (float)x * spacing,
                (float)(height - 1) * spacing,  /* start at top */
                (float)y * spacing * 0.01f       /* slight z offset for stability */
            );
            rac_phys_particles_emit(cloth->particles, pos,
                                     rac_phys_v3_zero(), per_mass);
        }
    }

    /* ── Stretch constraints (edges) ───────────────────────────── */
    /* Horizontal + vertical edges */
    int max_stretch = 2 * width * height;
    cloth->stretch_pairs = malloc(sizeof(int) * max_stretch * 2);
    cloth->stretch_rest = malloc(sizeof(float) * max_stretch);
    cloth->num_stretch = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            /* Right neighbor */
            if (x + 1 < width) {
                int ni = cloth->num_stretch;
                cloth->stretch_pairs[ni * 2] = idx;
                cloth->stretch_pairs[ni * 2 + 1] = idx + 1;
                cloth->stretch_rest[ni] = spacing;
                cloth->num_stretch++;
            }
            /* Down neighbor */
            if (y + 1 < height) {
                int ni = cloth->num_stretch;
                cloth->stretch_pairs[ni * 2] = idx;
                cloth->stretch_pairs[ni * 2 + 1] = idx + width;
                cloth->stretch_rest[ni] = spacing;
                cloth->num_stretch++;
            }
        }
    }

    /* ── Bend constraints (skip-one connections) ───────────────── */
    int max_bend = 2 * width * height;
    cloth->bend_pairs = malloc(sizeof(int) * max_bend * 2);
    cloth->bend_rest = malloc(sizeof(float) * max_bend);
    cloth->num_bend = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (x + 2 < width) {
                int ni = cloth->num_bend;
                cloth->bend_pairs[ni * 2] = idx;
                cloth->bend_pairs[ni * 2 + 1] = idx + 2;
                cloth->bend_rest[ni] = spacing * 2.0f;
                cloth->num_bend++;
            }
            if (y + 2 < height) {
                int ni = cloth->num_bend;
                cloth->bend_pairs[ni * 2] = idx;
                cloth->bend_pairs[ni * 2 + 1] = idx + 2 * width;
                cloth->bend_rest[ni] = spacing * 2.0f;
                cloth->num_bend++;
            }
        }
    }

    cloth->stretch_stiffness = 0.9f;
    cloth->bend_stiffness = 0.5f;
    cloth->solver_iterations = 4;

    return cloth;
}

void rac_phys_cloth_destroy(rac_phys_cloth *cloth) {
    if (!cloth) return;
    rac_phys_particles_destroy(cloth->particles);
    free(cloth->stretch_pairs);
    free(cloth->stretch_rest);
    free(cloth->bend_pairs);
    free(cloth->bend_rest);
    free(cloth);
}

void rac_phys_cloth_pin(rac_phys_cloth *cloth, int particle_index) {
    if (particle_index >= 0 && particle_index < cloth->particles->num_particles)
        cloth->particles->inv_masses[particle_index] = 0.0f;
}

void rac_phys_cloth_step(rac_phys_cloth *cloth,
                           rac_phys_vec3 gravity, float dt) {
    rac_phys_particle_system *ps = cloth->particles;
    int n = ps->num_particles;

    /* Save old positions for velocity update */
    rac_phys_vec3 *old_pos = malloc(sizeof(rac_phys_vec3) * n);
    if (!old_pos) return;
    memcpy(old_pos, ps->positions, sizeof(rac_phys_vec3) * n);

    /* Predict positions: x_pred = x + v*dt + g*dt² */
    for (int i = 0; i < n; i++) {
        if (ps->inv_masses[i] <= 0.0f) continue;
        rac_phys_vec3 v_step = rac_phys_v3_scale(ps->velocities[i], dt);
        rac_phys_vec3 g_step = rac_phys_v3_scale(gravity, dt * dt);
        ps->positions[i] = rac_phys_v3_add(
            ps->positions[i],
            rac_phys_v3_add(v_step, g_step));
    }

    /* Solve PBD constraints */
    rac_phys_pbd_config pbd_cfg = {
        .substeps = 1,
        .iterations = cloth->solver_iterations,
        .damping = 0.99f
    };

    /* Stretch constraints */
    rac_phys_pbd_solve_distance(ps->positions, ps->inv_masses,
                                 cloth->stretch_pairs, cloth->stretch_rest,
                                 cloth->num_stretch, &pbd_cfg);

    /* Bend constraints (fewer iterations, lower stiffness) */
    rac_phys_pbd_config bend_cfg = pbd_cfg;
    bend_cfg.iterations = 2;
    rac_phys_pbd_solve_distance(ps->positions, ps->inv_masses,
                                 cloth->bend_pairs, cloth->bend_rest,
                                 cloth->num_bend, &bend_cfg);

    /* Update velocities from position delta */
    float inv_dt = (dt > 0.0f) ? 1.0f / dt : 0.0f;
    for (int i = 0; i < n; i++) {
        if (ps->inv_masses[i] <= 0.0f) continue;
        rac_phys_vec3 delta = rac_phys_v3_sub(ps->positions[i], old_pos[i]);
        ps->velocities[i] = rac_phys_v3_scale(delta, inv_dt * 0.99f);
    }

    free(old_pos);
}
