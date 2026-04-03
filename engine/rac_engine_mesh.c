/*
 * rac_engine_mesh.c — Mesh System Implementation
 * All normal/AABB computations via RAC primitives.
 */

#include "rac_engine_mesh.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

void rac_mesh_registry_init(rac_mesh_registry *reg)
{
    memset(reg, 0, sizeof(*reg));
}

void rac_mesh_registry_cleanup(rac_mesh_registry *reg)
{
    for (int i = 0; i < reg->num_meshes; i++)
        rac_mesh_destroy(&reg->meshes[i]);
    reg->num_meshes = 0;
}

int rac_mesh_create(rac_mesh_registry *reg, int max_verts, int max_indices)
{
    if (reg->num_meshes >= RAC_MAX_MESHES) return -1;

    int id = reg->num_meshes++;
    rac_mesh *m = &reg->meshes[id];
    memset(m, 0, sizeof(*m));

    m->vertices = (rac_vertex *)calloc(max_verts, sizeof(rac_vertex));
    m->indices = (int *)calloc(max_indices, sizeof(int));
    m->num_vertices = 0;
    m->num_indices = 0;
    m->valid = (m->vertices && m->indices) ? 1 : 0;
    return id;
}

void rac_mesh_destroy(rac_mesh *mesh)
{
    free(mesh->vertices);
    free(mesh->indices);
    mesh->vertices = NULL;
    mesh->indices = NULL;
    mesh->valid = 0;
}

void rac_mesh_compute_aabb(rac_mesh *mesh)
{
    if (!mesh->valid || mesh->num_vertices == 0) return;

    rac_phys_vec3 mn = mesh->vertices[0].position;
    rac_phys_vec3 mx = mn;

    for (int i = 1; i < mesh->num_vertices; i++) {
        rac_phys_vec3 p = mesh->vertices[i].position;
        if (p.x < mn.x) mn.x = p.x;
        if (p.y < mn.y) mn.y = p.y;
        if (p.z < mn.z) mn.z = p.z;
        if (p.x > mx.x) mx.x = p.x;
        if (p.y > mx.y) mx.y = p.y;
        if (p.z > mx.z) mx.z = p.z;
    }
    mesh->aabb.min = mn;
    mesh->aabb.max = mx;
}

void rac_mesh_compute_normals(rac_mesh *mesh)
{
    if (!mesh->valid) return;

    /* Zero all normals */
    for (int i = 0; i < mesh->num_vertices; i++)
        mesh->vertices[i].normal = rac_phys_v3_zero();

    /* Accumulate face normals via rac cross product */
    for (int i = 0; i < mesh->num_indices; i += 3) {
        int i0 = mesh->indices[i];
        int i1 = mesh->indices[i + 1];
        int i2 = mesh->indices[i + 2];

        rac_phys_vec3 e1 = rac_phys_v3_sub(mesh->vertices[i1].position,
                                             mesh->vertices[i0].position);
        rac_phys_vec3 e2 = rac_phys_v3_sub(mesh->vertices[i2].position,
                                             mesh->vertices[i0].position);
        rac_phys_vec3 fn = rac_phys_v3_cross(e1, e2);

        mesh->vertices[i0].normal = rac_phys_v3_add(mesh->vertices[i0].normal, fn);
        mesh->vertices[i1].normal = rac_phys_v3_add(mesh->vertices[i1].normal, fn);
        mesh->vertices[i2].normal = rac_phys_v3_add(mesh->vertices[i2].normal, fn);
    }

    /* Normalize via rac_phys_v3_normalize (RAC-native) */
    for (int i = 0; i < mesh->num_vertices; i++)
        mesh->vertices[i].normal = rac_phys_v3_normalize(mesh->vertices[i].normal);
}

/* ── OBJ loader ────────────────────────────────────────────────────────── */

int rac_mesh_load_obj(rac_mesh_registry *reg, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    /* Temp storage */
    rac_phys_vec3 positions[RAC_MESH_MAX_VERTICES];
    rac_phys_vec3 normals[RAC_MESH_MAX_VERTICES];
    float uvs_u[RAC_MESH_MAX_VERTICES], uvs_v[RAC_MESH_MAX_VERTICES];
    int np = 0, nn = 0, nt = 0;

    int mesh_id = rac_mesh_create(reg, RAC_MESH_MAX_VERTICES, RAC_MESH_MAX_INDICES);
    if (mesh_id < 0) { fclose(f); return -1; }
    rac_mesh *mesh = &reg->meshes[mesh_id];

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3 && np < RAC_MESH_MAX_VERTICES)
                positions[np++] = rac_phys_v3(x, y, z);
        } else if (line[0] == 'v' && line[1] == 'n') {
            float x, y, z;
            if (sscanf(line + 3, "%f %f %f", &x, &y, &z) == 3 && nn < RAC_MESH_MAX_VERTICES)
                normals[nn++] = rac_phys_v3(x, y, z);
        } else if (line[0] == 'v' && line[1] == 't') {
            float u, v;
            if (sscanf(line + 3, "%f %f", &u, &v) == 2 && nt < RAC_MESH_MAX_VERTICES) {
                uvs_u[nt] = u;
                uvs_v[nt] = v;
                nt++;
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            /* Parse face: f v/vt/vn v/vt/vn v/vt/vn */
            int vi[4], ti[4], ni[4];
            int nv = 0;
            char *ptr = line + 2;

            while (nv < 4) {
                int v_idx = 0, t_idx = 0, n_idx = 0;
                int parsed = 0;

                if (sscanf(ptr, "%d/%d/%d%n", &v_idx, &t_idx, &n_idx, &parsed) >= 3) {
                } else if (sscanf(ptr, "%d//%d%n", &v_idx, &n_idx, &parsed) >= 2) {
                    t_idx = 0;
                } else if (sscanf(ptr, "%d/%d%n", &v_idx, &t_idx, &parsed) >= 2) {
                    n_idx = 0;
                } else if (sscanf(ptr, "%d%n", &v_idx, &parsed) >= 1) {
                    t_idx = 0; n_idx = 0;
                } else {
                    break;
                }

                vi[nv] = v_idx - 1;
                ti[nv] = t_idx - 1;
                ni[nv] = n_idx - 1;
                nv++;
                ptr += parsed;
                while (*ptr == ' ') ptr++;
                if (*ptr == '\0' || *ptr == '\n') break;
            }

            /* Emit vertices (triangulate quads) */
            for (int t = 0; t < nv - 2; t++) {
                int face[3] = { 0, t + 1, t + 2 };
                for (int fi = 0; fi < 3; fi++) {
                    int idx = face[fi];
                    if (mesh->num_vertices >= RAC_MESH_MAX_VERTICES) break;

                    rac_vertex vert;
                    memset(&vert, 0, sizeof(vert));
                    if (vi[idx] >= 0 && vi[idx] < np)
                        vert.position = positions[vi[idx]];
                    if (ni[idx] >= 0 && ni[idx] < nn)
                        vert.normal = normals[ni[idx]];
                    if (ti[idx] >= 0 && ti[idx] < nt) {
                        vert.u = uvs_u[ti[idx]];
                        vert.v = uvs_v[ti[idx]];
                    }

                    mesh->indices[mesh->num_indices++] = mesh->num_vertices;
                    mesh->vertices[mesh->num_vertices++] = vert;
                }
            }
        }
    }

    fclose(f);
    rac_mesh_compute_aabb(mesh);

    if (nn == 0)
        rac_mesh_compute_normals(mesh);

    return mesh_id;
}

/* ── Procedural generators ─────────────────────────────────────────────── */

int rac_mesh_gen_cube(rac_mesh_registry *reg, float size)
{
    int id = rac_mesh_create(reg, 24, 36);
    if (id < 0) return -1;
    rac_mesh *m = &reg->meshes[id];

    float h = size * 0.5f;

    /* 6 faces, 4 verts each = 24 verts */
    typedef struct { float x,y,z, nx,ny,nz, u,v; } fv;
    fv data[24] = {
        /* Front (+Z) */
        {-h,-h, h,  0, 0, 1,  0,0}, { h,-h, h,  0, 0, 1,  1,0},
        { h, h, h,  0, 0, 1,  1,1}, {-h, h, h,  0, 0, 1,  0,1},
        /* Back (-Z) */
        { h,-h,-h,  0, 0,-1,  0,0}, {-h,-h,-h,  0, 0,-1,  1,0},
        {-h, h,-h,  0, 0,-1,  1,1}, { h, h,-h,  0, 0,-1,  0,1},
        /* Right (+X) */
        { h,-h, h,  1, 0, 0,  0,0}, { h,-h,-h,  1, 0, 0,  1,0},
        { h, h,-h,  1, 0, 0,  1,1}, { h, h, h,  1, 0, 0,  0,1},
        /* Left (-X) */
        {-h,-h,-h, -1, 0, 0,  0,0}, {-h,-h, h, -1, 0, 0,  1,0},
        {-h, h, h, -1, 0, 0,  1,1}, {-h, h,-h, -1, 0, 0,  0,1},
        /* Top (+Y) */
        {-h, h, h,  0, 1, 0,  0,0}, { h, h, h,  0, 1, 0,  1,0},
        { h, h,-h,  0, 1, 0,  1,1}, {-h, h,-h,  0, 1, 0,  0,1},
        /* Bottom (-Y) */
        {-h,-h,-h,  0,-1, 0,  0,0}, { h,-h,-h,  0,-1, 0,  1,0},
        { h,-h, h,  0,-1, 0,  1,1}, {-h,-h, h,  0,-1, 0,  0,1},
    };

    for (int i = 0; i < 24; i++) {
        m->vertices[i].position = rac_phys_v3(data[i].x, data[i].y, data[i].z);
        m->vertices[i].normal = rac_phys_v3(data[i].nx, data[i].ny, data[i].nz);
        m->vertices[i].u = data[i].u;
        m->vertices[i].v = data[i].v;
    }
    m->num_vertices = 24;

    /* 6 faces * 2 triangles * 3 indices = 36 */
    for (int face = 0; face < 6; face++) {
        int base = face * 4;
        m->indices[face * 6 + 0] = base + 0;
        m->indices[face * 6 + 1] = base + 1;
        m->indices[face * 6 + 2] = base + 2;
        m->indices[face * 6 + 3] = base + 0;
        m->indices[face * 6 + 4] = base + 2;
        m->indices[face * 6 + 5] = base + 3;
    }
    m->num_indices = 36;

    rac_mesh_compute_aabb(m);
    return id;
}

int rac_mesh_gen_sphere(rac_mesh_registry *reg, float radius, int rings, int sectors)
{
    int nverts = (rings + 1) * (sectors + 1);
    int nidx = rings * sectors * 6;
    int id = rac_mesh_create(reg, nverts, nidx);
    if (id < 0) return -1;
    rac_mesh *m = &reg->meshes[id];

    /* Generate vertices using rac_rotate for sin/cos */
    for (int r = 0; r <= rings; r++) {
        /* phi = pi * r / rings — latitude angle */
        float phi = RAC_PI * (float)r / (float)rings;
        rac_vec2 phi_sc = rac_rotate((rac_vec2){1.0f, 0.0f}, phi);
        /* phi_sc.x = cos(phi), phi_sc.y = sin(phi) */
        float cos_phi = phi_sc.x;
        float sin_phi = phi_sc.y;

        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * RAC_PI * (float)s / (float)sectors;
            rac_vec2 th_sc = rac_rotate((rac_vec2){1.0f, 0.0f}, theta);
            float cos_th = th_sc.x;
            float sin_th = th_sc.y;

            rac_phys_vec3 n = rac_phys_v3(
                sin_phi * cos_th,
                cos_phi,
                sin_phi * sin_th
            );

            int idx = r * (sectors + 1) + s;
            m->vertices[idx].position = rac_phys_v3_scale(n, radius);
            m->vertices[idx].normal = n;
            m->vertices[idx].u = (float)s / (float)sectors;
            m->vertices[idx].v = (float)r / (float)rings;
        }
    }
    m->num_vertices = nverts;

    /* Generate indices */
    int ii = 0;
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < sectors; s++) {
            int cur = r * (sectors + 1) + s;
            int next = cur + sectors + 1;
            m->indices[ii++] = cur;
            m->indices[ii++] = next;
            m->indices[ii++] = cur + 1;
            m->indices[ii++] = cur + 1;
            m->indices[ii++] = next;
            m->indices[ii++] = next + 1;
        }
    }
    m->num_indices = ii;

    rac_mesh_compute_aabb(m);
    return id;
}

int rac_mesh_gen_plane(rac_mesh_registry *reg, float width, float depth, int subdiv)
{
    int n = subdiv + 1;
    int nverts = n * n;
    int nidx = subdiv * subdiv * 6;
    int id = rac_mesh_create(reg, nverts, nidx);
    if (id < 0) return -1;
    rac_mesh *m = &reg->meshes[id];

    float hw = width * 0.5f, hd = depth * 0.5f;

    for (int z = 0; z < n; z++) {
        for (int x = 0; x < n; x++) {
            int idx = z * n + x;
            float fx = -hw + width * (float)x / (float)subdiv;
            float fz = -hd + depth * (float)z / (float)subdiv;
            m->vertices[idx].position = rac_phys_v3(fx, 0.0f, fz);
            m->vertices[idx].normal = rac_phys_v3(0.0f, 1.0f, 0.0f);
            m->vertices[idx].u = (float)x / (float)subdiv;
            m->vertices[idx].v = (float)z / (float)subdiv;
        }
    }
    m->num_vertices = nverts;

    int ii = 0;
    for (int z = 0; z < subdiv; z++) {
        for (int x = 0; x < subdiv; x++) {
            int tl = z * n + x;
            m->indices[ii++] = tl;
            m->indices[ii++] = tl + n;
            m->indices[ii++] = tl + 1;
            m->indices[ii++] = tl + 1;
            m->indices[ii++] = tl + n;
            m->indices[ii++] = tl + n + 1;
        }
    }
    m->num_indices = ii;

    rac_mesh_compute_aabb(m);
    return id;
}

int rac_mesh_gen_cylinder(rac_mesh_registry *reg, float radius, float height, int segments)
{
    /* Side + top + bottom caps */
    int nverts = (segments + 1) * 2 + 2 + (segments + 1) * 2;
    int nidx = segments * 6 + segments * 3 * 2;  /* side quads + cap tris */
    int id = rac_mesh_create(reg, nverts, nidx);
    if (id < 0) return -1;
    rac_mesh *m = &reg->meshes[id];

    float hh = height * 0.5f;
    int vi = 0, ii = 0;

    /* Side vertices */
    for (int i = 0; i <= segments; i++) {
        float theta = 2.0f * RAC_PI * (float)i / (float)segments;
        rac_vec2 sc = rac_rotate((rac_vec2){1.0f, 0.0f}, theta);
        rac_phys_vec3 n = rac_phys_v3(sc.y, 0.0f, sc.x);

        /* Top ring */
        m->vertices[vi].position = rac_phys_v3(sc.y * radius, hh, sc.x * radius);
        m->vertices[vi].normal = n;
        m->vertices[vi].u = (float)i / (float)segments;
        m->vertices[vi].v = 0.0f;
        vi++;

        /* Bottom ring */
        m->vertices[vi].position = rac_phys_v3(sc.y * radius, -hh, sc.x * radius);
        m->vertices[vi].normal = n;
        m->vertices[vi].u = (float)i / (float)segments;
        m->vertices[vi].v = 1.0f;
        vi++;
    }

    /* Side indices */
    for (int i = 0; i < segments; i++) {
        int top = i * 2, bot = top + 1;
        int ntop = (i + 1) * 2, nbot = ntop + 1;
        m->indices[ii++] = top;
        m->indices[ii++] = bot;
        m->indices[ii++] = ntop;
        m->indices[ii++] = ntop;
        m->indices[ii++] = bot;
        m->indices[ii++] = nbot;
    }

    /* Top cap center */
    int tc = vi;
    m->vertices[vi].position = rac_phys_v3(0.0f, hh, 0.0f);
    m->vertices[vi].normal = rac_phys_v3(0.0f, 1.0f, 0.0f);
    vi++;

    /* Bottom cap center */
    int bc = vi;
    m->vertices[vi].position = rac_phys_v3(0.0f, -hh, 0.0f);
    m->vertices[vi].normal = rac_phys_v3(0.0f, -1.0f, 0.0f);
    vi++;

    /* Cap vertices and triangles */
    int top_ring_start = vi;
    for (int i = 0; i <= segments; i++) {
        float theta = 2.0f * RAC_PI * (float)i / (float)segments;
        rac_vec2 sc = rac_rotate((rac_vec2){1.0f, 0.0f}, theta);
        m->vertices[vi].position = rac_phys_v3(sc.y * radius, hh, sc.x * radius);
        m->vertices[vi].normal = rac_phys_v3(0.0f, 1.0f, 0.0f);
        vi++;
    }
    for (int i = 0; i < segments; i++) {
        m->indices[ii++] = tc;
        m->indices[ii++] = top_ring_start + i;
        m->indices[ii++] = top_ring_start + i + 1;
    }

    int bot_ring_start = vi;
    for (int i = 0; i <= segments; i++) {
        float theta = 2.0f * RAC_PI * (float)i / (float)segments;
        rac_vec2 sc = rac_rotate((rac_vec2){1.0f, 0.0f}, theta);
        m->vertices[vi].position = rac_phys_v3(sc.y * radius, -hh, sc.x * radius);
        m->vertices[vi].normal = rac_phys_v3(0.0f, -1.0f, 0.0f);
        vi++;
    }
    for (int i = 0; i < segments; i++) {
        m->indices[ii++] = bc;
        m->indices[ii++] = bot_ring_start + i + 1;
        m->indices[ii++] = bot_ring_start + i;
    }

    m->num_vertices = vi;
    m->num_indices = ii;
    rac_mesh_compute_aabb(m);
    return id;
}
