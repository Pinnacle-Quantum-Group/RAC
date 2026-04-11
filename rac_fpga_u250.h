#pragma once
#ifndef RAC_FPGA_U250_H
#define RAC_FPGA_U250_H

/*
 * rac_fpga_u250.h — RAC FPGA Backend Driver for Xilinx Alveo U250
 * Pinnacle Quantum Group — April 2026
 *
 * Host-side interface for the FIL-RAC VHDL accelerator on the Alveo U250.
 * All 17 RAC primitives are offloaded to hardware CORDIC engines via
 * PCIe Gen3 x16 XDMA with AXI-Lite BAR0 register access.
 *
 * FPGA design:  FIL/fil_rac/  (TDD/GHDL-first VHDL implementation)
 * Target:       xcu250-figd2104-2L-e, 300 MHz, 8 CORDIC engines
 * Interface:    PCIe BAR0 AXI-Lite registers (4 KB per engine)
 *
 * Usage:
 *   rac_context ctx = rac_create_context(RAC_BACKEND_FIL);
 *   // ... use standard rac_* API — dispatches to FPGA automatically
 *   rac_destroy_context(ctx);
 *
 * Build:
 *   Link with -lxdma or use the XDMA userspace driver.
 *   FPGA must be programmed with fil_rac bitstream before use.
 */

#include "rac.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── BAR0 Register Map (matches fil_rac_pkg.vhd) ────────────────────────── */

#define RAC_FPGA_REG_CTRL          0x000
#define RAC_FPGA_REG_STATUS        0x004
#define RAC_FPGA_REG_VERSION       0x008
#define RAC_FPGA_REG_OP            0x010
#define RAC_FPGA_REG_OPERAND_X     0x014
#define RAC_FPGA_REG_OPERAND_Y     0x018
#define RAC_FPGA_REG_THETA         0x01C
#define RAC_FPGA_REG_RESULT_X      0x020
#define RAC_FPGA_REG_RESULT_Y      0x024
#define RAC_FPGA_REG_RESULT_AUX    0x028
#define RAC_FPGA_REG_VEC_LEN       0x030
#define RAC_FPGA_REG_DMA_SRC       0x034
#define RAC_FPGA_REG_DMA_DST       0x038

/* CTRL register bits */
#define RAC_FPGA_CTRL_GO           (1 << 0)
#define RAC_FPGA_CTRL_RST_ENGINE   (1 << 1)

/* STATUS register bits */
#define RAC_FPGA_STATUS_BUSY       (1 << 0)
#define RAC_FPGA_STATUS_DONE       (1 << 1)
#define RAC_FPGA_STATUS_ERROR      (1 << 2)

/* Version: major.minor.patch as 8.8.16 */
#define RAC_FPGA_VERSION_EXPECTED  0x01000001

/* Per-engine BAR0 address stride (4 KB) */
#define RAC_FPGA_ENGINE_STRIDE     0x1000

/* Number of engines in default bitstream */
#define RAC_FPGA_NUM_ENGINES       8

/* ── Q16.16 conversion helpers ───────────────────────────────────────────── */

static inline int32_t rac_fpga_float_to_q16(float f) {
    return (int32_t)(f * 65536.0f);
}

static inline float rac_fpga_q16_to_float(int32_t q) {
    return (float)q / 65536.0f;
}

/* ── Device management ───────────────────────────────────────────────────── */

/*
 * rac_fpga_open: Open XDMA device and mmap BAR0.
 * Returns: file descriptor, or -1 on error.
 * device_path: e.g., "/dev/xdma0_user" (XDMA userspace char device)
 */
int rac_fpga_open(const char *device_path);

/*
 * rac_fpga_close: Unmap BAR0 and close device.
 */
void rac_fpga_close(int fd);

/*
 * rac_fpga_read_version: Read and verify FPGA version register.
 * Returns: version word, or 0 on mismatch/error.
 */
uint32_t rac_fpga_read_version(int fd);

/*
 * rac_fpga_dispatch: Execute a single RAC operation on an FPGA engine.
 *
 * engine_id: 0..NUM_ENGINES-1
 * op:        RAC opcode (RAC_OP_ROTATE, etc.)
 * x, y, theta: operands in float32 (converted to Q16.16 internally)
 * result_x, result_y, result_aux: output pointers (float32, converted from Q16.16)
 *
 * Returns: 0 on success, -1 on error/timeout.
 */
int rac_fpga_dispatch(int fd, int engine_id, rac_op_type op,
                      float x, float y, float theta,
                      float *result_x, float *result_y, float *result_aux);

/* ── FIL Tomasulo Dispatch Interface ─────────────────────────────────────────
 *
 * Geometric out-of-order execution via DSP48E2 cascade columns.
 * Port of FIL/research/FILEngine/fil_engine.c to VHDL.
 * VHDL implementation: FIL/fil_rac/rtl/fil_tomasulo.vhd
 *
 * Architecture: 4 DSP cascade columns (1 per SLR on U250), phi-cell
 * reservation stations with GlueCert dependency tracking, four-phase
 * execution (Issue/Broadcast/Execute/Commit), holonomy return path.
 */

/* AXI-Stream work input packing (128-bit tdata):
 *   [31:0]    x_vec   (Q16.16)
 *   [63:32]   y_vec   (Q16.16)
 *   [95:64]   angle   (Q16.16)
 *   [103:96]  work_id (8-bit tag)
 *   [127:104] reserved
 *
 * tuser[11:0]:
 *   [3:0]   edge_mask
 *   [7:4]   twist_mask
 *   [9:8]   col       (cascade column 0-3)
 *   [11:10] entry     (entry point 0-3)
 */

#define RAC_FIL_NUM_COLS       4
#define RAC_FIL_NUM_ENTRIES    4
#define RAC_FIL_MAX_PHI_CELLS  16
#define RAC_FIL_MAX_GLUE_CERTS 32

/* Phi-cell edge identifiers (matches VHDL FIL_EDGE_*) */
#define RAC_FIL_EDGE_LEFT   0
#define RAC_FIL_EDGE_RIGHT  1
#define RAC_FIL_EDGE_UP     2
#define RAC_FIL_EDGE_DOWN   3

/* Dispatch phase encoding (from phase_out[2:0]) */
#define RAC_FIL_PHASE_IDLE         0
#define RAC_FIL_PHASE_A            1  /* Interior (Issue) */
#define RAC_FIL_PHASE_B_PACK       2  /* Pack halos */
#define RAC_FIL_PHASE_B_IMPORT     3  /* Import halos (Broadcast) */
#define RAC_FIL_PHASE_C            4  /* Boundary (Execute) */
#define RAC_FIL_PHASE_D            5  /* Commit (Write Result) */
#define RAC_FIL_PHASE_HOLONOMY     6  /* Holonomy return */
#define RAC_FIL_PHASE_DONE         7  /* Epoch complete */

/*
 * rac_fil_issue: Submit a work item to the Tomasulo dispatch fabric.
 *
 * work_id:    8-bit tag returned with commit result
 * x, y:       operand vector (float32 → Q16.16)
 * angle:      rotation angle (float32 → Q16.16)
 * col:        target cascade column (0..3)
 * entry:      entry point within column (0..3)
 * edge_mask:  exposed edges bitmask (bit 0=L, 1=R, 2=U, 3=D)
 * twist_mask: Mobius twist flags per edge
 *
 * Returns: 0 on accept, -1 if backpressured (retry later).
 */
int rac_fil_issue(int fd, uint8_t work_id,
                  float x, float y, float angle,
                  int col, int entry,
                  uint8_t edge_mask, uint8_t twist_mask);

/*
 * rac_fil_load_cert: Load a GlueCert dependency entry.
 *
 * index:   cert table slot (0..31)
 * cell_a:  destination phi-cell (0..15)
 * cell_b:  source phi-cell (0..15)
 * edge_a:  destination edge (RAC_FIL_EDGE_*)
 * edge_b:  source edge (RAC_FIL_EDGE_*)
 * orient:  0=forward, 1=reversed
 * twist:   0=identity, 1=negate (Mobius seam)
 */
int rac_fil_load_cert(int fd, int index,
                      int cell_a, int cell_b,
                      int edge_a, int edge_b,
                      int orient, int twist);

/*
 * rac_fil_wait_commit: Block until a commit result is available.
 *
 * work_id:     returned work tag
 * accum:       accumulated result (Q16.16 → float32)
 * twist_count: holonomy twist count
 *
 * Returns: 0 on success, -1 on timeout.
 */
int rac_fil_wait_commit(int fd, uint8_t *work_id,
                        float *accum, uint16_t *twist_count);

/*
 * rac_fil_get_phase: Read current dispatch phase.
 * Returns: RAC_FIL_PHASE_* value.
 */
int rac_fil_get_phase(int fd);

#ifdef __cplusplus
}
#endif

#endif /* RAC_FPGA_U250_H */
