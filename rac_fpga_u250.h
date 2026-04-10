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

#ifdef __cplusplus
}
#endif

#endif /* RAC_FPGA_U250_H */
