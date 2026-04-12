// ─────────────────────────────────────────────────────────────────────────────
// rac_cordic_core.v — Synthesizable CORDIC core for the RAC ALU
// Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
//
// This module is a direct hardware translation of the inline functions in
// lib/c/rac_alu.c: _alu_run_circ_rot, _alu_run_circ_vec, _alu_run_hyp_rot.
// The iteration ROM is identical (atan_table, atanh_table). The state
// variables (x, y, z) are Q16.16 signed fixed-point — matching the
// rac_fpga_u250.h register interface and the Xilinx Alveo U250 backend.
//
// Operation:
//   - When `start` rises with `mode`/`dir` valid, the module latches
//     (x_in, y_in, z_in) into registers and enters the ITER state.
//   - Each subsequent clock performs one CORDIC micro-step. After 16
//     iterations (or 17 for hyperbolic — iter 4 and 13 repeat, so the
//     counter slips), `done` rises for one cycle and the outputs become
//     valid on the same edge.
//   - `done` drops on the next `start` or can be held with a simple
//     external latch.
//
// Throughput: one CORDIC result every 16-17 cycles. Pipelined variant
// (one result/cycle after a 16-cycle fill) is a mechanical unroll — see
// the rac_cordic_pipe.v derivation in README.md.
//
// Resource estimate (Xilinx Alveo U250, XCU250):
//   3 × 32-bit adders   (x, y, z path; shifters fold into LUTs)
//   4 × 32-bit registers
//   ~200 LUTs + ~120 FFs, ~200 MHz on -2 speed grade
//   Two ROMs (atan, atanh): 16 × 32 b each, implemented as distributed RAM
//
// Formal equivalence with rac_alu.c is established in
// lib/c/test_rac_xrac.c (3-way ALU ↔ microcode ↔ Xrac ISS equivalence);
// the RTL must match the same Q16.16 atan table entries.
// ─────────────────────────────────────────────────────────────────────────────

`default_nettype none

module rac_cordic_core #(
    parameter integer ITERS = 16,
    parameter integer WIDTH = 32
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Control
    input  wire                     start,       // one-cycle pulse
    input  wire [1:0]               mode,        // 00=CIRC, 01=HYP  (bit[1] reserved)
    input  wire                     dir,         // 0=ROTATE, 1=VECTORING

    // Operands (Q16.16 signed)
    input  wire signed [WIDTH-1:0]  x_in,
    input  wire signed [WIDTH-1:0]  y_in,
    input  wire signed [WIDTH-1:0]  z_in,

    // Results
    output reg  signed [WIDTH-1:0]  x_out,
    output reg  signed [WIDTH-1:0]  y_out,
    output reg  signed [WIDTH-1:0]  z_out,
    output reg                      done,
    output reg                      busy
);

    // ── Lookup tables (Q16.16) ──────────────────────────────────────────
    // atan(2^-i) × 2^16, i = 0 .. 15.
    // Values match lib/c/rac_alu.c::_alu_atan_table to ULP.
    //   atan(1)       = 0.7853981633974483  →  0x0000C90F
    //   atan(1/2)     = 0.4636476090008061  →  0x000076B1
    //   atan(1/4)     = 0.24497866312686414 →  0x00003EB6
    //   ...
    wire signed [WIDTH-1:0] atan_tbl [0:ITERS-1];
    assign atan_tbl[ 0] = 32'sh0000C90F;
    assign atan_tbl[ 1] = 32'sh000076B1;
    assign atan_tbl[ 2] = 32'sh00003EB6;
    assign atan_tbl[ 3] = 32'sh00001FD5;
    assign atan_tbl[ 4] = 32'sh00000FFA;
    assign atan_tbl[ 5] = 32'sh000007FF;
    assign atan_tbl[ 6] = 32'sh000003FF;
    assign atan_tbl[ 7] = 32'sh000001FF;
    assign atan_tbl[ 8] = 32'sh000000FF;
    assign atan_tbl[ 9] = 32'sh0000007F;
    assign atan_tbl[10] = 32'sh0000003F;
    assign atan_tbl[11] = 32'sh0000001F;
    assign atan_tbl[12] = 32'sh0000000F;
    assign atan_tbl[13] = 32'sh00000007;
    assign atan_tbl[14] = 32'sh00000003;
    assign atan_tbl[15] = 32'sh00000001;

    // atanh table (Q16.16) and physical shift index for hyperbolic mode.
    // Sequence honours Walther's iter-repeat at 4 and 13.
    //   iter i  →  shift = iter_map[i]  (1..14)
    //   atanh(2^-shift) × 2^16
    wire signed [WIDTH-1:0] atanh_tbl [0:ITERS-1];
    wire [4:0] hyp_shift [0:ITERS-1];
    assign atanh_tbl[ 0] = 32'sh00008C9F;  assign hyp_shift[ 0] = 5'd1;
    assign atanh_tbl[ 1] = 32'sh00004162;  assign hyp_shift[ 1] = 5'd2;
    assign atanh_tbl[ 2] = 32'sh00002027;  assign hyp_shift[ 2] = 5'd3;
    assign atanh_tbl[ 3] = 32'sh00001005;  assign hyp_shift[ 3] = 5'd4;
    assign atanh_tbl[ 4] = 32'sh00001005;  assign hyp_shift[ 4] = 5'd4;   // repeat
    assign atanh_tbl[ 5] = 32'sh00000801;  assign hyp_shift[ 5] = 5'd5;
    assign atanh_tbl[ 6] = 32'sh00000400;  assign hyp_shift[ 6] = 5'd6;
    assign atanh_tbl[ 7] = 32'sh00000200;  assign hyp_shift[ 7] = 5'd7;
    assign atanh_tbl[ 8] = 32'sh00000100;  assign hyp_shift[ 8] = 5'd8;
    assign atanh_tbl[ 9] = 32'sh00000080;  assign hyp_shift[ 9] = 5'd9;
    assign atanh_tbl[10] = 32'sh00000040;  assign hyp_shift[10] = 5'd10;
    assign atanh_tbl[11] = 32'sh00000020;  assign hyp_shift[11] = 5'd11;
    assign atanh_tbl[12] = 32'sh00000010;  assign hyp_shift[12] = 5'd12;
    assign atanh_tbl[13] = 32'sh00000008;  assign hyp_shift[13] = 5'd13;
    assign atanh_tbl[14] = 32'sh00000008;  assign hyp_shift[14] = 5'd13;  // repeat
    assign atanh_tbl[15] = 32'sh00000004;  assign hyp_shift[15] = 5'd14;

    // ── State ──────────────────────────────────────────────────────────
    reg signed [WIDTH-1:0] x_r, y_r, z_r;
    reg        [4:0]       iter;        // 0 .. ITERS-1
    reg        [1:0]       mode_r;
    reg                    dir_r;

    // Direction bit d ∈ {+1, -1}: choose from sign(z) (rotation mode)
    // or sign(-y) (vectoring mode). Branchless: check MSB.
    wire d_pos = (dir_r == 1'b0)
                 ? (z_r[WIDTH-1] == 1'b0)   // rotation: z >= 0
                 : (y_r[WIDTH-1] == 1'b1);  // vectoring: y < 0

    // Shift amount: hyperbolic uses the iter_map; circular uses iter itself.
    wire [4:0] shift_amt = (mode_r[0] == 1'b1)   // mode[0]=1 → hyperbolic
                           ? hyp_shift[iter]
                           : iter[4:0];

    // Table value to subtract from z
    wire signed [WIDTH-1:0] tbl_val = (mode_r[0] == 1'b1)
                                      ? atanh_tbl[iter]
                                      : atan_tbl[iter];

    // Shifted datapath (arithmetic right shift — sign extending)
    wire signed [WIDTH-1:0] x_sh = x_r >>> shift_amt;
    wire signed [WIDTH-1:0] y_sh = y_r >>> shift_amt;

    // ── FSM ────────────────────────────────────────────────────────────
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_r    <= {WIDTH{1'b0}};
            y_r    <= {WIDTH{1'b0}};
            z_r    <= {WIDTH{1'b0}};
            iter   <= 5'd0;
            mode_r <= 2'b00;
            dir_r  <= 1'b0;
            busy   <= 1'b0;
            done   <= 1'b0;
            x_out  <= {WIDTH{1'b0}};
            y_out  <= {WIDTH{1'b0}};
            z_out  <= {WIDTH{1'b0}};
        end else if (start && !busy) begin
            // Load operands
            x_r    <= x_in;
            y_r    <= y_in;
            z_r    <= z_in;
            mode_r <= mode;
            dir_r  <= dir;
            iter   <= 5'd0;
            busy   <= 1'b1;
            done   <= 1'b0;
        end else if (busy) begin
            // One micro-step per cycle
            // d · shifted is either +shifted or -shifted — a sign flip.
            // Circular: x' = x - d·y_sh,  y' = y + d·x_sh,  z' = z - d·tbl
            // Hyperbolic: x' = x + d·y_sh,  y' = y + d·x_sh,  z' = z - d·tbl
            if (mode_r[0] == 1'b1) begin
                // Hyperbolic
                if (d_pos) begin
                    x_r <= x_r + y_sh;
                    y_r <= y_r + x_sh;
                    z_r <= z_r - tbl_val;
                end else begin
                    x_r <= x_r - y_sh;
                    y_r <= y_r - x_sh;
                    z_r <= z_r + tbl_val;
                end
            end else begin
                // Circular
                if (d_pos) begin
                    x_r <= x_r - y_sh;
                    y_r <= y_r + x_sh;
                    z_r <= z_r - tbl_val;
                end else begin
                    x_r <= x_r + y_sh;
                    y_r <= y_r - x_sh;
                    z_r <= z_r + tbl_val;
                end
            end

            if (iter == ITERS - 1) begin
                busy  <= 1'b0;
                done  <= 1'b1;
                // Commit outputs on the final-iter edge
                x_out <= (mode_r[0] == 1'b1)
                         ? (d_pos ? x_r + y_sh : x_r - y_sh)
                         : (d_pos ? x_r - y_sh : x_r + y_sh);
                y_out <= (d_pos ? y_r + x_sh : y_r - x_sh);
                z_out <= (d_pos ? z_r - tbl_val : z_r + tbl_val);
            end else begin
                iter <= iter + 5'd1;
            end
        end else begin
            done <= 1'b0;
        end
    end

endmodule

`default_nettype wire
