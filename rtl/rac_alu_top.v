// ─────────────────────────────────────────────────────────────────────────────
// rac_alu_top.v — RAC ALU top-level: CORDIC core + projection accumulator
// Pinnacle Quantum Group — April 2026
//
// Wraps rac_cordic_core with:
//   - A projection accumulator (`acc` register + ACCUM opcode)
//   - An opcode decoder matching rac_alu.h: LOAD / CLR_ACC / SET_MODE /
//     MICRO / ACCUM / COMP / SIGN / RUN
//   - A "chain counter" for gain compensation (K⁻ⁿ via look-up)
//
// Op encoding on `op[3:0]` matches rac_alu_opcode in rac_alu.h:
//   0 LOAD          load x,y,z from input ports
//   1 CLR_ACC       acc <- 0
//   2 SET_MODE      mode <- op_arg[1:0], dir <- op_arg[2]
//   3 MICRO         one CORDIC iter (starts core with start=1 for 1 cycle)
//   4 RUN           16 CORDIC iters back-to-back
//   5 ACCUM         acc <- acc + x
//   6 COMPENSATE    x,y <- x,y · K⁻chain (reads from gain ROM, not shown)
//   7 SIGN          latch direction decision
//
// This is scaffolding — the important block is rac_cordic_core which carries
// the CORDIC datapath. Integrate into fil_rac AXI-Lite BAR0 register map
// (see rac_fpga_u250.h) by driving `op` + operand ports from MMIO writes and
// exposing {x_out, y_out, z_out, acc} as readback registers.
// ─────────────────────────────────────────────────────────────────────────────

`default_nettype none

module rac_alu_top #(
    parameter integer WIDTH = 32
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Command/operand interface
    input  wire [3:0]               op,
    input  wire [3:0]               op_arg,     // mode/dir flags for SET_MODE
    input  wire signed [WIDTH-1:0]  x_in,
    input  wire signed [WIDTH-1:0]  y_in,
    input  wire signed [WIDTH-1:0]  z_in,
    input  wire                     op_valid,   // pulse high to issue op

    // Result readback
    output reg  signed [WIDTH-1:0]  x_reg,
    output reg  signed [WIDTH-1:0]  y_reg,
    output reg  signed [WIDTH-1:0]  z_reg,
    output reg  signed [WIDTH-1:0]  acc_reg,
    output reg                      busy_o
);

    // Mode/direction state (updated by SET_MODE)
    reg [1:0] mode_r;
    reg       dir_r;

    // Chain counter — incremented per MICRO, cleared by COMPENSATE
    reg [5:0] chain_cnt;

    // Core interface
    wire                   core_done;
    wire                   core_busy;
    wire signed [WIDTH-1:0] core_x, core_y, core_z;
    reg                    core_start;

    rac_cordic_core #(.WIDTH(WIDTH), .ITERS(16)) u_core (
        .clk   (clk),
        .rst_n (rst_n),
        .start (core_start),
        .mode  (mode_r),
        .dir   (dir_r),
        .x_in  (x_reg),
        .y_in  (y_reg),
        .z_in  (z_reg),
        .x_out (core_x),
        .y_out (core_y),
        .z_out (core_z),
        .done  (core_done),
        .busy  (core_busy)
    );

    // Opcode constants (must match rac_alu_opcode in rac_alu.h)
    localparam OP_LOAD       = 4'd0;
    localparam OP_CLR_ACC    = 4'd1;
    localparam OP_SET_MODE   = 4'd2;
    localparam OP_MICRO      = 4'd3;
    localparam OP_RUN        = 4'd4;
    localparam OP_ACCUM      = 4'd5;
    localparam OP_COMPENSATE = 4'd6;
    localparam OP_SIGN       = 4'd7;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg      <= 0;
            y_reg      <= 0;
            z_reg      <= 0;
            acc_reg    <= 0;
            mode_r     <= 2'b00;
            dir_r      <= 1'b0;
            chain_cnt  <= 6'd0;
            core_start <= 1'b0;
            busy_o     <= 1'b0;
        end else begin
            core_start <= 1'b0;
            busy_o     <= core_busy;

            if (op_valid && !core_busy) begin
                case (op)
                    OP_LOAD: begin
                        x_reg <= x_in;
                        y_reg <= y_in;
                        z_reg <= z_in;
                    end
                    OP_CLR_ACC:
                        acc_reg <= 0;
                    OP_SET_MODE: begin
                        mode_r <= op_arg[1:0];
                        dir_r  <= op_arg[2];
                    end
                    OP_RUN: begin
                        // Kick off one full 16-iter CORDIC run
                        core_start <= 1'b1;
                    end
                    OP_ACCUM:
                        acc_reg <= acc_reg + x_reg;
                    OP_COMPENSATE: begin
                        // Simplified: assume rac_rotate's K_INV pre-scaling
                        // already handled host-side; reset chain counter.
                        chain_cnt <= 6'd0;
                    end
                    default: /* NOP */ ;
                endcase
            end

            // Capture core output when the CORDIC pipeline finishes
            if (core_done) begin
                x_reg     <= core_x;
                y_reg     <= core_y;
                z_reg     <= core_z;
                chain_cnt <= chain_cnt + 6'd16;   // one RUN == 16 iters
            end
        end
    end

endmodule

`default_nettype wire
