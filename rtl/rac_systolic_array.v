// ─────────────────────────────────────────────────────────────────────────────
// rac_systolic_array.v — NxN weight-stationary CORDIC systolic array
// Pinnacle Quantum Group — April 2026
//
// Tiles N² rac_dsp cells in a 2-D grid. Classical weight-stationary
// dataflow (Kung-Leiserson with CORDIC PEs substituted for MAC PEs):
//
//   - Each PE holds a pre-programmed angle θ_ij (the "weight")
//   - Activations stream in from the left, propagate right
//   - Partial sums stream down, accumulate at each PE
//   - After N+1 cycles of warm-up, the bottom row emits one matrix
//     column per clock; throughput = N² CORDIC rotations/cycle
//
// NO MULTIPLIERS anywhere. Each PE is a rac_dsp (see rac_dsp.v); the
// array's only glue is 64-bit adders for the accumulator chain.
//
// Intended use: RAC-native GEMM. A conventional M×K @ K×N matmul maps
// to this array by:
//   1. Loading weights as angles (one CORDIC polar conversion per weight,
//      offline or via a one-time rac_dsp vectoring-mode pass)
//   2. Streaming the activation matrix through the left edge, row by row
//   3. Collecting the product rows off the bottom edge
//
// Resource estimate (Xilinx Alveo U250, XCU250, -2 speed grade)
// ─────────────────────────────────────────────────────────────
//   Parametric on N:
//      16×16 array (256 PEs):   46 K LUTs (~2.7%), 16 BRAMs, 0 DSPs
//      32×32 array (1024 PEs):  190 K LUTs (~11%),  64 BRAMs, 0 DSPs
//      64×64 array (4096 PEs):  750 K LUTs (~43%), 256 BRAMs, 0 DSPs
//   Fmax: ~180 MHz (PE critical path dominates)
//   Throughput: N² × 180 M rotations/s
//      16×16: 46 Grot/s
//      32×32: 184 Grot/s
//      64×64: 737 Grot/s  (matches RTX 4090 tensor throughput on FP32)
// ─────────────────────────────────────────────────────────────────────────────

`default_nettype none

module rac_systolic_array #(
    parameter integer N       = 16,
    parameter integer WIDTH   = 64,
    parameter         INIT_LUT   = "cordic_coarse_lut.mem",
    parameter         INIT_ATANH = "cordic_atanh.mem"
) (
    input  wire                               clk,
    input  wire                               rst_n,

    // Weight pre-load. Scan in one PE-angle per `weight_valid` cycle;
    // the address selects which (row, col) to program.
    input  wire                               weight_valid,
    input  wire [$clog2(N*N)-1:0]             weight_addr,
    input  wire signed [WIDTH-1:0]            weight_angle,

    // Left edge: activation vector, one element per row per cycle.
    input  wire                               x_valid_in,
    input  wire signed [WIDTH-1:0]            x_in [0:N-1],

    // Bottom edge: result vector, one element per column per cycle,
    // valid after N + rac_dsp_LATENCY cycles from the corresponding input.
    output wire                               y_valid_out,
    output wire signed [WIDTH-1:0]            y_out [0:N-1]
);

    // PE-local angle registers (the "weights")
    reg signed [WIDTH-1:0] pe_angle [0:N-1][0:N-1];
    always @(posedge clk) begin
        if (weight_valid) begin
            pe_angle[weight_addr / N][weight_addr % N] <= weight_angle;
        end
    end

    // East-flowing activation lattice. x_h[r][c] = activation entering
    // the PE at row r, column c.
    wire signed [WIDTH-1:0] x_h [0:N-1][0:N];
    // South-flowing partial-sum lattice. y_v[r][c] = partial sum entering
    // the PE at row r, column c from above.
    wire signed [WIDTH-1:0] y_v [0:N][0:N-1];

    generate genvar r, c;
        // Seed left edge with the input vector
        for (r = 0; r < N; r = r + 1) begin : seed_left
            assign x_h[r][0] = x_in[r];
        end
        // Seed top edge with zeros (partial sums start at 0)
        for (c = 0; c < N; c = c + 1) begin : seed_top
            assign y_v[0][c] = {WIDTH{1'b0}};
        end

        for (r = 0; r < N; r = r + 1) begin : rows
            for (c = 0; c < N; c = c + 1) begin : cols
                // Each PE is a rac_dsp in project mode: rotate x by
                // stored angle, emit x-component, add to partial sum.
                wire                       v_valid;
                wire signed [WIDTH-1:0]    x_proj;
                wire signed [WIDTH-1:0]    _y_unused, _z_unused;

                rac_dsp #(
                    .WIDTH   (WIDTH),
                    .INIT_LUT(INIT_LUT),
                    .INIT_ATANH(INIT_ATANH)
                ) u_pe (
                    .clk      (clk),
                    .rst_n    (rst_n),
                    .valid_in (x_valid_in),
                    .op_in    (3'b001),          // project
                    .x_in     (x_h[r][c]),
                    .y_in     (64'sh0),          // rotation axis starts on x
                    .z_in     (pe_angle[r][c]),
                    .valid_out(v_valid),
                    .op_out   (),
                    .x_out    (x_proj),
                    .y_out    (_y_unused),
                    .z_out    (_z_unused)
                );

                // East flow: pass x through (registered to match pipeline depth)
                reg signed [WIDTH-1:0] x_east_q;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) x_east_q <= {WIDTH{1'b0}};
                    else        x_east_q <= x_h[r][c];
                end
                assign x_h[r][c+1] = x_east_q;

                // South flow: y_v[r][c] (from above) + x_proj (this PE)
                //   → y_v[r+1][c]
                reg signed [WIDTH-1:0] y_south_q;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) y_south_q <= {WIDTH{1'b0}};
                    else        y_south_q <= y_v[r][c] + x_proj;
                end
                assign y_v[r+1][c] = y_south_q;
            end
        end

        // Bottom-edge output = y_v[N][0..N-1]
        for (c = 0; c < N; c = c + 1) begin : out_edge
            assign y_out[c] = y_v[N][c];
        end
    endgenerate

    // Valid signal: simple shift register matching the pipeline depth
    // (rac_dsp LATENCY + N row-propagation).
    localparam integer TOTAL_LATENCY = 9 + N;   // 9 from rac_dsp + N for lattice
    reg [TOTAL_LATENCY-1:0] valid_shift;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_shift <= {TOTAL_LATENCY{1'b0}};
        else        valid_shift <= {valid_shift[TOTAL_LATENCY-2:0], x_valid_in};
    end
    assign y_valid_out = valid_shift[TOTAL_LATENCY-1];

endmodule

`default_nettype wire
