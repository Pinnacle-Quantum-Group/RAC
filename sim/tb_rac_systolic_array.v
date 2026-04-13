// ─────────────────────────────────────────────────────────────────────────────
// tb_rac_systolic_array.v — Verilog testbench for the N×N CORDIC GEMM array
// Pinnacle Quantum Group — April 2026
//
// Two modes, selected by the presence of gemm_input_batch.hex:
//
//   (a) Matrix-vector y = W·x         one x vector, one y vector
//   (b) Matrix-matrix  Y = W·X        K activations streamed serially,
//                                      one y vector per activation
//
// Size N and batch K are defaults (N=4, K=1) but overridable at iverilog
// compile time:
//
//   iverilog -g2012 -PN=16 -PK=8 -o tb ...
//
// The Makefile passes the `GEMM_N` and `GEMM_K` env vars through as -P.
//
// Inputs read at runtime:
//   gemm_weights.hex       N² angles in Q0.63 fraction-of-π (row-major)
//   gemm_input.hex         N activation values (single-vector mode)
//   gemm_input_batch.hex   K·N values (matrix-matrix mode, row-major by k)
//
// Output dumped:
//   rtl_gemm_outputs.hex   N·K values (one line per output element,
//                          K blocks of N)
// ─────────────────────────────────────────────────────────────────────────────

`default_nettype none
`timescale 1ns/1ps

module tb_rac_systolic_array;

    parameter integer N     = 4;
    parameter integer K     = 1;    // batch size (# of x vectors)
    parameter integer WIDTH = 64;

    // Per-vector pipeline drain: one x vector needs 2N+11 cycles
    // (rac_dsp LATENCY=12, east propagation N, south accumulation N,
    //  last column valid at 2N+11). Add slack.
    localparam integer VECTOR_LATENCY = 2*N + 16;

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n;

    // Weight programming
    reg                          weight_valid;
    reg  [$clog2(N*N)-1:0]       weight_addr;
    reg  signed [WIDTH-1:0]      weight_angle;

    // Activation — TB keeps the unpacked-array convenience view;
    // we pack/unpack into the module's flattened port at the boundary.
    reg                          x_valid_in;
    reg  signed [WIDTH-1:0]      x_in  [0:N-1];
    wire [N*WIDTH-1:0]           x_in_flat;
    genvar _pk;
    generate for (_pk = 0; _pk < N; _pk = _pk + 1) begin : pack_x_in
        assign x_in_flat[(_pk+1)*WIDTH-1 -: WIDTH] = x_in[_pk];
    end endgenerate

    // Result
    wire                         y_valid_out;
    wire [N*WIDTH-1:0]           y_out_flat;
    wire signed [WIDTH-1:0]      y_out [0:N-1];
    genvar _uu;
    generate for (_uu = 0; _uu < N; _uu = _uu + 1) begin : unpack_y_out
        assign y_out[_uu] = $signed(y_out_flat[(_uu+1)*WIDTH-1 -: WIDTH]);
    end endgenerate

    rac_systolic_array #(
        .N         (N),
        .WIDTH     (WIDTH),
        .INIT_LUT  ("cordic_coarse_lut.mem"),
        .INIT_ATANH("cordic_atanh.mem")
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .weight_valid (weight_valid),
        .weight_addr  (weight_addr),
        .weight_angle (weight_angle),
        .x_valid_in   (x_valid_in),
        .x_in         (x_in_flat),
        .y_valid_out  (y_valid_out),
        .y_out        (y_out_flat)
    );

    // ── Vector storage ───────────────────────────────────────────────
    reg [WIDTH-1:0] weights_mem [0:N*N-1];
    reg [WIDTH-1:0] input_mem   [0:K*N-1];   // up to K·N values
    reg [WIDTH-1:0] output_mem  [0:K*N-1];

    integer fin, fout, code;
    reg [2047:0] line;

    task load_hex_array(
        input [255*8:1] path,
        input [1:0]     which,       // 0 = weights, 1 = input (single or batch)
        output integer n_loaded
    );
        integer i, max_entries;
        reg [63:0] v;
        integer got;
        begin
            fin = $fopen(path, "r");
            if (fin == 0) begin
                n_loaded = 0;   // not an error for optional files
                $display("  (no %0s)", path);
            end else begin
                max_entries = (which == 0) ? (N*N) : (K*N);
                i = 0;
                while (!$feof(fin)) begin
                    code = $fgets(line, fin);
                    if (code <= 0) ;
                    else if (line[2047:2040] == "/" && line[2039:2032] == "/") ;
                    else begin
                        got = $sscanf(line, "%h", v);
                        if (got == 1) begin
                            if (i < max_entries) begin
                                if (which == 0) weights_mem[i] = v;
                                else            input_mem[i]   = v;
                            end
                            i = i + 1;
                        end
                    end
                end
                $fclose(fin);
                n_loaded = i;
                $display("  loaded %0d %0s", i,
                          (which == 0) ? "weights" : "inputs");
            end
        end
    endtask

    integer n_w, n_in_single, n_in_batch;
    integer k, i;
    integer total_in;

    initial begin
        rst_n        = 0;
        weight_valid = 0;
        weight_addr  = 0;
        weight_angle = 0;
        x_valid_in   = 0;
        for (i = 0; i < N; i = i + 1) x_in[i] = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("tb_rac_systolic_array: N=%0d K=%0d", N, K);

        // Load weights
        load_hex_array("gemm_weights.hex", 2'd0, n_w);
        if (n_w < N*N) begin
            $display("ERROR: need %0d weights, got %0d", N*N, n_w);
            $finish;
        end

        // Load batch-or-single input. Prefer batch if present.
        load_hex_array("gemm_input_batch.hex", 2'd1, n_in_batch);
        if (n_in_batch >= K*N) begin
            total_in = K * N;
            $display("  using batch input (K=%0d vectors)", K);
        end else begin
            load_hex_array("gemm_input.hex", 2'd1, n_in_single);
            if (n_in_single < N) begin
                $display("ERROR: need at least %0d inputs, got %0d",
                         N, n_in_single);
                $finish;
            end
            total_in = N;    // only one vector available
            if (K > 1) begin
                $display("  WARN: K=%0d but single-vector input; running K=1",
                         K);
            end
        end

        // Program PE angles, one per cycle
        for (i = 0; i < N*N; i = i + 1) begin
            weight_valid <= 1'b1;
            weight_addr  <= i[$clog2(N*N)-1:0];
            weight_angle <= $signed(weights_mem[i]);
            @(posedge clk);
        end
        weight_valid <= 1'b0;
        weight_addr  <= 0;
        weight_angle <= 0;
        @(posedge clk);

        // Stream K activation vectors through the array.
        // For each vector: hold x_in + x_valid_in for VECTOR_LATENCY cycles,
        // then sample y_out. This is deliberately conservative; a production
        // harness would overlap successive vectors in the pipeline.
        for (k = 0; k < (total_in / N); k = k + 1) begin
            for (i = 0; i < N; i = i + 1)
                x_in[i] = $signed(input_mem[k*N + i]);
            x_valid_in <= 1'b1;
            repeat (VECTOR_LATENCY) @(posedge clk);
            // Capture y_out
            for (i = 0; i < N; i = i + 1)
                output_mem[k*N + i] = y_out[i];
            // Deassert between vectors — lets the south accumulator reset
            // for the next vector (y_v[0][c] is tied to 0 combinationally).
            x_valid_in <= 1'b0;
            for (i = 0; i < N; i = i + 1) x_in[i] <= 0;
            repeat (VECTOR_LATENCY) @(posedge clk);
        end

        // Dump outputs — one line per element, K blocks of N
        fout = $fopen("rtl_gemm_outputs.hex", "w");
        $fwrite(fout, "// rac_systolic_array RTL outputs — N=%0d K=%0d\n",
                N, K);
        $fwrite(fout, "// format: y[k][c] in Q32.32 signed\n");
        for (k = 0; k < (total_in / N); k = k + 1) begin
            for (i = 0; i < N; i = i + 1) begin
                $fwrite(fout, "%016x\n", output_mem[k*N + i]);
            end
        end
        $fclose(fout);
        $display("wrote rtl_gemm_outputs.hex (%0d rows)", total_in);
        $finish;
    end

    // Timeout safety (4 × total time estimate)
    initial begin
        #1000000;
        $display("ERROR: testbench timeout");
        $finish;
    end

endmodule

`default_nettype wire
