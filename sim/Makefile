# sim/Makefile — rac_dsp cosim
#
# One-shot:   make all         → gen vectors, build RTL sim + C ref, cosim
# Pieces:     make vectors     → test_vectors.hex + golden.hex
#             make rtl         → build + run tb_rac_dsp under iverilog
#             make ref         → build + run rac_dsp_ref
#             make cosim       → diff rtl_outputs vs golden (+ optional C ref)
#             make clean       → remove generated artifacts

RTL_DIR := ../rtl

IVERILOG := iverilog -g2012
VVP      := vvp
CC       := cc
CFLAGS   := -O2 -Wall -Wextra -std=c99

PYTHON   := python3

# LUT memory files — the testbench + C ref both read these
LUT_MEMS := cordic_coarse_lut.mem cordic_atanh.mem

.PHONY: all vectors rtl ref cosim clean deps

all: cosim

deps:
	@for tool in $(firstword $(IVERILOG)) $(VVP) $(CC) $(PYTHON); do \
		command -v $$tool >/dev/null 2>&1 || \
			{ echo "MISSING: $$tool — apt install iverilog python3 build-essential"; exit 1; }; \
	done

# ── Generate test vectors + golden.hex ────────────────────────────────
vectors: test_vectors.hex golden.hex

test_vectors.hex golden.hex: gen_vectors.py
	$(PYTHON) gen_vectors.py

# ── Symlink LUT .mem files from rtl/ so tb + C ref can $readmemh them ─
$(LUT_MEMS): $(RTL_DIR)/cordic_coarse_lut.mem $(RTL_DIR)/cordic_atanh.mem
	ln -sf $(RTL_DIR)/cordic_coarse_lut.mem .
	ln -sf $(RTL_DIR)/cordic_atanh.mem .

# ── Build + run the RTL testbench ──────────────────────────────────────
rtl: rtl_outputs.hex

tb_rac_dsp: tb_rac_dsp.v $(RTL_DIR)/rac_dsp.v
	$(IVERILOG) -o $@ tb_rac_dsp.v $(RTL_DIR)/rac_dsp.v

rtl_outputs.hex: tb_rac_dsp test_vectors.hex $(LUT_MEMS)
	$(VVP) tb_rac_dsp

# ── Build + run the C reference ────────────────────────────────────────
ref: ref_outputs.hex

rac_dsp_ref: rac_dsp_ref.c
	$(CC) $(CFLAGS) -o $@ $<

ref_outputs.hex: rac_dsp_ref test_vectors.hex $(LUT_MEMS)
	./rac_dsp_ref test_vectors.hex cordic_coarse_lut.mem cordic_atanh.mem > $@

# ── Cosim diff ─────────────────────────────────────────────────────────
cosim: rtl_outputs.hex golden.hex
	$(PYTHON) cosim.py $(COSIM_ARGS)

# ── Cleanup ────────────────────────────────────────────────────────────
clean:
	rm -f tb_rac_dsp rac_dsp_ref \
	      test_vectors.hex golden.hex vectors.csv \
	      rtl_outputs.hex ref_outputs.hex \
	      $(LUT_MEMS)
