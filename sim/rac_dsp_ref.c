/*
 * rac_dsp_ref.c — single-DSP test driver.
 *
 * Reads test_vectors.hex, runs each case through rac_dsp_eval()
 * (implemented in rac_dsp_core.c), and prints one output row per
 * input in the same format as the RTL testbench. cosim.py diffs
 * against golden.hex.
 *
 * Build:  cc -O2 -std=c99 -Wall -Wextra -o rac_dsp_ref \
 *             rac_dsp_ref.c rac_dsp_core.c -lm
 * Run:    ./rac_dsp_ref test_vectors.hex coarse_lut.mem atan.mem atanh.mem
 *                   > ref_outputs.hex
 */

#include "rac_dsp_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "usage: %s test_vectors.hex coarse_lut.mem atan.mem atanh.mem\n",
            argv[0]);
        return 1;
    }
    int rc = rac_load_all_roms(argv[2], argv[3], argv[4]);
    if (rc != 0) return rc + 1;

    FILE *vf = fopen(argv[1], "r");
    if (!vf) { fprintf(stderr, "open %s\n", argv[1]); return 10; }

    printf("// rac_dsp C reference outputs\n");
    printf("// format: x_q3232 y_q3232 z_q3232\n");

    char line[512];
    while (fgets(line, sizeof(line), vf)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || (*p == '/' && p[1] == '/')) continue;
        unsigned op;
        uint64_t xh, yh, zh;
        if (sscanf(p, "%x %" SCNx64 " %" SCNx64 " %" SCNx64,
                   &op, &xh, &yh, &zh) != 4) continue;
        rac_q_t xo, yo, zo;
        rac_dsp_eval((rac_q_t)xh, (rac_q_t)yh, (rac_q_t)zh,
                     (int)(op & 7), &xo, &yo, &zo);
        printf("%016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
               (uint64_t)xo, (uint64_t)yo, (uint64_t)zo);
    }
    fclose(vf);
    return 0;
}
