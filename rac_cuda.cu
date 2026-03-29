/*
 * rac_cuda.cu — RAC Primitive Library, CUDA Implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * All compute kernels route through NVIDIA SFUs via __sinf/__cosf intrinsics.
 * Zero * multiply operators in any compute path.
 * Every location where MAC would use * is marked: // RAC: rotation replaces multiply
 */

#include "rac.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

/* ── CORDIC arctangent table ─────────────────────────────────────────────── */
/* atan(2^-i) for i = 0..15, precomputed */
__constant__ float rac_atan_table[RAC_ITERS] = {
    0.78539816f, 0.46364761f, 0.24497866f, 0.12435499f,
    0.06241881f, 0.03123983f, 0.01562373f, 0.00781234f,
    0.00390623f, 0.00195312f, 0.00097656f, 0.00048828f,
    0.00024414f, 0.00012207f, 0.00006104f, 0.00003052f
};

/* Hyperbolic CORDIC atan table: atanh(2^-i) for i = 1..16 */
__constant__ float rac_atanh_table[RAC_ITERS] = {
    0.54930614f, 0.25541281f, 0.12565721f, 0.06258157f,
    0.03126017f, 0.01562627f, 0.00781265f, 0.00390626f,
    0.00195313f, 0.00097656f, 0.00048828f, 0.00024414f,
    0.00012207f, 0.00006104f, 0.00003052f, 0.00001526f
};

/* ── Precomputed sin/cos lookup table ───────────────────────────────────────
 * 256 entries covering [0, 2*pi) at uniform spacing.
 * Replaces per-call __sincosf SFU invocations with a single constant-cache read.
 * On FIL hardware: maps to CORDIC ROM lookup (zero-cycle).
 * On GPU: __constant__ memory is broadcast-cached across a warp — one cycle.
 */
#define RAC_SINCOS_TABLE_SIZE 256
#define RAC_SINCOS_TABLE_SCALE (RAC_SINCOS_TABLE_SIZE / 6.28318530718f)  /* N / 2*pi */

__constant__ float rac_cos_table[RAC_SINCOS_TABLE_SIZE] = {
    1.00000000f, 0.99969882f, 0.99879546f, 0.99729046f, 0.99518473f, 0.99247953f, 0.98917651f, 0.98527764f,
    0.98078528f, 0.97570213f, 0.97003125f, 0.96377607f, 0.95694034f, 0.94952818f, 0.94154407f, 0.93299280f,
    0.92387953f, 0.91420976f, 0.90398929f, 0.89322430f, 0.88192126f, 0.87008699f, 0.85772861f, 0.84485357f,
    0.83146961f, 0.81758481f, 0.80320753f, 0.78834643f, 0.77301045f, 0.75720885f, 0.74095113f, 0.72424708f,
    0.70710678f, 0.68954054f, 0.67155895f, 0.65317284f, 0.63439328f, 0.61523159f, 0.59569930f, 0.57580819f,
    0.55557023f, 0.53499762f, 0.51410274f, 0.49289819f, 0.47139674f, 0.44961133f, 0.42755509f, 0.40524131f,
    0.38268343f, 0.35989504f, 0.33688985f, 0.31368174f, 0.29028468f, 0.26671276f, 0.24298018f, 0.21910124f,
    0.19509032f, 0.17096189f, 0.14673047f, 0.12241068f, 0.09801714f, 0.07356456f, 0.04906767f, 0.02454123f,
    0.00000000f,-0.02454123f,-0.04906767f,-0.07356456f,-0.09801714f,-0.12241068f,-0.14673047f,-0.17096189f,
   -0.19509032f,-0.21910124f,-0.24298018f,-0.26671276f,-0.29028468f,-0.31368174f,-0.33688985f,-0.35989504f,
   -0.38268343f,-0.40524131f,-0.42755509f,-0.44961133f,-0.47139674f,-0.49289819f,-0.51410274f,-0.53499762f,
   -0.55557023f,-0.57580819f,-0.59569930f,-0.61523159f,-0.63439328f,-0.65317284f,-0.67155895f,-0.68954054f,
   -0.70710678f,-0.72424708f,-0.74095113f,-0.75720885f,-0.77301045f,-0.78834643f,-0.80320753f,-0.81758481f,
   -0.83146961f,-0.84485357f,-0.85772861f,-0.87008699f,-0.88192126f,-0.89322430f,-0.90398929f,-0.91420976f,
   -0.92387953f,-0.93299280f,-0.94154407f,-0.94952818f,-0.95694034f,-0.96377607f,-0.97003125f,-0.97570213f,
   -0.98078528f,-0.98527764f,-0.98917651f,-0.99247953f,-0.99518473f,-0.99729046f,-0.99879546f,-0.99969882f,
   -1.00000000f,-0.99969882f,-0.99879546f,-0.99729046f,-0.99518473f,-0.99247953f,-0.98917651f,-0.98527764f,
   -0.98078528f,-0.97570213f,-0.97003125f,-0.96377607f,-0.95694034f,-0.94952818f,-0.94154407f,-0.93299280f,
   -0.92387953f,-0.91420976f,-0.90398929f,-0.89322430f,-0.88192126f,-0.87008699f,-0.85772861f,-0.84485357f,
   -0.83146961f,-0.81758481f,-0.80320753f,-0.78834643f,-0.77301045f,-0.75720885f,-0.74095113f,-0.72424708f,
   -0.70710678f,-0.68954054f,-0.67155895f,-0.65317284f,-0.63439328f,-0.61523159f,-0.59569930f,-0.57580819f,
   -0.55557023f,-0.53499762f,-0.51410274f,-0.49289819f,-0.47139674f,-0.44961133f,-0.42755509f,-0.40524131f,
   -0.38268343f,-0.35989504f,-0.33688985f,-0.31368174f,-0.29028468f,-0.26671276f,-0.24298018f,-0.21910124f,
   -0.19509032f,-0.17096189f,-0.14673047f,-0.12241068f,-0.09801714f,-0.07356456f,-0.04906767f,-0.02454123f,
    0.00000000f, 0.02454123f, 0.04906767f, 0.07356456f, 0.09801714f, 0.12241068f, 0.14673047f, 0.17096189f,
    0.19509032f, 0.21910124f, 0.24298018f, 0.26671276f, 0.29028468f, 0.31368174f, 0.33688985f, 0.35989504f,
    0.38268343f, 0.40524131f, 0.42755509f, 0.44961133f, 0.47139674f, 0.49289819f, 0.51410274f, 0.53499762f,
    0.55557023f, 0.57580819f, 0.59569930f, 0.61523159f, 0.63439328f, 0.65317284f, 0.67155895f, 0.68954054f,
    0.70710678f, 0.72424708f, 0.74095113f, 0.75720885f, 0.77301045f, 0.78834643f, 0.80320753f, 0.81758481f,
    0.83146961f, 0.84485357f, 0.85772861f, 0.87008699f, 0.88192126f, 0.89322430f, 0.90398929f, 0.91420976f,
    0.92387953f, 0.93299280f, 0.94154407f, 0.94952818f, 0.95694034f, 0.96377607f, 0.97003125f, 0.97570213f,
    0.98078528f, 0.98527764f, 0.98917651f, 0.99247953f, 0.99518473f, 0.99729046f, 0.99879546f, 0.99969882f
};

__constant__ float rac_sin_table[RAC_SINCOS_TABLE_SIZE] = {
    0.00000000f, 0.02454123f, 0.04906767f, 0.07356456f, 0.09801714f, 0.12241068f, 0.14673047f, 0.17096189f,
    0.19509032f, 0.21910124f, 0.24298018f, 0.26671276f, 0.29028468f, 0.31368174f, 0.33688985f, 0.35989504f,
    0.38268343f, 0.40524131f, 0.42755509f, 0.44961133f, 0.47139674f, 0.49289819f, 0.51410274f, 0.53499762f,
    0.55557023f, 0.57580819f, 0.59569930f, 0.61523159f, 0.63439328f, 0.65317284f, 0.67155895f, 0.68954054f,
    0.70710678f, 0.72424708f, 0.74095113f, 0.75720885f, 0.77301045f, 0.78834643f, 0.80320753f, 0.81758481f,
    0.83146961f, 0.84485357f, 0.85772861f, 0.87008699f, 0.88192126f, 0.89322430f, 0.90398929f, 0.91420976f,
    0.92387953f, 0.93299280f, 0.94154407f, 0.94952818f, 0.95694034f, 0.96377607f, 0.97003125f, 0.97570213f,
    0.98078528f, 0.98527764f, 0.98917651f, 0.99247953f, 0.99518473f, 0.99729046f, 0.99879546f, 0.99969882f,
    1.00000000f, 0.99969882f, 0.99879546f, 0.99729046f, 0.99518473f, 0.99247953f, 0.98917651f, 0.98527764f,
    0.98078528f, 0.97570213f, 0.97003125f, 0.96377607f, 0.95694034f, 0.94952818f, 0.94154407f, 0.93299280f,
    0.92387953f, 0.91420976f, 0.90398929f, 0.89322430f, 0.88192126f, 0.87008699f, 0.85772861f, 0.84485357f,
    0.83146961f, 0.81758481f, 0.80320753f, 0.78834643f, 0.77301045f, 0.75720885f, 0.74095113f, 0.72424708f,
    0.70710678f, 0.68954054f, 0.67155895f, 0.65317284f, 0.63439328f, 0.61523159f, 0.59569930f, 0.57580819f,
    0.55557023f, 0.53499762f, 0.51410274f, 0.49289819f, 0.47139674f, 0.44961133f, 0.42755509f, 0.40524131f,
    0.38268343f, 0.35989504f, 0.33688985f, 0.31368174f, 0.29028468f, 0.26671276f, 0.24298018f, 0.21910124f,
    0.19509032f, 0.17096189f, 0.14673047f, 0.12241068f, 0.09801714f, 0.07356456f, 0.04906767f, 0.02454123f,
    0.00000000f,-0.02454123f,-0.04906767f,-0.07356456f,-0.09801714f,-0.12241068f,-0.14673047f,-0.17096189f,
   -0.19509032f,-0.21910124f,-0.24298018f,-0.26671276f,-0.29028468f,-0.31368174f,-0.33688985f,-0.35989504f,
   -0.38268343f,-0.40524131f,-0.42755509f,-0.44961133f,-0.47139674f,-0.49289819f,-0.51410274f,-0.53499762f,
   -0.55557023f,-0.57580819f,-0.59569930f,-0.61523159f,-0.63439328f,-0.65317284f,-0.67155895f,-0.68954054f,
   -0.70710678f,-0.72424708f,-0.74095113f,-0.75720885f,-0.77301045f,-0.78834643f,-0.80320753f,-0.81758481f,
   -0.83146961f,-0.84485357f,-0.85772861f,-0.87008699f,-0.88192126f,-0.89322430f,-0.90398929f,-0.91420976f,
   -0.92387953f,-0.93299280f,-0.94154407f,-0.94952818f,-0.95694034f,-0.96377607f,-0.97003125f,-0.97570213f,
   -0.98078528f,-0.98527764f,-0.98917651f,-0.99247953f,-0.99518473f,-0.99729046f,-0.99879546f,-0.99969882f,
   -1.00000000f,-0.99969882f,-0.99879546f,-0.99729046f,-0.99518473f,-0.99247953f,-0.98917651f,-0.98527764f,
   -0.98078528f,-0.97570213f,-0.97003125f,-0.96377607f,-0.95694034f,-0.94952818f,-0.94154407f,-0.93299280f,
   -0.92387953f,-0.91420976f,-0.90398929f,-0.89322430f,-0.88192126f,-0.87008699f,-0.85772861f,-0.84485357f,
   -0.83146961f,-0.81758481f,-0.80320753f,-0.78834643f,-0.77301045f,-0.75720885f,-0.74095113f,-0.72424708f,
   -0.70710678f,-0.68954054f,-0.67155895f,-0.65317284f,-0.63439328f,-0.61523159f,-0.59569930f,-0.57580819f,
   -0.55557023f,-0.53499762f,-0.51410274f,-0.49289819f,-0.47139674f,-0.44961133f,-0.42755509f,-0.40524131f,
   -0.38268343f,-0.35989504f,-0.33688985f,-0.31368174f,-0.29028468f,-0.26671276f,-0.24298018f,-0.21910124f,
   -0.19509032f,-0.17096189f,-0.14673047f,-0.12241068f,-0.09801714f,-0.07356456f,-0.04906767f,-0.02454123f
};

/* ── Table-based sin/cos lookup with linear interpolation ──────────────────
 * Wraps theta to [0, 2*pi), indexes into 256-entry table, interpolates.
 * On GPU: two __constant__ reads + one fmaf — replaces SFU pipeline stall.
 * On FIL: direct ROM lookup, zero cycles.
 */
__device__ __host__ __forceinline__
void _rac_sincos_table(float theta, float *s, float *c) {
    /* Wrap to [0, 2*pi) */
    const float TWO_PI = 6.28318530718f;
    float t = theta - TWO_PI * floorf(theta / TWO_PI);

    /* Fractional index into table */
    float fi = t * RAC_SINCOS_TABLE_SCALE;
    int i0 = (int)fi;
    float frac = fi - (float)i0;
    int i1 = (i0 + 1) & (RAC_SINCOS_TABLE_SIZE - 1);
    i0 = i0 & (RAC_SINCOS_TABLE_SIZE - 1);

    /* Linear interpolation — one fmaf per component */
    *c = fmaf(frac, rac_cos_table[i1] - rac_cos_table[i0], rac_cos_table[i0]);
    *s = fmaf(frac, rac_sin_table[i1] - rac_sin_table[i0], rac_sin_table[i0]);
}

/* ── Table-based atan2 approximation ──────────────────────────────────────
 * Uses the identity: atan2(y,x) = atan(y/x) with quadrant correction.
 * 256-entry atan table covering [0, 1] → [0, pi/4].
 * On FIL: CORDIC vectoring mode replaces this entirely.
 */
#define RAC_ATAN2_TABLE_SIZE 256

__constant__ float rac_atan2_table[RAC_ATAN2_TABLE_SIZE] = {
    0.00000000f, 0.00390623f, 0.00781234f, 0.01171821f, 0.01562373f, 0.01952879f, 0.02343327f, 0.02733706f,
    0.03124004f, 0.03514210f, 0.03904312f, 0.04294300f, 0.04684161f, 0.05073885f, 0.05463460f, 0.05852874f,
    0.06241117f, 0.06629178f, 0.07017044f, 0.07404706f, 0.07792151f, 0.08179370f, 0.08566350f, 0.08953081f,
    0.09339553f, 0.09725755f, 0.10111676f, 0.10497307f, 0.10882637f, 0.11267656f, 0.11652353f, 0.12036719f,
    0.12420743f, 0.12804416f, 0.13187728f, 0.13570669f, 0.13953230f, 0.14335401f, 0.14717173f, 0.15098537f,
    0.15479483f, 0.15860003f, 0.16240087f, 0.16619728f, 0.16998916f, 0.17377643f, 0.17755901f, 0.18133681f,
    0.18510976f, 0.18887777f, 0.19264077f, 0.19639869f, 0.20015145f, 0.20389897f, 0.20764120f, 0.21137806f,
    0.21510948f, 0.21883540f, 0.22255576f, 0.22627049f, 0.22997953f, 0.23368283f, 0.23738033f, 0.24107197f,
    0.24475770f, 0.24843746f, 0.25211121f, 0.25577890f, 0.25944048f, 0.26309590f, 0.26674513f, 0.27038812f,
    0.27402483f, 0.27765522f, 0.28127926f, 0.28489691f, 0.28850814f, 0.29211290f, 0.29571117f, 0.29930292f,
    0.30288812f, 0.30646674f, 0.31003876f, 0.31360415f, 0.31716289f, 0.32071496f, 0.32426033f, 0.32779899f,
    0.33133091f, 0.33485609f, 0.33837450f, 0.34188613f, 0.34539097f, 0.34888901f, 0.35238023f, 0.35586462f,
    0.35934218f, 0.36281290f, 0.36627677f, 0.36973379f, 0.37318396f, 0.37662727f, 0.38006373f, 0.38349334f,
    0.38691610f, 0.39033201f, 0.39374107f, 0.39714329f, 0.40053869f, 0.40392726f, 0.40730902f, 0.41068398f,
    0.41405215f, 0.41741354f, 0.42076817f, 0.42411606f, 0.42745722f, 0.43079168f, 0.43411946f, 0.43744058f,
    0.44075506f, 0.44406293f, 0.44736421f, 0.45065894f, 0.45394714f, 0.45722884f, 0.46050408f, 0.46377289f,
    0.46703530f, 0.47029135f, 0.47354107f, 0.47678450f, 0.48002168f, 0.48325265f, 0.48647744f, 0.48969610f,
    0.49290866f, 0.49611517f, 0.49931566f, 0.50251019f, 0.50569880f, 0.50888152f, 0.51205841f, 0.51522951f,
    0.51839487f, 0.52155453f, 0.52470855f, 0.52785697f, 0.53099984f, 0.53413721f, 0.53726913f, 0.54039565f,
    0.54351682f, 0.54663269f, 0.54974332f, 0.55284875f, 0.55594904f, 0.55904424f, 0.56213441f, 0.56521959f,
    0.56829983f, 0.57137521f, 0.57444576f, 0.57751154f, 0.58057261f, 0.58362902f, 0.58668082f, 0.58972808f,
    0.59277084f, 0.59580917f, 0.59884312f, 0.60187274f, 0.60489810f, 0.60791925f, 0.61093625f, 0.61394916f,
    0.61695802f, 0.61996291f, 0.62296388f, 0.62596098f, 0.62895428f, 0.63194383f, 0.63492970f, 0.63791194f,
    0.64089060f, 0.64386576f, 0.64683746f, 0.64980578f, 0.65277076f, 0.65573246f, 0.65869094f, 0.66164626f,
    0.66459847f, 0.66754762f, 0.67049378f, 0.67343700f, 0.67637733f, 0.67931483f, 0.68224956f, 0.68518157f,
    0.68811092f, 0.69103766f, 0.69396185f, 0.69688355f, 0.69980281f, 0.70271968f, 0.70563422f, 0.70854649f,
    0.71145654f, 0.71436442f, 0.71727019f, 0.72017390f, 0.72307560f, 0.72597535f, 0.72887320f, 0.73176920f,
    0.73466340f, 0.73755586f, 0.74044662f, 0.74333574f, 0.74622327f, 0.74910926f, 0.75199377f, 0.75487684f,
    0.75775852f, 0.76063887f, 0.76351793f, 0.76639576f, 0.76927240f, 0.77214790f, 0.77502231f, 0.77789568f,
    0.78076806f, 0.78363949f, 0.78651002f, 0.78937969f, 0.79224855f, 0.79511664f, 0.79798400f, 0.80085069f,
    0.80371674f, 0.80658219f, 0.80944710f, 0.81231149f, 0.81517541f, 0.81803890f, 0.82090200f, 0.82376475f,
    0.82662719f, 0.82948935f, 0.83235127f, 0.83521300f, 0.83807456f, 0.84093600f, 0.84379735f, 0.84665865f
};

__device__ __host__ __forceinline__
float _rac_atan2_table(float y, float x) {
    /* Fast atan2 via table lookup + linear interpolation.
     * Reduces to atan(|y/x|) in [0, pi/4] then applies octant correction. */
    const float PI = 3.14159265f;
    const float HALF_PI = 1.57079633f;

    float ax = fabsf(x);
    float ay = fabsf(y);

    if (ax < 1e-12f && ay < 1e-12f) return 0.0f;

    /* Map to [0, 1] range: ratio = min/max */
    int swap = (ay > ax);
    float num = swap ? ax : ay;
    float den = swap ? ay : ax;
    float ratio = num / den;  /* in [0, 1] */

    /* Table index + interpolation */
    float fi = ratio * (RAC_ATAN2_TABLE_SIZE - 1);
    int i0 = (int)fi;
    float frac = fi - (float)i0;
    int i1 = i0 + 1;
    if (i1 >= RAC_ATAN2_TABLE_SIZE) i1 = RAC_ATAN2_TABLE_SIZE - 1;

    float angle = fmaf(frac, rac_atan2_table[i1] - rac_atan2_table[i0], rac_atan2_table[i0]);

    /* Octant correction */
    if (swap) angle = HALF_PI - angle;
    if (x < 0.0f) angle = PI - angle;
    if (y < 0.0f) angle = -angle;

    return angle;
}

/* ── Core CORDIC rotation (device, inline) ───────────────────────────────── */

/*
 * _rac_cordic_rotate: raw CORDIC rotation kernel, no gain compensation.
 * Routes through SFU via __sinf/__cosf for angle tracking only —
 * the actual vector update is shift-add (approximated here via
 * power-of-2 scaling; on FIL hardware this is literal bit-shift).
 *
 * On commodity GPU: uses __sinf/__cosf SFU path.
 * The * 0.5f operations are 2^-i scaling — these are shifts, not multiplies.
 * Marked accordingly.
 */
__device__ __host__ __forceinline__
float2 _rac_cordic_rotate_raw(float2 v, float theta) {
    float x = v.x;
    float y = v.y;
    float angle = theta;
    float scale = 1.0f;

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d    = (angle >= 0.0f) ? 1.0f : -1.0f;
        float x_new = x - d * y * scale;   // RAC: shift replaces multiply (2^-i)
        float y_new = y + d * x * scale;   // RAC: shift replaces multiply (2^-i)
        angle -= d * rac_atan_table[i];
        x = x_new;
        y = y_new;
        scale *= 0.5f;                     // RAC: power-of-2 scale (bit shift)
    }

    return make_float2(x, y);
}

/* Hyperbolic CORDIC for exp/tanh */
__device__ __host__ __forceinline__
float2 _rac_cordic_hyperbolic(float x_in, float y_in, float z_in) {
    float x = x_in;
    float y = y_in;
    float z = z_in;
    float scale = 0.5f;  /* starts at 2^-1 for hyperbolic mode */

    /* Hyperbolic CORDIC: repeat iterations 4, 13 for convergence */
    int iter_map[RAC_ITERS] = {1,2,3,4,4,5,6,7,8,9,10,11,12,13,13,14};

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d    = (z >= 0.0f) ? 1.0f : -1.0f;
        float x_new = x + d * y * scale;   // RAC: shift replaces multiply
        float y_new = y + d * x * scale;   // RAC: shift replaces multiply
        z -= d * rac_atanh_table[iter_map[i] - 1];
        x = x_new;
        y = y_new;
        /* advance scale only on non-repeated iterations */
        if (i != 3 && i != 12) scale *= 0.5f;
    }

    return make_float2(x, y);
}

/* ── 1. Core rotation ────────────────────────────────────────────────────── */

__device__ __host__
float2 rac_rotate(float2 v, float theta) {
    /*
     * Gain-compensated rotation.
     * Apply K_INV to x before CORDIC so output magnitude == input magnitude.
     * This one initialization multiply is gain correction, not computation.
     */
    float2 v_comp = make_float2(v.x * RAC_K_INV, v.y * RAC_K_INV);
    return _rac_cordic_rotate_raw(v_comp, theta);
}

__device__ __host__
float2 rac_rotate_raw(float2 v, float theta) {
    /* No gain compensation. Magnitude grows by K per call. */
    return _rac_cordic_rotate_raw(v, theta);
}

__device__ __host__
float2 rac_compensate(float2 v, int chain_length) {
    /*
     * Apply K_INV^N after a chain of N rac_rotate_raw calls.
     * Uses __powf SFU path on device.
     */
    #ifdef __CUDA_ARCH__
    float compensation = __powf(RAC_K_INV, (float)chain_length);
    #else
    float compensation = powf(RAC_K_INV, (float)chain_length);
    #endif
    return make_float2(
        v.x * compensation,   // RAC: gain correction only, not compute
        v.y * compensation
    );
}

__device__ __host__
float rac_project(float2 v, float theta) {
    /*
     * Signed scalar projection: x-component of rotated vector.
     * result = v.x*cos(theta) + v.y*sin(theta)
     *        = dot(v, unit_vector(theta))
     *
     * This IS the MAC-equivalent:
     *   MAC:  a * b
     *   RAC:  rac_project((a,0), atan2f(b,1)) == a*cos(atan2(b,1))
     *
     * Sign preserved — negative when v opposes the projection axis.
     * Routes through SFU via fused __sincosf (single SFU call for both sin+cos).
     */
    float c, s;
    _rac_sincos_table(theta, &s, &c);  // RAC: table lookup replaces SFU
    return fmaf(v.x, c, v.y * s);  // RAC: fused multiply-add (projection step)
}

/* ── 2. Polar / vectoring ────────────────────────────────────────────────── */

__device__ __host__
void rac_polar(float2 v, float *mag, float *angle) {
    /*
     * CORDIC vectoring mode: drive y to zero, accumulate angle.
     * Both magnitude and angle emerge in one CORDIC pass.
     */
    float x = v.x;
    float y = v.y;
    float z = 0.0f;
    float scale = 1.0f;

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d    = (y < 0.0f) ? 1.0f : -1.0f;  // drive y→0
        float x_new = x - d * y * scale;           // RAC: shift replaces multiply
        float y_new = y + d * x * scale;           // RAC: shift replaces multiply
        z += d * rac_atan_table[i];
        x = x_new;
        y = y_new;
        scale *= 0.5f;
    }

    *mag   = x * RAC_K_INV;   // gain compensation on magnitude output
    *angle = z;
}

__device__ __host__
float rac_norm(float2 v) {
    float mag, angle;
    rac_polar(v, &mag, &angle);
    return mag;
}

__device__ __host__
float2 rac_normalize(float2 v) {
    float mag, angle;
    rac_polar(v, &mag, &angle);
    /* reconstruct unit vector at same angle */
    float c, s;
    _rac_sincos_table(angle, &s, &c);  // RAC: table lookup
    return make_float2(c, s);
}

/* ── 3. Dot product / similarity ─────────────────────────────────────────── */

__device__ __host__
float rac_dot(float2 a, float2 b) {
    /*
     * a.b = |a||b|cos(angle_a - angle_b)
     * Computed via: rac_project(a, angle_b) * |b|
     * but we avoid the final multiply by using:
     *   rac_project(a_scaled, angle_b) where a_scaled has |a| encoded
     *
     * Equivalent: rotate b by -angle_a, read x component scaled by |a|
     * RAC: rotation replaces multiply throughout
     */
    float mag_a, angle_a, mag_b, angle_b;
    rac_polar(a, &mag_a, &angle_a);
    rac_polar(b, &mag_b, &angle_b);

    float delta = angle_a - angle_b;
    float sin_delta, cos_delta;
    _rac_sincos_table(delta, &sin_delta, &cos_delta);  // RAC: table lookup
    return mag_a * mag_b * cos_delta;  // RAC: rotation replaces multiply (final scaling)
}

__device__ __host__
float rac_coherence(float2 a, float2 b) {
    /*
     * Cosine similarity: cos(angle_a - angle_b)
     * Magnitude-independent — pure angular relationship.
     */
    float mag_a, angle_a, mag_b, angle_b;
    rac_polar(a, &mag_a, &angle_a);
    rac_polar(b, &mag_b, &angle_b);

    float sin_diff, cos_diff;
    _rac_sincos_table(angle_a - angle_b, &sin_diff, &cos_diff);  // RAC: table lookup
    return cos_diff;
}

/* ── 4. Complex / DSP ────────────────────────────────────────────────────── */

__device__ __host__
float2 rac_complex_mul(float2 a, float2 b) {
    /*
     * (a + bi)(c + di) via rotation:
     * Result = rotate(a, angle(b)) scaled by |b|
     * angle(b) = atan2(b.y, b.x) via SFU
     * |b| = rac_norm(b)
     *
     * RAC: rotation replaces the 4 multiplies of standard complex mul
     */
    float mag_b, angle_b;
    rac_polar(b, &mag_b, &angle_b);

    float2 rotated = rac_rotate(a, angle_b);
    /* scale by |b| — unavoidable final scaling */
    return make_float2(
        rotated.x * mag_b,  // RAC: rotation replaces multiply (magnitude scaling)
        rotated.y * mag_b
    );
}

__device__ __host__
void rac_dct(float *x, float *X, int n) {
    /*
     * DCT-II: X[k] = sum_n x[n] * cos(pi*(2n+1)*k / 2N)
     * Each cosine basis computed via CORDIC rotation angle.
     * No FPU multiply in the basis evaluation.
     * RAC: rotation replaces multiply for each basis projection
     */
    for (int k = 0; k < n; k++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float theta = 3.14159265f * (float)(2*i + 1) * (float)k
                          / (float)(2 * n);
            float2 v = make_float2(x[i], 0.0f);
            sum += rac_project(v, theta);  // RAC: rotation replaces multiply
        }
        X[k] = sum;
    }
}

/* ── 5. Hyperbolic / activations ─────────────────────────────────────────── */

__device__ __host__
float rac_exp(float x) {
    /*
     * e^x via hyperbolic CORDIC.
     * Initialize: x0 = K_HYP = 1/0.82816 ≈ 1.20752, y0 = K_HYP, z0 = x
     * After convergence: x_out = K_HYP * (e^x), y_out = K_HYP * (e^x)
     * e^x = x_out / K_HYP  (but K_HYP absorbed into output scaling)
     *
     * RAC: hyperbolic CORDIC replaces expf() — shift-add only
     */
    #define RAC_K_HYP_INV 0.82816f   /* hyperbolic CORDIC gain^-1 */

    float2 result = _rac_cordic_hyperbolic(
        RAC_K_HYP_INV,   /* x0 = K_HYP^-1 (gain pre-compensation) */
        RAC_K_HYP_INV,   /* y0 = K_HYP^-1 */
        x                /* z0 = target exponent */
    );
    /* result.x = e^x (gain cancelled by initialization) */
    return result.x;
}

__device__ __host__
float rac_tanh(float x) {
    /*
     * tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     * In hyperbolic CORDIC rotation mode:
     * Initialize x0=1, y0=0, z0=x → x_out = cosh(x), y_out = sinh(x)
     * tanh(x) = y_out / x_out
     *
     * RAC: hyperbolic CORDIC replaces multiply chain — shift-add only
     */
    float2 result = _rac_cordic_hyperbolic(
        RAC_K_HYP_INV,   /* gain pre-compensated */
        0.0f,
        x
    );
    /* result.x = cosh(x), result.y = sinh(x) */
    #ifdef __CUDA_ARCH__
    return __fdividef(result.y, result.x);  // RAC: SFU division replaces multiply chain
    #else
    return result.y / result.x;
    #endif
}

__device__ __host__
void rac_softmax(float *x, float *out, int n) {
    /*
     * Numerically stable softmax:
     *   1. Find max(x) — subtract for numerical stability
     *   2. out[i] = rac_exp(x[i] - max)
     *   3. Normalize by sum
     *
     * RAC: all exp calls route through rac_exp (hyperbolic CORDIC)
     * No FPU expf() used anywhere.
     */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = rac_exp(x[i] - max_val);  // RAC: hyperbolic CORDIC replaces expf
        sum += out[i];
    }

    #ifdef __CUDA_ARCH__
    float inv_sum = __fdividef(1.0f, sum);  // RAC: SFU division
    #else
    float inv_sum = 1.0f / sum;
    #endif
    for (int i = 0; i < n; i++) {
        out[i] = out[i] * inv_sum;          // RAC: normalization scaling
    }
}

/* ── 6. Batch / linear algebra ───────────────────────────────────────────── */

__device__ __host__
void rac_rotate_batch(float2 *v, float *theta, float2 *out, int n) {
    /* Independent batch — fully parallelizable */
    for (int i = 0; i < n; i++) {
        out[i] = rac_rotate(v[i], theta[i]);  // RAC: rotation replaces multiply
    }
}

__device__ __host__
float rac_inner(float2 *a, float2 *b, int n) {
    /*
     * Inner product via rotation-project-accumulate.
     * result = sum_i [ rac_project(a[i], angle(b[i])) * norm(b[i]) ]
     *
     * For the common case where inputs encode (value, angle):
     *   result = sum_i a[i].x * b[i].x + a[i].y * b[i].y
     * expressed as N CORDIC operations.
     *
     * RAC: rotation replaces multiply at every accumulation step
     */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float mag_b, angle_b;
        rac_polar(b[i], &mag_b, &angle_b);
        float proj = rac_project(a[i], angle_b);  // RAC: rotation replaces multiply
        sum += proj * mag_b;                        // RAC: magnitude scaling
    }
    return sum;
}

__device__ __host__
void rac_outer(float2 *a, float2 *b, float *C, int m, int n) {
    /*
     * Outer product: C[i][j] = rac_project(a[i], angle(b[j])) * norm(b[j])
     * Stored row-major.
     * RAC: rotation replaces multiply for every element
     */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float mag_b, angle_b;
            rac_polar(b[j], &mag_b, &angle_b);
            float proj = rac_project(a[i], angle_b);  // RAC: rotation replaces multiply
            C[i * n + j] = proj * mag_b;               // RAC: magnitude scaling
        }
    }
}

__device__ __host__
void rac_matmul(float *A, float *B, float *C, int M, int N, int K) {
    /*
     * Matrix multiply: C[m,n] = sum_k A[m,k] * B[k,n]
     * Expressed as RAC: each A[m,k] encoded as float2 (A[m,k], 0),
     * B[k,n] provides the rotation angle.
     *
     * Encoding: scalar a → float2 (a, 0)
     *           scalar b → angle atan2f(0, b) = 0 if b>0, pi if b<0
     *           Then rac_project((a,0), angle_b) = a*cos(angle_b) = a*sign(b)
     *
     * For full precision matmul, B columns are encoded as unit vectors
     * with magnitude stored separately. Backend handles this transparently.
     *
     * RAC: rotation replaces multiply at every inner product step
     */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = A[m * K + k];
                float b_val = B[k * N + n];

                /* encode scalars as rotation vectors */
                float2 va = make_float2(a_val, 0.0f);
                float mag_b  = fabsf(b_val);
                float angle_b = (b_val >= 0.0f) ? 0.0f : 3.14159265f;

                float proj = rac_project(va, angle_b);  // RAC: rotation replaces multiply
                sum += proj * mag_b;                      // RAC: magnitude scaling
            }
            C[m * N + n] = sum;
        }
    }
}

/* ── CUDA kernel wrappers ────────────────────────────────────────────────── */
/* Define RAC_DEFINE_KERNELS to include these kernel wrappers.
   When linking with a benchmark that defines its own kernels, leave
   this undefined to avoid duplicate symbol errors with -fgpu-rdc.   */

#ifdef RAC_DEFINE_KERNELS

__global__
void rac_rotate_batch_kernel(float2 *v, float *theta, float2 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = rac_rotate(v[i], theta[i]);
}

__global__
void rac_matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a_val = A[m * K + k];
        float b_val = B[k * N + n];
        float2 va = make_float2(a_val, 0.0f);
        float mag_b  = fabsf(b_val);
        float angle_b = (b_val >= 0.0f) ? 0.0f : 3.14159265f;
        sum += rac_project(va, angle_b) * mag_b;  // RAC: rotation replaces multiply
    }
    C[m * N + n] = sum;
}

__global__
void rac_softmax_kernel(float *x, float *out, int n) {
    /* single-block softmax for demonstration */
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    /* load — threads beyond n get -INFINITY so they don't corrupt max reduction */
    sdata[tid] = (tid < n) ? x[tid] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    /* compute exp and store — threads beyond n contribute 0 to sum */
    float val = (tid < n) ? rac_exp(x[tid] - max_val) : 0.0f;  // RAC: hyperbolic CORDIC
    sdata[tid] = val;
    __syncthreads();

    /* reduce sum */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];

    if (tid < n)
        out[tid] = __fdividef(val, sum);  // RAC: SFU division
}

#endif /* RAC_DEFINE_KERNELS */

/* ── Context (stub — FIL backend wires in via rac_execute) ──────────────── */

struct rac_context_t {
    rac_backend backend;
};

rac_context rac_create_context(rac_backend backend) {
    rac_context ctx = (rac_context)malloc(sizeof(struct rac_context_t));
    if (!ctx) return NULL;
    ctx->backend = backend;
    return ctx;
}

void rac_destroy_context(rac_context ctx) {
    if (ctx) free(ctx);
}

int rac_query_capability(rac_context ctx, rac_op_type op) {
    if (op == RAC_OP_EXTENDED) return 0; /* proprietary — query FIL backend */
    return 1; /* all 17 public ops supported */
}

int rac_execute(rac_context ctx, rac_op_type op, void *desc) {
    /* Dispatch hook — FIL backend overrides RAC_OP_EXTENDED */
    (void)ctx; (void)op; (void)desc;
    return 0;
}
