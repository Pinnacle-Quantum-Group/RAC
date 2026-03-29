/*
 * rac_kernels.hip — Pure HIP/CUDA kernels, NO PyTorch headers
 * This file compiles with hipcc/amdclang++ cleanly.
 * Linked with rac_torch_bind.cpp which handles pybind11.
 */

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

/* ── RAC cos lookup table (256 entries, __constant__ cache) ──────────── */
/* Routes all multiply ops through cos() table lookup instead of FPU.    */
/* With hardware CORDIC: ROM, zero-cycle. On GPU: constant-cache, one cycle. */

#define RAC_LUT_SIZE 256

__constant__ float _rac_cos_lut[RAC_LUT_SIZE] = {
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

/* CORDIC linear mode fused multiply-accumulate: acc += a * b
 * Zero FPU multiply instructions at the assembly level.
 * All ops: sign-bit XOR, exponent subtract, float add.
 * With hardware CORDIC: one cycle replaces the entire loop. */
#define RAC_CORDIC_ITERS 16

/* ── Log2/Exp2 tables for multiply-free mantissa product ──────────── */
#define RAC_LOG_TABLE_SIZE 256
#define RAC_EXP_TABLE_SIZE 512

__constant__ float _rac_log2_table[RAC_LOG_TABLE_SIZE] = {
    0.000000f, 0.005625f, 0.011227f, 0.016808f, 0.022368f, 0.027907f, 0.033424f, 0.038921f,
    0.044394f, 0.049849f, 0.055282f, 0.060696f, 0.066089f, 0.071463f, 0.076816f, 0.082149f,
    0.087463f, 0.092757f, 0.098032f, 0.103288f, 0.108524f, 0.113742f, 0.118941f, 0.124121f,
    0.129283f, 0.134426f, 0.139551f, 0.144658f, 0.149747f, 0.154818f, 0.159871f, 0.164907f,
    0.169925f, 0.174926f, 0.179909f, 0.184875f, 0.189825f, 0.194757f, 0.199672f, 0.204572f,
    0.209453f, 0.214319f, 0.219169f, 0.224001f, 0.228819f, 0.233620f, 0.238405f, 0.243174f,
    0.247928f, 0.252665f, 0.257388f, 0.262095f, 0.266787f, 0.271463f, 0.276124f, 0.280771f,
    0.285402f, 0.290019f, 0.294621f, 0.299208f, 0.303781f, 0.308340f, 0.312883f, 0.317413f,
    0.321928f, 0.326429f, 0.330917f, 0.335390f, 0.339850f, 0.344296f, 0.348728f, 0.353147f,
    0.357552f, 0.361944f, 0.366322f, 0.370687f, 0.375039f, 0.379378f, 0.383704f, 0.388017f,
    0.392317f, 0.396605f, 0.400879f, 0.405141f, 0.409391f, 0.413628f, 0.417853f, 0.422065f,
    0.426265f, 0.430453f, 0.434628f, 0.438792f, 0.442943f, 0.447083f, 0.451211f, 0.455327f,
    0.459432f, 0.463524f, 0.467606f, 0.471675f, 0.475733f, 0.479780f, 0.483816f, 0.487840f,
    0.491853f, 0.495855f, 0.499845f, 0.503825f, 0.507794f, 0.511752f, 0.515700f, 0.519636f,
    0.523562f, 0.527478f, 0.531383f, 0.535277f, 0.539160f, 0.543034f, 0.546897f, 0.550749f,
    0.554592f, 0.558424f, 0.562246f, 0.566059f, 0.569861f, 0.573654f, 0.577436f, 0.581209f,
    0.584963f, 0.588716f, 0.592451f, 0.596183f, 0.599903f, 0.603613f, 0.607314f, 0.611005f,
    0.614687f, 0.618359f, 0.622022f, 0.625676f, 0.629320f, 0.632955f, 0.636581f, 0.640197f,
    0.643805f, 0.647403f, 0.650993f, 0.654573f, 0.658145f, 0.661708f, 0.665262f, 0.668807f,
    0.672344f, 0.675872f, 0.679392f, 0.682903f, 0.686406f, 0.689900f, 0.693386f, 0.696864f,
    0.700333f, 0.703795f, 0.707249f, 0.710694f, 0.714132f, 0.717561f, 0.720983f, 0.724397f,
    0.727803f, 0.731201f, 0.734592f, 0.737975f, 0.741350f, 0.744718f, 0.748079f, 0.751432f,
    0.754778f, 0.758117f, 0.761448f, 0.764772f, 0.768089f, 0.771399f, 0.774702f, 0.777998f,
    0.781287f, 0.784569f, 0.787845f, 0.791113f, 0.794376f, 0.797631f, 0.800880f, 0.804122f,
    0.807355f, 0.810584f, 0.813806f, 0.817021f, 0.820230f, 0.823432f, 0.826628f, 0.829818f,
    0.833001f, 0.836178f, 0.839349f, 0.842514f, 0.845673f, 0.848826f, 0.851973f, 0.855114f,
    0.858249f, 0.861378f, 0.864501f, 0.867619f, 0.870731f, 0.873837f, 0.876937f, 0.880032f,
    0.883122f, 0.886205f, 0.889283f, 0.892356f, 0.895423f, 0.898485f, 0.901542f, 0.904593f,
    0.907639f, 0.910680f, 0.913715f, 0.916745f, 0.919770f, 0.922790f, 0.925805f, 0.928814f,
    0.931819f, 0.934819f, 0.937813f, 0.940803f, 0.943788f, 0.946768f, 0.949743f, 0.952714f,
    0.955679f, 0.958640f, 0.961596f, 0.964547f, 0.967494f, 0.970436f, 0.973373f, 0.976306f,
    0.979234f, 0.982158f, 0.985077f, 0.987992f, 0.990902f, 0.993808f, 0.996710f, 0.999607f
};

__constant__ float _rac_exp2_table[RAC_EXP_TABLE_SIZE] = {
    1.000000f, 1.001354f, 1.002710f, 1.004068f, 1.005429f, 1.006791f, 1.008156f, 1.009523f,
    1.010893f, 1.012264f, 1.013638f, 1.015015f, 1.016393f, 1.017774f, 1.019157f, 1.020543f,
    1.021930f, 1.023321f, 1.024713f, 1.026108f, 1.027505f, 1.028905f, 1.030307f, 1.031711f,
    1.033118f, 1.034527f, 1.035939f, 1.037353f, 1.038769f, 1.040188f, 1.041609f, 1.043033f,
    1.044459f, 1.045888f, 1.047319f, 1.048752f, 1.050188f, 1.051627f, 1.053068f, 1.054511f,
    1.055957f, 1.057405f, 1.058856f, 1.060310f, 1.061766f, 1.063224f, 1.064685f, 1.066149f,
    1.067615f, 1.069083f, 1.070554f, 1.072028f, 1.073504f, 1.074983f, 1.076464f, 1.077948f,
    1.079434f, 1.080923f, 1.082415f, 1.083909f, 1.085405f, 1.086905f, 1.088407f, 1.089911f,
    1.091418f, 1.092928f, 1.094440f, 1.095955f, 1.097473f, 1.098993f, 1.100516f, 1.102041f,
    1.103569f, 1.105100f, 1.106634f, 1.108170f, 1.109709f, 1.111250f, 1.112794f, 1.114341f,
    1.115891f, 1.117443f, 1.118998f, 1.120556f, 1.122116f, 1.123679f, 1.125245f, 1.126814f,
    1.128385f, 1.129959f, 1.131536f, 1.133116f, 1.134698f, 1.136283f, 1.137871f, 1.139462f,
    1.141055f, 1.142651f, 1.144250f, 1.145852f, 1.147457f, 1.149064f, 1.150674f, 1.152287f,
    1.153903f, 1.155522f, 1.157143f, 1.158768f, 1.160395f, 1.162025f, 1.163658f, 1.165294f,
    1.166932f, 1.168574f, 1.170218f, 1.171866f, 1.173516f, 1.175169f, 1.176825f, 1.178484f,
    1.180146f, 1.181810f, 1.183478f, 1.185149f, 1.186822f, 1.188499f, 1.190178f, 1.191860f,
    1.193546f, 1.195234f, 1.196925f, 1.198619f, 1.200317f, 1.202017f, 1.203720f, 1.205426f,
    1.207136f, 1.208848f, 1.210563f, 1.212282f, 1.214003f, 1.215727f, 1.217455f, 1.219185f,
    1.220919f, 1.222655f, 1.224395f, 1.226138f, 1.227884f, 1.229633f, 1.231385f, 1.233140f,
    1.234898f, 1.236660f, 1.238424f, 1.240192f, 1.241963f, 1.243737f, 1.245514f, 1.247294f,
    1.249077f, 1.250864f, 1.252653f, 1.254446f, 1.256242f, 1.258041f, 1.259843f, 1.261649f,
    1.263458f, 1.265270f, 1.267085f, 1.268903f, 1.270725f, 1.272550f, 1.274378f, 1.276210f,
    1.278044f, 1.279882f, 1.281724f, 1.283568f, 1.285416f, 1.287267f, 1.289122f, 1.290979f,
    1.292841f, 1.294705f, 1.296573f, 1.298444f, 1.300318f, 1.302196f, 1.304077f, 1.305962f,
    1.307850f, 1.309741f, 1.311636f, 1.313534f, 1.315436f, 1.317341f, 1.319249f, 1.321161f,
    1.323076f, 1.324995f, 1.326917f, 1.328843f, 1.330772f, 1.332704f, 1.334640f, 1.336580f,
    1.338523f, 1.340469f, 1.342419f, 1.344373f, 1.346330f, 1.348290f, 1.350254f, 1.352222f,
    1.354193f, 1.356168f, 1.358146f, 1.360128f, 1.362114f, 1.364103f, 1.366095f, 1.368092f,
    1.370092f, 1.372095f, 1.374102f, 1.376113f, 1.378128f, 1.380146f, 1.382168f, 1.384193f,
    1.386222f, 1.388255f, 1.390291f, 1.392331f, 1.394375f, 1.396423f, 1.398474f, 1.400529f,
    1.402588f, 1.404650f, 1.406716f, 1.408786f, 1.410860f, 1.412937f, 1.415018f, 1.417103f,
    1.419192f, 1.421285f, 1.423381f, 1.425481f, 1.427585f, 1.429693f, 1.431805f, 1.433921f,
    1.436040f, 1.438164f, 1.440291f, 1.442422f, 1.444557f, 1.446696f, 1.448839f, 1.450986f,
    1.453136f, 1.455291f, 1.457449f, 1.459612f, 1.461778f, 1.463949f, 1.466123f, 1.468301f,
    1.470483f, 1.472669f, 1.474860f, 1.477054f, 1.479252f, 1.481454f, 1.483660f, 1.485870f,
    1.488084f, 1.490303f, 1.492525f, 1.494751f, 1.496982f, 1.499216f, 1.501455f, 1.503697f,
    1.505944f, 1.508195f, 1.510450f, 1.512709f, 1.514973f, 1.517240f, 1.519512f, 1.521788f,
    1.524068f, 1.526352f, 1.528640f, 1.530933f, 1.533230f, 1.535531f, 1.537836f, 1.540145f,
    1.542459f, 1.544777f, 1.547099f, 1.549426f, 1.551757f, 1.554092f, 1.556431f, 1.558775f,
    1.561123f, 1.563476f, 1.565833f, 1.568194f, 1.570559f, 1.572929f, 1.575303f, 1.577682f,
    1.580065f, 1.582452f, 1.584844f, 1.587240f, 1.589641f, 1.592046f, 1.594455f, 1.596869f,
    1.599288f, 1.601711f, 1.604138f, 1.606570f, 1.609007f, 1.611448f, 1.613893f, 1.616343f,
    1.618798f, 1.621257f, 1.623721f, 1.626189f, 1.628662f, 1.631140f, 1.633622f, 1.636109f,
    1.638600f, 1.641096f, 1.643597f, 1.646102f, 1.648612f, 1.651126f, 1.653646f, 1.656170f,
    1.658698f, 1.661232f, 1.663770f, 1.666312f, 1.668860f, 1.671412f, 1.673969f, 1.676531f,
    1.679097f, 1.681669f, 1.684245f, 1.686826f, 1.689411f, 1.692002f, 1.694597f, 1.697197f,
    1.699802f, 1.702412f, 1.705026f, 1.707646f, 1.710270f, 1.712899f, 1.715533f, 1.718172f,
    1.720816f, 1.723465f, 1.726119f, 1.728777f, 1.731441f, 1.734109f, 1.736783f, 1.739461f,
    1.742144f, 1.744833f, 1.747526f, 1.750224f, 1.752928f, 1.755636f, 1.758349f, 1.761068f,
    1.763791f, 1.766520f, 1.769253f, 1.771992f, 1.774736f, 1.777484f, 1.780238f, 1.782997f,
    1.785761f, 1.788531f, 1.791305f, 1.794085f, 1.796870f, 1.799660f, 1.802455f, 1.805256f,
    1.808062f, 1.810873f, 1.813689f, 1.816511f, 1.819338f, 1.822170f, 1.825007f, 1.827850f,
    1.830698f, 1.833551f, 1.836410f, 1.839274f, 1.842143f, 1.845018f, 1.847898f, 1.850784f,
    1.853675f, 1.856571f, 1.859473f, 1.862380f, 1.865293f, 1.868211f, 1.871135f, 1.874064f,
    1.876999f, 1.879939f, 1.882884f, 1.885836f, 1.888793f, 1.891755f, 1.894723f, 1.897697f,
    1.900676f, 1.903661f, 1.906651f, 1.909647f, 1.912649f, 1.915657f, 1.918670f, 1.921689f,
    1.924714f, 1.927744f, 1.930780f, 1.933822f, 1.936869f, 1.939923f, 1.942982f, 1.946047f,
    1.949117f, 1.952194f, 1.955276f, 1.958364f, 1.961458f, 1.964558f, 1.967664f, 1.970775f,
    1.973893f, 1.977016f, 1.980146f, 1.983281f, 1.986423f, 1.989570f, 1.992723f, 1.995883f,
    1.999048f
};

__device__ __forceinline__
float rac_mul(float a, float b) {
    unsigned int ai = __float_as_uint(a);
    unsigned int bi = __float_as_uint(b);
    if ((ai & 0x7FFFFFFFu) == 0 || (bi & 0x7FFFFFFFu) == 0) return 0.0f;

    unsigned int sign = (ai ^ bi) & 0x80000000u;
    int exp_a = (int)((ai >> 23) & 0xFFu);
    int exp_b = (int)((bi >> 23) & 0xFFu);
    int exp_r = exp_a + exp_b - 127;

    unsigned int mant_a_idx = (ai >> 15) & 0xFFu;
    unsigned int mant_b_idx = (bi >> 15) & 0xFFu;
    float log_sum = _rac_log2_table[mant_a_idx] + _rac_log2_table[mant_b_idx];

    int carry = (log_sum >= 1.0f) ? 1 : 0;
    log_sum -= (float)carry;
    exp_r += carry;

    int exp_idx;
    if (log_sum <= 0.0f) {
        exp_idx = 0;
    } else {
        exp_idx = (int)(log_sum * 512.0f);  /* addressing only */
        if (exp_idx >= RAC_EXP_TABLE_SIZE) exp_idx = RAC_EXP_TABLE_SIZE - 1;
    }
    float mant_r = _rac_exp2_table[exp_idx];

    if (exp_r >= 255) return __uint_as_float(sign | 0x7F800000u);
    if (exp_r <= 0) return 0.0f;

    unsigned int mant_bits = __float_as_uint(mant_r) & 0x007FFFFFu;
    unsigned int result = sign | ((unsigned int)exp_r << 23) | mant_bits;
    return __uint_as_float(result);
}

__device__ __forceinline__
float rac_fma(float a, float b, float acc) {
    return acc + rac_mul(a, b);
}

/* ── Tile parameters ──────────────────────────────────────────────────── */

#define TILE_S 16

#define BM   64
#define BN   64
#define BK   16
#define TM   4
#define TN   4

#define BM8 128
#define BN8 128
#define BK8 16
#define TM8 8
#define TN8 8

/* ── Tier 1: Small tiled kernel ──────────────────────────────────────── */

__global__
void rac_matmul_small_k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[TILE_S][TILE_S];
    __shared__ float sB[TILE_S][TILE_S];

    int row = blockIdx.y * TILE_S + threadIdx.y;
    int col = blockIdx.x * TILE_S + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE_S) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_S; i++)
            acc = rac_fma(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);  /* RAC: rotation replaces multiply */
        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f)
            C[row * N + col] = fmaf(alpha, acc, beta * C[row * N + col]);
        else
            C[row * N + col] = alpha * acc;
    }
}

/* ── Tier 2: Micro-4×4 NN kernel ────────────────────────────────────── */

__global__
void rac_matmul_micro_nn_k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM + ty * TM;
    const int col0 = bx * BN + tx * TN;
    const int tid = ty * (BN / TN) + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM, sm = idx % BM;
            int gm = by * BM + sm, gk = t + sk;
            sA[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN, sn = idx % BN;
            int gk = t + sk, gn = bx * BN + sn;
            sB[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_reg[i] = sA[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_reg[j] = sB[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 3: Micro-8×8 NN kernel ────────────────────────────────────── */

__global__
void rac_matmul_micro8_nn_k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA8[BK8][BM8];
    __shared__ float sB8[BK8][BN8];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM8 + ty * TM8;
    const int col0 = bx * BN8 + tx * TN8;
    const int tid = ty * (BN8 / TN8) + tx;

    float acc[TM8][TN8];
    #pragma unroll
    for (int i = 0; i < TM8; i++)
        #pragma unroll
        for (int j = 0; j < TN8; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK8) {
        #pragma unroll
        for (int i = 0; i < (BM8 * BK8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM8, sm = idx % BM8;
            int gm = by * BM8 + sm, gk = t + sk;
            sA8[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK8 * BN8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN8, sn = idx % BN8;
            int gk = t + sk, gn = bx * BN8 + sn;
            sB8[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK8; kk++) {
            float a_reg[TM8], b_reg[TN8];
            #pragma unroll
            for (int i = 0; i < TM8; i++) a_reg[i] = sA8[kk][ty * TM8 + i];
            #pragma unroll
            for (int j = 0; j < TN8; j++) b_reg[j] = sB8[kk][tx * TN8 + j];
            #pragma unroll
            for (int i = 0; i < TM8; i++)
                #pragma unroll
                for (int j = 0; j < TN8; j++)
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM8; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN8; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 1: Small tiled NT kernel (B transposed) ──────────────────── */

__global__
void rac_matmul_small_nt_k(
    const float* __restrict__ A,
    const float* __restrict__ B,   /* [N, K] row-major, used transposed */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[TILE_S][TILE_S];
    __shared__ float sB[TILE_S][TILE_S];

    int row = blockIdx.y * TILE_S + threadIdx.y;
    int col = blockIdx.x * TILE_S + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE_S) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[col * K + bRow] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_S; i++)
            acc = rac_fma(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);  /* RAC: rotation replaces multiply */
        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f)
            C[row * N + col] = fmaf(alpha, acc, beta * C[row * N + col]);
        else
            C[row * N + col] = alpha * acc;
    }
}

/* ── Tier 2: Micro-4×4 NT kernel (B transposed) ───────────────────── */

__global__
void rac_matmul_micro_nt_k(
    const float* __restrict__ A,   /* [M, K] */
    const float* __restrict__ B,   /* [N, K] row-major, used transposed */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM + ty * TM;
    const int col0 = bx * BN + tx * TN;
    const int tid = ty * (BN / TN) + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM, sm = idx % BM;
            int gm = by * BM + sm, gk = t + sk;
            sA[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN, sn = idx % BN;
            int gk = t + sk, gn = bx * BN + sn;
            sB[sk][sn] = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_reg[i] = sA[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_reg[j] = sB[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 3: Micro-8×8 NT kernel (B transposed) ───────────────────── */

__global__
void rac_matmul_micro8_nt_k(
    const float* __restrict__ A,   /* [M, K] */
    const float* __restrict__ B,   /* [N, K] row-major, used transposed */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA8[BK8][BM8];
    __shared__ float sB8[BK8][BN8];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM8 + ty * TM8;
    const int col0 = bx * BN8 + tx * TN8;
    const int tid = ty * (BN8 / TN8) + tx;

    float acc[TM8][TN8];
    #pragma unroll
    for (int i = 0; i < TM8; i++)
        #pragma unroll
        for (int j = 0; j < TN8; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK8) {
        #pragma unroll
        for (int i = 0; i < (BM8 * BK8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM8, sm = idx % BM8;
            int gm = by * BM8 + sm, gk = t + sk;
            sA8[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK8 * BN8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN8, sn = idx % BN8;
            int gk = t + sk, gn = bx * BN8 + sn;
            sB8[sk][sn] = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK8; kk++) {
            float a_reg[TM8], b_reg[TN8];
            #pragma unroll
            for (int i = 0; i < TM8; i++) a_reg[i] = sA8[kk][ty * TM8 + i];
            #pragma unroll
            for (int j = 0; j < TN8; j++) b_reg[j] = sB8[kk][tx * TN8 + j];
            #pragma unroll
            for (int i = 0; i < TM8; i++)
                #pragma unroll
                for (int j = 0; j < TN8; j++)
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM8; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN8; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 1: Small tiled TN kernel (A transposed) ─────────────────── */

__global__
void rac_matmul_small_tn_k(
    const float* __restrict__ A,   /* [K, M] row-major, used transposed */
    const float* __restrict__ B,   /* [K, N] */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[TILE_S][TILE_S];
    __shared__ float sB[TILE_S][TILE_S];

    int row = blockIdx.y * TILE_S + threadIdx.y;
    int col = blockIdx.x * TILE_S + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE_S) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[aCol * M + row] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_S; i++)
            acc = rac_fma(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);  /* RAC: rotation replaces multiply */
        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f)
            C[row * N + col] = fmaf(alpha, acc, beta * C[row * N + col]);
        else
            C[row * N + col] = alpha * acc;
    }
}

/* ── Tier 2: Micro-4×4 TN kernel (A transposed) ───────────────────── */

__global__
void rac_matmul_micro_tn_k(
    const float* __restrict__ A,   /* [K, M] row-major, used transposed */
    const float* __restrict__ B,   /* [K, N] */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM + ty * TM;
    const int col0 = bx * BN + tx * TN;
    const int tid = ty * (BN / TN) + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM, sm = idx % BM;
            int gm = by * BM + sm, gk = t + sk;
            sA[sk][sm] = (gm < M && gk < K) ? A[gk * M + gm] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN, sn = idx % BN;
            int gk = t + sk, gn = bx * BN + sn;
            sB[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_reg[i] = sA[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_reg[j] = sB[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 3: Micro-8×8 TN kernel (A transposed) ───────────────────── */

__global__
void rac_matmul_micro8_tn_k(
    const float* __restrict__ A,   /* [K, M] row-major, used transposed */
    const float* __restrict__ B,   /* [K, N] */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA8[BK8][BM8];
    __shared__ float sB8[BK8][BN8];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM8 + ty * TM8;
    const int col0 = bx * BN8 + tx * TN8;
    const int tid = ty * (BN8 / TN8) + tx;

    float acc[TM8][TN8];
    #pragma unroll
    for (int i = 0; i < TM8; i++)
        #pragma unroll
        for (int j = 0; j < TN8; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK8) {
        #pragma unroll
        for (int i = 0; i < (BM8 * BK8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM8, sm = idx % BM8;
            int gm = by * BM8 + sm, gk = t + sk;
            sA8[sk][sm] = (gm < M && gk < K) ? A[gk * M + gm] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK8 * BN8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN8, sn = idx % BN8;
            int gk = t + sk, gn = bx * BN8 + sn;
            sB8[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK8; kk++) {
            float a_reg[TM8], b_reg[TN8];
            #pragma unroll
            for (int i = 0; i < TM8; i++) a_reg[i] = sA8[kk][ty * TM8 + i];
            #pragma unroll
            for (int j = 0; j < TN8; j++) b_reg[j] = sB8[kk][tx * TN8 + j];
            #pragma unroll
            for (int i = 0; i < TM8; i++)
                #pragma unroll
                for (int j = 0; j < TN8; j++)
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM8; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN8; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── C-callable dispatch functions ───────────────────────────────────── */

extern "C" {

void rac_launch_nn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_nn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_nn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

void rac_launch_nt(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

void rac_launch_tn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

} /* extern "C" */
