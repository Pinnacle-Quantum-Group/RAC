/*
 * rac_torch_bind.cpp — PyTorch pybind11 bindings (pure C++, no HIP)
 * Compiled with g++, NOT hipcc. Links against the .hip kernel .so.
 */

#include <torch/extension.h>
#include <vector>

#ifdef USE_ROCM
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAStream.h>
#endif

/* Kernel functions from rac_kernels.hip (compiled separately) */
extern "C" {
void rac_launch_nn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    void* stream);  /* hipStream_t is just a void* at ABI level */
void rac_launch_nt(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    void* stream);
void rac_launch_tn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    void* stream);
/* Tunable-precision variants. mode: 0=FAST (sign-XOR FMA, iters ignored),
 * 1=CORDIC (N=iters iterative micro-rotations, matches rac_cuda.cu),
 * 2=SHIFTADD (pure integer shift-add, N=iters, no FP multiplier). */
void rac_launch_nn_iters(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    int iters, int mode, void* stream);
void rac_launch_nt_iters(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    int iters, int mode, void* stream);
void rac_launch_tn_iters(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    int iters, int mode, void* stream);
/* no init needed */
}

/* Process-wide CORDIC iteration count + mode. Set from Python via
 * rac_cuda_ext.set_cordic_iters(n) / set_cordic_mode(m).
 * Defaults: iters=24, mode=0 (FAST — single sign-XOR FMA, hardware
 * multiplier engaged). */
#define RAC_MODE_FAST     0
#define RAC_MODE_CORDIC   1
#define RAC_MODE_SHIFTADD 2
static int g_cordic_iters = 24;
static int g_cordic_mode  = RAC_MODE_FAST;

/* Get the raw HIP/CUDA stream from PyTorch */
static void* _get_stream() {
#ifdef USE_ROCM
    return (void*)c10::hip::getCurrentHIPStream().stream();
#else
    return (void*)c10::cuda::getCurrentCUDAStream().stream();
#endif
}

/* ── ATen dispatch ───────────────────────────────────────────────────── */

torch::Tensor rac_matmul_forward(torch::Tensor A, torch::Tensor B,
                                  int iters, int mode) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "RAC: inputs must be GPU tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "RAC: float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "RAC: 2D tensors required");
    TORCH_CHECK(A.size(1) == B.size(0), "RAC: dimension mismatch");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    if (iters <= 0) iters = g_cordic_iters;
    if (mode  <  0) mode  = g_cordic_mode;
    rac_launch_nn_iters(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, 1.0f, 0.0f, iters, mode, _get_stream());

    return C;
}

torch::Tensor rac_linear_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int iters, int mode)
{
    TORCH_CHECK(input.is_cuda(), "RAC: input must be GPU tensor");

    auto orig_dtype = input.scalar_type();
    if (orig_dtype != torch::kFloat32) {
        input = input.to(torch::kFloat32);
        weight = weight.to(torch::kFloat32);
        if (bias.defined() && bias.numel() > 0) bias = bias.to(torch::kFloat32);
    }

    auto in_shape = input.sizes().vec();
    int out_features = weight.size(0);
    int in_features = weight.size(1);

    auto input_2d = input.reshape({-1, in_features}).contiguous();
    int M = input_2d.size(0);

    auto output = torch::empty({M, out_features}, input_2d.options());

    if (iters <= 0) iters = g_cordic_iters;
    if (mode  <  0) mode  = g_cordic_mode;

    /* NT kernel: C[M,out] = input[M,in] @ weight[out,in]^T — no transpose needed */
    rac_launch_nt_iters(
        input_2d.data_ptr<float>(), weight.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        M, out_features, in_features, 1.0f, 0.0f, iters, mode, _get_stream());

    if (bias.defined() && bias.numel() > 0)
        output.add_(bias);

    in_shape.back() = out_features;
    auto result = output.reshape(in_shape);
    if (orig_dtype != torch::kFloat32) result = result.to(orig_dtype);
    return result;
}

std::vector<torch::Tensor> rac_matmul_backward(
    torch::Tensor grad_C, torch::Tensor A, torch::Tensor B,
    bool need_grad_A, bool need_grad_B, int iters, int mode)
{
    grad_C = grad_C.contiguous();
    A = A.contiguous();
    B = B.contiguous();
    int M = A.size(0), K = A.size(1), N = B.size(1);

    if (iters <= 0) iters = g_cordic_iters;
    if (mode  <  0) mode  = g_cordic_mode;

    torch::Tensor grad_A, grad_B;

    if (need_grad_A) {
        /* grad_A[M,K] = grad_C[M,N] @ B[K,N]^T → NT kernel */
        grad_A = torch::empty({M, K}, A.options());
        rac_launch_nt_iters(
            grad_C.data_ptr<float>(), B.data_ptr<float>(),
            grad_A.data_ptr<float>(),
            M, K, N, 1.0f, 0.0f, iters, mode, _get_stream());
    }

    if (need_grad_B) {
        /* grad_B[K,N] = A[M,K]^T @ grad_C[M,N] → TN kernel with A=[M,K] treated as [K,M]^T */
        grad_B = torch::empty({K, N}, B.options());
        rac_launch_tn_iters(
            A.data_ptr<float>(), grad_C.data_ptr<float>(),
            grad_B.data_ptr<float>(),
            K, N, M, 1.0f, 0.0f, iters, mode, _get_stream());
    }

    return {grad_A, grad_B};
}

std::vector<torch::Tensor> rac_linear_backward(
    torch::Tensor grad_output, torch::Tensor input,
    torch::Tensor weight, bool need_bias_grad, int iters, int mode)
{
    int out_features = weight.size(0);
    int in_features = weight.size(1);

    if (iters <= 0) iters = g_cordic_iters;
    if (mode  <  0) mode  = g_cordic_mode;

    auto go = grad_output.reshape({-1, out_features}).contiguous();
    auto inp = input.reshape({-1, in_features}).contiguous();
    int M = inp.size(0);

    /* grad_input[M,in] = go[M,out] @ weight[out,in] → NN kernel (weight already [out,in]) */
    auto grad_input = torch::empty({M, in_features}, inp.options());
    rac_launch_nn_iters(
        go.data_ptr<float>(), weight.contiguous().data_ptr<float>(),
        grad_input.data_ptr<float>(),
        M, in_features, out_features, 1.0f, 0.0f, iters, mode, _get_stream());

    /* grad_weight[out,in] = go[M,out]^T @ inp[M,in] → TN kernel */
    auto grad_weight = torch::empty({out_features, in_features}, weight.options());
    rac_launch_tn_iters(
        go.data_ptr<float>(), inp.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        out_features, in_features, M, 1.0f, 0.0f, iters, mode, _get_stream());

    torch::Tensor grad_bias;
    if (need_bias_grad) grad_bias = go.sum(0);

    auto in_shape = input.sizes().vec();
    in_shape.back() = in_features;
    return {grad_input.reshape(in_shape), grad_weight, grad_bias};
}

/* ── pybind11 ────────────────────────────────────────────────────────── */

/* Process-wide iteration-count knob. Python calls this from
 * rac_set_precision(iters). Zero / negative means "use default (24)". */
static void rac_set_cordic_iters(int iters) {
    if (iters <= 0) { g_cordic_iters = 24; return; }
    if (iters > 24) iters = 24;
    if (iters < 1)  iters = 1;
    g_cordic_iters = iters;
}

static int rac_get_cordic_iters() { return g_cordic_iters; }

/* Process-wide compute-path selector. 0=FAST (sign-XOR FMA), 1=CORDIC
 * (rac_cuda.cu-style iterative rotation), 2=SHIFTADD (integer ALU only). */
static void rac_set_cordic_mode(int mode) {
    if (mode < 0 || mode > 2) mode = RAC_MODE_FAST;
    g_cordic_mode = mode;
}

static int rac_get_cordic_mode() { return g_cordic_mode; }

PYBIND11_MODULE(rac_cuda_ext, m) {
    m.doc() = "RAC: Rotation-Accumulate PyTorch Extension";
    /* RAC primitives use sign decomposition — no init needed */
    m.def("matmul_forward", &rac_matmul_forward, "RAC matmul forward",
          py::arg("A"), py::arg("B"),
          py::arg("iters")=0, py::arg("mode")=-1);
    m.def("matmul_backward", &rac_matmul_backward, "RAC matmul backward",
          py::arg("grad_C"), py::arg("A"), py::arg("B"),
          py::arg("need_grad_A")=true, py::arg("need_grad_B")=true,
          py::arg("iters")=0, py::arg("mode")=-1);
    m.def("linear_forward", &rac_linear_forward, "RAC linear forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("iters")=0, py::arg("mode")=-1);
    m.def("linear_backward", &rac_linear_backward, "RAC linear backward",
          py::arg("grad_output"), py::arg("input"), py::arg("weight"),
          py::arg("need_bias_grad"),
          py::arg("iters")=0, py::arg("mode")=-1);

    m.def("set_cordic_iters", &rac_set_cordic_iters,
          "Set the process-wide CORDIC iteration count (1..24). "
          "Higher = more precision + more cycles.",
          py::arg("iters"));
    m.def("get_cordic_iters", &rac_get_cordic_iters,
          "Return the current process-wide CORDIC iteration count.");
    m.def("set_cordic_mode", &rac_set_cordic_mode,
          "Set compute path: 0=FAST (sign-XOR FMA, hardware multiplier), "
          "1=CORDIC (iterative rotation, port of rac_cuda.cu), "
          "2=SHIFTADD (integer ALU only, no multiplier).",
          py::arg("mode"));
    m.def("get_cordic_mode", &rac_get_cordic_mode,
          "Return the current compute path mode (0/1/2).");

    m.attr("MODE_FAST")     = py::int_(RAC_MODE_FAST);
    m.attr("MODE_CORDIC")   = py::int_(RAC_MODE_CORDIC);
    m.attr("MODE_SHIFTADD") = py::int_(RAC_MODE_SHIFTADD);
}
