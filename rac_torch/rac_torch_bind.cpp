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
}

/* Get the raw HIP stream from PyTorch */
static void* _get_stream() {
    /* PyTorch's current stream, cast to raw pointer */
#ifdef USE_ROCM
    return nullptr;  /* default stream — synchronous on HIP */
#else
    return (void*)c10::cuda::getCurrentCUDAStream().stream();
#endif
}

/* ── ATen dispatch ───────────────────────────────────────────────────── */

torch::Tensor rac_matmul_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "RAC: inputs must be GPU tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "RAC: float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "RAC: 2D tensors required");
    TORCH_CHECK(A.size(1) == B.size(0), "RAC: dimension mismatch");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0), K = A.size(1), N = B.size(1);

    torch::cuda::synchronize();
    auto C = torch::zeros({M, N}, A.options());

    rac_launch_nn(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, 1.0f, 0.0f, _get_stream());

    torch::cuda::synchronize();
    return C;
}

torch::Tensor rac_linear_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
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
    auto weight_t = weight.t().contiguous();
    int M = input_2d.size(0);

    /* Sync to ensure weight transpose is complete before kernel reads it */
    torch::cuda::synchronize();

    auto output = torch::zeros({M, out_features}, input_2d.options());

    rac_launch_nn(
        input_2d.data_ptr<float>(), weight_t.data_ptr<float>(),
        output.data_ptr<float>(),
        M, out_features, in_features, 1.0f, 0.0f, _get_stream());

    /* Sync after kernel to ensure output is ready */
    torch::cuda::synchronize();

    if (bias.defined() && bias.numel() > 0)
        output.add_(bias);

    in_shape.back() = out_features;
    auto result = output.reshape(in_shape);
    if (orig_dtype != torch::kFloat32) result = result.to(orig_dtype);
    return result;
}

std::vector<torch::Tensor> rac_matmul_backward(
    torch::Tensor grad_C, torch::Tensor A, torch::Tensor B,
    bool need_grad_A, bool need_grad_B)
{
    grad_C = grad_C.contiguous();
    A = A.contiguous();
    B = B.contiguous();
    int M = A.size(0), K = A.size(1), N = B.size(1);

    torch::Tensor grad_A, grad_B;

    if (need_grad_A) {
        grad_A = torch::empty({M, K}, A.options());
        auto Bt = B.t().contiguous();
        rac_launch_nn(
            grad_C.data_ptr<float>(), Bt.data_ptr<float>(),
            grad_A.data_ptr<float>(),
            M, K, N, 1.0f, 0.0f, _get_stream());
    }

    if (need_grad_B) {
        grad_B = torch::empty({K, N}, B.options());
        auto At = A.t().contiguous();
        rac_launch_nn(
            At.data_ptr<float>(), grad_C.data_ptr<float>(),
            grad_B.data_ptr<float>(),
            K, N, M, 1.0f, 0.0f, _get_stream());
    }

    return {grad_A, grad_B};
}

std::vector<torch::Tensor> rac_linear_backward(
    torch::Tensor grad_output, torch::Tensor input,
    torch::Tensor weight, bool need_bias_grad)
{
    int out_features = weight.size(0);
    int in_features = weight.size(1);

    auto go = grad_output.reshape({-1, out_features}).contiguous();
    auto inp = input.reshape({-1, in_features}).contiguous();
    int M = inp.size(0);

    auto grad_input = torch::empty({M, in_features}, inp.options());
    rac_launch_nn(
        go.data_ptr<float>(), weight.contiguous().data_ptr<float>(),
        grad_input.data_ptr<float>(),
        M, in_features, out_features, 1.0f, 0.0f, _get_stream());

    auto grad_weight = torch::empty({out_features, in_features}, weight.options());
    auto got = go.t().contiguous();
    rac_launch_nn(
        got.data_ptr<float>(), inp.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        out_features, in_features, M, 1.0f, 0.0f, _get_stream());

    torch::Tensor grad_bias;
    if (need_bias_grad) grad_bias = go.sum(0);

    auto in_shape = input.sizes().vec();
    in_shape.back() = in_features;
    return {grad_input.reshape(in_shape), grad_weight, grad_bias};
}

/* ── pybind11 ────────────────────────────────────────────────────────── */

PYBIND11_MODULE(rac_cuda_ext, m) {
    m.doc() = "RAC: Rotation-Accumulate PyTorch Extension";
    m.def("matmul_forward", &rac_matmul_forward, "RAC matmul forward");
    m.def("matmul_backward", &rac_matmul_backward, "RAC matmul backward",
          py::arg("grad_C"), py::arg("A"), py::arg("B"),
          py::arg("need_grad_A")=true, py::arg("need_grad_B")=true);
    m.def("linear_forward", &rac_linear_forward, "RAC linear forward");
    m.def("linear_backward", &rac_linear_backward, "RAC linear backward");
}
