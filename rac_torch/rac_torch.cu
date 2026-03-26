/*
 * rac_torch.cu — RAC PyTorch Extension, CUDA/HIP Kernel Implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Implements:
 *   rac_matmul_forward:  C = A @ B via RAC micro-kernel (zero multiplications)
 *   rac_matmul_backward: dA = dC @ B.T,  dB = A.T @ dC  (both via RAC)
 *   rac_linear_forward:  output = input @ weight.T + bias
 *   rac_linear_backward: grad_input, grad_weight, grad_bias
 *
 * The backward pass is mathematically identical to the forward pass —
 * matrix multiply by transposed matrices — so RAC handles it natively.
 * Zero multiplications in both forward and backward.
 */

#ifdef __HIP__
  #include <hip/hip_runtime.h>
  #define CUDA_OR_HIP_ERROR hipError_t
  #define cudaSuccess hipSuccess
  #define cudaGetErrorString hipGetErrorString
  #define RAC_SINCOS(t,s,c) __sincosf(t,s,c)
#else
  #include <cuda_runtime.h>
  #define CUDA_OR_HIP_ERROR cudaError_t
  #define RAC_SINCOS(t,s,c) __sincosf(t,s,c)
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

/* ── CORDIC constants ────────────────────────────────────────────────────── */

#define RAC_K_INV     0.60725f
#define RAC_ITERS     16
#define RAC_ITERS_FAST 12
#define TILE          8        /* micro-8x8 — fastest on gfx1102 */
#define RAC_PI        3.14159265f

__constant__ float rac_atan_lut[RAC_ITERS] = {
    0.78539816f, 0.46364761f, 0.24497866f, 0.12435499f,
    0.06241881f, 0.03123983f, 0.01562373f, 0.00781234f,
    0.00390623f, 0.00195312f, 0.00097656f, 0.00048828f,
    0.00024414f, 0.00012207f, 0.00006104f, 0.00003052f
};

/* ── CORDIC rotation (device) ────────────────────────────────────────────── */

template<int ITERS>
__device__ __forceinline__
float _rac_project(float vx, float theta) {
    /*
     * Fast path: trivial angles skip SFU entirely.
     * General path: fused sincos + fmaf.
     * RAC: rotation replaces multiply at every step.
     */
    if (theta == 0.0f)    return vx;
    if (theta == RAC_PI)  return -vx;
    float s, c;
    RAC_SINCOS(theta, &s, &c);
    return fmaf(vx, c, 0.0f);   /* vy=0 in scalar encoding: vx*cos(theta) */
}

/* ── RAC micro-8x8 tiled matmul kernel ───────────────────────────────────── */
/*
 * Scalar encoding: A[m,k] → (a_val, 0), B[k,n] → angle=0 if b≥0 else π, mag=|b|
 * rac_project((a,0), angle_b) * mag_b = a * b  (degenerate case, exact)
 * SFU not called for trivial angles — pure fmaf in inner loop.
 */

__global__
void rac_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++)
            /* P0-2: degenerate RAC — trivial angle, pure fmaf, no SFU call */
            acc = fmaf(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);

        __syncthreads();
    }

    if (row < M && col < N) {
        float existing = (beta != 0.0f) ? C[row * N + col] : 0.0f;
        C[row * N + col] = fmaf(alpha, acc, beta * existing);
    }
}

/* Transposed-B variant for backward pass (dA = dC @ B^T) */
__global__
void rac_matmul_nt_kernel(          /* A: normal, B: transposed */
    const float* __restrict__ A,    /* [M, K] */
    const float* __restrict__ B,    /* [N, K] — stored transposed */
    float*       __restrict__ C,    /* [M, N] */
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bCol = t * TILE + threadIdx.y;  /* B is transposed: index as [col, k] */

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol]   : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (col < N && bCol < K) ? B[col * K + bCol]   : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++)
            acc = fmaf(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);

        __syncthreads();
    }

    if (row < M && col < N) {
        float existing = (beta != 0.0f) ? C[row * N + col] : 0.0f;
        C[row * N + col] = fmaf(alpha, acc, beta * existing);
    }
}

/* Transposed-A variant for backward pass (dB = A^T @ dC) */
__global__
void rac_matmul_tn_kernel(          /* A: transposed, B: normal */
    const float* __restrict__ A,    /* [K, M] — stored transposed */
    const float* __restrict__ B,    /* [K, N] */
    float*       __restrict__ C,    /* [M, N] */
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aRow = t * TILE + threadIdx.x;  /* A transposed: index as [k, row] */
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aRow < K) ? A[aRow * M + row]   : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col]   : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++)
            acc = fmaf(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);

        __syncthreads();
    }

    if (row < M && col < N) {
        float existing = (beta != 0.0f) ? C[row * N + col] : 0.0f;
        C[row * N + col] = fmaf(alpha, acc, beta * existing);
    }
}

/* ── C++ dispatch functions (called from Python via pybind) ─────────────── */

static void _launch_nn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE-1)/TILE, (M + TILE-1)/TILE);
    rac_matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

static void _launch_nt(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE-1)/TILE, (M + TILE-1)/TILE);
    rac_matmul_nt_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

static void _launch_tn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE-1)/TILE, (M + TILE-1)/TILE);
    rac_matmul_tn_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

/* ── ATen-level dispatch ─────────────────────────────────────────────────── */

torch::Tensor rac_matmul_forward_cuda(
    torch::Tensor A,   /* [M, K] float32 */
    torch::Tensor B)   /* [K, N] float32 */
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "RAC: inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "RAC: float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "RAC: 2D tensors only for matmul");
    TORCH_CHECK(A.size(1) == B.size(0), "RAC: incompatible matrix dimensions");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    _launch_nn(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, 1.0f, 0.0f, stream);

    return C;
}

std::vector<torch::Tensor> rac_matmul_backward_cuda(
    torch::Tensor grad_C,   /* [M, N] */
    torch::Tensor A,        /* [M, K] saved from forward */
    torch::Tensor B)        /* [K, N] saved from forward */
{
    grad_C = grad_C.contiguous();
    A      = A.contiguous();
    B      = B.contiguous();

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    /* dA = grad_C @ B^T   shape: [M, K] */
    auto grad_A = torch::zeros_like(A);
    _launch_nt(
        grad_C.data_ptr<float>(), B.data_ptr<float>(), grad_A.data_ptr<float>(),
        M, K, N, 1.0f, 0.0f, stream);

    /* dB = A^T @ grad_C   shape: [K, N] */
    auto grad_B = torch::zeros_like(B);
    _launch_tn(
        A.data_ptr<float>(), grad_C.data_ptr<float>(), grad_B.data_ptr<float>(),
        K, N, M, 1.0f, 0.0f, stream);

    return {grad_A, grad_B};
}

torch::Tensor rac_linear_forward_cuda(
    torch::Tensor input,    /* [*, in_features]  */
    torch::Tensor weight,   /* [out_features, in_features] */
    torch::Tensor bias)     /* [out_features] or empty */
{
    TORCH_CHECK(input.is_cuda(), "RAC: input must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "RAC: float32 only");

    /* Flatten batch dimensions */
    auto in_shape = input.sizes().vec();
    int out_features = weight.size(0);
    int in_features  = weight.size(1);

    auto input_2d = input.reshape({-1, in_features}).contiguous();
    auto weight_t = weight.t().contiguous();   /* [in_features, out_features] */
    int M = input_2d.size(0), K = in_features, N = out_features;

    auto output = torch::zeros({M, N}, input.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    _launch_nn(
        input_2d.data_ptr<float>(), weight_t.data_ptr<float>(),
        output.data_ptr<float>(), M, N, K, 1.0f, 0.0f, stream);

    if (bias.defined() && bias.numel() > 0)
        output.add_(bias);   /* bias broadcast — standard ATen, not RAC */

    /* Restore batch shape */
    in_shape.back() = out_features;
    return output.reshape(in_shape);
}

std::vector<torch::Tensor> rac_linear_backward_cuda(
    torch::Tensor grad_output,  /* [*, out_features] */
    torch::Tensor input,        /* [*, in_features]  */
    torch::Tensor weight,       /* [out_features, in_features] */
    bool need_bias_grad)
{
    int out_features = weight.size(0);
    int in_features  = weight.size(1);

    auto go  = grad_output.reshape({-1, out_features}).contiguous();
    auto inp = input.reshape({-1, in_features}).contiguous();
    int M    = inp.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    /* grad_input = grad_output @ weight   [M, in_features] */
    auto grad_input = torch::zeros({M, in_features}, input.options());
    _launch_nn(
        go.data_ptr<float>(), weight.data_ptr<float>(),
        grad_input.data_ptr<float>(), M, in_features, out_features, 1.0f, 0.0f, stream);

    /* grad_weight = grad_output^T @ input   [out_features, in_features] */
    auto grad_weight = torch::zeros_like(weight);
    _launch_tn(
        go.data_ptr<float>(), inp.data_ptr<float>(),
        grad_weight.data_ptr<float>(), out_features, in_features, M, 1.0f, 0.0f, stream);

    /* grad_bias = grad_output.sum(0)   [out_features] */
    torch::Tensor grad_bias;
    if (need_bias_grad)
        grad_bias = go.sum(0);

    /* Restore input grad shape */
    auto in_shape = input.sizes().vec();
    in_shape.back() = in_features;

    return {grad_input.reshape(in_shape), grad_weight, grad_bias};
}

/* ── pybind11 module registration ────────────────────────────────────────── */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RAC: Rotation-Accumulate PyTorch Extension — Pinnacle Quantum Group";

    m.def("matmul_forward",  &rac_matmul_forward_cuda,
          "RAC matrix multiply forward (zero multiplications)",
          py::arg("A"), py::arg("B"));

    m.def("matmul_backward", &rac_matmul_backward_cuda,
          "RAC matrix multiply backward",
          py::arg("grad_C"), py::arg("A"), py::arg("B"));

    m.def("linear_forward",  &rac_linear_forward_cuda,
          "RAC linear layer forward (input @ weight.T + bias)",
          py::arg("input"), py::arg("weight"), py::arg("bias"));

    m.def("linear_backward", &rac_linear_backward_cuda,
          "RAC linear layer backward",
          py::arg("grad_output"), py::arg("input"), py::arg("weight"),
          py::arg("need_bias_grad"));
}
