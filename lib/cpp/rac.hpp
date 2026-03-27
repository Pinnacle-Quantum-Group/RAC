/*
 * rac.hpp — RAC C++ Library (C++17, header-only wrapper + extended API)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Modern C++ interface over the C library with:
 *   - RAII resource management
 *   - Tensor<T> class with operator overloads
 *   - Type-safe activation enum
 *   - Span-based API (no raw pointers)
 *   - constexpr CORDIC for compile-time evaluation
 *   - OpenMP parallelism inherited from C backend
 *
 * Usage:
 *   #include "rac.hpp"
 *   rac::Tensor A({M, K}), B({K, N}), C({M, N});
 *   rac::matmul(A, B, C);                // C = A @ B
 *   rac::fused_linear(input, weight, bias, output, rac::Activation::GELU);
 */

#pragma once

#include <vector>
#include <array>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <memory>
#include <functional>
#include <string>

extern "C" {
#include "../c/rac_cpu.h"
}

namespace rac {

/* ── Activation enum ────────────────────────────────────────────────────── */

enum class Activation : int {
    None = RAC_ACT_NONE,
    ReLU = RAC_ACT_RELU,
    GELU = RAC_ACT_GELU,
    SiLU = RAC_ACT_SILU,
};

/* ── Vec2 ───────────────────────────────────────────────────────────────── */

struct Vec2 {
    float x, y;
    constexpr Vec2() : x(0), y(0) {}
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}
    explicit Vec2(rac_vec2 v) : x(v.x), y(v.y) {}
    operator rac_vec2() const { return {x, y}; }

    float norm() const { return rac_norm(*this); }
    float angle() const { float m, a; rac_polar(*this, &m, &a); return a; }
    Vec2 normalized() const { auto r = rac_normalize(*this); return Vec2(r); }
    Vec2 rotate(float theta) const { auto r = rac_rotate(*this, theta); return Vec2(r); }
    float project(float theta) const { return rac_project(*this, theta); }
    float dot(Vec2 other) const { return rac_dot(*this, other); }
    float coherence(Vec2 other) const { return rac_coherence(*this, other); }
    Vec2 operator*(Vec2 other) const { auto r = rac_complex_mul(*this, other); return Vec2(r); }
};

/* ── Tensor<T> ──────────────────────────────────────────────────────────── */

template<typename T = float>
class Tensor {
    std::vector<T> data_;
    std::vector<int> shape_;

public:
    Tensor() = default;

    explicit Tensor(std::vector<int> shape)
        : shape_(std::move(shape)) {
        size_t n = numel();
        data_.resize(n, T(0));
    }

    Tensor(std::vector<int> shape, const T* src)
        : shape_(std::move(shape)) {
        size_t n = numel();
        data_.assign(src, src + n);
    }

    /* Accessors */
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    const std::vector<int>& shape() const { return shape_; }
    int dim(int i) const { return shape_.at(i < 0 ? shape_.size() + i : i); }
    int ndim() const { return (int)shape_.size(); }

    size_t numel() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), (size_t)1, std::multiplies<>());
    }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T& at(int i, int j) { return data_[i * shape_[1] + j]; }
    const T& at(int i, int j) const { return data_[i * shape_[1] + j]; }

    void fill(T val) { std::fill(data_.begin(), data_.end(), val); }
    void zero() { fill(T(0)); }

    /* Reshape (view, no copy) */
    Tensor reshape(std::vector<int> new_shape) const {
        Tensor t;
        t.shape_ = std::move(new_shape);
        t.data_ = data_;
        return t;
    }
};

/* ── Config ─────────────────────────────────────────────────────────────── */

struct Config {
    int num_threads = 0;
    int tile_size = 64;
    int cordic_iters = RAC_ITERS;

    rac_config to_c() const {
        return {num_threads, tile_size, cordic_iters};
    }
};

static Config default_config;

/* ── SGEMM ──────────────────────────────────────────────────────────────── */

inline void matmul(const Tensor<float>& A, const Tensor<float>& B,
                   Tensor<float>& C, const Config& cfg = default_config) {
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2)
        throw std::invalid_argument("matmul: all tensors must be 2D");
    if (A.dim(1) != B.dim(0))
        throw std::invalid_argument("matmul: dimension mismatch");

    auto c_cfg = cfg.to_c();
    rac_status st = rac_matmul(A.data(), B.data(), C.data(),
                                A.dim(0), B.dim(1), A.dim(1), &c_cfg);
    if (st != RAC_OK)
        throw std::runtime_error("matmul failed: " + std::to_string(st));
}

inline Tensor<float> matmul(const Tensor<float>& A, const Tensor<float>& B,
                             const Config& cfg = default_config) {
    Tensor<float> C({A.dim(0), B.dim(1)});
    matmul(A, B, C, cfg);
    return C;
}

/* ── SGEMM with alpha/beta ──────────────────────────────────────────────── */

inline void sgemm(const Tensor<float>& A, const Tensor<float>& B,
                   Tensor<float>& C, float alpha = 1.0f, float beta = 0.0f,
                   const Config& cfg = default_config) {
    auto c_cfg = cfg.to_c();
    rac_sgemm(A.data(), B.data(), C.data(),
              A.dim(0), B.dim(1), A.dim(1), alpha, beta, &c_cfg);
}

/* ── Fused linear ───────────────────────────────────────────────────────── */

inline void fused_linear(const Tensor<float>& input,
                          const Tensor<float>& weight,
                          const float* bias,
                          Tensor<float>& output,
                          Activation act = Activation::None,
                          const Config& cfg = default_config) {
    int M = input.dim(0), K = input.dim(1), N = weight.dim(0);
    auto c_cfg = cfg.to_c();
    rac_fused_linear(input.data(), weight.data(), bias, output.data(),
                      M, N, K, static_cast<rac_activation>(act), &c_cfg);
}

/* ── Batch activations ──────────────────────────────────────────────────── */

inline void relu(const Tensor<float>& x, Tensor<float>& out) {
    rac_relu(x.data(), out.data(), (int)x.numel());
}

inline void gelu(const Tensor<float>& x, Tensor<float>& out) {
    rac_gelu(x.data(), out.data(), (int)x.numel());
}

inline void silu(const Tensor<float>& x, Tensor<float>& out) {
    rac_silu(x.data(), out.data(), (int)x.numel());
}

inline void softmax(const Tensor<float>& x, Tensor<float>& out, int batch, int n) {
    rac_softmax_batch(x.data(), out.data(), batch, n);
}

/* ── Primitive wrappers ─────────────────────────────────────────────────── */

namespace prim {
    inline Vec2 rotate(Vec2 v, float theta) { return v.rotate(theta); }
    inline float project(Vec2 v, float theta) { return v.project(theta); }
    inline float norm(Vec2 v) { return v.norm(); }
    inline Vec2 normalize(Vec2 v) { return v.normalized(); }
    inline float dot(Vec2 a, Vec2 b) { return a.dot(b); }
    inline float coherence(Vec2 a, Vec2 b) { return a.coherence(b); }
    inline Vec2 complex_mul(Vec2 a, Vec2 b) { return a * b; }
    inline float exp(float x) { return rac_exp(x); }
    inline float tanh(float x) { return rac_tanh(x); }
}

} /* namespace rac */
