/-
  RAC — Chained CORDIC Error Accumulation Bounds
  Pinnacle Quantum Group — April 2026

  Proves that accumulated rounding error in n chained CORDIC
  rotations grows at most linearly: total_error ≤ n * 2^{-k}
  where k is the number of CORDIC iterations per rotation.
  Reference: rac_cuda.cu, lib/c/rac_cordic.c
-/
import Mathlib

noncomputable section
open Finset BigOperators Filter Topology

namespace RAC.Numerical.ErrorAccumulation

/-! ## 1. Single-Step CORDIC Error Bound -/

def singleStepError (k : ℕ) : ℝ := (2 : ℝ)⁻¹ ^ k

theorem singleStepError_pos (k : ℕ) : 0 < singleStepError k := by
  unfold singleStepError; positivity

theorem singleStepError_decreasing : StrictAnti singleStepError := by
  intro i j hij
  unfold singleStepError
  exact pow_lt_pow_right_of_lt_one (by norm_num : (0:ℝ) < (2:ℝ)⁻¹)
    (by norm_num : (2:ℝ)⁻¹ < 1) hij

theorem singleStepError_le_one (k : ℕ) : singleStepError k ≤ 1 := by
  unfold singleStepError
  exact pow_le_one k (by norm_num) (by norm_num)

/-! ## 2. Chained Error Accumulation -/

def chainedError (n k : ℕ) : ℝ := ↑n * singleStepError k

theorem chainedError_bound (n k : ℕ) :
    chainedError n k = ↑n * (2 : ℝ)⁻¹ ^ k := by
  unfold chainedError singleStepError
  rfl

theorem chainedError_nonneg (n k : ℕ) : 0 ≤ chainedError n k := by
  unfold chainedError
  exact mul_nonneg (Nat.cast_nonneg _) (le_of_lt (singleStepError_pos k))

/-! ## 3. Error Decreases with More Iterations -/

theorem more_iterations_less_error (n k₁ k₂ : ℕ) (hk : k₁ < k₂) (hn : 0 < n) :
    chainedError n k₂ < chainedError n k₁ := by
  unfold chainedError
  apply mul_lt_mul_of_pos_left (singleStepError_decreasing hk)
  exact Nat.cast_pos.mpr hn

/-! ## 4. Error Convergence to Zero -/

theorem error_vanishes_with_precision (n : ℕ) (ε : ℝ) (hε : 0 < ε) :
    ∃ k : ℕ, chainedError n k < ε := by
  -- chainedError n k = ↑n * (1/2)^k, and (1/2)^k → 0, so the product → 0.
  have h_pow : Tendsto (fun k : ℕ => ((2 : ℝ)⁻¹) ^ k) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  have h_chain : Tendsto (fun k : ℕ => chainedError n k) atTop (nhds 0) := by
    have h_mul := h_pow.const_mul (↑n : ℝ)
    simpa [chainedError, singleStepError, mul_zero] using h_mul
  -- Eventually (chainedError n k < ε); extract a witness.
  rcases (h_chain.eventually (Iio_mem_nhds hε)).exists with ⟨K, hK⟩
  exact ⟨K, hK⟩

/-! ## 5. Composition of Error Bounds -/

def composedError (errors : Fin n → ℝ) : ℝ := ∑ i, errors i

theorem triangle_inequality_chain (errors : Fin n → ℝ)
    (hbound : ∀ i, |errors i| ≤ singleStepError k) :
    |composedError errors| ≤ ↑n * singleStepError k := by
  unfold composedError
  calc |∑ i, errors i| ≤ ∑ i, |errors i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i : Fin n, singleStepError k := Finset.sum_le_sum (fun i _ => hbound i)
    _ = ↑n * singleStepError k := by simp [Finset.sum_const, Finset.card_fin]

/-! ## 6. Practical Bounds for Common Configurations -/

theorem error_16iter_bound (n : ℕ) :
    chainedError n 16 = ↑n * (2 : ℝ)⁻¹ ^ 16 := by
  exact chainedError_bound n 16

theorem error_24iter_bound (n : ℕ) :
    chainedError n 24 = ↑n * (2 : ℝ)⁻¹ ^ 24 := by
  exact chainedError_bound n 24

end RAC.Numerical.ErrorAccumulation
