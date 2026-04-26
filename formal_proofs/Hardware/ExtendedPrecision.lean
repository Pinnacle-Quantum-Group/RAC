-- Variable-iteration CORDIC: 8(edge)/16(infer)/24(train), error monotone decreasing
import Mathlib
namespace RAC.Hardware.ExtendedPrecision

def cordic_error (n : Nat) : ℝ := (2 : ℝ)^(-(n : ℤ) + 1)

/-- Larger iteration count → strictly smaller error. Strict monotonicity of
    `2^k` (with `k : ℤ`, base `2 > 1`) applied to the decreasing exponent
    `-n + 1`. -/
theorem error_monotone (m n : Nat) (h : m < n) : cordic_error n < cordic_error m := by
  unfold cordic_error
  have hcast : (m : ℤ) < (n : ℤ) := by exact_mod_cast h
  have hexp : (-(n : ℤ) + 1) < (-(m : ℤ) + 1) := by linarith
  exact zpow_lt_zpow_right₀ (by norm_num : (1 : ℝ) < 2) hexp
  -- NEEDS_VERIFICATION: in older Mathlib this lemma is `zpow_lt_zpow_right`
  -- (no ₀ suffix). Both have signature `1 < a → m < n → a^m < a^n`.

/-- 24-iteration CORDIC matches FP32 mantissa precision: error = 2^(-23). -/
theorem training_matches_fp32 : cordic_error 24 = (2 : ℝ)^(-(23:ℤ)) := by
  show (2 : ℝ)^(-(24 : ℤ) + 1) = (2 : ℝ)^(-(23 : ℤ))
  norm_num

theorem precision_hierarchy :
    cordic_error 24 < cordic_error 16 ∧ cordic_error 16 < cordic_error 8 := by
  exact ⟨error_monotone 16 24 (by norm_num), error_monotone 8 16 (by norm_num)⟩

end RAC.Hardware.ExtendedPrecision
