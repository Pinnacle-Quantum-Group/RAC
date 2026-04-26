/-
  RAC — Arctan Foundation Lemmas (CORDIC support)
  Pinnacle Quantum Group — April 2026

  Mathlib v4.5.0 ships `tan_arctan`, `arctan_zero`, `strictMonoOn_tan`,
  `arctan_mem_Ioo`, `arctan_lt_pi_div_two`, `Real.le_tan` — but NOT
  the elementary corollaries `arctan_pos`, `arctan_strictMono`, or
  `arctan_le_self_of_nonneg` that CORDIC convergence proofs need.

  We derive them once here so downstream modules
  (Cordic/Convergence.lean, Cordic/PrecisionKnob.lean,
   TableLookup/Atan2Correctness.lean, …) can use them as oneliners.

  All three are textbook (Volder 1959, Walther 1971), provable in
  v4.5.0 from primitives via the standard inverse-of-strict-mono
  argument plus `le_tan` from Bounds.lean.
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

noncomputable section
open Real Set

namespace RAC.Cordic.ArctanFacts

/-- `arctan` is strictly monotone on all of ℝ.
    Proof: `tan` is strict mono on `(-π/2, π/2)`, and `arctan` lands
    in this interval (by `arctan_mem_Ioo`) with `tan (arctan x) = x`.
    By contrapositive of mono of `tan`, `arctan` is strict mono. -/
theorem arctan_strictMono : StrictMono arctan := by
  intro x y hxy
  by_contra hle
  push_neg at hle
  have hx_in : arctan x ∈ Ioo (-(π / 2)) (π / 2) := arctan_mem_Ioo x
  have hy_in : arctan y ∈ Ioo (-(π / 2)) (π / 2) := arctan_mem_Ioo y
  have h_tan_le : tan (arctan y) ≤ tan (arctan x) := by
    rcases eq_or_lt_of_le hle with heq | hlt
    · rw [heq]
    · exact (strictMonoOn_tan hy_in hx_in hlt).le
  rw [tan_arctan, tan_arctan] at h_tan_le
  linarith

/-- `0 < arctan x` for `x > 0`. -/
theorem arctan_pos {x : ℝ} (hx : 0 < x) : 0 < arctan x := by
  rw [← arctan_zero]
  exact arctan_strictMono hx

/-- `0 ≤ arctan x` for `x ≥ 0`. -/
theorem arctan_nonneg {x : ℝ} (hx : 0 ≤ x) : 0 ≤ arctan x := by
  rcases eq_or_lt_of_le hx with rfl | hlt
  · rw [arctan_zero]
  · exact (arctan_pos hlt).le

/-- `arctan y ≤ y` for `y ≥ 0`.  Direct from `Real.le_tan`:
    `arctan y ≤ tan (arctan y) = y` whenever `arctan y ∈ [0, π/2)`.
    The interval condition follows from `arctan_nonneg` and
    `arctan_lt_pi_div_two`. -/
theorem arctan_le_self_of_nonneg {y : ℝ} (hy : 0 ≤ y) : arctan y ≤ y := by
  have h1 : 0 ≤ arctan y := arctan_nonneg hy
  have h2 : arctan y < π / 2 := arctan_lt_pi_div_two y
  calc arctan y ≤ tan (arctan y) := Real.le_tan h1 h2
    _ = y := tan_arctan y

/-- For natural i, `(2 : ℝ)⁻¹ ^ i > 0`. Convenience lemma. -/
theorem inv_two_pow_pos (i : ℕ) : (0 : ℝ) < (2 : ℝ)⁻¹ ^ i := by
  positivity

/-- The `(2 : ℝ)⁻¹ ^ ·` family is strictly anti on ℕ.  Used to chain
    with `arctan_strictMono` for the CORDIC table monotonicity. -/
theorem inv_two_pow_strictAnti : StrictAnti (fun i : ℕ => (2 : ℝ)⁻¹ ^ i) := by
  intro i j hij
  -- (2⁻¹)^j < (2⁻¹)^i iff i < j (since 2⁻¹ ∈ (0, 1))
  have h0 : (0 : ℝ) < (2 : ℝ)⁻¹ := by norm_num
  have h1 : (2 : ℝ)⁻¹ < 1 := by norm_num
  exact pow_lt_pow_of_lt_one h0 h1 hij

end RAC.Cordic.ArctanFacts
