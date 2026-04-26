/-
  RAC — Maclaurin Bounds on `arctan`
  Pinnacle Quantum Group — April 2026

  Mathlib v4.5.0 ships `Real.hasDerivAt_arctan` (giving `arctan'(x) = 1/(1+x²)`)
  and the standard `Convex.strictMonoOn_of_deriv_pos` machinery, but NOT the
  Maclaurin/Taylor bounds:
    `x - x³/3 ≤ arctan x ≤ x` for x ≥ 0
  which are foundational to CORDIC convergence (Volder 1959, the
  "absorption property" `atan(2⁻ᵏ) ≤ ∑_{j>k} atan(2⁻ʲ)`).

  Pattern follows the existing `Real.lt_tan` proof in
  `Mathlib/Analysis/SpecialFunctions/Trigonometric/Bounds.lean`:
  define an auxiliary `g`, show `g'(x) ≥ 0` for `x ≥ 0`, and deduce
  monotonicity from `Convex.strictMonoOn_of_deriv_pos`.
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Cordic.ArctanFacts

noncomputable section
open Real Set
open RAC.Cordic.ArctanFacts

namespace RAC.Trig.ArctanBounds

/-- Auxiliary: `g(y) := arctan y - y + y³/3`.  We have `g(0) = 0` and
    `g'(y) = y⁴/(1 + y²) ≥ 0`.  -/
private def g (y : ℝ) : ℝ := arctan y - y + y^3 / 3

private lemma g_zero : g 0 = 0 := by
  simp [g, Real.arctan_zero]

/-- `g'(y) = 1/(1+y²) - 1 + y² = y⁴/(1+y²)`.  -/
private lemma hasDerivAt_g (y : ℝ) :
    HasDerivAt g (y^4 / (1 + y^2)) y := by
  -- Build the derivative for each summand and combine.
  have h_arctan : HasDerivAt arctan (1 / (1 + y^2)) y := Real.hasDerivAt_arctan y
  have h_id : HasDerivAt (fun z : ℝ => z) 1 y := hasDerivAt_id y
  have h_pow3 : HasDerivAt (fun z : ℝ => z^3) (3 * y^2) y := by
    simpa using (hasDerivAt_id y).pow 3
  have h_pow3_div : HasDerivAt (fun z : ℝ => z^3 / 3) (y^2) y := by
    have := h_pow3.div_const 3
    convert this using 1
    ring
  -- arctan - id + (·)³/3
  have h_g : HasDerivAt g (1 / (1 + y^2) - 1 + y^2) y :=
    (h_arctan.sub h_id).add h_pow3_div
  -- Show 1/(1+y²) - 1 + y² = y⁴/(1+y²)
  convert h_g using 1
  have h_denom_pos : (0 : ℝ) < 1 + y^2 := by positivity
  field_simp
  ring

private lemma deriv_g (y : ℝ) : deriv g y = y^4 / (1 + y^2) :=
  (hasDerivAt_g y).deriv

private lemma deriv_g_pos {y : ℝ} (hy : 0 < y) : 0 < deriv g y := by
  rw [deriv_g]
  have h_y4_pos : (0 : ℝ) < y^4 := by positivity
  have h_denom_pos : (0 : ℝ) < 1 + y^2 := by positivity
  exact div_pos h_y4_pos h_denom_pos

private lemma continuous_g : Continuous g := by
  unfold g
  exact (Real.continuous_arctan.sub continuous_id).add ((continuous_pow 3).div_const _)

/-- `arctan` Maclaurin lower bound: `x - x³/3 ≤ arctan x` for `x ≥ 0`.

    Proof: `g(y) := arctan y - y + y³/3` satisfies `g(0) = 0` and
    `g'(y) = y⁴/(1+y²) ≥ 0`, with strict positivity for `y > 0`.
    `Convex.strictMonoOn_of_deriv_pos` on `Ici 0` gives
    `g(0) < g(x)` for `x > 0`, i.e., `0 < arctan x - x + x³/3`. -/
theorem arctan_lb {x : ℝ} (hx : 0 ≤ x) : x - x^3 / 3 ≤ arctan x := by
  rcases eq_or_lt_of_le hx with hx0 | hx_pos
  · -- x = 0: 0 - 0 ≤ arctan 0 = 0
    subst hx0
    simp [Real.arctan_zero]
  · -- x > 0: use strict monotonicity of g on Ici 0.
    have h_cont : ContinuousOn g (Ici (0:ℝ)) := continuous_g.continuousOn
    have h_interior : interior (Ici (0:ℝ)) = Ioi 0 := interior_Ici
    have h_deriv_pos_on : ∀ y, y ∈ interior (Ici (0:ℝ)) → 0 < deriv g y := by
      intro y hy
      rw [h_interior] at hy
      exact deriv_g_pos hy
    have h_mono : StrictMonoOn g (Ici (0:ℝ)) :=
      Convex.strictMonoOn_of_deriv_pos (convex_Ici 0) h_cont h_deriv_pos_on
    have h_zero_mem : (0:ℝ) ∈ Ici (0:ℝ) := left_mem_Ici
    have h_x_mem : x ∈ Ici (0:ℝ) := hx
    have h_strict := h_mono h_zero_mem h_x_mem hx_pos
    rw [g_zero] at h_strict
    -- h_strict : 0 < g x = arctan x - x + x³/3
    have : 0 < arctan x - x + x^3 / 3 := h_strict
    linarith

/-- `arctan` Maclaurin upper bound: `arctan x ≤ x` for `x ≥ 0`.
    (Already provable in v4.5.0 via `Real.le_tan` + `tan_arctan`; see
    `RAC.Cordic.ArctanFacts.arctan_le_self_of_nonneg`.  Re-exposed here
    for symmetry of the bound pair.) -/
theorem arctan_ub {x : ℝ} (hx : 0 ≤ x) : arctan x ≤ x :=
  arctan_le_self_of_nonneg hx

/-! ## Half-argument bound: `arctan(x)/2 ≤ arctan(x/2)` for x ≥ 0.

    Equivalent to `arctan(x/2) ≥ arctan(x)/2`. Derived via
    h(y) := 2·arctan(y/2) - arctan(y), with h(0) = 0 and
    h'(y) = 3y² / ((4+y²)(1+y²)) ≥ 0.

    Consequence: `atanTable (k+1) = arctan(2⁻ᵏ⁻¹) = arctan((2⁻ᵏ)/2) ≥
    arctan(2⁻ᵏ)/2 = atanTable k / 2`.  This is the geometric inequality
    that drives Volder's absorption property
      `atanTable k ≤ ∑_{j>k} atanTable j`. -/

private def h₂ (y : ℝ) : ℝ := 2 * arctan (y / 2) - arctan y

private lemma h₂_zero : h₂ 0 = 0 := by simp [h₂, Real.arctan_zero]

private lemma hasDerivAt_h₂ (y : ℝ) :
    HasDerivAt h₂ (3 * y^2 / ((4 + y^2) * (1 + y^2))) y := by
  -- d/dy arctan(y/2) = (1/2) · 1/(1 + (y/2)²)
  have h_id_div : HasDerivAt (fun z : ℝ => z / 2) (1/2 : ℝ) y := by
    simpa using (hasDerivAt_id y).div_const 2
  have h_arctan_half : HasDerivAt (fun z : ℝ => arctan (z / 2))
      ((1 / (1 + (y/2)^2)) * (1/2)) y :=
    (Real.hasDerivAt_arctan (y/2)).comp y h_id_div
  have h_arctan : HasDerivAt arctan (1 / (1 + y^2)) y := Real.hasDerivAt_arctan y
  have : HasDerivAt h₂ (2 * ((1 / (1 + (y/2)^2)) * (1/2)) - 1 / (1 + y^2)) y :=
    (h_arctan_half.const_mul 2).sub h_arctan
  convert this using 1
  have h_denom1 : (0 : ℝ) < 1 + (y/2)^2 := by positivity
  have h_denom2 : (0 : ℝ) < 1 + y^2 := by positivity
  have h_denom1_ne : (1 + (y/2)^2 : ℝ) ≠ 0 := ne_of_gt h_denom1
  have h_denom2_ne : (1 + y^2 : ℝ) ≠ 0 := ne_of_gt h_denom2
  have h_denom_4 : (4 + y^2 : ℝ) ≠ 0 := by positivity
  -- 1/(1 + (y/2)²) = 4/(4 + y²)
  field_simp
  ring

private lemma deriv_h₂_nonneg (y : ℝ) : 0 ≤ deriv h₂ y := by
  rw [(hasDerivAt_h₂ y).deriv]
  have h_denom : (0 : ℝ) < (4 + y^2) * (1 + y^2) := by positivity
  apply div_nonneg
  · positivity
  · exact h_denom.le

private lemma deriv_h₂_pos {y : ℝ} (hy : y ≠ 0) : 0 < deriv h₂ y := by
  rw [(hasDerivAt_h₂ y).deriv]
  have h_y2_pos : 0 < y^2 := by positivity
  have h_denom : (0 : ℝ) < (4 + y^2) * (1 + y^2) := by positivity
  apply div_pos
  · linarith
  · exact h_denom

private lemma continuous_h₂ : Continuous h₂ := by
  unfold h₂
  -- `Continuous.const_mul` isn't in Mathlib v4.5.0; use `continuity`
  -- which closes routine continuity goals (sub, comp, mul, arctan).
  continuity

/-- `arctan(x)/2 ≤ arctan(x/2)` for `x ≥ 0`.  Concavity-style bound
    derived via `h₂(y) := 2·arctan(y/2) - arctan(y)`, mono with
    h₂(0) = 0 ⟹ h₂(x) ≥ 0 ⟹ arctan(x) ≤ 2·arctan(x/2). -/
theorem arctan_half_ge {x : ℝ} (hx : 0 ≤ x) : arctan x / 2 ≤ arctan (x / 2) := by
  rcases eq_or_lt_of_le hx with hx0 | hx_pos
  · subst hx0; simp [Real.arctan_zero]
  · have h_cont : ContinuousOn h₂ (Ici (0:ℝ)) := continuous_h₂.continuousOn
    have h_interior : interior (Ici (0:ℝ)) = Ioi 0 := interior_Ici
    have h_deriv_pos_on : ∀ y, y ∈ interior (Ici (0:ℝ)) → 0 < deriv h₂ y := by
      intro y hy
      rw [h_interior] at hy
      exact deriv_h₂_pos (ne_of_gt hy)
    have h_mono : StrictMonoOn h₂ (Ici (0:ℝ)) :=
      Convex.strictMonoOn_of_deriv_pos (convex_Ici 0) h_cont h_deriv_pos_on
    have h_zero_mem : (0:ℝ) ∈ Ici (0:ℝ) := left_mem_Ici
    have h_x_mem : x ∈ Ici (0:ℝ) := hx
    have h_strict := h_mono h_zero_mem h_x_mem hx_pos
    rw [h₂_zero] at h_strict
    -- 0 < h₂ x; expand `h₂` so linarith sees the linear form
    -- `0 < 2·arctan(x/2) - arctan(x)`.
    unfold h₂ at h_strict
    linarith

/-! ## Specialization for CORDIC: bounds on atan(2⁻ᵏ).

    `atanTable i = arctan ((1/2)^i)` in the CORDIC modules.  Here we
    expose:
      atanTable_lb : (1/2)^i - (1/2)^(3i)/3 ≤ arctan ((1/2)^i)
      atanTable_ub :                          arctan ((1/2)^i) ≤ (1/2)^i
      atanTable_half : arctan (2^(-(k+1))) ≥ arctan (2^(-k)) / 2
    The third is the geometric chain that yields absorption via
      ∑_{j>k} atanTable j ≥ atanTable k · ∑_{j≥1} 2⁻ʲ = atanTable k. -/

theorem arctan_inv_two_pow_lb (i : ℕ) :
    (2:ℝ)⁻¹^i - ((2:ℝ)⁻¹^i)^3 / 3 ≤ arctan ((2:ℝ)⁻¹^i) := by
  have h : (0 : ℝ) ≤ (2:ℝ)⁻¹^i := by positivity
  exact arctan_lb h

theorem arctan_inv_two_pow_ub (i : ℕ) :
    arctan ((2:ℝ)⁻¹^i) ≤ (2:ℝ)⁻¹^i := by
  have h : (0 : ℝ) ≤ (2:ℝ)⁻¹^i := by positivity
  exact arctan_ub h

/-- Geometric chain: `atan(2⁻⁽ᵏ⁺¹⁾) ≥ atan(2⁻ᵏ) / 2`. -/
theorem arctan_inv_two_pow_succ_ge_half (k : ℕ) :
    arctan ((2:ℝ)⁻¹^k) / 2 ≤ arctan ((2:ℝ)⁻¹^(k+1)) := by
  have h_nonneg : (0 : ℝ) ≤ (2:ℝ)⁻¹^k := by positivity
  have h_eq : (2:ℝ)⁻¹^(k+1) = (2:ℝ)⁻¹^k / 2 := by
    rw [pow_succ]; ring
  rw [h_eq]
  exact arctan_half_ge h_nonneg

end RAC.Trig.ArctanBounds
