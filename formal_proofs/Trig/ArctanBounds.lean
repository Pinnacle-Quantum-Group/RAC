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
import RAC.Cordic.ArctanFacts

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
  unfold_let g
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

/-! ## Specialization for CORDIC: bounds on atan(2⁻ᵏ).

    `atanTable i = arctan ((1/2)^i)` in the CORDIC modules.  Here we
    expose:
      atanTable_lb : (1/2)^i - (1/2)^(3i)/3 ≤ arctan ((1/2)^i)
      atanTable_ub :                          arctan ((1/2)^i) ≤ (1/2)^i
    which are the inputs to the absorption-property analysis.  -/

theorem arctan_inv_two_pow_lb (i : ℕ) :
    (2:ℝ)⁻¹^i - ((2:ℝ)⁻¹^i)^3 / 3 ≤ arctan ((2:ℝ)⁻¹^i) := by
  have h : (0 : ℝ) ≤ (2:ℝ)⁻¹^i := by positivity
  exact arctan_lb h

theorem arctan_inv_two_pow_ub (i : ℕ) :
    arctan ((2:ℝ)⁻¹^i) ≤ (2:ℝ)⁻¹^i := by
  have h : (0 : ℝ) ≤ (2:ℝ)⁻¹^i := by positivity
  exact arctan_ub h

end RAC.Trig.ArctanBounds
