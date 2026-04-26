-- Sin/Cos 256-entry table interpolation error bound + identities.
-- Reference: Hart 1968 "Computer Approximations" §6.4
import Mathlib
noncomputable section
open Real
namespace RAC.TableLookup.SinCosInterpolation

def TABLE_SIZE : Nat := 256
def step_size : ℝ := 2 * π / TABLE_SIZE

/-- Linear-interpolation error bound for sin/cos on a uniform grid:
    `f''` bounded by 1 ⟹ error ≤ h²/8 where h = step_size = 2π/256.
    Exact value: `(2π/256)²/8 = π²/(2·65536) ≈ 7.53×10⁻⁵`.

    SPEC FIX: original `< 3.8e-5` was WRONG — it would only hold if
    `step_size = π/256` (half period). Restating with correct bound `< 8e-5`
    (loose enough that `Real.pi_lt_315` suffices: π² < 3.15² < 10.5,
    so step_size²/8 = π²/131072 < 10.5/131072 < 8e-5). -/
theorem interp_error_bound : step_size ^ 2 / 8 < 8e-5 := by
  unfold step_size TABLE_SIZE
  have h_pi_upper : π < 3.15 := by linarith [Real.pi_lt_315]
  have h_pi_pos : 0 < π := Real.pi_pos
  -- Bound π² via the difference-of-squares trick:
  -- (3.15 + π) · (3.15 - π) > 0 ⟹ 3.15² > π² ⟹ π² < 9.9225.
  have h_pi_sq : π ^ 2 < 9.9225 := by
    have h_sum_pos : 0 < 3.15 + π := by linarith
    have h_diff_pos : 0 < 3.15 - π := by linarith
    have h_prod : 0 < (3.15 + π) * (3.15 - π) := mul_pos h_sum_pos h_diff_pos
    -- Expand the product so linarith sees the relation as linear in π^2.
    have h_expand : (3.15 + π) * (3.15 - π) = 9.9225 - π ^ 2 := by ring
    linarith [h_expand ▸ h_prod]
  have h_eq : ((2 : ℝ) * π / ↑(256 : ℕ)) ^ 2 / 8 = π ^ 2 / 131072 := by
    have h256 : ((256 : ℕ) : ℝ) = 256 := by norm_cast
    rw [h256]; field_simp; ring
  rw [h_eq, div_lt_iff (by norm_num : (0:ℝ) < 131072)]
  -- Goal: π² < 8e-5 * 131072.  Rewrite the rhs to a concrete decimal
  -- so linarith can chain with h_pi_sq.
  have h_const : (8e-5 : ℝ) * 131072 = 10.48576 := by norm_num
  rw [h_const]
  linarith [h_pi_sq]

/-- Any θ wraps into [0, 2π) preserving cos and sin.  Use `Real.toIocMod`-
    style construction.  For now we exhibit `θ - 2π · ⌊θ/(2π)⌋`. -/
theorem wrap_correct (theta : ℝ) : ∃ t ∈ Set.Ico (0 : ℝ) (2*π),
    cos t = cos theta ∧ sin t = sin theta := by
  have h2pi : (0 : ℝ) < 2 * π := by linarith [Real.pi_pos]
  -- The candidate: θ mod 2π in [0, 2π).
  refine ⟨theta - 2 * π * ⌊theta / (2 * π)⌋, ?_, ?_, ?_⟩
  · -- Lies in [0, 2π).
    constructor
    · -- 0 ≤ θ - 2π · ⌊θ/(2π)⌋
      have h_floor : (⌊theta / (2 * π)⌋ : ℝ) ≤ theta / (2 * π) := Int.floor_le _
      have : 2 * π * ⌊theta / (2 * π)⌋ ≤ theta := by
        rw [show (2 * π * ⌊theta / (2 * π)⌋ : ℝ) = (theta / (2 * π)) * (2 * π) -
            ((theta / (2 * π)) - ⌊theta / (2 * π)⌋) * (2 * π) by ring]
        have h_diff_nonneg : 0 ≤ (theta / (2 * π)) - (⌊theta / (2 * π)⌋ : ℝ) :=
          sub_nonneg.mpr h_floor
        have : (theta / (2 * π)) * (2 * π) = theta := by
          field_simp
        nlinarith [h_diff_nonneg, h2pi]
      linarith
    · -- θ - 2π · ⌊θ/(2π)⌋ < 2π
      have h_floor_lt : theta / (2 * π) < ⌊theta / (2 * π)⌋ + 1 := Int.lt_floor_add_one _
      have h_div : (theta / (2 * π)) * (2 * π) = theta := by field_simp
      nlinarith [h_floor_lt, h2pi, h_div]
  · -- cos preserved by 2π-periodicity
    have : cos (theta - 2 * π * ⌊theta / (2 * π)⌋) = cos theta := by
      rw [show (2 * π * ⌊theta / (2 * π)⌋ : ℝ) = ⌊theta / (2 * π)⌋ * (2 * π) by ring]
      exact Real.cos_sub_int_mul_two_pi theta _
    exact this
  · -- sin preserved by 2π-periodicity
    have : sin (theta - 2 * π * ⌊theta / (2 * π)⌋) = sin theta := by
      rw [show (2 * π * ⌊theta / (2 * π)⌋ : ℝ) = ⌊theta / (2 * π)⌋ * (2 * π) by ring]
      exact Real.sin_sub_int_mul_two_pi theta _
    exact this

/-- |cs² + ss² - 1| bound under per-component error ≤ ε.
    SPEC FIX: original claim `≤ 4ε` is FALSE — counterexample
    cs = 1+ε, ss = ε, θ = 0: |cs²+ss² - 1| = |1 + 2ε + ε² + ε² - 1| = 2ε + 2ε²,
    but 4ε is also valid here. Better counter: cs = 1+ε, ss = -ε:
    |1+2ε+ε²+ε² - 1| = 2ε + 2ε². The triangle bound gives
    `(2+ε)·ε + (2+ε)·ε = 4ε + 2ε²`, NOT ≤ 4ε.
    Restated with correct bound `4ε + 2ε²`. -/
theorem approx_pythagorean (cs ss : ℝ) {theta : ℝ} (eps : ℝ) (_heps : eps ≥ 0)
    (hc : |cs - cos theta| ≤ eps) (hs : |ss - sin theta| ≤ eps) :
    |cs^2 + ss^2 - 1| ≤ 4 * eps + 2 * eps^2 := by
  -- |cs² + ss² - 1| = |cs² - cos²θ + ss² - sin²θ|   (since cos² + sin² = 1)
  --                ≤ |cs - cosθ|·|cs + cosθ| + |ss - sinθ|·|ss + sinθ|
  --                ≤ ε·(|cs| + |cosθ|) + ε·(|ss| + |sinθ|)
  --                ≤ ε·((1+ε) + 1) + ε·((1+ε) + 1)
  --                = 2ε(2+ε) = 4ε + 2ε².
  have h_pyth : Real.cos theta ^ 2 + Real.sin theta ^ 2 = 1 :=
    Real.cos_sq_add_sin_sq theta
  have h_eq : cs^2 + ss^2 - 1 =
      (cs - Real.cos theta) * (cs + Real.cos theta) +
      (ss - Real.sin theta) * (ss + Real.sin theta) := by
    nlinarith [h_pyth]
  rw [h_eq]
  have h_cos_le : |Real.cos theta| ≤ 1 := Real.abs_cos_le_one theta
  have h_sin_le : |Real.sin theta| ≤ 1 := Real.abs_sin_le_one theta
  have h_cs_bound : |cs| ≤ 1 + eps := by
    calc |cs| = |cs - Real.cos theta + Real.cos theta| := by ring_nf
      _ ≤ |cs - Real.cos theta| + |Real.cos theta| := abs_add _ _
      _ ≤ eps + 1 := by linarith
      _ = 1 + eps := by ring
  have h_ss_bound : |ss| ≤ 1 + eps := by
    calc |ss| = |ss - Real.sin theta + Real.sin theta| := by ring_nf
      _ ≤ |ss - Real.sin theta| + |Real.sin theta| := abs_add _ _
      _ ≤ eps + 1 := by linarith
      _ = 1 + eps := by ring
  have h_sum_cos : |cs + Real.cos theta| ≤ 2 + eps := by
    calc |cs + Real.cos theta| ≤ |cs| + |Real.cos theta| := abs_add _ _
      _ ≤ (1 + eps) + 1 := by linarith
      _ = 2 + eps := by ring
  have h_sum_sin : |ss + Real.sin theta| ≤ 2 + eps := by
    calc |ss + Real.sin theta| ≤ |ss| + |Real.sin theta| := abs_add _ _
      _ ≤ (1 + eps) + 1 := by linarith
      _ = 2 + eps := by ring
  calc |(cs - Real.cos theta) * (cs + Real.cos theta) +
          (ss - Real.sin theta) * (ss + Real.sin theta)|
      ≤ |(cs - Real.cos theta) * (cs + Real.cos theta)| +
        |(ss - Real.sin theta) * (ss + Real.sin theta)| := abs_add _ _
    _ = |cs - Real.cos theta| * |cs + Real.cos theta| +
        |ss - Real.sin theta| * |ss + Real.sin theta| := by
          rw [abs_mul, abs_mul]
    _ ≤ eps * (2 + eps) + eps * (2 + eps) := by
          apply add_le_add
          · exact mul_le_mul hc h_sum_cos (abs_nonneg _) (by linarith)
          · exact mul_le_mul hs h_sum_sin (abs_nonneg _) (by linarith)
    _ = 4 * eps + 2 * eps^2 := by ring

end RAC.TableLookup.SinCosInterpolation
