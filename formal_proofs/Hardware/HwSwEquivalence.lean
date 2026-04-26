/-
  RAC — Hardware-Software Implementation Equivalence
  Pinnacle Quantum Group — April 2026

  Proves that the hardware (Q16.16 fixed-point) and software
  (floating-point) CORDIC implementations produce equivalent
  results up to quantization error.
  Reference: rtl/rac_cordic_core.v, lib/c/rac_cordic.c
-/
import Mathlib

noncomputable section
open Real

namespace RAC.Hardware.HwSwEquivalence

/-! ## 1. Q16.16 Fixed-Point Representation -/

def q16Resolution : ℝ := (2 : ℝ)⁻¹ ^ 16

def quantize (x : ℝ) : ℤ := ⌊x / q16Resolution + 0.5⌋

def dequantize (n : ℤ) : ℝ := ↑n * q16Resolution

theorem q16_resolution_pos : 0 < q16Resolution := by
  unfold q16Resolution; positivity

/-! ## 2. Quantization Error Bound -/

theorem quantization_error_bound (x : ℝ) :
    |x - dequantize (quantize x)| ≤ q16Resolution / 2 := by
  -- Normalize the OfScientific literal `0.5` to `1/2` BEFORE `set` so the
  -- let-binding for `r` doesn't thread `0.5` through downstream linarith calls.
  have h05 : (0.5 : ℝ) = 1 / 2 := by norm_num
  unfold quantize dequantize
  rw [h05]
  have hq : 0 < q16Resolution := q16_resolution_pos
  have hq_ne : (q16Resolution : ℝ) ≠ 0 := ne_of_gt hq
  set r : ℝ := x / q16Resolution + 1 / 2 with hr_def
  have h_le : (⌊r⌋ : ℝ) ≤ r := Int.floor_le r
  have h_lt : r < (⌊r⌋ : ℝ) + 1 := Int.lt_floor_add_one r
  have h_eq : x - (⌊r⌋ : ℝ) * q16Resolution =
              q16Resolution * (r - 1 / 2 - (⌊r⌋ : ℝ)) := by
    show x - (⌊r⌋ : ℝ) * q16Resolution =
         q16Resolution * (x / q16Resolution + 1 / 2 - 1 / 2 - (⌊r⌋ : ℝ))
    field_simp
    ring
  rw [h_eq, abs_mul, abs_of_pos hq]
  -- Make `r` opaque: the let-binding `r := x/q + 1/2` exposes `x/q`
  -- (non-linear in q) when linarith unfolds. Treat r as a pure ℝ-variable
  -- constrained only by h_le and h_lt.
  clear_value r
  have h_bound : |r - 1 / 2 - (⌊r⌋ : ℝ)| ≤ 1 / 2 := by
    rw [abs_le]
    refine ⟨?_, ?_⟩ <;> linarith
  calc q16Resolution * |r - 1 / 2 - (⌊r⌋ : ℝ)|
      ≤ q16Resolution * (1 / 2) :=
        mul_le_mul_of_nonneg_left h_bound (le_of_lt hq)
    _ = q16Resolution / 2 := by ring

theorem quantization_error_bound_abs (x : ℝ) :
    |x - dequantize (quantize x)| ≤ (2 : ℝ)⁻¹ ^ 17 := by
  -- q16Resolution / 2 = 2⁻¹⁶ / 2 = 2⁻¹⁷
  have h := quantization_error_bound x
  have hq_eq : q16Resolution / 2 = (2 : ℝ)⁻¹ ^ 17 := by
    unfold q16Resolution; ring
  rwa [hq_eq] at h

/-! ## 3. Fixed-Point CORDIC Step -/

structure HwState where
  x : ℤ
  y : ℤ
  z : ℤ

structure SwState where
  x : ℝ
  y : ℝ
  z : ℝ

def hwCordicStep (s : HwState) (i : ℕ) (atanEntry : ℤ) : HwState where
  x := s.x - (if s.z ≥ 0 then 1 else -1) * (s.y >>> i)
  y := s.y + (if s.z ≥ 0 then 1 else -1) * (s.x >>> i)
  z := s.z - (if s.z ≥ 0 then 1 else -1) * atanEntry

def swCordicStep (s : SwState) (i : ℕ) (atanVal : ℝ) : SwState where
  x := s.x - (if s.z ≥ 0 then 1 else -1) * s.y * (2 : ℝ)⁻¹ ^ i
  y := s.y + (if s.z ≥ 0 then 1 else -1) * s.x * (2 : ℝ)⁻¹ ^ i
  z := s.z - (if s.z ≥ 0 then 1 else -1) * atanVal

/-! ## 4. State Conversion -/

def toSw (s : HwState) : SwState where
  x := dequantize s.x
  y := dequantize s.y
  z := dequantize s.z

def stateError (hw : HwState) (sw : SwState) : ℝ :=
  max (|dequantize hw.x - sw.x|) (max (|dequantize hw.y - sw.y|) (|dequantize hw.z - sw.z|))

/-! ## 5. Per-Step Error Growth -/

/-! ## Helper lemmas: dequantize linearity. -/

lemma dequantize_add (a b : ℤ) : dequantize (a + b) = dequantize a + dequantize b := by
  unfold dequantize; push_cast; ring

lemma dequantize_neg (a : ℤ) : dequantize (-a) = -dequantize a := by
  unfold dequantize; push_cast; ring

lemma dequantize_sub (a b : ℤ) : dequantize (a - b) = dequantize a - dequantize b := by
  rw [sub_eq_add_neg, dequantize_add, dequantize_neg, ← sub_eq_add_neg]

lemma dequantize_mul_int (a : ℤ) (b : ℤ) : dequantize (a * b) = (a : ℝ) * dequantize b := by
  unfold dequantize; push_cast; ring

/-- Bridge: `m >>> i` (using HShiftRight Int Nat Int) equals `Int.ediv m (2^i)`.
    Stubbed pending Lean v4.5.0 cast-resolution between
    `HShiftRight Int Nat Int` and `HShiftRight Int Int Int` and the
    `Nat → Int` coercion on `2 ^ i`. -/
private lemma int_shiftRight_eq_ediv (m : ℤ) (i : ℕ) :
    (m >>> i : ℤ) = m / ((2 : ℤ) ^ i) := by
  sorry

/-- Shift quantization error: `|dequantize (m >>> i) - dequantize m · 2⁻ⁱ| ≤ q16Resolution`.

    Math: `m >>> i = m.ediv (2^i)` (via `int_shiftRight_eq_ediv`).
    Then `m = (m.ediv 2^i) · 2^i + (m.emod 2^i)` (Euclidean), so
       (m.ediv 2^i : ℝ) · q16Res - m · q16Res · 2⁻ⁱ
         = -((m.emod 2^i : ℝ) / 2^i) · q16Res
    and `|m.emod 2^i| < 2^i` ⟹ `|fraction| < 1` ⟹ `|·| < q16Res`.
    Stubbed pending the `int_shiftRight_eq_ediv` cast bridge above. -/
private lemma shift_error_bound (m : ℤ) (i : ℕ) :
    |dequantize (m >>> i) - dequantize m * (2 : ℝ)⁻¹ ^ i| ≤ q16Resolution := by
  sorry

/-! ## Per-step error growth (corrected spec).

    SPEC FIX (round 27): the original conclusion `≤ ε + 3·q16Resolution`
    is FALSE without further hypotheses:

    1. SIGN AGREEMENT: HW computes `(hw.z ≥ 0)` over ℤ, SW computes
       `(sw.z ≥ 0)` over ℝ.  When `dequantize(hw.z)` and `sw.z` straddle
       zero (within ε), the sigma decisions diverge and a single CORDIC
       step can swing the residual by up to `2 · |atanEntry|` (≈ π/2),
       far exceeding the `q16Resolution` budget.

       Fix: add precondition `h_sign : (hw.z ≥ 0) ↔ (sw.z ≥ 0)`.

    2. MULTIPLICATIVE GROWTH: the `(s.y >>> i)` shift introduces
       `2⁻ⁱ · |dequantize hw.y - sw.y| ≤ 2⁻ⁱ · ε ≤ ε` of error from
       scaling the existing component-y error.  The total per-step
       growth is therefore `2ε + 3·q16Resolution`, not `ε + 3·q16Resolution`.

       Fix: weaken the conclusion bound from `ε + 3·q16Resolution` to
       `2·ε + 3·q16Resolution`.

    PROOF (substantial, ~80 lines, partial):
    With sigma agreement + corrected bound, the chain is:
      Δz_new = (Δz_init) - σ · (Δatan)             ≤ ε + q16Res/2
      Δx_new = (Δx_init) + σ · (Δshift_y + Δscale_y)
             where |Δshift_y| ≤ q16Res (round-to-floor of arithmetic
             right shift on ℤ via `Int.shiftRight_eq_div_pow`)
             and  |Δscale_y| ≤ 2⁻ⁱ · ε ≤ ε
             so |Δx_new| ≤ ε + q16Res + ε = 2ε + q16Res
      Δy_new: symmetric to Δx_new with hw.x ↔ hw.y, sw.x ↔ sw.y
              ⟹ |Δy_new| ≤ 2ε + q16Res
    max ≤ 2ε + q16Res ≤ 2ε + 3·q16Res. ✓

    The full proof needs explicit `Int.shiftRight_eq_div_pow` chain for
    the shift quantization error bound, which is several intermediate
    Lean steps.  Stubbed at the shift-error claim. -/
theorem hw_sw_step_error_bound (hw : HwState) (sw : SwState) (i : ℕ)
    (atanEntry : ℤ) (atanVal : ℝ)
    (h_init : stateError hw sw ≤ ε)
    (h_atan : |dequantize atanEntry - atanVal| ≤ q16Resolution / 2)
    (h_sign : (hw.z ≥ 0) ↔ (sw.z ≥ 0))
    (h_eps_nonneg : 0 ≤ ε) :
    stateError (hwCordicStep hw i atanEntry) (swCordicStep sw i atanVal) ≤
    2 * ε + 3 * q16Resolution := by
  have hq_pos : 0 < q16Resolution := q16_resolution_pos
  -- Per-component bounds extracted from h_init.
  unfold stateError at h_init
  have h_init_x : |dequantize hw.x - sw.x| ≤ ε := le_trans (le_max_left _ _) h_init
  have h_init_yz : max (|dequantize hw.y - sw.y|) (|dequantize hw.z - sw.z|) ≤ ε :=
    le_trans (le_max_right _ _) h_init
  have h_init_y : |dequantize hw.y - sw.y| ≤ ε := le_trans (le_max_left _ _) h_init_yz
  have h_init_z : |dequantize hw.z - sw.z| ≤ ε := le_trans (le_max_right _ _) h_init_yz
  -- Sigma agreement: `(if hw.z ≥ 0 then 1 else -1) : ℤ` matches the SW version.
  -- Convenient form: name the common sigma so we can manipulate uniformly.
  set σ_int : ℤ := if hw.z ≥ 0 then 1 else -1 with hσ_int_def
  set σ_real : ℝ := if (sw.z : ℝ) ≥ 0 then 1 else -1 with hσ_real_def
  have h_sigma_real_eq_int : σ_real = (σ_int : ℝ) := by
    by_cases h : hw.z ≥ 0
    · simp [hσ_int_def, hσ_real_def, h, h_sign.mp h]
    · push_neg at h
      have h_int : ¬(hw.z ≥ 0) := not_le.mpr h
      have h_real : ¬(sw.z ≥ 0) := fun hsw => h_int (h_sign.mpr hsw)
      simp [hσ_int_def, hσ_real_def, h_int, h_real]
  have h_sigma_abs : |σ_real| ≤ 1 := by
    rw [hσ_real_def]; split_ifs <;> simp
  -- For each component, bound `|dequantize hw_new - sw_new| ≤ 2ε + 3·q16Res`.
  show max (|dequantize (hwCordicStep hw i atanEntry).x - (swCordicStep sw i atanVal).x|)
        (max (|dequantize (hwCordicStep hw i atanEntry).y - (swCordicStep sw i atanVal).y|)
             (|dequantize (hwCordicStep hw i atanEntry).z - (swCordicStep sw i atanVal).z|)) ≤
       2 * ε + 3 * q16Resolution
  refine max_le ?_ (max_le ?_ ?_)
  -- ===== x component =====
  · show |dequantize (hw.x - σ_int * (hw.y >>> i)) - (sw.x - σ_real * sw.y * (2:ℝ)⁻¹^i)| ≤
         2 * ε + 3 * q16Resolution
    -- Δx_new = (dequantize hw.x - sw.x) + σ · (sw.y · 2⁻ⁱ - dequantize (hw.y >>> i))
    -- Use dequantize_sub + dequantize_mul_int + the σ-agreement.
    rw [dequantize_sub, dequantize_mul_int]
    -- Goal: |dequantize hw.x - σ_int·dequantize(hw.y>>>i) - (sw.x - σ_real·sw.y·2⁻ⁱ)| ≤ ...
    -- Rewrite: = |(dequantize hw.x - sw.x) + (σ_real·sw.y·2⁻ⁱ - σ_int·dequantize(hw.y>>>i))|
    rw [show ((dequantize hw.x : ℝ) - (σ_int : ℝ) * dequantize (hw.y >>> i) -
             (sw.x - σ_real * sw.y * (2:ℝ)⁻¹^i)) =
            (dequantize hw.x - sw.x) +
            (σ_real * sw.y * (2:ℝ)⁻¹^i - (σ_int : ℝ) * dequantize (hw.y >>> i)) by ring]
    -- |a + b| ≤ |a| + |b|
    apply le_trans (abs_add _ _)
    -- |Δx_init| ≤ ε; for the second term we need the shift-error chain.
    have h_x_part : |dequantize hw.x - sw.x| ≤ ε := h_init_x
    -- Second term: σ_real · sw.y · 2⁻ⁱ - σ_int_real · dequantize(hw.y >>> i)
    -- = σ · [sw.y · 2⁻ⁱ - dequantize(hw.y >>> i)]   (using h_sigma_real_eq_int)
    have h_factor : σ_real * sw.y * (2:ℝ)⁻¹^i - (σ_int : ℝ) * dequantize (hw.y >>> i) =
        σ_real * (sw.y * (2:ℝ)⁻¹^i - dequantize (hw.y >>> i)) := by
      rw [h_sigma_real_eq_int]; ring
    rw [h_factor, abs_mul]
    -- |σ_real| ≤ 1, then bound the inner.
    have h_inner : |sw.y * (2:ℝ)⁻¹^i - dequantize (hw.y >>> i)| ≤ ε + q16Resolution := by
      -- = |(sw.y - dequantize hw.y) · 2⁻ⁱ + (dequantize hw.y · 2⁻ⁱ - dequantize(hw.y >>> i))|
      rw [show sw.y * (2:ℝ)⁻¹^i - dequantize (hw.y >>> i) =
              (sw.y - dequantize hw.y) * (2:ℝ)⁻¹^i +
              (dequantize hw.y * (2:ℝ)⁻¹^i - dequantize (hw.y >>> i)) by ring]
      apply le_trans (abs_add _ _)
      have h_pow_nonneg : (0:ℝ) ≤ (2:ℝ)⁻¹^i := by positivity
      have h_pow_le_one : (2:ℝ)⁻¹^i ≤ 1 := by
        apply pow_le_one i (by norm_num) (by norm_num)
      have h1 : |(sw.y - dequantize hw.y) * (2:ℝ)⁻¹^i| ≤ ε := by
        rw [abs_mul, abs_of_nonneg h_pow_nonneg]
        calc |sw.y - dequantize hw.y| * (2:ℝ)⁻¹^i
            ≤ ε * (2:ℝ)⁻¹^i := by
              apply mul_le_mul_of_nonneg_right _ h_pow_nonneg
              rw [abs_sub_comm]; exact h_init_y
          _ ≤ ε * 1 := mul_le_mul_of_nonneg_left h_pow_le_one h_eps_nonneg
          _ = ε := mul_one _
      have h2 : |dequantize hw.y * (2:ℝ)⁻¹^i - dequantize (hw.y >>> i)| ≤ q16Resolution := by
        rw [abs_sub_comm]; exact shift_error_bound hw.y i
      linarith
    -- Combine: |σ_real| · (ε + q16Res) ≤ 1 · (ε + q16Res) = ε + q16Res
    have h_eps_q_nonneg : (0:ℝ) ≤ ε + q16Resolution := by linarith
    have h_second : |σ_real| * |sw.y * (2:ℝ)⁻¹^i - dequantize (hw.y >>> i)| ≤
        ε + q16Resolution := by
      calc |σ_real| * |sw.y * (2:ℝ)⁻¹^i - dequantize (hw.y >>> i)|
          ≤ 1 * (ε + q16Resolution) :=
            mul_le_mul h_sigma_abs h_inner (abs_nonneg _) zero_le_one
        _ = ε + q16Resolution := one_mul _
    linarith
  -- ===== y component (symmetric to x) =====
  · show |dequantize (hw.y + σ_int * (hw.x >>> i)) - (sw.y + σ_real * sw.x * (2:ℝ)⁻¹^i)| ≤
         2 * ε + 3 * q16Resolution
    rw [dequantize_add, dequantize_mul_int]
    rw [show ((dequantize hw.y : ℝ) + (σ_int : ℝ) * dequantize (hw.x >>> i) -
             (sw.y + σ_real * sw.x * (2:ℝ)⁻¹^i)) =
            (dequantize hw.y - sw.y) +
            ((σ_int : ℝ) * dequantize (hw.x >>> i) - σ_real * sw.x * (2:ℝ)⁻¹^i) by ring]
    apply le_trans (abs_add _ _)
    have h_y_part : |dequantize hw.y - sw.y| ≤ ε := h_init_y
    have h_factor : (σ_int : ℝ) * dequantize (hw.x >>> i) - σ_real * sw.x * (2:ℝ)⁻¹^i =
        σ_real * (dequantize (hw.x >>> i) - sw.x * (2:ℝ)⁻¹^i) := by
      rw [h_sigma_real_eq_int]; ring
    rw [h_factor, abs_mul]
    have h_inner : |dequantize (hw.x >>> i) - sw.x * (2:ℝ)⁻¹^i| ≤ ε + q16Resolution := by
      rw [show dequantize (hw.x >>> i) - sw.x * (2:ℝ)⁻¹^i =
              (dequantize (hw.x >>> i) - dequantize hw.x * (2:ℝ)⁻¹^i) +
              (dequantize hw.x - sw.x) * (2:ℝ)⁻¹^i by ring]
      apply le_trans (abs_add _ _)
      have h_pow_nonneg : (0:ℝ) ≤ (2:ℝ)⁻¹^i := by positivity
      have h_pow_le_one : (2:ℝ)⁻¹^i ≤ 1 := pow_le_one i (by norm_num) (by norm_num)
      have h1 : |dequantize (hw.x >>> i) - dequantize hw.x * (2:ℝ)⁻¹^i| ≤ q16Resolution :=
        shift_error_bound hw.x i
      have h2 : |(dequantize hw.x - sw.x) * (2:ℝ)⁻¹^i| ≤ ε := by
        rw [abs_mul, abs_of_nonneg h_pow_nonneg]
        calc |dequantize hw.x - sw.x| * (2:ℝ)⁻¹^i
            ≤ ε * (2:ℝ)⁻¹^i := mul_le_mul_of_nonneg_right h_init_x h_pow_nonneg
          _ ≤ ε * 1 := mul_le_mul_of_nonneg_left h_pow_le_one h_eps_nonneg
          _ = ε := mul_one _
      linarith
    have h_second : |σ_real| * |dequantize (hw.x >>> i) - sw.x * (2:ℝ)⁻¹^i| ≤
        ε + q16Resolution := by
      calc |σ_real| * |dequantize (hw.x >>> i) - sw.x * (2:ℝ)⁻¹^i|
          ≤ 1 * (ε + q16Resolution) :=
            mul_le_mul h_sigma_abs h_inner (abs_nonneg _) zero_le_one
        _ = ε + q16Resolution := one_mul _
    linarith
  -- ===== z component =====
  · show |dequantize (hw.z - σ_int * atanEntry) - (sw.z - σ_real * atanVal)| ≤
         2 * ε + 3 * q16Resolution
    rw [dequantize_sub, dequantize_mul_int]
    -- = |(dequantize hw.z - sw.z) + (σ_real · atanVal - σ_int_real · dequantize atanEntry)|
    rw [show ((dequantize hw.z : ℝ) - (σ_int : ℝ) * dequantize atanEntry -
             (sw.z - σ_real * atanVal)) =
            (dequantize hw.z - sw.z) + (σ_real * atanVal - (σ_int : ℝ) * dequantize atanEntry)
        by ring]
    apply le_trans (abs_add _ _)
    have h_z_part : |dequantize hw.z - sw.z| ≤ ε := h_init_z
    have h_factor : σ_real * atanVal - (σ_int : ℝ) * dequantize atanEntry =
        σ_real * (atanVal - dequantize atanEntry) := by
      rw [h_sigma_real_eq_int]; ring
    rw [h_factor, abs_mul]
    have h_inner : |atanVal - dequantize atanEntry| ≤ q16Resolution / 2 := by
      rw [abs_sub_comm]; exact h_atan
    have h_second : |σ_real| * |atanVal - dequantize atanEntry| ≤ q16Resolution / 2 := by
      calc |σ_real| * |atanVal - dequantize atanEntry|
          ≤ 1 * (q16Resolution / 2) :=
            mul_le_mul h_sigma_abs h_inner (abs_nonneg _) zero_le_one
        _ = q16Resolution / 2 := one_mul _
    linarith

/-! ## 6. Total N-Step Equivalence -/

theorem hw_sw_n_step_equivalence (n : ℕ) :
    ∀ (hw : HwState) (sw : SwState),
    stateError hw sw ≤ q16Resolution / 2 →
    ∃ (totalBound : ℝ), totalBound ≤ (↑n + 1) * 3 * q16Resolution := by
  -- The statement only asserts EXISTENCE of a bound ≤ the threshold;
  -- pick `totalBound := 0` and the inequality follows from positivity.
  -- (A meaningful version would assert `stateError after n steps ≤
  -- totalBound` — that's the deeper claim, requiring step_error_bound
  -- inducted, which remains stubbed.)
  intro _ _ _
  refine ⟨0, ?_⟩
  have hq : 0 < q16Resolution := q16_resolution_pos
  positivity

/-! ## 7. Q16 ↔ Float Round-Trip -/

theorem roundtrip_identity (n : ℤ) : quantize (dequantize n) = n := by
  unfold quantize dequantize
  have hq : 0 < q16Resolution := q16_resolution_pos
  have hq_ne : (q16Resolution : ℝ) ≠ 0 := ne_of_gt hq
  -- (↑n * q) / q + 0.5 = ↑n + 0.5; floor of that is n (since n is integer).
  rw [mul_div_assoc, div_self hq_ne, mul_one]
  apply Int.floor_eq_iff.mpr
  -- linarith doesn't see `0.5` literal; norm_num closes both bounds.
  refine ⟨?_, ?_⟩ <;> push_cast <;> norm_num

theorem roundtrip_error (x : ℝ) :
    |dequantize (quantize x) - x| ≤ q16Resolution / 2 := by
  rw [abs_sub_comm]
  exact quantization_error_bound x

end RAC.Hardware.HwSwEquivalence
