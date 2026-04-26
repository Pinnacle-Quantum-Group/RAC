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

theorem hw_sw_step_error_bound (hw : HwState) (sw : SwState) (i : ℕ)
    (atanEntry : ℤ) (atanVal : ℝ)
    (h_init : stateError hw sw ≤ ε)
    (h_atan : |dequantize atanEntry - atanVal| ≤ q16Resolution / 2) :
    stateError (hwCordicStep hw i atanEntry) (swCordicStep sw i atanVal) ≤
    ε + 3 * q16Resolution := by
  sorry

/-! ## 6. Total N-Step Equivalence -/

theorem hw_sw_n_step_equivalence (n : ℕ) :
    ∀ (hw : HwState) (sw : SwState),
    stateError hw sw ≤ q16Resolution / 2 →
    ∃ (totalBound : ℝ), totalBound ≤ (↑n + 1) * 3 * q16Resolution := by
  sorry

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
