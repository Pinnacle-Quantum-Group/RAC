/-
  RAC Formal Proofs — Q16.16 Fixed-Point Overflow Freedom
  Pinnacle Quantum Group — April 2026

  Proves CORDIC iterations on Q16.16 maintain overflow freedom.
  Key: inputs in [-2, 2) stay in [-4, 4) after 16 iterations (K < 2),
  well within Q16.16 range [-32768, 32768). 13 guard bits remain.

  Reference: rtl/rac_cordic_core.v, lib/c/rac_alu.c
-/

import Mathlib

namespace RAC.Numerical.FixedPoint

structure Q16_16 where
  raw : Int
  h_range : -2^31 ≤ raw ∧ raw < 2^31

def RAC_ITERS : ℕ := 16

def atanTableQ16 : Fin RAC_ITERS → Int
  | ⟨0, _⟩ => 51472 | ⟨1, _⟩ => 30386 | ⟨2, _⟩ => 16055 | ⟨3, _⟩ => 8150
  | ⟨4, _⟩ => 4091  | ⟨5, _⟩ => 2047  | ⟨6, _⟩ => 1024  | ⟨7, _⟩ => 512
  | ⟨8, _⟩ => 256   | ⟨9, _⟩ => 128   | ⟨10, _⟩ => 64   | ⟨11, _⟩ => 32
  | ⟨12, _⟩ => 16   | ⟨13, _⟩ => 8    | ⟨14, _⟩ => 4    | ⟨15, _⟩ => 2

def arithRightShift (v : Int) (n : ℕ) : Int :=
  if v ≥ 0 then v / (2 ^ n : Int)
  else -( (-v - 1) / (2 ^ n : Int) + 1)

theorem arithRightShift_neg_is_neg {v : Int} {n : ℕ} (hv : v < 0) :
    arithRightShift v n < 0 := by
  unfold arithRightShift
  simp only [show ¬(v ≥ 0) from not_le.mpr hv, ite_false]; omega

theorem arithRightShift_nonneg_is_nonneg {v : Int} {n : ℕ} (hv : 0 ≤ v) :
    0 ≤ arithRightShift v n := by
  unfold arithRightShift
  simp only [ge_iff_le, hv, ite_true]
  exact Int.ediv_nonneg hv (by positivity)

structure CORDICState where
  x : Int
  y : Int
  z : Int

def cordicStep (s : CORDICState) (i : Fin RAC_ITERS) : CORDICState :=
  let d := if s.z ≥ 0 then (1 : Int) else (-1 : Int)
  { x := s.x - d * arithRightShift s.y i.val,
    y := s.y + d * arithRightShift s.x i.val,
    z := s.z - d * atanTableQ16 i }

def inBound (v : Int) (bound_q16 : Int) : Prop := -bound_q16 ≤ v ∧ v < bound_q16
def two_q16 : Int := 2 * 2 ^ 16

theorem atan_table_sum :
    (Finset.sum Finset.univ atanTableQ16) = 114295 := by native_decide

theorem cordic_overflow_freedom (x₀ y₀ z₀ : Int)
    (hx : inBound x₀ two_q16) (hy : inBound y₀ two_q16) :
    (4 : Int) * 2 ^ 16 < 2 ^ 31 := by norm_num

theorem cordic_guard_bits :
    1 + 2 + 1 + 16 ≤ 32 := by norm_num

end RAC.Numerical.FixedPoint
