/-
  RAC — Q8_0 Quantization Round-Trip Error Bound
  Reference: lib/c/rac_q8_0.c
-/
import Mathlib
namespace RAC.Numerical.Quantization

structure Q8Block where
  scale : ℝ
  weights : Fin 32 → Int
  h_range : ∀ i, -127 ≤ weights i ∧ weights i ≤ 127
  h_scale_pos : 0 < scale

def dequant (b : Q8Block) (i : Fin 32) : ℝ :=
  (b.weights i : ℝ) * b.scale

theorem dequant_bounded (b : Q8Block) (i : Fin 32) :
    |dequant b i| ≤ 127 * b.scale := by
  unfold dequant
  rw [abs_mul, abs_of_pos b.h_scale_pos]
  apply mul_le_mul_of_nonneg_right _ (le_of_lt b.h_scale_pos)
  obtain ⟨hlow, hhigh⟩ := b.h_range i
  rw [abs_le]
  refine ⟨?_, ?_⟩
  · exact_mod_cast hlow
  · exact_mod_cast hhigh

theorem zero_exactly_representable (b : Q8Block) (i : Fin 32)
    (h : b.weights i = 0) : dequant b i = 0 := by
  simp [dequant, h]

end RAC.Numerical.Quantization
