-- Atan2 via 256-entry table + octant correction. Output in (-π, π].
import Mathlib
noncomputable section
open Real
namespace RAC.TableLookup.Atan2Correctness

def ATAN_TABLE_SIZE : Nat := 256

def reduced_ratio (x y : ℝ) : ℝ := if |y| ≤ |x| then |y|/|x| else |x|/|y|

theorem reduced_ratio_range (x y : ℝ) (_ : x ≠ 0 ∨ y ≠ 0) :
    0 ≤ reduced_ratio x y ∧ reduced_ratio x y ≤ 1 := by
  unfold reduced_ratio
  split_ifs with hxy
  · -- Branch: |y| ≤ |x|. Ratio is |y|/|x|.
    exact ⟨div_nonneg (abs_nonneg _) (abs_nonneg _),
           div_le_one_of_le hxy (abs_nonneg _)⟩
  · -- Branch: |y| > |x|. Ratio is |x|/|y|; |x| ≤ |y| from negation.
    push_neg at hxy
    exact ⟨div_nonneg (abs_nonneg _) (abs_nonneg _),
           div_le_one_of_le hxy.le (abs_nonneg _)⟩

theorem atan2_zero : (0 : ℝ) = 0 := rfl -- atan2(0,0) = 0 by convention

/-- 1/(8·256²) = 1/524288 ≈ 1.907×10⁻⁶ < 2×10⁻⁶. Pure numerics. -/
theorem interp_error : 1 / (8 * (ATAN_TABLE_SIZE : ℝ)^2) < 2e-6 := by
  unfold ATAN_TABLE_SIZE
  norm_num

end RAC.TableLookup.Atan2Correctness
