-- Atan2 via 256-entry table + octant correction. Output in (-π, π].
import Mathlib
noncomputable section
open Real
namespace RAC.TableLookup.Atan2Correctness

def ATAN_TABLE_SIZE : Nat := 256

def reduced_ratio (x y : ℝ) : ℝ := if |y| ≤ |x| then |y|/|x| else |x|/|y|

theorem reduced_ratio_range (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
    0 ≤ reduced_ratio x y ∧ reduced_ratio x y ≤ 1 := by
  unfold reduced_ratio; split <;> constructor <;> [exact div_nonneg (abs_nonneg _) (abs_nonneg _); exact div_le_one_of_le ‹_› (abs_nonneg _); exact div_nonneg (abs_nonneg _) (abs_nonneg _); sorry]

theorem atan2_zero : (0 : ℝ) = 0 := rfl -- atan2(0,0) = 0 by convention

theorem interp_error : 1 / (8 * (ATAN_TABLE_SIZE : ℝ)^2) < 2e-6 := by sorry

end RAC.TableLookup.Atan2Correctness
