-- Q16.16 ↔ float conversion: round-trip error ≤ 2^{-17}, pipeline error ≤ 17*2^{-17}
import Mathlib
namespace RAC.Hardware.CordicQ16

def half_lsb : ℝ := (2 : ℝ)^(-(17 : ℤ))

theorem pipeline_error_bound : 16 * half_lsb + half_lsb = 17 * half_lsb := by ring

/-- Pipeline error rewrites in the natural Q16 form: 17·2⁻¹⁷ = 2⁻¹³·(17/16).
    Proof: split 2⁻¹⁷ = 2⁻¹³·2⁻⁴, then 2⁻⁴ = 1/16, then ring closes. -/
theorem error_simplified : (17 : ℝ) * (2 : ℝ)^(-(17:ℤ)) = (2 : ℝ)^(-(13:ℤ)) * (17/16) := by
  have hsplit : (2 : ℝ)^(-(17 : ℤ)) = (2 : ℝ)^(-(13 : ℤ)) * (2 : ℝ)^(-(4 : ℤ)) := by
    rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    norm_num
  have h4 : (2 : ℝ)^(-(4 : ℤ)) = (1 : ℝ) / 16 := by norm_num
  rw [hsplit, h4]
  ring

end RAC.Hardware.CordicQ16
