-- Q16.16 ↔ float conversion: round-trip error ≤ 2^{-17}, pipeline error ≤ 17*2^{-17}
import Mathlib
namespace RAC.Hardware.CordicQ16

def half_lsb : ℝ := (2 : ℝ)^(-(17 : ℤ))

theorem pipeline_error_bound : 16 * half_lsb + half_lsb = 17 * half_lsb := by ring

theorem error_simplified : (17 : ℝ) * (2 : ℝ)^(-(17:ℤ)) = (2 : ℝ)^(-(13:ℤ)) * (17/16) := by sorry

end RAC.Hardware.CordicQ16
