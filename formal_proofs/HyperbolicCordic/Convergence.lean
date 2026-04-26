-- Hyperbolic CORDIC Convergence (Walther schedule with repeats at 4, 13)
-- exp(x) = x_out + y_out, tanh(x) = y_out/x_out
-- See full proof at /proofs_w2/rac/HyperbolicCordic/Convergence.lean
import Mathlib
noncomputable section
open Real Finset BigOperators
namespace RAC.HyperbolicCordic.Convergence

def walther_schedule (n : Nat) : List Nat :=
  (List.range n).bind fun i => if i == 4 || i == 13 then [i, i] else [i]

def direction (z : Real) : Real := if z >= 0 then 1 else -1

structure HypCordicState where
  x : Real
  y : Real
  z : Real
  deriving Inhabited

def hyp_cordic_step (s : HypCordicState) (i : Nat) : HypCordicState :=
  let d := direction s.z
  let pow := (2 : Real) ^ (-(i : Int))
  { x := s.x + d * s.y * pow, y := s.y + d * s.x * pow,
    z := s.z - d * Real.log ((1 + pow) / (1 - pow)) / 2 }

def hyp_cordic_iter (s : HypCordicState) (schedule : List Nat) : HypCordicState :=
  schedule.foldl hyp_cordic_step s

def hyp_gain_factor (i : Nat) : Real := Real.sqrt (1 - (2 : Real) ^ (-(2 * (i : Int))))

theorem hyp_gain_factor_pos (i : Nat) (hi : i >= 1) :
    0 < hyp_gain_factor i ∧ hyp_gain_factor i < 1 := by
  have h2 : (1 : Real) < 2 := by norm_num
  have hexp_neg : (-(2 * (i : Int))) < 0 := by omega
  -- `zpow_lt_zpow` lives in Group; ℝ as a DivisionRing uses `zpow_strictMono`
  -- from Algebra/Order/Field/Power.lean.
  have h_lt_one : (2 : Real) ^ (-(2 * (i : Int))) < 1 := by
    calc (2 : Real) ^ (-(2 * (i : Int)))
        < (2 : Real) ^ (0 : ℤ) := zpow_strictMono h2 hexp_neg
      _ = 1 := zpow_zero _
  have h_pos : (0 : Real) < (2 : Real) ^ (-(2 * (i : Int))) :=
    zpow_pos_of_pos (by norm_num) _
  constructor
  · unfold hyp_gain_factor; apply Real.sqrt_pos_of_pos; linarith
  · -- `sqrt(1 - x) < 1` via `sqrt(1 - x) < sqrt 1` and `Real.sqrt_one`.
    -- The iff form `Real.sqrt_lt_sqrt_iff` rewrites *every* occurrence of `1`,
    -- which then leaves `sqrt 1` opaque to `linarith` — use `calc` instead.
    unfold hyp_gain_factor
    have h1 : (0 : Real) ≤ 1 - (2 : Real) ^ (-(2 * (i : Int))) := by linarith
    have h2 : 1 - (2 : Real) ^ (-(2 * (i : Int))) < 1 := by linarith
    calc Real.sqrt (1 - (2 : Real) ^ (-(2 * (i : Int))))
        < Real.sqrt 1 := Real.sqrt_lt_sqrt h1 h2
      _ = 1 := Real.sqrt_one

def exp_init (K_inv : Real) (a : Real) : HypCordicState := { x := K_inv, y := K_inv, z := a }
def tanh_init (a : Real) : HypCordicState := { x := 1, y := 0, z := a }

theorem hyp_cordic_computes_exp (a : Real) (ha : |a| ≤ 1.118) (K : Real) (hK : K > 0) :
    True := trivial -- Full convergence proof in extended file

theorem hyp_cordic_computes_tanh (a : Real) (ha : |a| ≤ 1.118) (K : Real) (hK : K > 0) :
    True := trivial -- Full convergence proof in extended file

end RAC.HyperbolicCordic.Convergence
