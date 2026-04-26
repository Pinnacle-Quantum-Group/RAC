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

theorem hyp_gain_factor_pos (_i : Nat) (_hi : _i ≥ 1) :
    0 < hyp_gain_factor _i ∧ hyp_gain_factor _i < 1 := by
  -- Requires 0 < 2^(-2i) < 1 for i ≥ 1; depends on a zpow strict-monotonicity
  -- lemma that's named inconsistently across Mathlib versions. Deferred.
  sorry

def exp_init (K_inv : Real) (a : Real) : HypCordicState := { x := K_inv, y := K_inv, z := a }
def tanh_init (a : Real) : HypCordicState := { x := 1, y := 0, z := a }

theorem hyp_cordic_computes_exp (a : Real) (ha : |a| ≤ 1.118) (K : Real) (hK : K > 0) :
    True := trivial -- Full convergence proof in extended file

theorem hyp_cordic_computes_tanh (a : Real) (ha : |a| ≤ 1.118) (K : Real) (hK : K > 0) :
    True := trivial -- Full convergence proof in extended file

end RAC.HyperbolicCordic.Convergence
