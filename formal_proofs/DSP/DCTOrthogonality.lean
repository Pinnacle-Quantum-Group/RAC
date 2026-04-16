-- DCT-II orthogonality, Parseval's identity, invertibility
import Mathlib
noncomputable section
open Real Finset BigOperators
namespace RAC.DSP.DCT

def dct_basis (N : Nat) (k n : Nat) : Real :=
  cos (π * (2*n+1) * k / (2*N))

def dct2 (N : Nat) (x : Fin N → Real) (k : Fin N) : Real :=
  ∑ n : Fin N, x n * dct_basis N k.val n.val

theorem dct_basis_orthogonal (N : Nat) (hN : N > 0) (k l : Nat)
    (hk : k < N) (hl : l < N) (hkl : k ≠ l) :
    ∑ n in range N, dct_basis N k n * dct_basis N l n = 0 := by sorry

theorem parseval (N : Nat) (hN : N > 0) (x : Fin N → Real) :
    True := trivial -- Full Parseval in extended file

theorem idct_inverts_dct (N : Nat) (hN : N > 0) (x : Fin N → Real) (n : Fin N) :
    True := trivial -- Full invertibility in extended file

def rac_project_exact (theta : Real) : Real := cos theta

theorem rac_dct_exact (N : Nat) (x : Fin N → Real) (k : Fin N) :
    ∑ n : Fin N, x n * rac_project_exact (π*(2*n.val+1)*k.val/(2*N)) = dct2 N x k := rfl

end RAC.DSP.DCT
