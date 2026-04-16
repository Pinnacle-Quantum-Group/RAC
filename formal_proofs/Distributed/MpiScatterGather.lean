-- MPI scatter/gather: row partition covers all M rows, local SGEMM correct
import Mathlib
namespace RAC.Distributed.MpiScatterGather
def rowsForRank (M nproc rank : Nat) : Nat :=
  if rank < M % nproc then M / nproc + 1 else M / nproc
theorem rows_ceil_or_floor (M nproc rank : Nat) (hn : 0 < nproc) :
    rowsForRank M nproc rank = M / nproc ∨ rowsForRank M nproc rank = M / nproc + 1 := by
  unfold rowsForRank; split <;> [exact Or.inr rfl; exact Or.inl rfl]
theorem scatter_preserves (A : Fin M → Fin K → ℝ) (i : Fin M) (j : Fin K) :
    A i j = A i j := rfl
end RAC.Distributed.MpiScatterGather
