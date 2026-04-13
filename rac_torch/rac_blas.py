"""rac_blas.py — PyTorch BLAS API mirror for RAC.

Single-precision, row-major BLAS surface (Levels 1, 2, 3) implemented on top
of PyTorch primitives. Each function matches the semantics of the rac_blas.h
C API (Pinnacle Quantum Group RAC library) and operates on torch.Tensor.

In-place ops modify their output tensors directly. Float16 / bfloat16 inputs
are promoted to fp32 for compute and cast back. Works on any device.
"""

import enum
import math
from typing import Tuple

import torch


# ── Enums (CBLAS-compatible numeric values) ──────────────────────────────────


class BlasOp(enum.IntEnum):
    NO_TRANS = 111
    TRANS = 112


class BlasUpLo(enum.IntEnum):
    UPPER = 121
    LOWER = 122


class BlasDiag(enum.IntEnum):
    NON_UNIT = 131
    UNIT = 132


class BlasSide(enum.IntEnum):
    LEFT = 141
    RIGHT = 142


# ── Private helpers ──────────────────────────────────────────────────────────


def _promote(t: torch.Tensor) -> torch.Tensor:
    """Promote half/bfloat16 tensors to fp32 for compute (no-op otherwise).

    Returned tensor lives on the same device as the input.
    """
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.float()
    return t


def _make_sym(A: torch.Tensor, uplo: BlasUpLo) -> torch.Tensor:
    """Build the full symmetric N×N matrix from the stored triangle of A."""
    Af = _promote(A)
    n = Af.shape[0]
    if uplo == BlasUpLo.UPPER:
        U = torch.triu(Af)
        # Diagonal must not be doubled when we add the transpose.
        diag = torch.diag(torch.diagonal(Af))
        return U + U.t() - diag
    else:
        L = torch.tril(Af)
        diag = torch.diag(torch.diagonal(Af))
        return L + L.t() - diag


def _make_tri(A: torch.Tensor, uplo: BlasUpLo, diag: BlasDiag) -> torch.Tensor:
    """Build a working triangular copy of A.

    If diag == UNIT, the diagonal is forced to 1 (off-diagonal stored values
    in the chosen triangle are kept; values in the other triangle are zeroed).
    """
    Af = _promote(A)
    if uplo == BlasUpLo.UPPER:
        T = torch.triu(Af)
    else:
        T = torch.tril(Af)
    if diag == BlasDiag.UNIT:
        n = T.shape[0]
        idx = torch.arange(n, device=T.device)
        T = T.clone()
        T[idx, idx] = 1.0
    return T


def _writeback(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Copy src into dst, casting if dst is half/bfloat16. Returns dst."""
    if dst.dtype != src.dtype:
        dst.copy_(src.to(dst.dtype))
    else:
        dst.copy_(src)
    return dst


# ── Level 1 BLAS ─────────────────────────────────────────────────────────────


def saxpy(alpha: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """y := alpha * x + y (in-place)."""
    if y.dtype in (torch.float16, torch.bfloat16) or x.dtype != y.dtype:
        result = _promote(y) + float(alpha) * _promote(x).to(_promote(y).dtype)
        _writeback(y, result)
    else:
        y.add_(x, alpha=float(alpha))
    return y


def sdot(x: torch.Tensor, y: torch.Tensor) -> float:
    """dot := sum_i x_i * y_i (returns Python float)."""
    return torch.dot(_promote(x).flatten(), _promote(y).flatten()).item()


def snrm2(x: torch.Tensor) -> float:
    """nrm2 := sqrt(sum_i x_i^2)."""
    return torch.linalg.vector_norm(_promote(x).flatten()).item()


def sasum(x: torch.Tensor) -> float:
    """asum := sum_i |x_i|."""
    return _promote(x).abs().sum().item()


def isamax(x: torch.Tensor) -> int:
    """Return 0-based index of element with maximum |x_i|."""
    return int(torch.argmax(_promote(x).flatten().abs()).item())


def sscal(alpha: float, x: torch.Tensor) -> torch.Tensor:
    """x := alpha * x (in-place)."""
    if x.dtype in (torch.float16, torch.bfloat16):
        _writeback(x, _promote(x) * float(alpha))
    else:
        x.mul_(float(alpha))
    return x


def scopy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """y := x (in-place; y must be same shape as x)."""
    _writeback(y, _promote(x).to(_promote(y).dtype))
    return y


def sswap(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Swap contents of x and y in place; return (x, y)."""
    tmp = x.clone()
    _writeback(x, y)
    _writeback(y, tmp)
    return x, y


def srot(
    x: torch.Tensor, y: torch.Tensor, c: float, s: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Givens rotation in place:

        x_i := c*x_i + s*y_i
        y_i := c*y_i - s*x_i
    """
    xf = _promote(x).clone()
    yf = _promote(y).clone()
    new_x = float(c) * xf + float(s) * yf
    new_y = float(c) * yf - float(s) * xf
    _writeback(x, new_x)
    _writeback(y, new_y)
    return x, y


def srotg(a: float, b: float) -> Tuple[float, float, float, float]:
    """Construct a Givens rotation. Returns (c, s, r, z).

    Pure scalar algorithm (Lawson-Hanson / BLAS reference) so this function
    does not take or return tensors.
    """
    a = float(a)
    b = float(b)
    roe = a if abs(a) > abs(b) else b
    scale = abs(a) + abs(b)
    if scale == 0.0:
        c = 1.0
        s = 0.0
        r = 0.0
        z = 0.0
    else:
        r = scale * math.sqrt((a / scale) ** 2 + (b / scale) ** 2)
        r = math.copysign(r, roe)
        c = a / r
        s = b / r
        if abs(a) > abs(b):
            z = s
        elif c != 0.0:
            z = 1.0 / c
        else:
            z = 1.0
    return c, s, r, z


# ── Level 2 BLAS ─────────────────────────────────────────────────────────────


def sgemv(
    trans: BlasOp,
    alpha: float,
    A: torch.Tensor,
    x: torch.Tensor,
    beta: float,
    y: torch.Tensor,
) -> torch.Tensor:
    """y := alpha * op(A) * x + beta * y (in-place on y)."""
    Af = _promote(A)
    xf = _promote(x).flatten()
    yf = _promote(y).flatten()
    if trans == BlasOp.NO_TRANS:
        Aop = Af
    else:
        Aop = Af.t()
    new_y = float(alpha) * (Aop @ xf) + float(beta) * yf
    _writeback(y, new_y.view_as(y))
    return y


def sger(
    alpha: float,
    x: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    """A := alpha * x * y^T + A (rank-1 update, in-place on A)."""
    xf = _promote(x).flatten()
    yf = _promote(y).flatten()
    update = float(alpha) * torch.outer(xf, yf)
    new_A = _promote(A) + update
    _writeback(A, new_A)
    return A


def ssymv(
    uplo: BlasUpLo,
    alpha: float,
    A: torch.Tensor,
    x: torch.Tensor,
    beta: float,
    y: torch.Tensor,
) -> torch.Tensor:
    """y := alpha * A * x + beta * y, A symmetric (only `uplo` triangle read)."""
    A_full = _make_sym(A, uplo)
    xf = _promote(x).flatten()
    yf = _promote(y).flatten()
    new_y = float(alpha) * (A_full @ xf) + float(beta) * yf
    _writeback(y, new_y.view_as(y))
    return y


def ssyr(
    uplo: BlasUpLo,
    alpha: float,
    x: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    """A := alpha * x * x^T + A; only `uplo` triangle of A is touched."""
    xf = _promote(x).flatten()
    outer = float(alpha) * torch.outer(xf, xf)
    if uplo == BlasUpLo.UPPER:
        masked = torch.triu(outer)
    else:
        masked = torch.tril(outer)
    new_A = _promote(A) + masked
    _writeback(A, new_A)
    return A


def ssyr2(
    uplo: BlasUpLo,
    alpha: float,
    x: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    """A := alpha * (x*y^T + y*x^T) + A; only `uplo` triangle of A is touched."""
    xf = _promote(x).flatten()
    yf = _promote(y).flatten()
    outer = float(alpha) * (torch.outer(xf, yf) + torch.outer(yf, xf))
    if uplo == BlasUpLo.UPPER:
        masked = torch.triu(outer)
    else:
        masked = torch.tril(outer)
    new_A = _promote(A) + masked
    _writeback(A, new_A)
    return A


def strmv(
    uplo: BlasUpLo,
    trans: BlasOp,
    diag: BlasDiag,
    A: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """x := op(A) * x, A triangular (in-place on x)."""
    T = _make_tri(A, uplo, diag)
    if trans == BlasOp.TRANS:
        T = T.t()
    xf = _promote(x).flatten()
    new_x = T @ xf
    _writeback(x, new_x.view_as(x))
    return x


def strsv(
    uplo: BlasUpLo,
    trans: BlasOp,
    diag: BlasDiag,
    A: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Solve op(A) * x = b in place (b passed via x); A triangular."""
    T = _make_tri(A, uplo, diag)
    upper = uplo == BlasUpLo.UPPER
    unit = diag == BlasDiag.UNIT
    if trans == BlasOp.TRANS:
        # Solve A^T x = b by passing transposed operand. The triangle flips:
        # transpose of an upper-triangular matrix is lower-triangular.
        T_use = T.t()
        upper_use = not upper
    else:
        T_use = T
        upper_use = upper
    xf = _promote(x).flatten().unsqueeze(-1)
    sol = torch.linalg.solve_triangular(
        T_use, xf, upper=upper_use, unitriangular=unit
    ).squeeze(-1)
    _writeback(x, sol.view_as(x))
    return x


# ── Level 3 BLAS ─────────────────────────────────────────────────────────────


def sgemm(
    transA: BlasOp,
    transB: BlasOp,
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: torch.Tensor,
) -> torch.Tensor:
    """C := alpha * op(A) * op(B) + beta * C (in-place on C)."""
    Af = _promote(A)
    Bf = _promote(B)
    Cf = _promote(C)
    opA = Af.t() if transA == BlasOp.TRANS else Af
    opB = Bf.t() if transB == BlasOp.TRANS else Bf
    new_C = float(alpha) * (opA @ opB) + float(beta) * Cf
    _writeback(C, new_C)
    return C


def ssymm(
    side: BlasSide,
    uplo: BlasUpLo,
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: torch.Tensor,
) -> torch.Tensor:
    """C := alpha * A * B + beta * C  (LEFT)
       C := alpha * B * A + beta * C  (RIGHT)
    A is symmetric; only `uplo` triangle is read."""
    A_full = _make_sym(A, uplo)
    Bf = _promote(B)
    Cf = _promote(C)
    if side == BlasSide.LEFT:
        new_C = float(alpha) * (A_full @ Bf) + float(beta) * Cf
    else:
        new_C = float(alpha) * (Bf @ A_full) + float(beta) * Cf
    _writeback(C, new_C)
    return C


def ssyrk(
    uplo: BlasUpLo,
    trans: BlasOp,
    alpha: float,
    A: torch.Tensor,
    beta: float,
    C: torch.Tensor,
) -> torch.Tensor:
    """C := alpha * op(A) * op(A)^T + beta * C; only `uplo` triangle of C updated."""
    Af = _promote(A)
    Cf = _promote(C)
    if trans == BlasOp.NO_TRANS:
        tmp = Af @ Af.t()
    else:
        tmp = Af.t() @ Af
    new_C = float(beta) * Cf
    update = float(alpha) * tmp
    if uplo == BlasUpLo.UPPER:
        mask = torch.triu(torch.ones_like(update, dtype=torch.bool))
    else:
        mask = torch.tril(torch.ones_like(update, dtype=torch.bool))
    new_C = torch.where(mask, new_C + update, new_C)
    _writeback(C, new_C)
    return C


def ssyr2k(
    uplo: BlasUpLo,
    trans: BlasOp,
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: torch.Tensor,
) -> torch.Tensor:
    """C := alpha * (op(A)*op(B)^T + op(B)*op(A)^T) + beta*C; uplo triangle only."""
    Af = _promote(A)
    Bf = _promote(B)
    Cf = _promote(C)
    if trans == BlasOp.NO_TRANS:
        tmp = Af @ Bf.t() + Bf @ Af.t()
    else:
        tmp = Af.t() @ Bf + Bf.t() @ Af
    new_C = float(beta) * Cf
    update = float(alpha) * tmp
    if uplo == BlasUpLo.UPPER:
        mask = torch.triu(torch.ones_like(update, dtype=torch.bool))
    else:
        mask = torch.tril(torch.ones_like(update, dtype=torch.bool))
    new_C = torch.where(mask, new_C + update, new_C)
    _writeback(C, new_C)
    return C


def strmm(
    side: BlasSide,
    uplo: BlasUpLo,
    trans: BlasOp,
    diag: BlasDiag,
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """B := alpha * op(A) * B  (LEFT)
       B := alpha * B * op(A)  (RIGHT)
    A triangular; result in B."""
    A_op = _make_tri(A, uplo, diag)
    if trans == BlasOp.TRANS:
        A_op = A_op.t()
    Bf = _promote(B)
    if side == BlasSide.LEFT:
        new_B = float(alpha) * (A_op @ Bf)
    else:
        new_B = float(alpha) * (Bf @ A_op)
    _writeback(B, new_B)
    return B


def strsm(
    side: BlasSide,
    uplo: BlasUpLo,
    trans: BlasOp,
    diag: BlasDiag,
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Solve op(A) * X = alpha * B (LEFT)  or  X * op(A) = alpha * B (RIGHT).

    Result overwrites B. A is triangular and assumed non-singular.
    """
    T = _make_tri(A, uplo, diag)
    upper = uplo == BlasUpLo.UPPER
    unit = diag == BlasDiag.UNIT
    if trans == BlasOp.TRANS:
        T_use = T.t()
        upper_use = not upper
    else:
        T_use = T
        upper_use = upper
    rhs = float(alpha) * _promote(B)
    if side == BlasSide.LEFT:
        # Solve T_use * X = rhs   ->  X is M x N
        sol = torch.linalg.solve_triangular(
            T_use, rhs, upper=upper_use, unitriangular=unit, left=True
        )
    else:
        # Solve X * T_use = rhs
        sol = torch.linalg.solve_triangular(
            T_use, rhs, upper=upper_use, unitriangular=unit, left=False
        )
    _writeback(B, sol)
    return B


# ── Public API ───────────────────────────────────────────────────────────────


__all__ = [
    # Enums
    "BlasOp",
    "BlasUpLo",
    "BlasDiag",
    "BlasSide",
    # Level 1
    "saxpy",
    "sdot",
    "snrm2",
    "sasum",
    "isamax",
    "sscal",
    "scopy",
    "sswap",
    "srot",
    "srotg",
    # Level 2
    "sgemv",
    "sger",
    "ssymv",
    "ssyr",
    "ssyr2",
    "strmv",
    "strsv",
    # Level 3
    "sgemm",
    "ssymm",
    "ssyrk",
    "ssyr2k",
    "strmm",
    "strsm",
]


# ── Smoke test ───────────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Exercise all 23 wrappers on small random tensors; print OK/FAIL per name."""
    torch.manual_seed(0)

    passed = 0
    failed = 0
    results = []

    def _run(name, fn):
        nonlocal passed, failed
        try:
            fn()
            results.append(f"OK   {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            results.append(f"FAIL {name}: {exc!r}")
            failed += 1

    # Common test fixtures
    def mk_vec(n=8):
        return torch.randn(n, dtype=torch.float32)

    def mk_mat(m, n):
        return torch.randn(m, n, dtype=torch.float32)

    # ── Level 1 ──────────────────────────────────────────────────────────
    _run("saxpy", lambda: saxpy(0.5, mk_vec(), mk_vec()))
    _run("sdot", lambda: sdot(mk_vec(), mk_vec()))
    _run("snrm2", lambda: snrm2(mk_vec()))
    _run("sasum", lambda: sasum(mk_vec()))
    _run("isamax", lambda: isamax(mk_vec()))
    _run("sscal", lambda: sscal(2.0, mk_vec()))
    _run("scopy", lambda: scopy(mk_vec(), mk_vec()))
    _run("sswap", lambda: sswap(mk_vec(), mk_vec()))
    _run("srot", lambda: srot(mk_vec(), mk_vec(), 0.6, 0.8))
    _run("srotg", lambda: srotg(3.0, 4.0))

    # ── Level 2 ──────────────────────────────────────────────────────────
    _run(
        "sgemv",
        lambda: sgemv(
            BlasOp.NO_TRANS, 1.0, mk_mat(3, 4), mk_vec(4), 0.5, mk_vec(3)
        ),
    )
    _run(
        "sger",
        lambda: sger(0.7, mk_vec(3), mk_vec(4), mk_mat(3, 4)),
    )
    # Symmetric / triangular ops use square A
    _run(
        "ssymv",
        lambda: ssymv(
            BlasUpLo.UPPER, 1.0, mk_mat(4, 4), mk_vec(4), 0.0, mk_vec(4)
        ),
    )
    _run(
        "ssyr",
        lambda: ssyr(BlasUpLo.LOWER, 1.0, mk_vec(4), mk_mat(4, 4)),
    )
    _run(
        "ssyr2",
        lambda: ssyr2(
            BlasUpLo.UPPER, 1.0, mk_vec(4), mk_vec(4), mk_mat(4, 4)
        ),
    )
    # For triangular ops, build a well-conditioned matrix for solves.
    A_tri = mk_mat(4, 4) + 4.0 * torch.eye(4)
    _run(
        "strmv",
        lambda: strmv(
            BlasUpLo.UPPER, BlasOp.NO_TRANS, BlasDiag.NON_UNIT,
            A_tri.clone(), mk_vec(4),
        ),
    )
    _run(
        "strsv",
        lambda: strsv(
            BlasUpLo.UPPER, BlasOp.NO_TRANS, BlasDiag.NON_UNIT,
            A_tri.clone(), mk_vec(4),
        ),
    )

    # ── Level 3 ──────────────────────────────────────────────────────────
    _run(
        "sgemm",
        lambda: sgemm(
            BlasOp.NO_TRANS, BlasOp.NO_TRANS,
            1.0, mk_mat(3, 4), mk_mat(4, 3), 0.5, mk_mat(3, 3),
        ),
    )
    _run(
        "ssymm",
        lambda: ssymm(
            BlasSide.LEFT, BlasUpLo.UPPER,
            1.0, mk_mat(4, 4), mk_mat(4, 3), 0.0, mk_mat(4, 3),
        ),
    )
    _run(
        "ssyrk",
        lambda: ssyrk(
            BlasUpLo.UPPER, BlasOp.NO_TRANS,
            1.0, mk_mat(4, 3), 0.5, mk_mat(4, 4),
        ),
    )
    _run(
        "ssyr2k",
        lambda: ssyr2k(
            BlasUpLo.LOWER, BlasOp.NO_TRANS,
            1.0, mk_mat(4, 3), mk_mat(4, 3), 0.5, mk_mat(4, 4),
        ),
    )
    A_tri_sq = mk_mat(4, 4) + 4.0 * torch.eye(4)
    _run(
        "strmm",
        lambda: strmm(
            BlasSide.LEFT, BlasUpLo.UPPER, BlasOp.NO_TRANS, BlasDiag.NON_UNIT,
            1.0, A_tri_sq.clone(), mk_mat(4, 3),
        ),
    )
    _run(
        "strsm",
        lambda: strsm(
            BlasSide.LEFT, BlasUpLo.UPPER, BlasOp.NO_TRANS, BlasDiag.NON_UNIT,
            1.0, A_tri_sq.clone(), mk_mat(4, 3),
        ),
    )

    for line in results:
        print(line)
    total = passed + failed
    print(f"\nSummary: {passed}/{total} passed, {failed} failed")


if __name__ == "__main__":
    _smoke_test()
