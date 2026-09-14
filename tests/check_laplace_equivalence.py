"""Check LaplaceOperator against a literal transcription of the reference code.

The reference (`Laplace-Neural-Operator/2D_Diffusion/main.py`, `PR2d`) is
transcribed below exactly, including its index pairing, and evaluated densely at
a size small enough for its O(C^2 * prod(M) * prod(N)) tensors. Our factorized
operator must reproduce it to floating-point tolerance.

Run: python tests/check_laplace_equivalence.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from laplace_operator import LaplaceOperator


def reference_pr2d(x, pole1, pole2, residue):
    """Literal transcription of PR2d.forward on a normalized [0,1) domain."""
    N1, N2 = x.shape[-2:]
    ty = torch.arange(N1, dtype=torch.float64) / N1
    tx = torch.arange(N2, dtype=torch.float64) / N2
    alpha = torch.fft.fft2(x, dim=[-2, -1])
    # float64 frequencies: the upstream default is float32, which would make
    # this comparison measure the reference's own rounding, not our algebra.
    f1 = torch.fft.fftfreq(N1, d=1.0 / N1, dtype=torch.float64)
    f2 = torch.fft.fftfreq(N2, d=1.0 / N2, dtype=torch.float64)
    lambda1 = ((2j * math.pi) * f1.to(torch.complex128)).reshape(-1, 1, 1, 1)
    lambda2 = ((2j * math.pi) * f2.to(torch.complex128)).reshape(-1, 1, 1, 1)

    term1 = torch.div(1, torch.einsum("pbix,qbik->pqbixk",
                                      torch.sub(lambda1, pole1), torch.sub(lambda2, pole2)))
    Hw = torch.einsum("bixk,pqbixk->pqbixk", residue, term1)
    Pk = Hw
    output_residue1 = torch.einsum("biox,oxikpq->bkox", alpha, Hw)
    output_residue2 = torch.einsum("biox,oxikpq->bkpq", alpha, Pk)

    x1 = torch.fft.ifft2(output_residue1, s=(N1, N2)).real
    t1 = torch.einsum("bip,kz->bipz", pole1, ty.to(torch.complex128).reshape(1, -1))
    t2 = torch.einsum("biq,kx->biqx", pole2, tx.to(torch.complex128).reshape(1, -1))
    t3 = torch.einsum("bipz,biqx->bipqzx", torch.exp(t1), torch.exp(t2))
    x2 = torch.einsum("kbpq,bipqzx->kizx", output_residue2, t3).real / N1 / N2
    return x1 + x2


def main():
    torch.manual_seed(0)
    failures = 0
    for C, M1, M2, N1, N2 in [(3, 2, 2, 6, 6), (4, 3, 2, 8, 5), (2, 4, 4, 7, 9)]:
        op = LaplaceOperator(C, 2, (M1, M2), stable_poles=False, channel_chunk=2).to(torch.complex128)
        x = torch.randn(2, C, N1, N2, dtype=torch.float64)
        with torch.no_grad():
            ours = op(x)
            theirs = reference_pr2d(x, op.poles[0], op.poles[1], op.residue)
        err = (ours - theirs).abs().max().item()
        scale = theirs.abs().max().item()
        ok = err <= 1e-9 * max(scale, 1.0)
        failures += not ok
        print(f"C={C} M=({M1},{M2}) N=({N1},{N2})  max|ours-ref|={err:.3e} "
              f"rel={err / max(scale, 1e-30):.3e}  {'OK' if ok else 'FAIL'}")

    # chunking must not change the result
    op = LaplaceOperator(6, 2, (3, 3), stable_poles=False, channel_chunk=6).to(torch.complex128)
    x = torch.randn(2, 6, 7, 7, dtype=torch.float64)
    with torch.no_grad():
        full = op(x)
        op.channel_chunk = 1
        chunked = op(x)
    err = (full - chunked).abs().max().item()
    print(f"channel_chunk 6 vs 1: max diff {err:.3e}  {'OK' if err < 1e-12 else 'FAIL'}")
    failures += err >= 1e-12

    # stable_poles must force decaying transients
    op = LaplaceOperator(3, 2, (2, 2), stable_poles=True)
    with torch.no_grad():
        op.poles[0].real.fill_(5.0)
    assert (op._pole(0).real <= 0).all(), "stable_poles did not clamp"
    print("stable_poles clamps positive real parts: OK")

    # 3-D path runs and gradients flow
    op = LaplaceOperator(4, 3, (2, 2, 2), channel_chunk=2)
    x = torch.randn(1, 4, 6, 6, 5, requires_grad=True)
    y = op(x)
    y.sum().backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in op.parameters()]
    print(f"3-D forward {tuple(y.shape)}, finite grads on all {len(grads)} params: "
          f"{'OK' if all(grads) and x.grad is not None else 'FAIL'}")
    failures += not (all(grads) and x.grad is not None)

    print("\nALL CHECKS PASSED" if not failures else f"\n{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
