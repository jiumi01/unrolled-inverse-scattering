# -*- coding: utf-8 -*-
"""_probe_fdvj4.py — confirm that the non-smoothness is the UNDER-CONVERGED CG solve.

_probe_fdvj3 established: block D (no iterative solve) matches the analytic JVP to 2.7e-12,
while block A (through the 15-step CG contrast solve) has ||Delta f|| pinned near 0.03 for
every step in 1e-3 .. 1e-10 -- i.e. the executed map is not smooth at those scales, so no
difference quotient can converge.

Hypothesis: the CG is not converged in 15 iterations, and the iterate is dominated by
round-off amplification through an ill-conditioned A.  Test: rerun both the residual
measurement and the eps ladder for n_cg = 15, 50, 200, 600.  If the residual drops by
orders of magnitude and FD then agrees with JVP, the diagnosis is confirmed and the
remedy is a converged contrast solve.
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_grad_enabled(True)

from isp_core import UnrolledISP
from spec_tools import JacOp, as_float64_model
from run_bjac import make_f, flat_of
from run_experiments import load_data, init_state, NX, NY, N_INC

DT = torch.float64
BATCH = 2
EPS = (1e-2, 1e-3, 1e-4, 1e-6, 1e-8)


def build():
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True), Gb, Gs, Ei)
    y_b = y_te[:BATCH].to(torch.complex128)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
    c0, w0 = init_state(m, y_b, BATCH)
    return m, y_b, c0.to(DT), w0


def cg_state(m, E, y, n_cg):
    """Return (x, relative residual ||b-Ax||/||b||) for the regularised LS system, with the
    same A and b as _cg_ls (intermediate reshapes done correctly this time)."""
    with torch.no_grad():
        x = m._cg_ls(E, y, n_cg=n_cg)
        B, I, n = E.shape
        gsd = (torch.abs(m.Gs) ** 2).sum(dim=0)

        def apply_A(v):
            Ex = E * v.unsqueeze(1)
            mm = torch.matmul(m.Gs, Ex.reshape(-1, n).T).T.reshape(B, I, -1)
            Gm = torch.matmul(m.Gs.conj().T, mm.reshape(-1, mm.shape[-1]).T
                              ).T.reshape(B, I, n)
            return (E.conj() * Gm).real.sum(dim=1)

        b = (E.conj() * torch.matmul(
            m.Gs.conj().T, y.reshape(-1, y.shape[-1]).T).T.reshape(B, I, -1)).real.sum(dim=1)
        lam = m.cg_lam_rel * ((torch.abs(E) ** 2) * gsd).sum(dim=1) + 1e-12
        r = b - (apply_A(x) + lam * x)
        return x, float(r.norm() / (b.norm() + 1e-30))


def main():
    m, y_b, c0, w0 = build()
    m.theta = 1.0
    zr = flat_of(c0, w0)
    n = NX * NY

    with torch.no_grad():
        w_t = (1 - 0.5) * w0 + 0.5 * c0.unsqueeze(1) * (m.Ei.unsqueeze(0) + m._Gb_apply(w0))
        E2 = m.Ei.unsqueeze(0) + m._Gb_apply(w_t)
    print("=== CG convergence of the contrast solve ===")
    for nc in (15, 50, 150, 400, 800):
        _, res = cg_state(m, E2, y_b, nc)
        print("   n_cg=%4d   ||b-Ax||/||b|| = %.4e" % (nc, res))

    orig = m._cg_ls
    for n_cg in (15, 50, 200, 600):
        m._cg_ls = (lambda nc: (lambda E, y, **kw: orig(E, y, n_cg=nc)))(n_cg)
        f = make_f(m, 0.5, 0.5, 1.0, y_b, False, which="A")
        op = JacOp(f, zr, in_idx=torch.arange(0, n), out_slice=slice(0, n), name="A")
        g = torch.Generator().manual_seed(0)
        x = torch.randn(op.Nin, generator=g, dtype=DT, device=op.dev)
        x = x / x.norm()
        V = op._to_states(x)
        Ja = op.jvp(x)
        na = float(Ja.norm())
        print("=== block A, n_cg=%d :  ||Ja||=%.6e ===" % (n_cg, na))
        for eps in EPS:
            with torch.no_grad():
                Df = (op._raw(op.zr + eps * V).reshape(-1)
                      - op._raw(op.zr - eps * V).reshape(-1))
            ndf = float(Df.norm())
            nf = ndf / (2 * eps)
            rel = float((Ja - Df / (2 * eps)).norm() / nf) if nf > 1e-20 else float("nan")
            print("   eps=%.0e  ||Df||=%.4e  ||Df||/(2eps)=%.4e  rel.diff(Ja)=%.3e"
                  % (eps, ndf, nf, rel))
    m._cg_ls = orig


if __name__ == "__main__":
    main()
