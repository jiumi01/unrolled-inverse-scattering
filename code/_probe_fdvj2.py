# -*- coding: utf-8 -*-
"""_probe_fdvj2.py — decide WHICH side of the FD-vs-JVP disagreement is trustworthy.

Three independent questions, in increasing order of decisiveness:

 Q1  Adjoint consistency.  For random v,w: <Jv,w> = <v,J^T w> to round-off?  This tests
     jvp and vjp against EACH OTHER (not against FD) and needs no smoothness assumption.

 Q2  Is the contrast solve converged?  Print ||b - A x|| / ||b|| after n_cg = 15 and after
     n_cg = 400.  If 15 steps leave a large relative residual, the executed layer is an
     under-converged CG iterate, which is exactly the regime where a difference quotient
     stops converging.

 Q3  Re-run the FD with the solve converged (n_cg = 400).  If FD now agrees with JVP, the
     instrument is correct and the earlier 100% disagreement was the under-converged
     solve, not AD.
"""
import os, sys
import numpy as np
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


def build(proj=True):
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=proj), Gb, Gs, Ei)
    y_b = y_te[:BATCH].to(torch.complex128)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
    c0, w0 = init_state(m, y_b, BATCH)
    return m, y_b, c0.to(DT), w0


def cg_residual(m, E, y, n_cg):
    """relative residual ||b - A x||/||b|| of the CG iterate, recomputed independently
    from the same A and b used inside _cg_ls."""
    with torch.no_grad():
        x = m._cg_ls(E, y, n_cg=n_cg)

        def apply_A(v):
            Ex = E * v.unsqueeze(1)
            mm = torch.matmul(m.Gs, Ex.reshape(-1, Ex.shape[-1]).T).T.reshape(E.shape)
            Gm = torch.matmul(m.Gs.conj().T, mm.reshape(-1, mm.shape[-1]).T).T.reshape(E.shape)
            return (E.conj() * Gm).real.sum(dim=1)

        b = (E.conj() * torch.matmul(
            m.Gs.conj().T, y.reshape(-1, y.shape[-1]).T).T.reshape(E.shape)).real.sum(dim=1)
        diagA = ((torch.abs(E) ** 2) * (torch.abs(m.Gs) ** 2).sum(dim=0)).sum(dim=1)
        lam = m.cg_lam_rel * diagA + 1e-12
        r = b - (apply_A(x) + lam * x)
        return float(r.norm() / (b.norm() + 1e-30))


def main():
    m, y_b, c0, w0 = build(True)
    zr = flat_of(c0, w0)
    n = NX * NY
    m.theta = 1.0

    # ---- Q1 adjoint consistency ------------------------------------------------
    fA = make_f(m, 0.5, 0.5, 1.0, y_b, False, which="A")
    op = JacOp(fA, zr, in_idx=torch.arange(0, n), out_slice=slice(0, n), name="A")
    g = torch.Generator().manual_seed(1)
    v = torch.randn(op.Nin, generator=g, dtype=DT, device=op.dev); v /= v.norm()
    w = torch.randn(op.Nout, generator=g, dtype=DT, device=op.dev); w /= w.norm()
    Jv, JTw = op.jvp(v), op.vjp(w)
    print("=== Q1 adjoint identity <Jv,w> vs <v,J^T w> ===")
    print("   <Jv,w>=% .12e   <v,J^Tw>=% .12e   rel=%.2e"
          % (float((Jv * w).sum()), float((v * JTw).sum()),
             abs(float((Jv * w).sum()) - float((v * JTw).sum()))
             / (abs(float((Jv * w).sum())) + 1e-30)))

    # ---- Q2 CG convergence ----------------------------------------------------
    with torch.no_grad():
        E = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
        w_t = (1 - 0.5) * w0 + 0.5 * c0.unsqueeze(1) * (m.Ei.unsqueeze(0) + m._Gb_apply(w0))
        E2 = m.Ei.unsqueeze(0) + m._Gb_apply(w_t)
    print("=== Q2 relative residual of the contrast CG solve ===")
    for nc in (5, 15, 50, 150, 400):
        print("   n_cg=%4d  ||b-Ax||/||b|| = %.3e" % (nc, cg_residual(m, E2, y_b, nc)))

    # ---- Q3 FD with a converged solve -----------------------------------------
    orig = m._cg_ls
    for n_cg in (15, 400):
        m._cg_ls = (lambda nc: (lambda E, y, **kw: orig(E, y, n_cg=nc)))(n_cg)
        f = make_f(m, 0.5, 0.5, 1.0, y_b, False, which="A")
        op2 = JacOp(f, zr, in_idx=torch.arange(0, n), out_slice=slice(0, n), name="A")
        gg = torch.Generator().manual_seed(0)
        x = torch.randn(op2.Nin, generator=gg, dtype=DT, device=op2.dev); x /= x.norm()
        V = op2._to_states(x)
        Ja = op2.jvp(x)
        print("=== Q3 block A, n_cg=%d ===" % n_cg)
        for eps in (1e-3, 1e-4, 1e-5, 1e-6):
            with torch.no_grad():
                Jf = (op2._raw(op2.zr + eps * V).reshape(-1)
                      - op2._raw(op2.zr - eps * V).reshape(-1)) / (2 * eps)
            nf = float(Jf.norm())
            print("   eps=%.0e  ||Ja||=%.4e ||Jf||=%.4e  rel.diff=%.3e"
                  % (eps, float(Ja.norm()), nf,
                     float((Ja - Jf).norm() / nf) if nf > 1e-20 else float("nan")))
    m._cg_ls = orig


if __name__ == "__main__":
    main()
