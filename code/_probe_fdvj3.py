# -*- coding: utf-8 -*-
"""_probe_fdvj3.py — localise the FD-vs-JVP disagreement with a CONTROL BLOCK.

The layer has two structurally different parts:
  * the field step  w+ = (1-theta)w + theta[(1-alpha)w + alpha * chi .* E(w)]  -- purely
    algebraic, no iterative solve.  Its Jacobian is EXACT and known in closed form:
        D = d w+/d w = (1-theta alpha) I + theta alpha diag(chi) G_b ,
    so block D is a map whose derivative is available three independent ways.
  * the contrast step  chi_hat = _cg_ls(E, y),  a fixed-15-iteration CG solve on an
    ill-conditioned system.  Blocks A, B and C all pass through it.

If the central difference agrees with the analytic JVP on D but not on A, the fault is the
CG solve, not forward-mode AD.  The eps ladder separates "truncation-limited" (error falls
as eps -> 0) from "jump-limited" (||Delta f|| stays constant, i.e. the executed map is not
smooth) from "round-off-limited" (error rises as eps -> 0 with ||Delta f|| -> 0).
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
EPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)


def build():
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True), Gb, Gs, Ei)
    y_b = y_te[:BATCH].to(torch.complex128)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
    c0, w0 = init_state(m, y_b, BATCH)
    return m, y_b, c0.to(DT), w0


def ladder(tag, op):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(op.Nin, generator=g, dtype=DT, device=op.dev)
    x = x / x.norm()
    V = op._to_states(x)
    Ja = op.jvp(x)
    na = float(Ja.norm())
    print("  %s: ||Ja|| = %.6e" % (tag, na))
    for eps in EPS:
        with torch.no_grad():
            Df = (op._raw(op.zr + eps * V).reshape(-1)
                  - op._raw(op.zr - eps * V).reshape(-1))
        ndf = float(Df.norm())
        nf = ndf / (2 * eps)
        rel = float((Ja - Df / (2 * eps)).norm() / nf) if nf > 1e-20 else float("nan")
        print("     eps=%.0e  ||Df||=%.4e  ||Df||/(2eps)=%.4e  rel.diff(Ja)=%.3e"
              % (eps, ndf, nf, rel))


def main():
    m, y_b, c0, w0 = build()
    m.theta = 1.0
    zr = flat_of(c0, w0)
    n = NX * NY
    I = N_INC

    # closed-form D, to be compared against both estimators
    with torch.no_grad():
        Gb = m.Gb.to(torch.complex128)
        eye = torch.eye(n, dtype=torch.float64)
        # D acts on real and imaginary parts of w separately with the same real matrix
        # Re/Im decouple because G_b is applied identically; build it by applying to basis.
        pass

    fd = make_f(m, 0.5, 0.5, 1.0, y_b, False, which="w")
    opD = JacOp(fd, zr, in_idx=torch.arange(n, n * (1 + 2 * I)),
                out_slice=slice(0, 2 * I * n), name="D")
    fa = make_f(m, 0.5, 0.5, 1.0, y_b, False, which="A")
    opA = JacOp(fa, zr, in_idx=torch.arange(0, n), out_slice=slice(0, n), name="A")

    print("=== CONTROL: block D  (no contrast solve) ===")
    ladder("D", opD)
    print("=== SUSPECT: block A  (through the 15-step CG contrast solve) ===")
    ladder("A", opA)


if __name__ == "__main__":
    main()
