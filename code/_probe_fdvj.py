# -*- coding: utf-8 -*-
"""_probe_fdvj.py — why do the analytic JVP and the central finite difference disagree
by 100% in the float64 audit, when both are supposed to differentiate the same map?

Hypotheses to separate:
  H1  the analytic JVP is zero (a forward-mode AD failure through the CG loop);
  H2  the finite difference is the broken side (step too small / clamp boundary);
  H3  the CLAMP is the culprit: Pi projects onto [0, chi_max], and a large fraction of
      the output sits exactly on the boundary, where clamp is non-differentiable.  The
      analytic derivative and the secant of the difference quotient then legitimately
      disagree at those entries, and no step size fixes it.

Distinguishing evidence printed below:
  * ||Ja||, ||Jf||  (which side is zero?)
  * fraction of output entries pinned at chi_max (how much of the map is clamped?)
  * the same comparison with the clamp DISABLED (m.proj=False) -- if the disagreement
    vanishes there, H3 is confirmed and the whole issue is the nondifferentiable
    projection, not the instrument.
  * an eps ladder on the unprojected map (H2).
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


def dir_and_fd(op, eps):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(op.Nin, generator=g, dtype=DT, device=op.dev)
    x = x / x.norm()
    V = op._to_states(x)
    Ja = op.jvp(x)
    with torch.no_grad():
        Jf = (op._raw(op.zr + eps * V).reshape(-1)
              - op._raw(op.zr - eps * V).reshape(-1)) / (2 * eps)
    nf = float(Jf.norm())
    rel = float((Ja - Jf).norm() / nf) if nf > 1e-20 else float("nan")
    return float(Ja.norm()), nf, rel


def run(tag, proj):
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=proj), Gb, Gs, Ei)
    y_b = y_te[:BATCH].to(torch.complex128)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
    c0, w0 = init_state(m, y_b, BATCH)
    c0 = c0.to(DT)
    zr = flat_of(c0, w0)
    n = NX * NY
    print("=" * 72)
    print("%s  (proj=%s)" % (tag, proj))
    with torch.no_grad():
        fA = make_f(m, 0.5, 0.5, 1.0, y_b, False, which="A")
        out = fA(zr)
        pinned = float((out >= m.chi_max - 1e-12).float().mean())
        print("  output: max=%.6f  frac pinned at chi_max=%.4f  min=%.6f"
              % (float(out.max()), pinned, float(out.min())))
    fA = make_f(m, 0.5, 0.5, 1.0, y_b, False, which="A")
    op = JacOp(fA, zr, in_idx=torch.arange(0, n), out_slice=slice(0, n), name="A")
    for eps in (1e-4, 1e-5, 1e-6, 1e-7):
        na, nf, rel = dir_and_fd(op, eps)
        print("  block A eps=%.0e : ||Ja||=%.6e ||Jf||=%.6e  rel.diff=%.3e" % (eps, na, nf, rel))


if __name__ == "__main__":
    run("BLOCK A, projection ON", True)
    run("BLOCK A, projection OFF", False)
