# -*- coding: utf-8 -*-
"""Is sigma_full(alpha=0.05) = 22.77 real, or a float32/ill-conditioning artefact?

Analytic limit: as alpha -> 0 the layer tends to diag((1-gamma)I, I), so sigma_max -> 1.
Checks: (a) Lanczos float32, (b) power iteration float32, (c) Lanczos float64,
(d) analytic (jvp) vs finite-difference directional derivatives.
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(True)
from spec_tools import JacOp, sigma_max_L, sigma_max_power
from run_bjac import make_f, flat_of, THETA
from run_experiments import load_data, init_state, NX, NY, N_INC
from isp_core import UnrolledISP

Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
DEV = "cpu"
B = 16


def build(dtype):
    m = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV)
    m = m.to(dtype)
    yb = y_te[:B].to(DEV).to(dtype)
    if not hasattr(m, "_gs_diag"):
        with torch.no_grad():
            m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), yb[:1])
    chi0, w0 = init_state(m, yb, B)
    chi0, w0 = chi0.to(dtype), w0.to(dtype)
    return m, yb, chi0, w0


def P(msg):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", "probe_alpha_prog.txt"), "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)


P("alpha |  lanczos-f32  power-f32 |  lanczos-f64  power-f64 | FD-vs-jvp rel.diff")
for a in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
    row = []
    for dtype in (torch.float32, torch.float64):
        m, yb, chi0, w0 = build(dtype)
        zr = flat_of(chi0, w0).to(dtype)
        f = make_f(m, a, a, THETA, yb, False, which="full")
        op = JacOp(f, zr, name="a%.2f" % a)
        s, nv, info = sigma_max_L(op, tol=1e-9 if dtype == torch.float64 else 1e-6)
        sp, _ = sigma_max_power(op, iters=150)
        row += [s, sp]
        if dtype == torch.float32:
            # directional-derivative cross-check: analytic JVP vs central FD
            torch.manual_seed(0)
            v = torch.randn_like(zr); v = v / v.norm()
            Ja = op.jvp(v.reshape(-1))
            eps = 1e-4
            Jf = (f(zr + eps * v) - f(zr - eps * v)) / (2 * eps)
            rel = float((Ja - Jf.reshape(-1)).norm() / (Jf.norm() + 1e-30))
            f32_rel = rel
        del m, op
    P("%.2f  |  %11.5f %11.5f |  %11.5f %11.5f |  %.3e"
      % (a, row[0], row[1], row[2], row[3], f32_rel))
