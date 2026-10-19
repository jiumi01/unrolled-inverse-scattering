# -*- coding: utf-8 -*-
"""_probe_sigmaC.py — instrument self-check for the coupling-block singular value.

Question: for the physics core, is sigma_max(d w+/d chi) equal to alpha*max_r|E|?
Analytic claim: w+ = w + theta*(w_t - w), w_t = (1-a)w + a*chi*Ew  =>  d w+/d chi = theta*a*diag(Ew).
Probe computes the single-coordinate central-difference columns directly, and also
J*u for random unit u supported on the chi block, with B=1 and B=16.
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(False)
from isp_core import UnrolledISP
from run_experiments import load_data, init_state, NX, NY, N_INC
from run_bjac import core_parts, flat_of

DEV = "cpu"
Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)
m = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV)

for B in (1, 16):
    chi_b, y_b = chi_te_d[:B], y_te_d[:B]
    chi0, w0 = init_state(m, y_b, B)
    zr = flat_of(chi0, w0)
    n, I = m.nx * m.ny, m.n_inc
    a = g = 0.5
    Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
    print(f"--- B={B}  max|Ew|={float(Ew.abs().max()):.6f}  alpha*max|E|={a*float(Ew.abs().max()):.6f}")

    # direct columns of d w+/d chi
    d = 1e-6
    cols = []
    for r in range(0, n, 97):
        e = torch.zeros_like(zr); e[:, r] = 1.0
        _, wp, _, _ = core_parts(m, zr + d * e, a, g, 1.0, y_b)
        _, wm, _, _ = core_parts(m, zr - d * e, a, g, 1.0, y_b)
        cols.append(((wp - wm) / (2 * d)))
    # analytic column for the same r values
    rs = list(range(0, n, 97))
    ana = [a * Ew[:, :, r] for r in rs]
    for k, r in enumerate(rs):
        nc = cols[k].norm(dim=(1, 2)).max().item()
        # analytic: per-sample Frobenius norm of alpha*Ew[:,:,r]
        na = a * Ew[:, :, r].abs().norm(dim=1).max().item()
        print(f"   r={r:4d}  ||col||_F(dchi)={nc:.8f}   alpha*||Ew[:,:,r]||_F={na:.8f}"
              f"  ratio={nc/max(na,1e-30):.6f}")

    # J*u for random unit u on the chi block
    for trial in range(3):
        u = torch.zeros_like(zr); u[:, :n] = torch.randn(B, n); u = u / u.norm()
        _, wp, _, _ = core_parts(m, zr + d * u, a, g, 1.0, y_b)
        _, wm, _, _ = core_parts(m, zr - d * u, a, g, 1.0, y_b)
        Ju = (wp - wm) / (2 * d)
        print(f"   trial{trial}: ||J u||={Ju.norm():.6f}  bound alpha*max|E|={a*float(Ew.abs().max()):.6f}"
              f"  ratio={Ju.norm()/ (a*float(Ew.abs().max())):.4f}")
