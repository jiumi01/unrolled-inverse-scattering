# -*- coding: utf-8 -*-
"""Verify: exact gains with damping caps + extrapolation stability."""
import numpy as np, torch, os
from isp_core import UnrolledISP, exact_gain, exact_cycle_gain
import run_experiments as R

Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = R.load_data()
chi_te_d, y_te_d = chi_te.to(R.DEV), y_te.to(R.DEV)
m = UnrolledISP(8, Gb, Gs, Ei, R.NX, R.NY, R.N_INC, mode="constrained", eps_bar=0.03).to(R.DEV)
R.train(m, chi_tr, y_tr, epochs=8, log=lambda s: None)
print("short-trained RRE=%.4f" % R.eval_model(m, chi_te_d, y_te_d))
print("alpha caps:", np.round((m.a_max * torch.sigmoid(m.a)).detach().cpu().numpy(), 3))
print("gamma caps:", np.round((m.g_max * torch.sigmoid(m.g)).detach().cpu().numpy(), 3))
chi_b = chi_te_d[:50]; y_b = y_te_d[:50]
with torch.no_grad():
    chi0, w0 = R.init_state(m, y_b, 50)
    gs = []
    for k in range(8):
        gs.append(round(exact_gain(m, k, chi0, w0, y_b), 4))
        chi0, w0 = m.layer(k, chi0, w0, y_b)
    print("exact per-layer gains:", gs)
    chi0b, w0b = R.init_state(m, y_b, 50)
    print("exact cycle gain:", round(exact_cycle_gain(m, chi0b, w0b, y_b), 4))
    # cyclic extrapolation
    chi0, w0 = R.init_state(m, y_b, 50)
    for k in range(8):
        chi0, w0 = m.layer(k, chi0, w0, y_b)
    print("after K=8 RRE=%.4f |chi|=%.4f" % (R.rre(chi0, chi_b), chi0.norm(dim=1).mean()))
    for step in range(1, 81):
        kk = (step - 1) % 8
        chi0, w0 = m.layer(kk, chi0, w0, y_b)
        if step % 20 == 0:
            res = (((chi0 - prev).norm(dim=1)) / (chi0.norm(dim=1) + 1e-12)).mean().item()
            print("step %3d RRE=%.4f |chi|=%.4f" % (step, R.rre(chi0, chi_b), chi0.norm(dim=1).mean()))
        prev = chi0.clone()
# appended: rho measurement
from isp_core import exact_cycle_rho
with torch.no_grad():
    chi0r, w0r = R.init_state(m, y_b, 50)
    print("cycle rho:", round(exact_cycle_rho(m, chi0r, w0r, y_b, m=30), 4))
