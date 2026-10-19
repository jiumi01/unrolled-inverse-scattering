# -*- coding: utf-8 -*-
"""Scan cg_lam_rel: exact cycle gain, extrapolation behavior, short-train RRE."""
import numpy as np, torch, os
from isp_core import UnrolledISP, exact_gain, exact_cycle_gain
import run_experiments as R

Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = R.load_data()
chi_te_d, y_te_d = chi_te.to(R.DEV), y_te.to(R.DEV)
for lam_rel in [0.05, 0.1, 0.25, 0.5]:
    m = UnrolledISP(8, Gb, Gs, Ei, R.NX, R.NY, R.N_INC, mode="constrained",
                    eps_bar=0.03, cg_lam_rel=lam_rel).to(R.DEV)
    R.train(m, chi_tr, y_tr, epochs=8, log=lambda s: None)
    rre_tr = R.eval_model(m, chi_te_d, y_te_d)
    chi_b = chi_te_d[:50]; y_b = y_te_d[:50]
    with torch.no_grad():
        chi0, w0 = R.init_state(m, y_b, 50)
        cyc = exact_cycle_gain(m, chi0, w0, y_b)
        gs = []
        for k in range(8):
            gs.append(round(exact_gain(m, k, chi0, w0, y_b), 3))
            chi0, w0 = m.layer(k, chi0, w0, y_b)
        rres = []
        for step in range(1, 81):
            kk = (step - 1) % 8
            chi0, w0 = m.layer(kk, chi0, w0, y_b)
            if step in (1, 40, 80):
                rres.append(round(R.rre(chi0, chi_b), 4))
    print("lam_rel=%.2f  RRE@8ep=%.4f  cycle_gain=%.3f  layers=%s  extrapRRE(1/40/80)=%s"
          % (lam_rel, rre_tr, cyc, gs[:4], rres))
