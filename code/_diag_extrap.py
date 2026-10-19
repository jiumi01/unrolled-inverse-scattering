# -*- coding: utf-8 -*-
"""Diagnose depth-extrapolation drift of the constrained model."""
import numpy as np, torch, os, json
from isp_core import UnrolledISP, estimate_gain
import run_experiments as R

Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = R.load_data()
chi_te_d, y_te_d = chi_te.to(R.DEV), y_te.to(R.DEV)
m = UnrolledISP(8, Gb, Gs, Ei, R.NX, R.NY, R.N_INC, mode="constrained", eps_bar=0.05).to(R.DEV)
m.load_state_dict(torch.load(os.path.join(R.RES, "ckpt_constrained_K8.pt")))
m.eval()
a = torch.sigmoid(m.a).detach().cpu().numpy(); g = torch.sigmoid(m.g).detach().cpu().numpy()
print("learned alpha:", np.round(a, 3))
print("learned gamma:", np.round(g, 3))
with torch.no_grad():
    chi_b = chi_te_d[:50]; y_b = y_te_d[:50]
    chi0, w0 = R.init_state(m, y_b, 50)
    print("init RRE=%.4f  |chi|=%.4f" % (R.rre(chi0, chi_b), chi0.norm(dim=1).mean()))
    for k in range(8):
        chi0, w0 = m.layer(k, chi0, w0, y_b)
    print("after K=8 RRE=%.4f  |chi|=%.4f" % (R.rre(chi0, chi_b), chi0.norm(dim=1).mean()))
    # cyclic extrapolation with diagnostics
    for step in range(1, 41):
        kk = (step - 1) % 8
        prev = chi0.clone()
        chi0, w0 = m.layer(kk, chi0, w0, y_b)
        if step % 5 == 0:
            res = (((chi0 - prev).norm(dim=1)) / (chi0.norm(dim=1) + 1e-12)).mean().item()
            print("step %3d res=%.3e RRE=%.4f |chi|=%.4f  |w|=%.4f"
                  % (step, res, R.rre(chi0, chi_b), chi0.norm(dim=1).mean(), w0.norm(dim=-1).mean()))
    # layer gain at the drifted state
    print("gain at drifted state (layer0):", round(estimate_gain(m, 0, chi0, w0, y_b), 4))
    chi_s, w_s = R.init_state(m, y_b, 50)
    print("gain at init state   (layer0):", round(estimate_gain(m, 0, chi_s, w_s, y_b), 4))
