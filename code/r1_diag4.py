# -*- coding: utf-8 -*-
"""诊断4：投影开启/关闭下，核对 Weyl 步 sigma_max(DT_k) >= sigma_max(J) - Lip(H_k)。"""
import os, sys, torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
torch.set_grad_enabled(True)
import run_gain_decomp as R
from isp_core import UnrolledISP
from spec_tools import as_float64_model
from run_experiments import load_data, init_state, NX, NY, N_INC

DEV = "cpu"
Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
chi_te_d = chi_te.to(DEV).to(torch.float64); y_te_d = y_te.to(DEV).to(torch.complex128)
m = as_float64_model(UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
                     Gb, Gs, Ei)
with torch.no_grad():
    m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
chi0, w0 = init_state(m, y_te_d[:16], 16); chi0 = chi0.to(torch.float64)
zr = R.flat_state(chi0, w0)

for proj in (True, False):
    m1 = as_float64_model(UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained",
                                      proj=proj).to(DEV), Gb, Gs, Ei)
    with torch.no_grad():
        m1._cg_ls(m1.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
    core, cp, _, _ = R.s_of(R.make_flatten(m1, 0, 0.5, 0.5, False, y_te_d[:16], "full"),
                            zr, name="core")
    mk = as_float64_model(UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained",
                                      eps_bar=0.03, proj=proj).to(DEV), Gb, Gs, Ei)
    mk.load_state_dict(torch.load("results/ckpt_constrained_K8.pt", map_location=DEV)); mk.eval()
    for k in (3, 7):
        lv, _, _, _ = R.s_of(R.make_H(mk, k, y_te_d[:16]), zr, name="H%d" % k)
        cv, _, _, _ = R.s_of(R.make_flatten(mk, k, 0.5, 0.5, True, y_te_d[:16], "full"),
                             zr, name="c%d" % k)
        print("proj=%-5s  k=%d : core=%.6f  Lip(H)=%.6f  composite=%.6f   "
              "Weyl-RHS=core-Lip=%.6f   Weyl %s"
              % (proj, k, core, lv, cv, core - lv, "OK" if cv >= core - lv - 1e-6 else "VIOLATED"),
              flush=True)
