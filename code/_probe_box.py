# -*- coding: utf-8 -*-
"""Probe: does the projection box (0.6) actually bind, and what is the realized |chi|_inf?"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import UnrolledISP
from run_experiments import load_data, init_state, rre, NX, NY, N_INC

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)
print("train chi range", float(chi_tr.min()), float(chi_tr.max()))
print("test  chi range", float(chi_te.min()), float(chi_te.max()))

for mode in ["constrained", "unconstrained"]:
    ck = os.path.join(RES, f"ckpt_{mode}_K8.pt")
    if not os.path.exists(ck):
        continue
    for cmax in [0.6, 0.5]:
        m = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03, chi_max=cmax).to(DEV)
        m.load_state_dict(torch.load(ck, map_location=DEV)); m.eval()
        with torch.no_grad():
            chi0, w0 = init_state(m, y_te_d, 100)
            c, w = m(chi0, w0, y_te_d)
            print(f"  {mode} box={cmax}: RRE={rre(c, chi_te_d):.4f}  max|chi|={c.abs().max().item():.4f} "
                  f"min(chi)={c.min().item():.4f}")
        # long rollout with this box
        with torch.no_grad():
            chi0, w0 = init_state(m, y_te_d[:50], 50)
            for k in range(800):
                chi0, w0 = m.layer(k % 8, chi0, w0, y_te_d[:50])
            print(f"      after 800 layers: max|chi|={chi0.abs().max().item():.3f} "
                  f"rre={rre(chi0, chi_te_d[:50]):.4f} finite={bool(torch.isfinite(chi0).all())}")
        del m
