# -*- coding: utf-8 -*-
"""诊断：用 run_gain_decomp 的 layer_fixed+spec_tools（作者仪器）只复算 A_physics_core。"""
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
chi0, w0 = init_state(m, y_te_d[:16], 16)
chi0 = chi0.to(torch.float64)
zr = R.flat_state(chi0, w0)
f_val, f_pow, nv, info = R.s_of(R.make_flatten(m, 0, 0.5, 0.5, False, y_te_d[:16], "full"),
                                zr, name="full")
print("gain_decomp-A replicate: sigma_max_full=%.6f (power %.6f) solver=%s matvecs=%d"
      % (f_val, f_pow, info, nv))
print("  JSON A_physics_core.sigma_max_full = 14.418558")
