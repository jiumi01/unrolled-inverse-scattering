# -*- coding: utf-8 -*-
"""对照：用作者 float64 路径（仅用于定位操作点）复算 gain_decomp 的操作点 E。"""
import os, sys, torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
torch.set_grad_enabled(True)
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
with torch.no_grad():
    Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
    colnorm = Ew.abs().pow(2).sum(dim=1).sqrt()
    maxcol = float(colnorm.max().item()); maxabs = float(Ew.abs().max().item())
print("gain_decomp-path (float64 model) max_colnorm_E=%.10f max_abs_E=%.10f ratio=%.10f"
      % (maxcol, maxabs, maxcol / maxabs))
print("chi0 max=%.6f" % float(chi0.abs().max()))
