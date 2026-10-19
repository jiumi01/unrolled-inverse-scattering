# -*- coding: utf-8 -*-
"""诊断2：用 run_gain_decomp 的 frozen 函数复算 B/C（受约束，layer 3）与 A。"""
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
# A
fA, pA, nA, iA = R.s_of(R.make_flatten(m, 0, 0.5, 0.5, False, y_te_d[:16], "full"), zr, name="full")
print("frozen gain_decomp A: sigma_max_full=%.6f (power %.6f)  [JSON 14.418558]" % (fA, pA), flush=True)
# chi-block of A
ac, apc, _, _ = R.s_of(R.make_flatten(m, 0, 0.5, 0.5, False, y_te_d[:16], "chi"),
                       zr, inj=torch.arange(NX * NY), out=slice(0, NX * NY), name="A")
print("frozen gain_decomp A.chi_block=%.6f  [JSON 3.218399 ; bjac sigma_A 3.252599]" % ac, flush=True)
del m

ck = os.path.join("results", "ckpt_constrained_K8.pt")
mk = as_float64_model(UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=0.03).to(DEV),
                      Gb, Gs, Ei)
mk.load_state_dict(torch.load(ck, map_location=DEV))
mk.eval()
zr = R.flat_state(chi0, w0)
for k in (3,):
    lv, lp, _, _ = R.s_of(R.make_H(mk, k, y_te_d[:16]), zr, name="H%d" % k)
    cv, cp, _, _ = R.s_of(R.make_flatten(mk, k, 0.5, 0.5, True, y_te_d[:16], "full"), zr, name="comp%d" % k)
    print("frozen gain_decomp C[constrained k=%d]: Lip(H)=%.6f  composite=%.6f (power %.6f)"
          % (k, lv, cv, cp), flush=True)
print("JSON: B_constrained.Lip_H[7]=0.319217 ; C_constrained.sigma_max_composite_max=13.031655")
