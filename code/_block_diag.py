# -*- coding: utf-8 -*-
"""Block-wise exact gains via FD-JVP + autograd-VJP power iteration."""
import numpy as np, torch, os
from isp_core import UnrolledISP, _flat_layer_fn, _power_sigma_fd
import run_experiments as R

Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = R.load_data()
chi_te_d, y_te_d = chi_te.to(R.DEV), y_te.to(R.DEV)
m = UnrolledISP(8, Gb, Gs, Ei, R.NX, R.NY, R.N_INC, mode="constrained",
                eps_bar=0.03, cg_lam_rel=0.5).to(R.DEV)
R.train(m, chi_tr, y_tr, epochs=8, log=lambda s: None)
m.eval()
n = R.NX * R.NY
B, I = 16, R.N_INC
chi_b = chi_te_d[:B]; y_b = y_te_d[:B]
with torch.no_grad():
    chi0, w0 = R.init_state(m, y_b, B)
f = _flat_layer_fn(m, 0, y_b)
zr = torch.cat([chi0, w0.real.reshape(B, -1), w0.imag.reshape(B, -1)], dim=1)
D = zr.shape[1]
chi_idx = torch.arange(0, n, device=R.DEV)
w_idx = torch.arange(n, D, device=R.DEV)
print("sigma(J) full      =", round(_power_sigma_fd(f, zr), 4))
print("sigma(J) chi->all  =", round(_power_sigma_fd(f, zr, in_mask=chi_idx), 4))
print("sigma(J) w->all    =", round(_power_sigma_fd(f, zr, in_mask=w_idx), 4))
print("sigma(J) chi->chi  =", round(_power_sigma_fd(f, zr, in_mask=chi_idx, out_slice=slice(0, n)), 4))
print("sigma(J) w->chi    =", round(_power_sigma_fd(f, zr, in_mask=w_idx, out_slice=slice(0, n)), 4))
print("sigma(J) chi->w    =", round(_power_sigma_fd(f, zr, in_mask=chi_idx, out_slice=slice(n, D)), 4))
print("sigma(J) w->w      =", round(_power_sigma_fd(f, zr, in_mask=w_idx, out_slice=slice(n, D)), 4))
