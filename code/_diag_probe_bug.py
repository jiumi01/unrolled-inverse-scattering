# -*- coding: utf-8 -*-
"""Why did the FIRST version of _probe_T1_blockLL report a spurious 6.8e-3 for the
projection-OFF control?  Both expressions are evaluated on the SAME tangent here."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(True)
from isp_core import UnrolledISP
from spec_tools import JacOp, as_float64_model
from run_experiments import load_data, init_state, NX, NY, N_INC
from run_gain_decomp import make_flatten, make_H, flat_state

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = GAMMA = 0.5; BATCH = 16

Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
y_d = y_te.to(DEV).to(torch.complex128); chi_te_d = chi_te.to(DEV).to(torch.float64)
m = as_float64_model(UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained",
                                 eps_bar=0.03, proj=True).to(DEV), Gb, Gs, Ei)
m.load_state_dict(torch.load(os.path.join(RES, "ckpt_constrained_K8.pt"), map_location=DEV))
m.eval()
with torch.no_grad():
    m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_d[:1])
chi0, w0 = init_state(m, y_d[:BATCH], BATCH); chi0 = chi0.to(torch.float64)
zr = flat_state(chi0, w0); n, I = m.nx * m.ny, m.n_inc

for proj in (True, False):
    m.proj = proj
    of = JacOp(make_flatten(m, 0, ALPHA, GAMMA, True, y_d[:BATCH], "full"), zr)
    oc = JacOp(make_flatten(m, 0, ALPHA, GAMMA, False, y_d[:BATCH], "full"), zr)
    oh = JacOp(make_H(m, 0, y_d[:BATCH]), zr)
    torch.manual_seed(0)
    v = torch.randn(of.Nin, dtype=zr.dtype, device=DEV); v = v / v.norm()
    DT, J, H = of.jvp(v), oc.jvp(v), oh.jvp(v)
    print("proj=%-5s  len(DT)=%d len(J)=%d len(H)=%d  of.B=%d of.R=%d oh.R=%d"
          % (proj, DT.numel(), J.numel(), H.numel(), of.B, of.R, oh.R))
    # --- the OLD (flat) expression -------------------------------------------
    DH_old = torch.cat([H, torch.zeros_like(DT[n * of.B:])])
    rel_old = (DT - J - DH_old).norm().item() / (DT.norm().item() + 1e-30)
    # --- the NEW (block) expression ------------------------------------------
    DTb, Jb, Hb = DT.reshape(BATCH, -1), J.reshape(BATCH, -1), H.reshape(BATCH, n)
    rel_new = ((DTb[:, :n] - Jb[:, :n] - Hb).norm() ** 2
               + (DTb[:, n:] - Jb[:, n:]).norm() ** 2).sqrt().item() \
        / (DTb.norm().item() + 1e-30)
    print("            rel_old=%.6e   rel_new=%.6e   len(DH_old)=%d (need %d)"
          % (rel_old, rel_new, DH_old.numel(), DT.numel()), flush=True)
