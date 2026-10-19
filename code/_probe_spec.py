# -*- coding: utf-8 -*-
"""Timing + validation probe for spec_tools before rewriting the audit."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(True)
from spec_tools import JacOp, sigma_max_L, sigma_max_power, spectral_radius
from run_bjac import make_f, flat_of, THETA, ALPHA0, GAMMA0
from run_experiments import load_data, init_state, NX, NY, N_INC
from isp_core import UnrolledISP

t0 = time.time()
# ---- L0: random linear map, dimension check
B, D, R = 1, 400, 260
torch.manual_seed(0)
L = torch.randn(R, D) / np.sqrt(D)
zr = torch.randn(B, D)
op = JacOp(lambda z: z @ L.T, zr, name="rand")
print("op dims: Nin=%d Nout=%d (expect %d, %d)" % (op.Nin, op.Nout, B*D, B*R))
s, nv, info = sigma_max_L(op)
svd = float(torch.linalg.matrix_norm(L, 2))
print("L0 sigma_L=%.10f  SVD=%.10f  rel.err=%.2e  matvecs=%d  %s" % (s, svd, abs(s-svd)/svd, nv, info))

# ---- physics
Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
DEV = "cpu"
m = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV)
with torch.no_grad():  # pre-warm the lazily-cached diag(Gs^H Gs)
    m._cg_ls(m.Ei.unsqueeze(0).expand(1,-1,-1), y_te[:1].to(DEV))
y_b = y_te[:16].to(DEV); chi_b = chi_te[:16].to(DEV)
chi0, w0 = init_state(m, y_b, 16)
zr = flat_of(chi0, w0)
n, I = NX*NY, N_INC
idx_chi = torch.arange(0, n); idx_w = torch.arange(n, n*(1+2*I))

for tag, kw in [("C (w<-chi)", dict(in_idx=idx_chi, out_slice=slice(0, 2*I*n))),
                ("A (chi<-chi)", dict(in_idx=idx_chi, out_slice=slice(0, n))),
                ("full", dict())]:
    f = make_f(m, ALPHA0, GAMMA0, THETA, y_b, False,
               which=("w" if tag.startswith("C") else "A" if tag.startswith("A") else "full"))
    t1 = time.time()
    op = JacOp(f, zr, name=tag, **kw)
    s, nv, info = sigma_max_L(op, tol=1e-5)
    print("[%s] sigma=%.4f matvecs=%d t=%.1fs (%s)" % (tag, s, nv, time.time()-t1, info))

# closed form for C
with torch.no_grad():
    Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
    maxcol = float(Ew.abs().pow(2).sum(dim=1).sqrt().max())
print("closed form sigma_C = %.6f" % (ALPHA0*THETA*maxcol))
print("total %.0fs" % (time.time()-t0))
