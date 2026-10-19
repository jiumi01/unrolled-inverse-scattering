# -*- coding: utf-8 -*-
"""_probe_dtype.py — verify the float64 conversion keeps the physics complex, before
spending 20 minutes on the audit.  Expected: Ei complex128, and a one-layer float32-vs-
float64 relative difference of O(1e-6) (float32 round-off), not O(1e-1)."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
from isp_core import UnrolledISP
from spec_tools import as_float64_model, dtype_consistency
from run_experiments import load_data, init_state, NX, NY, N_INC

Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
print("Ei numpy dtype:", getattr(Ei, "dtype", type(Ei)))
# demonstrate the trap on a throwaway model, because Module.to() mutates in place
tmp = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True)
print("before .to(float64): Ei", tmp.Ei.dtype, "Gb", tmp.Gb.dtype)
tmp.to(torch.float64)
print("after  .to(float64): Ei", tmp.Ei.dtype, "Gb", tmp.Gb.dtype, "<- the trap")
m32 = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True)
m = as_float64_model(UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained",
                                 proj=True), Gb, Gs, Ei)
print("after  as_float64_model: Ei", m.Ei.dtype, "Gb", m.Gb.dtype,
      "corr weight", next(m.parameters()).dtype)
y_b32 = y_te[:16].to(torch.complex64)
y_b64 = y_te[:16].to(torch.complex128)
with torch.no_grad():
    m32._cg_ls(m32.Ei.unsqueeze(0).expand(1, -1, -1), y_b32[:1])
    c0, w0 = init_state(m32, y_b32, 16)
with torch.no_grad():
    m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b64[:1])
c0d, w0d = init_state(m, y_b64, 16)
print("state dtypes:", c0.dtype, w0.dtype, "|", c0d.dtype, w0d.dtype)
res = dtype_consistency(m32, m, w0d.detach(), y_b64, chi=c0d.detach())
for k, v in res.items():
    print("  %-18s %s" % (k, ("%.3e" % v) if isinstance(v, float) else v))
