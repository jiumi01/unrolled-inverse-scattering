# -*- coding: utf-8 -*-
"""_probe_setup.py — how long does the audit preamble cost?

_recompute_f64.py, _run_L45.py and run_seeds.py all regenerate their dataset in
load_data() (make_dataset(300, ...) + make_dataset(100, ...)) with 400 dense MoM solves.
Their progress files have stayed empty for tens of minutes while the box also runs
another project's training, so it matters whether that silence is the preamble or a hang.
This probes the three stages separately and extrapolates.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
from isp_core import build_matrices, make_dirs, make_dataset, UnrolledISP
from run_experiments import NX, NY, N_INC, init_state

print("threads:", torch.get_num_threads(), flush=True)

t = time.time()
Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
dirs = make_dirs(N_INC)
print("build_matrices+make_dirs : %.1fs" % (time.time() - t), flush=True)

t = time.time()
chi, y, Ei = make_dataset(5, Gb, Gs, pts, dirs, 2 * np.pi, NX, NY, seed=1, snr_db=20.0)
dt = time.time() - t
print("make_dataset(5 samples)  : %.1fs  -> 300 samples ~ %.0fs, 400 ~ %.0fs"
      % (dt, dt / 5 * 300, dt / 5 * 400), flush=True)

t = time.time()
m = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to("cpu")
print("model construct          : %.1fs" % (time.time() - t), flush=True)
buf = [n for n, b in m.named_buffers()]
print("buffers:", buf, flush=True)

t = time.time()
yb = torch.from_numpy(y[:16]).to(torch.complex64)
c0, w0 = init_state(m, yb, 16)
print("init_state (16)          : %.1fs   chi0 dtype=%s" % (time.time() - t, c0.dtype), flush=True)

try:
    m64 = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(torch.float64)
    print("model .to(float64)       : OK; Ei dtype=%s" % (m64.Ei.dtype,), flush=True)
except Exception as e:
    print("model .to(float64)       : FAILED ->", type(e).__name__, e, flush=True)
