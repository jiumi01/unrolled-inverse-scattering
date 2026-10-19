# -*- coding: utf-8 -*-
"""Quick sanity probe: classical vs short-trained constrained model."""
import numpy as np, torch, time
from isp_core import build_matrices, make_dirs, make_dataset, UnrolledISP
import run_experiments as R

t0 = time.time()
Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = R.load_data()
print("||Gb||2 =", round(np.linalg.norm(Gb, 2), 4), " chi_max =", float(chi_tr.max()))
chi_te_d, y_te_d = chi_te.to(R.DEV), y_te.to(R.DEV)
for K in [1, 4, 8]:
    m = UnrolledISP(K, Gb, Gs, Ei, R.NX, R.NY, R.N_INC, mode="classical").to(R.DEV)
    print("classical K=%d RRE=%.4f" % (K, R.eval_model(m, chi_te_d, y_te_d)))
m = UnrolledISP(4, Gb, Gs, Ei, R.NX, R.NY, R.N_INC, mode="constrained", eps_bar=0.05).to(R.DEV)
R.train(m, chi_tr, y_tr, epochs=8)
print("constrained K=4 (8 ep) RRE=%.4f" % R.eval_model(m, chi_te_d, y_te_d))
m = UnrolledISP(4, Gb, Gs, Ei, R.NX, R.NY, R.N_INC, mode="unconstrained", eps_bar=0.05).to(R.DEV)
R.train(m, chi_tr, y_tr, epochs=8)
print("unconstrained K=4 (8 ep) RRE=%.4f" % R.eval_model(m, chi_te_d, y_te_d))
print("elapsed %.0fs" % (time.time() - t0))
