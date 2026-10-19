# -*- coding: utf-8 -*-
"""
run_classical_sweep.py — fairness check for the classical baseline.

The unrolled networks learn damping factors bounded by a_max=0.5 / g_max=0.6, whereas the
classical baseline of run_experiments.py uses the full step alpha=gamma=1. A referee may
therefore object that the baseline is not tuned. Here we sweep alpha=gamma over a grid at
several depths and report the best achievable classical RRE, so the comparison can be made
against the *best* classical configuration rather than one arbitrary setting.

Also reports the same for the unrolled depth horizon (K=8, 24, 50, 100) to show whether
a tuned classical method would close the gap.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import UnrolledISP
from run_experiments import load_data, init_state, rre, NX, NY, N_INC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
KS = [8, 24, 50, 100]


def run_classical(K, alpha, Gb, Gs, Ei, chi_te, y_te):
    """Classical alternating iteration with a prescribed damping, K layers."""
    m = UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, mode="classical").to(DEV)
    a = torch.tensor(alpha, device=DEV)
    with torch.no_grad():
        def layer(chi, w, y):
            w_t = (1 - a) * w + a * chi.unsqueeze(1) * (m.Ei.unsqueeze(0) + m._Gb_apply(w))
            E = m.Ei.unsqueeze(0) + m._Gb_apply(w_t)
            chi_hat = m._cg_ls(E, y)
            chi_t = (1 - a) * chi + a * chi_hat
            return chi_t.clamp(0.0, m.chi_max), w_t

        chi0, w0 = init_state(m, y_te, chi_te.shape[0])
        cc, ww = chi0, w0
        for _ in range(K):
            cc, ww = layer(cc, ww, y_te)
        e = rre(cc, chi_te)
    del m
    return e


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)
    out = {}
    for K in KS:
        row = {}
        for a in ALPHAS:
            row[str(a)] = run_classical(K, a, Gb, Gs, Ei, chi_te_d, y_te_d)
        best_a = min(row, key=lambda k: row[k])
        out[str(K)] = {"rre": row, "best_alpha": best_a, "best_rre": row[best_a]}
        print(f"K={K:3d}  best alpha={best_a}: {row[best_a]:.4f}   " +
              " ".join(f"{a}:{row[str(a)]:.3f}" for a in ALPHAS))
    with open(os.path.join(RES, "classical_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[classical sweep done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
