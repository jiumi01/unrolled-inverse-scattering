# -*- coding: utf-8 -*-
"""_xcheck_operators.py -- instrument cross-check BEFORE the regeneration chain.

Two float64 audit drivers quote the "physics-core full-layer Jacobian at the
Born-LS init state, alpha=gamma=0.5, batch 16":

  * run_bjac.make_f        -> used by _ngc_audit.py, run_bjac.py, run_prec_audit.py
  * run_gain_decomp.make_flatten (layer_fixed) -> used by run_gain_decomp.py,
                                                  run_gain_decomp_pair.py

On the post-fix dataset these two paths disagree by ~3% (3.1234 vs 3.2193), which
would make every cross-driver "agreement" percentage in the manuscript meaningless.
This script decides which construction is the outlier BEFORE any regeneration:

  1. map equivalence: f1(z) == f2(z) at several random z (the audited FUNCTIONS);
  2. spectrum: sigma_max via both constructions at the same state, Lanczos + power;
  3. state equivalence: init_state vs the pair driver's born_ls_state at n_cg=15.

CPU-only, no checkpoint, ~2 min.  Writes results/xcheck_operators.json.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_grad_enabled(True)

from isp_core import UnrolledISP
from spec_tools import JacOp, sigma_max_L, sigma_max_power, as_float64_model
from run_experiments import load_data, init_state, NX, NY, N_INC
import run_bjac as RB
import run_gain_decomp as RG
import run_gain_decomp_pair as RP

DEV = "cpu"
BATCH = 16
ALPHA = GAMMA = 0.5
THETA = 1.0
RES = os.path.normpath(os.path.join(HERE, "..", "results"))


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    y_b = y_te[:BATCH].to(DEV).to(torch.complex128)
    log("[data] chi_te %s  y_te %s  batch=%d" % (tuple(chi_te.shape), tuple(y_te.shape), BATCH))

    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
        Gb, Gs, Ei)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
    chi0, w0 = init_state(m, y_b, BATCH)
    chi0 = chi0.to(torch.float64)
    assert w0.is_complex(), "complex field lost"

    z1 = RB.flat_of(chi0, w0)
    z2 = RG.flat_state(chi0, w0)
    log("[state] flat_of vs flat_state max abs diff: %.3e" % float((z1 - z2).abs().max()))

    f1 = RB.make_f(m, ALPHA, GAMMA, THETA, y_b, False, which="full")
    f2 = RG.make_flatten(m, 0, ALPHA, GAMMA, False, y_b, "full")

    # ---- 1. map equivalence at random z ------------------------------------- #
    g = torch.Generator().manual_seed(7)
    zz = torch.randn(z1.shape, generator=g, dtype=torch.float64)
    zz = zz / zz.norm()
    o1, o2 = f1(zz), f2(zz)
    map_rel = float((o1 - o2).norm() / o1.norm())
    log("[map] f1 vs f2 at random z: rel diff = %.3e" % map_rel)

    # ---- 2. spectrum at the init state, both constructions ------------------ #
    op1 = JacOp(f1, z1, name="bjac_make_f")
    s1, nv1, info1 = sigma_max_L(op1, tol=1e-6)
    s1p, _ = sigma_max_power(op1, iters=300)
    del op1
    op2 = JacOp(f2, z2, name="gd_make_flatten")
    s2, nv2, info2 = sigma_max_L(op2, tol=1e-6)
    s2p, _ = sigma_max_power(op2, iters=300)
    del op2
    log("[spec] make_f:        lanczos=%.6f power=%.6f (%s, %d matvecs)" % (s1, s1p, info1, nv1))
    log("[spec] make_flatten:  lanczos=%.6f power=%.6f (%s, %d matvecs)" % (s2, s2p, info2, nv2))
    log("[spec] relative gap:  %.3e" % (abs(s1 - s2) / max(s1, s2)))

    # ---- 3. state: init_state vs born_ls_state(15) -------------------------- #
    y_b_dev = y_b.to(RG.DEV)
    _, chi0p, w0p = RP.born_ls_state("constrained", Gb, Gs, Ei, y_b_dev, 15)
    chi0p, w0p = chi0p.to(DEV), w0p.to(DEV)
    st = float((chi0 - chi0p).abs().max()) + float((w0 - w0p).abs().max())
    log("[state] init_state vs born_ls_state(15): max abs diff = %.3e" % st)

    out = {
        "batch": BATCH, "alpha": ALPHA, "gamma": GAMMA, "theta": THETA,
        "dtype": "float64", "device": DEV,
        "map_rel_diff_random_z": map_rel,
        "state_flat_diff": float((z1 - z2).abs().max()),
        "sigma_make_f": {"lanczos": float(s1), "power": float(s1p),
                         "solver": info1, "matvecs": int(nv1)},
        "sigma_make_flatten": {"lanczos": float(s2), "power": float(s2p),
                               "solver": info2, "matvecs": int(nv2)},
        "sigma_rel_gap": abs(s1 - s2) / max(s1, s2),
        "state_init_vs_born_ls_15": st,
        "verdict": ("constructions agree" if (map_rel < 1e-10 and abs(s1 - s2) / max(s1, s2) < 1e-4)
                    else "CONSTRUCTIONS DISAGREE -- investigate before quoting cross-driver agreement"),
        "wallclock_s": time.time() - t0,
    }
    with open(os.path.join(RES, "xcheck_operators.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("[done] %.0fs -> results/xcheck_operators.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
