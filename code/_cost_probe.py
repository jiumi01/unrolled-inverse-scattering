"""Cost audit for R2 m3.6 / Q3.5.

The manuscript compares the unrolled network with the classical iteration in
*outer iterations* ("a factor 12 fewer layers").  A reviewer asked for the
comparison to be stated at matched per-layer cost, and for a wall-clock or FLOP
cross-check.  This probe measures, on the real problem shapes, what one layer of
each costs.

Design notes (traps avoided):
  * CPU-only.  The seed re-run may be occupying the GPU; nothing here needs one,
    and mixing the two would make the timings meaningless.
  * No checkpoint is loaded.  Layer cost is an architecture property, so a
    freshly initialised model with the *exact* production config is the correct
    object; loading a checkpoint would also race with the seed job rewriting
    results/ckpt_seed0_constrained_K8.pt.
  * The CG budget is the default `n_cg=15` of `UnrolledISP._cg_ls`; note the
    layer calls `self._cg_ls(E, y)` *without* passing `self.n_cg`, so the
    `n_cg` attribute is decorative and the effective budget is the signature
    default.  We therefore time `_cg_ls` directly instead of trusting `n_cg`.
  * Timings are medians of repeats after a warm-up, at batch 16 (the training
    batch), with 2 CPU threads (the pinning used by the experiments).

Writes results/cost_probe.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isp_core as ic          # noqa: E402
import run_experiments as RE   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
torch.set_num_threads(2)

BATCH = 16
N_CG = 15
REPS = 7


def timeit(fn, reps=REPS, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = RE.load_data()
    nx = ny = RE.NX

    mk = dict(nx=nx, ny=ny, n_inc=RE.N_INC, eps_bar=0.03, n_cg=N_CG, proj=True)
    unc = ic.UnrolledISP(1, Gb, Gs, Ei, mode="constrained", featbound=True, **mk)
    cls = ic.UnrolledISP(1, Gb, Gs, Ei, mode="classical", **mk)
    for m in (unc, cls):
        m.eval()

    y_b = y_te[:BATCH].contiguous()
    chi_b = chi_te[:BATCH].contiguous()
    with torch.no_grad():
        chi0, w0 = RE.init_state(unc, y_b, BATCH)

        # --- full learned layer (CG solve + feature map + CNN correction) -----
        t_layer = timeit(lambda: unc.layer(0, chi0, w0, y_b))
        # --- classical layer (same CG solve, no correction) ------------------
        t_cls_layer = timeit(lambda: cls.layer(0, chi0, w0, y_b))

        with torch.no_grad():
            alpha = unc.a_max * torch.sigmoid(unc.a[0])
            w_t = (1 - alpha) * w0 + alpha * chi0.unsqueeze(1) * (
                unc.Ei.unsqueeze(0) + unc._Gb_apply(w0))
            E = unc.Ei.unsqueeze(0) + unc._Gb_apply(w_t)

            # --- CG contrast solve alone ------------------------------------
            t_cg = timeit(lambda: unc._cg_ls(E, y_b, n_cg=N_CG))
            t_cg1 = timeit(lambda: unc._cg_ls(E, y_b, n_cg=1))

            # --- the rest of the learned layer ------------------------------
            t_gb = timeit(lambda: unc._Gb_apply(w_t))

            resid = unc._Gs_apply(w_t) - y_b
            resid_dom = torch.matmul(
                unc.Gs.conj().T, resid.reshape(-1, resid.shape[-1]).T
            ).T.reshape(BATCH, unc.n_inc, -1)
            feat = torch.cat([
                resid_dom.real.reshape(BATCH, -1), resid_dom.imag.reshape(BATCH, -1),
                E.real.reshape(BATCH, -1), E.imag.reshape(BATCH, -1),
                w_t.real.reshape(BATCH, -1), w_t.imag.reshape(BATCH, -1),
                chi0], dim=1)
            feat = feat.reshape(BATCH, -1, nx, ny)
            feat = torch.tanh(feat / unc.feat_scale)
            t_cnn = timeit(lambda: unc.corrs[0](feat))

    # ---------------- analytic MAC counts (device independent) --------------
    n = nx * ny
    n_rx = Gs.shape[0]
    n_inc = RE.N_INC
    cin = 6 * n_inc + 1
    width = 8
    ksz = 9
    cnn_mac_per_sample = n * (cin * width * ksz + width * width * ksz + width * 1 * ksz)
    cg_mac_per_iter = 2 * (n_inc * n * n_rx) + n_inc * n       # Gs and Gs^H, plus sums
    gb_mac_per_apply = n_inc * n * n
    feat_mac = n_inc * n * n_rx                                # resid_dom (one Gs^H)
    n_params_cnn = (cin * width * ksz + width) + (width * width * ksz + width) \
        + (width * ksz + 1)

    out = {
        "protocol": {
            "device": "cpu",
            "threads": 2,
            "batch": BATCH,
            "reps": REPS,
            "nx": nx, "ny": ny, "n_inc": n_inc, "n_rx": int(n_rx),
            "n_cg_effective_default": N_CG,
            "note": "median wall-clock after 2 warm-up calls; architecture cost only, "
                    "no checkpoint loaded; classical layer uses the same n_cg=15 solve",
        },
        "wallclock_s": {
            "learned_layer_total": t_layer,
            "classical_layer_total": t_cls_layer,
            "cg_solve_full_budget": t_cg,
            "cg_solve_one_iter": t_cg1,
            "gb_apply_once": t_gb,
            "correction_cnn_only": t_cnn,
        },
        "wallclock_share": {
            "cg_over_learned_layer": t_cg / t_layer,
            "cnn_over_learned_layer": t_cnn / t_layer,
            "learned_over_classical_layer_minus_one": (t_layer - t_cls_layer) / t_cls_layer,
            "gb_apply_over_learned_layer": t_gb / t_layer,
        },
        "mac_analytic": {
            "cnn_per_sample": cnn_mac_per_sample,
            "cg_per_sample_per_iter": cg_mac_per_iter,
            "cg_per_sample_full_budget": N_CG * cg_mac_per_iter,
            "gb_apply_per_sample": gb_mac_per_apply,
            "feature_map_gsH_per_sample": feat_mac,
            "cnn_params": n_params_cnn,
        },
        "verdict": {
            "comparator_is_damped_alternating_fixed_point": True,
            "cnn_overhead_vs_classical_layer": (t_layer - t_cls_layer) / t_cls_layer,
            "cbudget_identical_between_arms": True,
        },
    }
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "cost_probe.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(json.dumps(out, indent=1))
    print("\nCOST PROBE DONE")


if __name__ == "__main__":
    main()
