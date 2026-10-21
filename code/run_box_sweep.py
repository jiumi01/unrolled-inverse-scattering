# -*- coding: utf-8 -*-
"""run_box_sweep.py -- E-class experiment: dose-response of the certificates in chi_max.

Reviewer item E of the pre-submission report asked for the certificate quantities to be
recomputed at *different contrast upper bounds*, so that the reader can see which
conclusions are structural and which are properties of the operating point we chose.

Three boxes are added on top of the archived chi_max = 0.5 run:
    chi_max in {0.6, 0.8, 1.0}
For each box, both arms (constrained / unconstrained, K = 8, 40 epochs) are retrained on
data drawn with *that* box, so the training distribution and the projection set move
together, which is what a practitioner changing the contrast prior would do.

Per (box, arm) we record: test RRE, the 800-layer rollout's max|chi| over the horizon and
its finiteness.  The theory side is closed-form and is NOT refitted:
    beta0 = chi_max ||Gb||_2                 (weak-scattering gate)
    alpha_c = (theta * E)^-1                 (independent of chi_max)
so the tables say exactly where the contraction certificate switches off.

NOTE.  make_dataset() caches on (n, Gb, Gs, Ei, nx, ny, seed, snr) -- chi_max is NOT in
that key, so reusing it here would silently hand back the 0.25-0.5 data.  The data are
therefore generated locally with random_contrast(chi_max=box) and cached under a filename
that carries the box.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import (build_matrices, make_dirs, random_contrast, forward_exact,
                      add_noise, UnrolledISP, incident_fields)
from run_experiments import init_state, rre, train, eval_model, NX, NY, N_INC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(RES, "boxsw")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

BOXES = [0.6, 0.8, 1.0]
K_TRAIN = 8
EPOCHS = 40
N_CYCLE = 100                 # 8 x 100 = 800 unrolled layers
ARMS = ["constrained", "unconstrained"]


def pairs(box, n, seed, snr_db, Gb, Gs, Ei):
    """(chi, y) with chi = random_contrast(..., chi_max=box); cached per box."""
    tag = "n%d_s%d_snr%g" % (n, seed, snr_db)
    path = os.path.join(CACHE, "box%.1f_%s.npz" % (box, tag))
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        rng = np.random.default_rng(seed)
        chis, ys = [], []
        for _ in range(n):
            chi = random_contrast(NX, NY, rng, chi_max=box)
            _, _, y = forward_exact(Gb, Gs, Ei, chi)
            ys.append(add_noise(y, snr_db, rng))
            chis.append(chi)
        chis = np.stack(chis).astype(np.float32)
        ys = np.stack(ys).astype(np.complex64)
        np.savez_compressed(path, chis=chis, ys=ys)
    z = np.load(path)
    return z["chis"], z["ys"]


def rollout(model, chi_b, y_b, n_layers, K0):
    """Same cyclic rollout as run_ablation.rollout_cfg (kept identical on purpose)."""
    with torch.no_grad():
        chi0, w0 = init_state(model, y_b, chi_b.shape[0])
        maxc = []
        for k in range(n_layers):
            chi0, w0 = model.layer(k % K0, chi0, w0, y_b)
            mc = chi0.abs().max().item()
            maxc.append(mc)
            if not np.isfinite(mc) or mc > 1e6:
                return dict(rre_extrap=None, maxabs_chi=mc, maxabs_chi_over=max(maxc),
                            finite_extrap=False, broke_at=k)
        return dict(rre_extrap=rre(chi0, chi_b), maxabs_chi=maxc[-1],
                    maxabs_chi_over=max(maxc), finite_extrap=True, broke_at=None)


def main():
    t0 = time.time()
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    kb = 2 * np.pi
    # incident fields must be the same object the archived dataset uses
    Ei = incident_fields(pts, dirs, kb)
    gbnorm = float(np.linalg.norm(Gb, 2))

    out = {"gb_norm": gbnorm, "boxes": BOXES, "epochs": EPOCHS, "K": K_TRAIN,
           "runs": {}, "control": {}}
    # archived chi_max = 0.5 operating point, for continuity of the table
    out["control"]["beta0_at_0p5"] = 0.5 * gbnorm
    out["control"]["beta0_at_1"] = gbnorm
    out["control"]["box_edge_beta0_one"] = 1.0 / gbnorm      # chi_max at which beta0 = 1

    for box in BOXES:
        chi_tr, y_tr = pairs(box, 300, 1, 20.0, Gb, Gs, Ei)
        chi_te, y_te = pairs(box, 100, 2, 300.0, Gb, Gs, Ei)
        chi_tr = torch.from_numpy(chi_tr).float()
        y_tr = torch.from_numpy(y_tr).to(torch.complex64)
        chi_te_d = torch.from_numpy(chi_te).float().to(DEV)
        y_te_d = torch.from_numpy(y_te).to(torch.complex64).to(DEV)

        beta0 = box * gbnorm
        out["runs"]["box%.1f" % box] = {"chi_max": box, "beta0": beta0,
                                        "beta0_lt_one": bool(beta0 < 1.0),
                                        "chi_data_max": float(chi_tr.max()),
                                        "maxabs_chi_gt": float(chi_te_d.abs().max())}
        for mode in ARMS:
            tag = "box%.1f_%s" % (box, mode)
            print("\n===== %s =====" % tag, flush=True)
            torch.manual_seed(1234); np.random.seed(1234)
            m = UnrolledISP(K_TRAIN, Gb, Gs, Ei, NX, NY, N_INC, mode=mode,
                            eps_bar=0.03, chi_max=box).to(DEV)
            rec = {"chi_max": box, "mode": mode, "beta0": beta0,
                   "beta0_lt_one": bool(beta0 < 1.0), "chi_max_box": float(m.chi_max)}
            train(m, chi_tr, y_tr, epochs=EPOCHS, log=lambda *a: None)
            rec["rre"] = eval_model(m, chi_te_d, y_te_d)
            rec["finite"] = bool(np.isfinite(rec["rre"]))
            m.eval()
            rec.update(rollout(m, chi_te_d[:50], y_te_d[:50], K_TRAIN * N_CYCLE, K_TRAIN))
            print("  RRE=%.4f  800-layer max|chi|=%.3f (over horizon %.3f) finite=%s"
                  % (rec["rre"], rec["maxabs_chi"], rec["maxabs_chi_over"],
                     rec["finite_extrap"]), flush=True)
            out["runs"][tag] = rec
            with open(os.path.join(RES, "box_sweep.json"), "w") as f:
                json.dump(out, f, indent=1)
            del m
            torch.cuda.empty_cache()

    print("[box sweep done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
