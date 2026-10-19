# -*- coding: utf-8 -*-
"""run_seeds.py — seed-replication of the headline protocol (answers R3's
"single result cannot support a binary bounded/divergent conclusion").

Protocol, for each repeat r = 0,1,2:
  * training set  : regenerated with a seed of (1, 11, 12), 300 samples, 20 dB.  r=0 reuses
                    the main protocol's data seed 1, so its dataset is identical to the main
                    run; r=1,2 change the data as well;
  * test set      : FIXED (seed 2, 300 dB, 100 samples), so all repeats are scored on the
                    identical evaluation set and remain comparable to the main depth curve;
  * model init / batch order: torch.manual_seed(100 + r), which differs from the main
                    pipeline's 1234 -- so the r=0 numbers are NOT expected to equal
                    depth_curve.json to the last digit, and the residual gap between the two
                    is itself the init/batch-order component of the variability.

Variants (all K = 8, 40 epochs, identical optimiser/schedule/parameter count):
  * constrained      (budgeted:  eps_bar = 0.03, proj on,  feature-bounded trunk)
  * unconstrained    (no budget: proj on,  feature-bounded trunk)
  * unc_projoff_fboff(the arm that leaves the feasible box in Ablation I: proj off AND feat-bd
                    off.  Note it does NOT diverge: it stays numerically finite over the whole
                    horizon, as reported in Sec. VI-C(c))

Quantities per (variant, r): test RRE; 800-layer extrapolation RRE and max|chi|;
noise sensitivity at 20 dB; finiteness of the rollout.

Writes results/seeds.json incrementally (after every (variant, r) pair) and saves
results/ckpt_seed{r}_{variant}_K8.pt, so a recycled session never loses completed work.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import UnrolledISP, build_matrices, make_dirs, make_dataset
from run_experiments import (RES, DEV, NX, NY, N_INC, init_state, rre, train)

VARIANTS = {
    "constrained":       dict(mode="constrained",   proj=True,  featbound=True,  eps_bar=0.03),
    "unconstrained":     dict(mode="unconstrained", proj=True,  featbound=True,  eps_bar=0.03),
    "unc_projoff_fboff": dict(mode="unconstrained", proj=False, featbound=False, eps_bar=0.03),
}
SEEDS = [(0, 1), (1, 11), (2, 12)]
K = 8
EPOCHS = 40

# Single source of truth for the r=0 reproducibility note.  R3 m3.12/m3.15: an earlier version
# of this string claimed that r=0 "must reproduce depth_curve.json", which the same file's
# r0_constrained.rre_K8 = 0.40180 against depth_curve.json:constrained_K8 = 0.40673 falsifies --
# the training-data seed is shared but the initialisation/batch order is not.  The artifact
# results/seeds.json carries this exact string; code/_restamp_seeds_note.py re-stamps a stale
# artifact from this constant and code/_gate_deliverable.py asserts the two are equal.
SEED_NOTE = ("r=0 reuses the main protocol's TRAINING-DATA seed (1), so the dataset is "
             "identical to the main run; the initialisation seed deliberately differs "
             "(100+r here against 1234 in the main pipeline), and r=1,2 change the "
             "training-data seed as well. This file is a protocol-variability replicate, "
             "NOT a decomposition of variability into components: an identical-seed "
             "re-run of this same driver on the identical code also moves the numbers, "
             "by an amount that is not small compared with the across-seed spread "
             "reported here (see results/repro_floor.json, measured between two runs of "
             "the identical driver). "
             "The across-seed standard deviation must therefore not be read as an error "
             "bar on this pipeline; the stable content of the replication is the pair of "
             "qualitative verdicts -- every run numerically finite, and the admissible-box "
             "verdict identical in every repeat.")


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def build_mats():
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    return Gb, Gs, pts, dirs


def datasets(Gb, Gs, pts, dirs, seed_tr):
    kb = 2 * np.pi
    chi_tr, y_tr, Ei = make_dataset(300, Gb, Gs, pts, dirs, kb, NX, NY,
                                    seed=seed_tr, snr_db=20.0)
    chi_te, y_te, _ = make_dataset(100, Gb, Gs, pts, dirs, kb, NX, NY,
                                   seed=2, snr_db=300.0)
    return (torch.from_numpy(chi_tr).float(), torch.from_numpy(y_tr).to(torch.complex64),
            torch.from_numpy(chi_te).float(), torch.from_numpy(y_te).to(torch.complex64),
            Ei)


@torch.no_grad()
def eval_rre(model, chi_te, y_te, batch=50):
    model.eval()
    errs = []
    for i in range(0, chi_te.shape[0], batch):
        cb, yb = chi_te[i:i + batch].to(DEV), y_te[i:i + batch].to(DEV)
        c0, w0 = init_state(model, yb, cb.shape[0])
        ck, _ = model(c0, w0, yb)
        errs.append(rre(ck, cb))
    return float(np.mean(errs))


@torch.no_grad()
def eval_extrap(model, chi_te, y_te, K0=8, steps=100, n=50):
    """Cyclic depth extrapolation; returns final RRE, max|chi|, finiteness."""
    model.eval()
    cb, yb = chi_te[:n].to(DEV), y_te[:n].to(DEV)
    c0, w0 = init_state(model, yb, n)
    finite = True
    for k in range(steps):
        c0, w0 = model.layer(k % K0, c0, w0, yb)
        if not torch.isfinite(c0).all():
            finite = False
            break
    if not finite:
        return dict(rre=float("nan"), maxabs_chi=float("inf"), finite=False)
    return dict(rre=rre(c0, cb), maxabs_chi=float(c0.abs().max().item()), finite=True)


@torch.no_grad()
def eval_noise(model, chi_te, y_te, snr_db=20.0, n=50, delta=0.05):
    model.eval()
    cb, yb = chi_te[:n].to(DEV), y_te[:n].to(DEV)
    c0, w0 = init_state(model, yb, n)
    c_clean, _ = model(c0, w0, yb)
    rng = np.random.default_rng(7)
    scale = float(yb.abs().mean().item())
    dy = torch.from_numpy(
        (rng.standard_normal(yb.shape) + 1j * rng.standard_normal(yb.shape))
    ).to(torch.complex64).to(DEV) * (scale * delta)
    c_pert, _ = model(c0, w0, yb + dy)
    num = (c_pert - c_clean).norm(dim=1)
    den = c_clean.norm(dim=1) + 1e-12
    return float((num / den).mean().item())


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs = build_mats()
    gbn = float(np.linalg.norm(Gb, 2))
    out = {"config": {"K": K, "epochs": EPOCHS, "seeds": [s for _, s in SEEDS],
                      "variants": list(VARIANTS), "gb_norm2": gbn,
                      "beta0": 0.5 * gbn, "device": DEV,
                      "test_set": "fixed seed=2, 100 samples",
                      "note": SEED_NOTE},
           "runs": {}}

    for r, seed_tr in SEEDS:
        chi_tr, y_tr, chi_te, y_te, Ei = datasets(Gb, Gs, pts, dirs, seed_tr)
        chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)
        print("[seed run %d] training-data seed=%d" % (r, seed_tr), flush=True)
        for vname, cfg in VARIANTS.items():
            key = "r%d_%s" % (r, vname)
            torch.manual_seed(100 + r)
            np.random.seed(200 + r)
            m = UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, **cfg).to(DEV)
            train(m, chi_tr, y_tr, epochs=EPOCHS, log=lambda *a: None)
            rec = {"variant": vname, "config": {k: v for k, v in cfg.items()},
                   "seed": seed_tr, "repeat": r}
            rec["rre_K8"] = eval_rre(m, chi_te_d, y_te_d)
            rec["extrap"] = eval_extrap(m, chi_te_d, y_te_d, K0=K)
            rec["noise_sens_20dB"] = eval_noise(m, chi_te_d, y_te_d, 20.0)
            torch.save(m.state_dict(), os.path.join(RES, "ckpt_seed%d_%s_K8.pt" % (r, vname)))
            out["runs"][key] = rec
            save("seeds.json", out)
            print("  [%s] RRE=%.4f  extrap RRE=%.4f  max|chi|=%.3f  finite=%s  sens=%.4f"
                  % (vname, rec["rre_K8"], rec["extrap"]["rre"], rec["extrap"]["maxabs_chi"],
                     rec["extrap"]["finite"], rec["noise_sens_20dB"]), flush=True)
            del m

    # ---- aggregate
    def agg(path):
        vals = [r[path] for r in out["runs"].values() if r.get(path) is not None]
        vals = [v for v in vals if v == v]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}

    summ = {}
    for vname in VARIANTS:
        d = {}
        for q in ("rre_K8", "noise_sens_20dB"):
            vs = [out["runs"]["r%d_%s" % (r, vname)][q] for r, _ in SEEDS
                  if q in out["runs"]["r%d_%s" % (r, vname)]]
            vs = [v for v in vs if v == v]
            if vs:
                d[q] = {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "n": len(vs)}
        ex = [out["runs"]["r%d_%s" % (r, vname)]["extrap"] for r, _ in SEEDS]
        d["extrap_rre"] = {"mean": float(np.mean([e["rre"] for e in ex if e["rre"] == e["rre"]])),
                           "n": sum(1 for e in ex if e["rre"] == e["rre"])}
        d["extrap_finite_count"] = "%d/%d" % (sum(1 for e in ex if e["finite"]), len(ex))
        d["extrap_maxabs_chi_max"] = max(e["maxabs_chi"] for e in ex)
        d["beta0_measured"] = d["extrap_maxabs_chi_max"] * gbn
        summ[vname] = d
    out["summary"] = summ
    save("seeds.json", out)
    print("== summary ==")
    for k, v in summ.items():
        print(" ", k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                  for kk, vv in v.items() if kk != "config"}))
    print("[seeds done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
