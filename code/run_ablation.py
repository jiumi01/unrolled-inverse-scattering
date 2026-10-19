# -*- coding: utf-8 -*-
"""
run_ablation.py — single-variable ablations isolating which ingredient supplies BOUNDEDNESS
of the unrolled rollout, and which supplies the a-priori certificate.

Arms (all K=8, identical seed/schedule; only the listed flags differ):
  A1 unconstrained, proj=on,  featbound=on    (main unconstrained baseline)
  A2 unconstrained, proj=off, featbound=on    (does the projection matter?)
  A3 unconstrained, proj=off, featbound=off   (is feature bounding the mechanism?)
  A4 constrained,   proj=on,  featbound=on    (main constrained baseline)
  A5 constrained,   proj=off, featbound=on

Section B: inference-time projection toggle on the FROZEN proj=True checkpoints, which
separates the inference-time role of the projection from its training-time role.

Reported per arm: short-horizon RRE, measured per-layer/cycle gain, 800-layer cyclic
extrapolation (final RRE, max|chi|, max|w|, max|chi| over the horizon, finiteness).
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import build_matrices, make_dirs, make_dataset, UnrolledISP, exact_gain, exact_cycle_gain
from run_experiments import load_data, init_state, rre, train, eval_model, NX, NY, N_INC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K_TRAIN = 8
N_CYCLE = 100           # 8 x 100 = 800 unrolled layers

ARMS = [
    # --- seed 1: full 2x2x2 factorial (mode x proj x featbound) ---
    ("unc_pOn_fbOn_s1",  dict(mode="unconstrained", proj=True,  featbound=True),  1),
    ("unc_pOn_fbOff_s1", dict(mode="unconstrained", proj=True,  featbound=False), 1),
    ("unc_pOff_fbOn_s1", dict(mode="unconstrained", proj=False, featbound=True),  1),
    ("unc_pOff_fbOff_s1", dict(mode="unconstrained", proj=False, featbound=False), 1),
    ("cons_pOn_fbOn_s1", dict(mode="constrained",   proj=True,  featbound=True),  1),
    ("cons_pOn_fbOff_s1", dict(mode="constrained",  proj=True,  featbound=False), 1),
    ("cons_pOff_fbOn_s1", dict(mode="constrained",  proj=False, featbound=True),  1),
    ("cons_pOff_fbOff_s1", dict(mode="constrained", proj=False, featbound=False), 1),
    # --- seed replicates on the two decisive corners ---
    ("unc_pOn_fbOn_s2",  dict(mode="unconstrained", proj=True,  featbound=True),  2),
    ("unc_pOn_fbOn_s3",  dict(mode="unconstrained", proj=True,  featbound=True),  3),
    ("unc_pOff_fbOff_s2", dict(mode="unconstrained", proj=False, featbound=False), 2),
    ("unc_pOff_fbOff_s3", dict(mode="unconstrained", proj=False, featbound=False), 3),
]

# canonical tag used by the paper's tables/figures -> arm tag
CANON = {
    ("unconstrained", True, True): "unc_pOn_fbOn_s1",
    ("unconstrained", False, True): "unc_pOff_fbOn_s1",
    ("unconstrained", False, False): "unc_pOff_fbOff_s1",
    ("unconstrained", True, False): "unc_pOn_fbOff_s1",
    ("constrained", True, True): "cons_pOn_fbOn_s1",
    ("constrained", False, True): "cons_pOff_fbOn_s1",
    ("constrained", False, False): "cons_pOff_fbOff_s1",
    ("constrained", True, False): "cons_pOn_fbOff_s1",
}


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def rollout_cfg(model, chi_b, y_b, n_layers, K0):
    """Cyclic rollout with per-step diagnostics."""
    with torch.no_grad():
        chi0, w0 = init_state(model, y_b, chi_b.shape[0])
        maxc, maxw = [], []
        for k in range(n_layers):
            kk = (k % K0) if model.mode != "classical" else k
            chi0, w0 = model.layer(kk, chi0, w0, y_b)
            mc = chi0.abs().max().item(); mw = w0.abs().max().item()
            maxc.append(mc); maxw.append(mw)
            if not np.isfinite(mc) or not np.isfinite(mw) or mc > 1e6 or mw > 1e6:
                return dict(rre_extrap=None, maxabs_chi=mc, maxabs_w=mw,
                            maxabs_chi_over=max(maxc), finite_extrap=False, broke_at=k)
        e = rre(chi0, chi_b)
    return dict(rre_extrap=e, maxabs_chi=maxc[-1], maxabs_w=maxw[-1],
                maxabs_chi_over=max(maxc), finite_extrap=True, broke_at=None)


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)
    gbnorm = float(np.linalg.norm(Gb, 2))
    out = {}

    for tag, cfg, seed in ARMS:
        print(f"\n===== ablation {tag}  {cfg}  seed={seed} =====")
        torch.manual_seed(seed * 1000 + 1234); np.random.seed(seed * 1000 + 1234)
        m = UnrolledISP(K_TRAIN, Gb, Gs, Ei, NX, NY, N_INC, eps_bar=0.03, **cfg).to(DEV)
        rec = dict(cfg); rec["chi_max_box"] = m.chi_max; rec["seed"] = seed
        try:
            train(m, chi_tr, y_tr, epochs=40)
            torch.save(m.state_dict(), os.path.join(RES, f"ckpt_abl_{tag}.pt"))
            rec["rre_short"] = eval_model(m, chi_te_d, y_te_d)
            rec["finite_short"] = bool(np.isfinite(rec["rre_short"]))
        except Exception as ex:
            rec.update(rre_short=None, finite_short=False, train_error=repr(ex)[:200])
        print(f"  short-horizon RRE={rec.get('rre_short')}")

        try:
            m.eval()
            chi_b, y_b = chi_te_d[:16], y_te_d[:16]
            z = init_state(m, y_b, 16)
            c, w = z
            gs = []
            for k in range(K_TRAIN):
                gs.append(exact_gain(m, k, c, w, y_b))
                c, w = m.layer(k, c, w, y_b)
            rec["gain_layer"] = [float(x) for x in gs]
            rec["gain_cycle"] = float(exact_cycle_gain(m, *z, y_b))
        except Exception as ex:
            rec.update(gain_layer=None, gain_cycle=None, gain_error=repr(ex)[:200])

        try:
            rec.update(rollout_cfg(m, chi_te_d[:50], y_te_d[:50], K_TRAIN * N_CYCLE, K_TRAIN))
        except Exception as ex:
            rec.update(rre_extrap=None, finite_extrap=False, extrap_error=repr(ex)[:200])
        print(f"  extrap 800: finite={rec.get('finite_extrap')} rre={rec.get('rre_extrap')} "
              f"max|chi|={rec.get('maxabs_chi')} (over horizon {rec.get('maxabs_chi_over')}) "
              f"gain_cycle={rec.get('gain_cycle')}")
        out[tag] = rec
        del m
        save("ablation.json", out)

    # ---- Section B: inference-time projection toggle on frozen proj=on weights ----
    print("\n===== Section B: inference-time toggle on frozen weights =====")
    for mode in ["unconstrained", "constrained"]:
        ck = os.path.join(RES, f"ckpt_{mode}_K8.pt")
        if not os.path.exists(ck):
            print(f"  skip {mode} (no checkpoint)")
            continue
        for proj in [True, False]:
            tag = f"{mode}_frozen_proj{int(proj)}"
            m = UnrolledISP(K_TRAIN, Gb, Gs, Ei, NX, NY, N_INC, mode=mode,
                            eps_bar=0.03, proj=proj).to(DEV)
            m.load_state_dict(torch.load(ck, map_location=DEV)); m.eval()
            rec = dict(mode=mode, proj=bool(proj), frozen=True, source_ckpt=f"ckpt_{mode}_K8.pt")
            rec["rre_short"] = eval_model(m, chi_te_d, y_te_d)
            rec["finite_short"] = bool(np.isfinite(rec["rre_short"]))
            rec.update(rollout_cfg(m, chi_te_d[:50], y_te_d[:50], K_TRAIN * N_CYCLE, K_TRAIN))
            print(f"  {tag}: short RRE={rec['rre_short']:.4f} finite={rec['finite_extrap']} "
                  f"rre_extrap={rec['rre_extrap']} max|chi|={rec.get('maxabs_chi')}")
            out[tag] = rec
            del m
            save("ablation.json", out)

    print(f"\n[ablation done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
