# -*- coding: utf-8 -*-
"""_retrain_cg200.py -- regenerate the converged-solve training control checkpoint.

Why this file exists.  run_cg_sweep.py Part 2 needs a model TRAINED at n_cg=200
(results/ckpt_cg200_constrained_K8.pt) to complete its train x eval truncation
matrix: a model trained at the implemented n_cg=15 (ckpt_40ep_constrained_K8.pt,
produced by _retrain_depth.py) against one trained at a converged solve, both
evaluated across the inference-time n_cg grid.  The archived checkpoint was
produced before the isp_core.py correction of the contrast solve (forward_exact's
least-squares column scaling), so it cannot be reused: the layer it trained is
not the layer the manuscript certifies.

Protocol (identical to the embedded trainer of run_cg_sweep.py Part 2, with the
seed made explicit):
  * K=8, constrained, eps_bar=0.03, projection on -- the paper's configuration;
  * the model's own _cg_ls is patched to a settable-step-count CG solve
    (run_cg_sweep.cg_solve/patch) at n_cg=200 for BOTH the Born-LS initial state
    and every layer of the forward pass, i.e. training happens at the converged
    solve, not only inference;
  * run_experiments.train for 40 epochs (Adam 1e-3, bs=16, grad-norm clip 1e3),
    the same uniform 40-epoch schedule as _retrain_depth.py;
  * torch.manual_seed(1234) / np.random.seed(1234) immediately before the model
    construction -- the SAME arm seed as _retrain_depth.py.  train() consumes the
    global RNG only through torch.randperm per epoch, so the two arms (n_cg=15
    and n_cg=200) see identical initialisation and identical batch order: the
    only thing that differs across the pair is the contrast-solve step count,
    which is exactly the single variable the train x eval matrix is about.

float32 on GPU, like every training driver in this project (the audit drivers
are the float64 ones; see run_gain_decomp.py).

Writes results/ckpt_cg200_constrained_K8.pt and results/cg200_retrain.json.
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE
from isp_core import UnrolledISP
from run_cg_sweep import patch, CKPT_B

HERE = os.path.dirname(os.path.abspath(__file__))
RES = RE.RES
DEV = RE.DEV                      # training chain: GPU when available
K = 8
N_CG_TRAIN = 200
EPOCHS = 40
ARM_SEED = 1234                   # identical to _retrain_depth.py's per-arm seed


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = RE.load_data()
    chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)

    # Explicit per-arm seed: the train15/train200 pair must differ ONLY in n_cg.
    torch.manual_seed(ARM_SEED)
    np.random.seed(ARM_SEED)
    mb = UnrolledISP(K, Gb, Gs, Ei, RE.NX, RE.NY, RE.N_INC,
                     mode="constrained", eps_bar=0.03).to(DEV)
    st = patch(mb, N_CG_TRAIN)    # converged solve for init AND forward, while training

    # train() resolves init_state as a module global, so patching RE.init_state to
    # the model's own (patched) solver makes the Born-LS state use n_cg=200 too.
    import run_experiments as _RE

    ORIG_INIT = _RE.init_state

    def _init(m, y, batch):
        with torch.no_grad():
            E = m.Ei.unsqueeze(0).expand(batch, -1, -1)
            c0 = m._cg_ls(E, y)
            c0 = c0.clamp(0.0, m.chi_max)
            w0_ = c0.unsqueeze(1) * m.Ei.unsqueeze(0)
        return c0, w0_

    _RE.init_state = _init
    print("  training constrained K=8 at n_cg=%d, %d epochs, seed=%d"
          % (N_CG_TRAIN, EPOCHS, ARM_SEED), flush=True)
    RE.train(mb, chi_tr, y_tr, epochs=EPOCHS, log=lambda *a: print("   ", *a, flush=True))
    torch.save(mb.state_dict(), CKPT_B)

    # --- sanity record: evaluate the new checkpoint at both ends of the grid ----
    ev = {}
    for ncg in (15, 200):
        patch(mb, ncg)
        ev[str(ncg)] = RE.eval_model(mb, chi_te_d, y_te_d)
        print("  eval at n_cg=%3d: RRE=%.5f" % (ncg, ev[str(ncg)]), flush=True)
    _RE.init_state = ORIG_INIT      # restore (defensive; process exits anyway)

    out = {
        "checkpoint": os.path.relpath(CKPT_B, os.path.normpath(os.path.join(HERE, ".."))),
        "K": K, "mode": "constrained", "eps_bar": 0.03,
        "n_cg_train": N_CG_TRAIN, "epochs": EPOCHS, "arm_seed": ARM_SEED,
        "device": DEV, "dtype": "float32 (training chain)",
        "train_cg_resid_last": st["resid"],
        "eval_rre": ev,
        "protocol": ("identical to run_cg_sweep.py Part 2 embedded trainer; explicit "
                     "seed shared with _retrain_depth.py so the train15/train200 pair "
                     "differs only in the contrast-solve step count"),
        "provenance_md5": {
            "isp_core.py": _md5(os.path.join(HERE, "isp_core.py")),
            "run_experiments.py": _md5(os.path.join(HERE, "run_experiments.py")),
            "run_cg_sweep.py": _md5(os.path.join(HERE, "run_cg_sweep.py")),
        },
        "wallclock_s": time.time() - t0,
    }
    with open(os.path.join(RES, "cg200_retrain.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[retrain cg200 done] %.0fs" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
