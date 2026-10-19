# -*- coding: utf-8 -*-
"""Compare the terminal training loss of the deployed n_cg=15 network and the
n_cg=200 control, using train()'s exact loss definition and the full training set.
Both are evaluated at their own training truncation, eval mode (no BN/dropout in the
architecture, so eval == train loss)."""
import os, sys, json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import UnrolledISP
from run_experiments import load_data, init_state, NX, NY, N_INC, RES
import run_cg_sweep as CS

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def loss_of(ckpt, n_cg):
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    m = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=0.03).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV))
    m.eval()
    CS.patch(m, n_cg)                       # evaluate at its OWN training truncation
    chi_tr_d, y_tr_d = chi_tr.to(DEV), y_tr.to(DEV)
    n, bs, lam = chi_tr.shape[0], 16, 0.5
    tot = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            c0, w0 = init_state(m, y_tr_d[i:i + bs], min(bs, n - i))
            ck, wk = m(c0, w0, y_tr_d[i:i + bs])
            Gsw = torch.matmul(m.Gs, wk.reshape(-1, wk.shape[-1]).T).T.reshape(y_tr_d[i:i + bs].shape)
            tot += (((ck - chi_tr_d[i:i + bs]) ** 2).mean()
                    + lam * (Gsw - y_tr_d[i:i + bs]).abs().pow(2).mean()).item()
    return tot / max(1, (n // bs))


out = {}
for tag, ck, ncg in [("train15", CS.CKPT_A, 15), ("train200", CS.CKPT_B, 200)]:
    v = loss_of(ck, ncg)
    out[tag] = v
    print("%-9s n_cg=%3d  terminal train loss = %.5f" % (tag, ncg, v), flush=True)

out["ratio"] = out["train200"] / out["train15"]
out["loss_note"] = ("train()'s exact objective, chi MSE + 0.5 * data-consistency MSE, "
                    "full 300-sample training set, eval mode, both models at their own "
                    "training truncation; produced by code/_cgloss_probe.py")
with open(os.path.join(RES, "cg_loss.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote results/cg_loss.json", flush=True)
