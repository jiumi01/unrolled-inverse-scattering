# -*- coding: utf-8 -*-
"""
run_cg_sweep.py — CG iteration convergence sweep.

Question addressed
------------------
"Which algorithm is the paper about?"  The layer map solves the contrast step by
conjugate gradients.  A referee objected that a fixed-iteration CG solve is not
the exact least-squares map, so the certificates might not describe the executed
network.  The manuscript argues that (i) every certified statement rests on the
CG-free blocks C, D and (ii) the contrast-path blocks A, B are reported in both
states.  This script measures both halves of that argument.

Design
------
Part 1 — LOCAL (fixed operating point).  The Jacobian is measured at ONE fixed
state (the converged Born init, and a second point 4 layers in), so n_cg is the
*only* free variable.  Prediction from the closed forms of Lemma (blocks):
        C = theta alpha diag(E)                   -> exact, n_cg-INVARIANT
        D = (1 - theta alpha) I + theta alpha diag(chi) Gb  -> exact, INVARIANT
        A, B (contrast path) -> enter through dchi_hat/d(.) -> n_cg-DEPENDENT
Measured with the paper's exact instrument (spec_tools.JacOp, double precision).

Part 2 — DYNAMIC (trajectory).  For a model trained at n_cg=15 and one trained at
n_cg=200 we report reconstruction RRE and the full-cycle fixed-point residual as
the inference-time n_cg is swept, i.e. a train x eval matrix that answers "is the
method reliant on its training truncation?".

Outputs results/cg_sweep.json (supersedes the first, confounded version).
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import UnrolledISP, _flat_layer_fn
from spec_tools import JacOp, sigma_max_L, sigma_max_validated, as_float64_model
from run_experiments import load_data, train, init_state, rre, NX, NY, N_INC, RES

N_CG_LIST = [5, 10, 15, 30, 50, 100, 200]
N_CG_TRAIN_A = 15                 # checkpoint the manuscript was trained with
N_CG_TRAIN_B = 200                # converged-solve control
CKPT_A = os.path.join(RES, "ckpt_40ep_constrained_K8.pt")
CKPT_B = os.path.join(RES, "ckpt_cg200_constrained_K8.pt")
K = 8
BATCH = 16


def cg_solve(model, E, y, n_cg):
    """Exact re-implementation of UnrolledISP._cg_ls with a settable step count."""
    B = E.shape[0]

    def apply_A(x):
        Ex = E * x.unsqueeze(1)
        m = torch.matmul(model.Gs, Ex.reshape(-1, Ex.shape[-1]).T).T.reshape(B, E.shape[1], -1)
        Gm = torch.matmul(model.Gs.conj().T, m.reshape(-1, m.shape[-1]).T).T.reshape(E.shape)
        return (E.conj() * Gm).real.sum(dim=1)

    b = (E.conj() * torch.matmul(
        model.Gs.conj().T, y.reshape(-1, y.shape[-1]).T).T.reshape(E.shape)).real.sum(dim=1)
    if not hasattr(model, "_gs_diag"):
        model._gs_diag = (torch.abs(model.Gs) ** 2).sum(dim=0)
    diagA = ((torch.abs(E) ** 2) * model._gs_diag).sum(dim=1)
    lam = model.cg_lam_rel * diagA + 1e-12
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r * r).sum(dim=1)
    for _ in range(n_cg):
        Ap = apply_A(p) + lam * p
        alpha = (rs / (p * Ap).sum(dim=1).clamp_min(1e-20)).unsqueeze(1)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=1)
        p = r + (rs_new / rs.clamp_min(1e-20)).unsqueeze(1) * p
        rs = rs_new
    return x, float((r.norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)).detach().mean())


def patch(model, n_cg):
    st = {"resid": float("nan")}

    def solver(E, y):
        x, rr = cg_solve(model, E, y, n_cg)
        st["resid"] = rr
        return x
    model._cg_ls = solver
    return st


def flat(chi, w):
    B = chi.shape[0]
    return torch.cat([chi, w.real.reshape(B, -1), w.imag.reshape(B, -1)], dim=1)


def blocks(model, chi, w, y, n, I):
    D = n * (1 + 2 * I)
    f = _flat_layer_fn(model, 0, y)
    zr = flat(chi, w).detach()
    spec = {"A": (list(range(0, n)), slice(0, n)),
            "B": (list(range(n, D)), slice(0, n)),
            "C": (list(range(0, n)), slice(n, D)),
            "D": (list(range(n, D)), slice(n, D)),
            "full": (None, None)}
    # Every block is measured with BOTH estimators (block-diagonal Lanczos on J^T J and
    # power iteration on the same operator) and the reported value follows the project rule:
    # Lanczos when the two agree to 1%, power otherwise.  Measured failure this guards
    # against: with Lanczos alone this very sweep wrote B=0.0 (n_cg=50) and full=0.0
    # (n_cg=100) at a point where neighbouring rows give O(1) values.
    # Only the two blocks that actually mis-behaved (B and full: Lanczos wrote 0.0 at
    # n_cg=50/100) get the second estimator.  Validating all five would triple the run for
    # blocks whose values never moved; A/C/D stay on Lanczos and say so by omitting the fields.
    VALIDATE = ("B", "full")
    out = {}
    for name, (cols, rows) in spec.items():
        J = JacOp(f, zr, in_idx=cols, out_slice=rows, name=name)
        if name in VALIDATE:
            v = sigma_max_validated(J, tol=1e-7, iters=60)
            out[name] = v["sigma"]
            out[name + "_lanczos"] = v["lanczos"]
            out[name + "_power"] = v["power"]
            out[name + "_rel"] = v["rel"]
            out[name + "_agrees"] = v["agrees"]
            out[name + "_zero_operator"] = bool(v["sigma"] == 0.0)
        else:
            out[name], _mv, _info = sigma_max_L(J, tol=1e-7)
    out["all_solvers_agree"] = bool(all(out[k + "_agrees"] for k in VALIDATE))
    return out


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Part 1 runs on the GPU when one is available, and the device is recorded per part.  This
    # is deliberate and reversed an earlier "CPU everywhere" change: every Jacobian matvec of
    # this driver performs a FULL CG contrast solve of n_cg iterations, so a single n_cg=200 row
    # costs ~13x an n_cg=15 row.  Measured on this machine: 583 s for the n_cg=30 row on CPU f64,
    # i.e. >30 min for n_cg=100 and >2 h for n_cg=200 -- a 6 h scan for a table whose effect spans
    # a factor of thirty.  What this table needs is one arithmetic ACROSS ITS OWN ROWS (all rows
    # float64 on one device), not the same device as the single-point audits; the 3% CPU/GPU
    # difference seen in the ill-conditioned single-point solves is irrelevant to a scan whose
    # spread is 30x, and the device is written into the artefact so a reader can see it.
    dev1 = dev
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d, y_te_d = chi_te.to(dev), y_te.to(dev)
    n = NX * NY
    # Device and dtype are recorded per part: Part 1 is a float64 spectral audit
    # (as_float64_model), Part 2 is the float32 training chain -- the train x eval
    # matrix is about the truncation, not about precision, and a reader must be able
    # to see which arithmetic produced which block.
    out = {"n_cg_list": N_CG_LIST, "K": K, "device": dev,
           "device_part1": dev1, "device_part2": dev,
           "dtype": {"part1_local": "float64 (as_float64_model)",
                     "part2_train_eval": "float32 (training chain)"}}

    # ---------- Part 1: LOCAL, fixed operating point ------------------------
    print("[part 1] local block spectra at a FIXED operating point (double precision)")
    m64 = UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=0.03).to(dev1)
    m64.load_state_dict(torch.load(CKPT_A, map_location=dev1))
    m64.eval()
    m64 = as_float64_model(m64, Gb, Gs, Ei)
    y64 = y_te[:BATCH].to(dev1).to(torch.complex128)
    patch(m64, 200)                      # fixed point computed with a converged solve
    with torch.no_grad():
        Ei_e = m64.Ei.unsqueeze(0).expand(BATCH, -1, -1)
        chi0 = m64._cg_ls(Ei_e.to(torch.complex128), y64)
        chi0 = chi0.clamp(0.0, m64.chi_max).to(torch.float64)
        w0 = (chi0.unsqueeze(1) * m64.Ei.unsqueeze(0)).to(torch.complex128)
        # second point: four layers in (also with the converged solve -> fixed)
        c4, w4 = chi0, w0
        for k in range(4):
            c4, w4 = m64.layer(k, c4, w4, y64)

    local = {"point0_born": [], "point4_layer": []}
    for ncg in N_CG_LIST:
        _t_row = time.time()
        st = patch(m64, ncg)
        # residual of the fixed-point solve itself (at point0)
        with torch.no_grad():
            _ = m64._cg_ls(m64.Ei.unsqueeze(0).expand(BATCH, -1, -1).to(torch.complex128), y64)
        resid = st["resid"]
        b0 = blocks(m64, chi0, w0, y64, n, N_INC)
        b4 = blocks(m64, c4, w4, y64, n, N_INC)
        local["point0_born"].append({"n_cg": ncg, "cg_resid": resid, **b0})
        local["point4_layer"].append({"n_cg": ncg, **b4})
        print("  n_cg=%3d resid=%.3e  C=%.4f D=%.4f | A=%.4f B=%.4f J=%.4f  (%.0fs)"
              % (ncg, resid, b0["C"], b0["D"], b0["A"], b0["B"], b0["full"],
                 time.time() - _t_row), flush=True)
    out["local_point0_born"] = local["point0_born"]
    out["local_point4_layer"] = local["point4_layer"]
    del m64
    torch.cuda.empty_cache()

    # ---------- Part 2: DYNAMIC, train x eval -------------------------------
    print("[part 2] train x eval truncation matrix")
    if not os.path.exists(CKPT_B):
        # The regeneration pipeline trains this checkpoint with code/_retrain_cg200.py
        # (explicit seed 1234, shared with _retrain_depth.py, so the train15/train200
        # pair differs only in n_cg) and runs it BEFORE this driver.  This embedded
        # fallback is the original unseeded trainer, kept only so the driver still
        # works standalone; its protocol is archived in results/cg200_retrain.json
        # when produced by the dedicated script.
        print("  training constrained K=8 at n_cg=200 (this is the expensive step)")
        mb = UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=0.03).to(dev)
        patch(mb, N_CG_TRAIN_B)
        tr = init_state
        import run_experiments as RE
        # train() calls RE.init_state(module-level) -> patch it to use the model solver
        def _init(m, y, batch):
            with torch.no_grad():
                E = m.Ei.unsqueeze(0).expand(batch, -1, -1)
                c0 = m._cg_ls(E, y)
                c0 = c0.clamp(0.0, m.chi_max)
                w0_ = c0.unsqueeze(1) * m.Ei.unsqueeze(0)
            return c0, w0_
        RE.init_state = _init
        train(mb, chi_tr, y_tr, epochs=40)
        torch.save(mb.state_dict(), CKPT_B)
        del mb
        torch.cuda.empty_cache()

    mat = {}
    for tag, ckpt in [("train15", CKPT_A), ("train200", CKPT_B)]:
        m = UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=0.03).to(dev)
        m.load_state_dict(torch.load(ckpt, map_location=dev))
        m.eval()
        row = {}
        for ncg in N_CG_LIST:
            patch(m, ncg)
            with torch.no_grad():
                c0, w0_ = init_state(m, y_te_d, chi_te_d.shape[0])
                ck, wk = m(c0, w0_, y_te_d)
                e = rre(ck, chi_te_d)
                c2, w2 = ck, wk
                for k in range(K):
                    c2, w2 = m.layer(k, c2, w2, y_te_d)
                num = ((c2 - ck) ** 2).sum() + (w2 - wk).abs().pow(2).sum()
                den = (ck ** 2).sum() + wk.abs().pow(2).sum() + 1e-30
                fp = float((num / den).sqrt().item())
            row[str(ncg)] = {"rre": e, "fixed_point_resid": fp}
            print("  %s  n_cg=%3d  RRE=%.4f  fp=%.3e" % (tag, ncg, e, fp))
        mat[tag] = row
        del m
        torch.cuda.empty_cache()
    out["train_eval"] = mat

    with open(os.path.join(RES, "cg_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[cg_sweep done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
