# -*- coding: utf-8 -*-
"""_chi_lip_audit.py — round-1 review R2-M2.6 / Q2.4 / Q2.5.

Three questions the review asked, all about the *object* being measured:

(1) Cor. fidelity states its hypothesis on the PHYSICAL map F (the layer with the learned
    correction switched off, H = 0).  L45.json measured the chi -> chi block of the map WITH
    the correction (G_hat).  This script measures BOTH, so the attribution "the hypothesis is
    falsified by our own measurement" can name the right object.

(2) The chi-map Lipschitz constant of the full state map is the norm of the whole row block
    [A B] (input = all of (chi, Re w, Im w), output = chi); measuring only the chi -> chi block
    A UNDERSTATES it.  Both are reported.

(3) M_J = sup_m ||J^m|| was truncated at m in {1,2} for a K = 8 composite.  For a non-normal
    operator the transient peak need not be at m = 2; here ms = 1..8.

Protocol: identical to _run_L45.py -- float64, K = 8 cyclic composite, batch 16 test states,
learned step sizes (learned=True) AND the fixed reference alpha = gamma = 1/2, as-implemented
contrast solve (n_cg = 15), the same immutable checkpoints.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "chi_lip.json")
DEV = "cpu"
DT = torch.float64
K8 = 8
TOL = 1e-4
MS = (1, 2, 4, 8)   # K = 8 costs 2L J-products per matvec; 4 probes up to the full horizon

from isp_core import UnrolledISP
from spec_tools import (JacOp, as_float64_model, sigma_max_L, sigma_max_sq,
                        spectral_radius, sigma_max_validated, rho_power)
from run_bjac import make_f, flat_of, THETA, BATCH
from run_experiments import load_data, init_state, NX, NY, N_INC

VARIANTS = (("constrained", "ckpt_constrained_K8.pt", True),
            ("unconstrained", "ckpt_unconstrained_K8.pt", False))
MAPS = (("G_hat", True), ("F", False))


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(RES, "chi_lip_progress.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def row_and_block(mk, y_b, zr, n, I, use_corr, which_alpha):
    """Return the chi-map Lipschitz constant (norm of the whole [A B] row block) and the
    chi -> chi block quantities, for one map and one step-size protocol."""
    if which_alpha == "learned":
        f = make_f(mk, 0.5, 0.5, THETA, y_b, use_corr, K=K8, learned=True, which="full")
    else:
        f = make_f(mk, 0.5, 0.5, THETA, y_b, use_corr, K=K8, learned=False, which="full")
    in_all = torch.arange(0, n * (1 + 2 * I))
    in_chi = torch.arange(0, n)
    out_chi = torch.arange(0, n)
    J_row = JacOp(f, zr, in_idx=in_all, out_slice=out_chi, name="chi_row")
    J_cc = JacOp(f, zr, in_idx=in_chi, out_slice=out_chi, name="chi_cc")
    # Cost/benefit note (2026-10-19).  sigma_chi_to_chi has an INDEPENDENT external cross-check
    # already: L45.json reports sigma_max_chi_block = 0.467 for the same composite from a different
    # driver, and chi_lip returns 0.4674.  A second in-process estimator here would double a 40-min
    # run for no new information, so the sigma values stay on Lanczos.  The spectral radius has NO
    # external check and it is the quantity whose sign of conclusion moved, so it alone is measured
    # twice: IRAM (eigs, which can lock onto a non-dominant eigenvalue of a non-normal operator
    # without saying so) and 60 steps of power iteration on J itself.  Reported value = IRAM when
    # the two agree to 2%, otherwise the power value, with the flag recorded either way.
    lip_row, mv_row, _ = sigma_max_L(J_row, tol=TOL)
    sig_cc, mv_cc, _ = sigma_max_L(J_cc, tol=TOL)
    rho_cc, mv_rho, _ = spectral_radius(J_cc, tol=TOL)
    rho_pw, _ = rho_power(J_cc, iters=60)
    rho_cc = float(rho_cc) if np.isfinite(rho_cc) else 0.0
    rho_pw = float(rho_pw) if np.isfinite(rho_pw) else 0.0
    rho_hi = max(rho_cc, rho_pw)
    rho_rel = (abs(rho_cc - rho_pw) / rho_hi) if rho_hi > 0 else 0.0
    rho_ok = bool(rho_rel <= 2e-2)
    return dict(lip_chi_row=float(lip_row), sigma_chi_to_chi=float(sig_cc),
                rho_chi_to_chi=(rho_cc if rho_ok else rho_pw),
                rho_lanczos=rho_cc, rho_power=rho_pw, rho_rel=float(rho_rel),
                rho_agrees=rho_ok,
                row_matvecs=int(mv_row), cc_matvecs=int(mv_cc),
                rho_matvecs=int(mv_rho))


ONLY = {a for a in sys.argv[1:] if not a.startswith("-")}
DO_TRANS = "notrans" not in sys.argv[1:]


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    n = NX * NY
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}

    m1 = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
        Gb, Gs, Ei)
    y_b = y_te[:BATCH].to(DEV).to(torch.complex128)
    with torch.no_grad():
        m1._cg_ls(m1.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
        chi0, w0 = init_state(m1, y_b, BATCH)
    chi0 = chi0.to(DT)
    assert w0.is_complex()

    arc = os.path.join(RES, "ckpt_archive_r1")
    for tag, ck, proj in VARIANTS:
        if ONLY and tag not in ONLY:
            continue
        path = os.path.join(arc, ck)
        if not os.path.exists(path):
            path = os.path.join(RES, ck)
        if not os.path.exists(path):
            log("  [skip] %s" % path)
            continue
        mk = as_float64_model(
            UnrolledISP(K8, Gb, Gs, Ei, NX, NY, N_INC,
                        mode=("constrained" if tag == "constrained" else "unconstrained"),
                        proj=proj).to(DEV), Gb, Gs, Ei)
        mk.load_state_dict(torch.load(path, map_location=DEV))
        mk.eval()
        with torch.no_grad():
            mk._cg_ls(mk.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
        zr = flat_of(chi0, w0)

        rec = {"K": K8, "proj": bool(proj), "dtype": "float64", "batch": BATCH,
               "tol": TOL, "solver": "eigsh/IRAM on J^T J (sigma), eigs (rho)",
               "n_cg": int(getattr(mk, "n_cg", 15))}
        for mname, use_corr in MAPS:
            for proto in ("learned", "fixed_ag_0.5"):
                key = "%s_%s" % (mname, proto)
                t1 = time.time()
                rec[key] = row_and_block(mk, y_b, zr, n, N_INC, use_corr, proto)
                log("  %-12s %-14s lip_chi_row=%.4f  sigma_cc=%.4f  rho_cc=%.4f  (%.0fs)"
                    % (tag, key, rec[key]["lip_chi_row"], rec[key]["sigma_chi_to_chi"],
                       rec[key]["rho_chi_to_chi"], time.time() - t1))
                # INCREMENTAL WRITE.  The transient block below is the expensive one (L = 8
                # costs 16 J-products per matvec); writing only at the end of the variant
                # means a slow transient throws away ~11 min of completed measurements.
                out["chi_lip_" + tag] = rec
                json.dump(out, open(OUT, "w"), indent=1)

        # transient sup_m ||J^m|| up to m = K, on the full composite (with correction).
        # CLI:  python _chi_lip_audit.py [variant ...] [notrans]
        #   variant  -- process only the named variant(s) ("constrained" / "unconstrained")
        #   notrans  -- skip the transient probe (it is the expensive part)
        if DO_TRANS:
            ffull = make_f(mk, 0.5, 0.5, THETA, y_b, True, K=K8, learned=True, which="full")
            Jfull = JacOp(ffull, zr, name="J_K8")
            tr = {}
            for L in MS:
                t1 = time.time()
                tr[L] = sigma_max_sq(Jfull, L, tol=1e-3)
                log("    m=%d  ||J^m||=%.4f  (%.0fs)" % (L, tr[L], time.time() - t1))
            rec["transient_full_ms"] = {str(k): float(v) for k, v in tr.items()}
            rec["M_J_full"] = float(max(tr.values()))
            rec["M_J_argmax_m"] = int(max(tr.items(), key=lambda kv: kv[1])[0])
            rec["M_J_truncated_m2"] = float(max(v for k, v in tr.items() if k <= 2))
            log("  %-12s transient m=1..%d: peak %.4f at m=%d (m<=2 gives %.4f)"
                % (tag, K8, rec["M_J_full"], rec["M_J_argmax_m"], rec["M_J_truncated_m2"]))
        out["chi_lip_" + tag] = rec
        json.dump(out, open(OUT, "w"), indent=1)

    log("[chi_lip] done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
