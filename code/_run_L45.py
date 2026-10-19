# -*- coding: utf-8 -*-
"""_run_L45.py — the composite-Jacobian part of the certification audit (L4/L5).

Split out from run_bjac.py because a full audit pass is dominated by the K=8 composite
(each matvec costs 8 layer applications, each of which runs the 15-step CG contrast
solve) and because the composite section is the only one that needs trained checkpoints.

Differences from the first version:
  * it writes results/L45.json, its OWN file, instead of read-modify-writing
    results/bjac.json.  Two audit jobs running concurrently would otherwise clobber
    each other's keys (read at t0, dump at t1 > t0);
  * it runs in FLOAT64.  The contrast step inverts a Tikhonov system with
    lambda_min ~ 1e-5, so A^{-1} amplifies a directional derivative by ~1e5 and a
    float32 derivative through the CG loop loses significant digits; the same
    precision effect that made the float32 alpha sweep report 22.77 at alpha = 0.05
    also applies to the composite Jacobian, and rho_chi > 1 is a published claim.

For the constrained variant the projection is ON, i.e. the operator audited is the one
actually deployed: the clamp makes the derivative vanish on coordinates pinned at the
box edges, which is the honest as-deployed Jacobian.

Usage: python _run_L45.py
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(True)
from isp_core import UnrolledISP
from spec_tools import (JacOp, sigma_max_L, sigma_max_power, spectral_radius,
                         transient_M, as_float64_model)
from run_bjac import make_f, flat_of, THETA, CHI_MAX, BATCH
from run_experiments import load_data, init_state, NX, NY, N_INC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "L45.json")
DEV = "cpu"
DT = torch.float64
K8 = 8
COMP_TOL = 1e-4
COMP_POWER = 15
COMP_MS = (1, 2)
VARIANTS = (("constrained", "ckpt_constrained_K8.pt", True),
            ("unconstrained", "ckpt_unconstrained_K8.pt", False))


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(RES, "L45_progress.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    n = NX * NY
    gbn = float(np.linalg.norm(Gb, 2))

    # ---- reference single-layer state (same batch of test states as the rest of the audit)
    m1 = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
        Gb, Gs, Ei)
    y_b = y_te[:BATCH].to(DEV).to(torch.complex128)     # complex field, not float64
    with torch.no_grad():
        m1._cg_ls(m1.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
        chi0, w0 = init_state(m1, y_b, BATCH)
    chi0 = chi0.to(DT)              # w0 stays complex128
    assert w0.is_complex(), "float64 conversion lost the complex field state"

    arc = os.path.join(RES, "ckpt_archive_r1")
    for tag, ck, proj in VARIANTS:
        path = os.path.join(arc, ck)
        if not os.path.exists(path):            # fall back to the live directory
            path = os.path.join(RES, ck)
        if not os.path.exists(path):
            log("  [skip] %s" % path)
            continue
        log("=== K=%d cyclic composite: %s (proj=%s, float64) ===" % (K8, tag, proj))
        mk = as_float64_model(
            UnrolledISP(K8, Gb, Gs, Ei, NX, NY, N_INC,
                        mode=("constrained" if tag == "constrained" else "unconstrained"),
                        proj=proj).to(DEV), Gb, Gs, Ei)
        mk.load_state_dict(torch.load(path, map_location=DEV))   # float32 -> float64 cast
        mk.eval()
        with torch.no_grad():
            mk._cg_ls(mk.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
            c8, w8 = mk(chi0, w0, y_b)
            maxchi = float(c8.abs().max().item())
        zr = flat_of(chi0, w0)

        fc = make_f(mk, 0.5, 0.5, THETA, y_b, True, K=K8, learned=True, which="full")

        def chi_only(z, _fc=fc, _n=n):
            return _fc(z)[:, :_n]

        Jc = JacOp(fc, zr, name="comp-" + tag)
        s_comp, nv_c, info_c = sigma_max_L(Jc, tol=COMP_TOL)
        sp_comp, _ = sigma_max_power(Jc, iters=COMP_POWER)
        log("  sigma_max(composite)=%.6f (power %.6f) matvecs=%d %s"
            % (s_comp, sp_comp, nv_c, info_c))
        rho_comp, nv_r, info_r = spectral_radius(Jc, tol=COMP_TOL)
        log("  rho(composite)=%.6f matvecs=%d %s" % (rho_comp, nv_r, info_r))

        Jx = JacOp(chi_only, zr, in_idx=torch.arange(n), out_slice=slice(0, n),
                   name="chichi-" + tag)
        rho_chi, nv_rc, info_rc = spectral_radius(Jx, tol=COMP_TOL)
        s_chi, _, _ = sigma_max_L(Jx, tol=COMP_TOL)
        log("  rho_chi(composite)=%.6f  sigma_max(chi->chi)=%.6f  matvecs=%d %s"
            % (rho_chi, s_chi, nv_rc, info_rc))

        tr = transient_M(Jc, ms=COMP_MS)
        good = [v for v in tr.values() if v == v]
        MJ = max(good) if good else float("nan")
        log("  transient ||J^m|| %s  -> M_J=%.4f" % (tr, MJ))

        rec = {
            "sigma_max_composite": float(s_comp), "sigma_max_composite_power": float(sp_comp),
            "sigma_matvecs": nv_c, "sigma_solver": info_c,
            "rho_composite": float(rho_comp), "rho_matvecs": nv_r, "rho_solver": info_r,
            "rho_chi_composite": float(rho_chi), "sigma_max_chi_block": float(s_chi),
            "rho_chi_matvecs": nv_rc, "rho_chi_solver": info_rc,
            "transient_norms": {str(k): v for k, v in tr.items()}, "M_J_approx": MJ,
            "gap_ratio_sigma_over_rho": (s_comp / rho_comp if rho_comp else float("nan")),
            "maxabs_chi_after_cycle": maxchi, "K": K8, "proj": bool(proj),
            "rho_chi_lt_one": bool(rho_chi < 1.0),
            "dtype": "float64", "batch": BATCH, "tol": COMP_TOL,
            "note": ("rho_chi is the spectral radius of the chi -> chi block of the K-layer "
                     "composite Jacobian, i.e. the quantity whose being < 1 is the hypothesis "
                     "of Cor. fidelity; M_J = sup_{m in %s} ||J^m|| is the transient factor "
                     "that the geometric bound rho^m omits.  Float64 because the CG contrast "
                     "solve amplifies directional derivatives by ~1/lambda_min." % (list(COMP_MS),))}
        out["L4_%s_composite" % tag] = rec
        json.dump(out, open(OUT, "w"), indent=1)
        del mk, Jc, Jx

    ck = out.get("L4_constrained_composite")
    if ck:
        out["L5_non_normality"] = {
            "variant": "constrained K=8",
            "sigma_max_composite": ck["sigma_max_composite"],
            "rho_composite": ck["rho_composite"],
            "gap_ratio_sigma_over_rho": ck["gap_ratio_sigma_over_rho"],
            "M_J_approx": ck["M_J_approx"],
            "transient_norms": ck["transient_norms"],
            "dtype": "float64",
            "note": ("rho^m <= ||J^m|| <= kappa_2(V) rho^m; the observed gap sigma_max/rho is "
                     "the price of reading a spectral radius as a contraction rate on a "
                     "non-normal layer.")}
        uk = out.get("L4_unconstrained_composite")
        if uk:
            out["L5_non_normality_unconstrained"] = {
                "sigma_max_composite": uk["sigma_max_composite"],
                "rho_composite": uk["rho_composite"],
                "gap_ratio_sigma_over_rho": uk["gap_ratio_sigma_over_rho"],
                "M_J_approx": uk["M_J_approx"],
                "dtype": "float64"}
    json.dump(out, open(OUT, "w"), indent=1)
    log("[L4/L5 done] %.0fs -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
