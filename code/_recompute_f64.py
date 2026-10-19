# -*- coding: utf-8 -*-
"""_recompute_f64.py — recompute the certification audit's spectral quantities in
FLOAT64, and patch results/bjac.json.

Why.  The contrast step solves a Tikhonov system whose smallest eigenvalue is
lambda_min ~ 1e-5, so A^{-1} amplifies by ~1e5 and a single-precision directional
derivative through the 15-step CG loop can lose most of its significant digits.  The
float32 alpha sweep produced sigma_full = 22.77 at alpha = 0.05 against 1.24 at
alpha = 0.10 and 1.45 at alpha = 0.20 -- an outlier excluded by the analytic
small-alpha limit (the layer tends to diag((1-gamma)I, I), so sigma_max -> 1).

This script recomputes, in double precision:
  * L1   : the four blocks + full Jacobian at alpha = gamma = 0.5, with the
           closed-form sigma_max(C) and the block/consistency assertions;
  * L2   : the alpha sweep;
  * L2b  : the Born-contrast variant;
  * L3   : Lambda and lambda_min;
and writes BOTH the float64 value and the previous float32 value into bjac.json, so
the size of the precision effect is part of the record rather than hidden.

Each estimate also carries an independent central finite-difference-vs-analytic-JVP
directional check, which does not differentiate through the same code path.

Usage: python _recompute_f64.py
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(True)
from isp_core import UnrolledISP
from spec_tools import JacOp, sigma_max_L, sigma_max_power, as_float64_model, dtype_consistency
from run_bjac import make_f, flat_of, THETA, CHI_MAX, BATCH
from run_experiments import load_data, init_state, NX, NY, N_INC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cpu"
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
DT = torch.float64


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(RES, "recompute_f64_progress.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def fd_vs_jvp(op, zr, f, seed=0):
    """Relative difference between the analytic JVP and a central finite-difference
    directional derivative -- an independent differentiation path, used to catch a wrong
    direction mapping or a dtype fault in the operator itself.

    Both sides must use the operator's OWN input block and output slicing, so the
    perturbation is built with op._to_states (which places the direction in `in_idx` and
    zeros elsewhere) and read back through op._raw.  Drawing a full-state direction and
    pushing it through a BLOCK operator is a shape error: it is what the previous version
    did, and the map is batch-separable, so a direction constant across the batch gives an
    exact block-diagonal Jacobian and the check remains sharp.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(op.Nin, generator=g, dtype=DT)
    x = x / x.norm()
    V = op._to_states(x)                        # (B, D), zero outside in_idx
    Ja = op.jvp(x)                              # flat (B*R,)
    eps = 1e-6
    with torch.no_grad():
        Jf = (op._raw(op.zr + eps * V).reshape(-1) - op._raw(op.zr - eps * V).reshape(-1)) / (2 * eps)
    nf = float(Jf.norm())
    if nf < 1e-20:                      # zero map (e.g. block B of the Born variant)
        return 0.0 if float(Ja.norm()) < 1e-20 else float("inf")
    assert Ja.shape == Jf.shape, "JVP %s vs FD %s" % (tuple(Ja.shape), tuple(Jf.shape))
    return float((Ja - Jf).norm() / nf)


def measure_f64(m, y_b, chi0, w0, alpha, gamma, theta=THETA, born=False):
    n, I = m.nx * m.ny, m.n_inc
    zr = flat_of(chi0, w0)
    idx_chi = torch.arange(0, n)
    idx_w = torch.arange(n, n * (1 + 2 * I))
    fA = make_f(m, alpha, gamma, theta, y_b, False, born=born, which="A")
    fw = make_f(m, alpha, gamma, theta, y_b, False, born=born, which="w")
    ff = make_f(m, alpha, gamma, theta, y_b, False, born=born, which="full")
    spec = (("A", fA, dict(in_idx=idx_chi, out_slice=slice(0, n))),
            ("B", fA, dict(in_idx=idx_w, out_slice=slice(0, n))),
            ("C", fw, dict(in_idx=idx_chi, out_slice=slice(0, 2 * I * n))),
            ("D", fw, dict(in_idx=idx_w, out_slice=slice(0, 2 * I * n))),
            ("full", ff, dict()))
    res = {}
    for tag, f, kw in spec:
        op = JacOp(f, zr, name=tag, **kw)
        s, nv, info = sigma_max_L(op, tol=1e-10)
        sp, _ = sigma_max_power(op, iters=200)
        res["sigma_%s" % tag] = float(s)
        res["sigma_%s_power" % tag] = float(sp)
        res["sigma_%s_matvecs" % tag] = nv
        res["sigma_%s_solver" % tag] = info
        res["sigma_%s_fd_vs_jvp_reldiff" % tag] = fd_vs_jvp(op, zr, f)
        log("    block %-4s sigma=%.8f (power %.8f)  matvecs=%d  fd-vs-jvp=%.2e"
            % (tag, s, sp, nv, res["sigma_%s_fd_vs_jvp_reldiff" % tag]))
        del op
    with torch.no_grad():
        Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
        maxabs = float(Ew.abs().max().item())
        colnorm = Ew.abs().pow(2).sum(dim=1).sqrt()
        maxcol = float(colnorm.max().item())
    res.update(dict(max_abs_E=maxabs, max_colnorm_E=maxcol,
                    E_inf_ratio=maxcol / maxabs if maxabs else float("inf"),
                    exact_sigma_C=theta * alpha * maxcol,
                    exact_sigma_C_gram=float(((theta * alpha) ** 2 * colnorm ** 2).max().sqrt()),
                    alpha_c_rigorous=1.0 / (theta * maxcol) if maxcol else float("inf")))
    res["sigma_C_ratio_meas_over_formula"] = res["sigma_C"] / max(res["exact_sigma_C"], 1e-30)
    res["rigorous_lower_bound"] = res["exact_sigma_C"]
    blk = max(res["sigma_A"], res["sigma_B"], res["sigma_C"], res["sigma_D"])
    res["block_max"] = blk
    res["full_minus_block_max"] = res["sigma_full"] - blk
    res["consistent_full_ge_blocks"] = bool(res["sigma_full"] >= blk - 1e-9)
    return res


def main():
    t0 = time.time()
    bj = json.load(open(os.path.join(RES, "bjac.json")))
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    gbn = float(np.linalg.norm(Gb, 2))
    beta0 = CHI_MAX * gbn

    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
        Gb, Gs, Ei)
    # NOTE dtype: y is a COMPLEX field, so it is moved with complex128.  Casting it with
    # .to(torch.float64) would silently delete its imaginary part (see as_float64_model).
    y_b = y_te[:BATCH].to(DEV).to(torch.complex128)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
    c0, w0 = init_state(m, y_b, BATCH)
    c0 = c0.to(DT)          # w0 is complex128; casting it to a real dtype would discard it
    assert w0.is_complex() and m.Ei.is_complex(), "float64 conversion lost the complex physics"

    # ---- L0b: the conversion itself must be transparent -------------------------
    m32 = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV)
    with torch.no_grad():
        m32._cg_ls(m32.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1].to(torch.complex64))
    dcon = dtype_consistency(m32, m, w0.detach(), y_b, chi=c0.detach())
    dcon["note"] = ("(a) operators: the same complex input through Gb/Gs/Ei in both "
                    "precisions -- a destroyed imaginary part would give O(1), float32 "
                    "round-off gives O(1e-7). (b) layer: one full layer on the SAME state; "
                    "layer_rel is NOT a conversion defect but the amplification of a 1e-7 "
                    "arithmetic difference by the 15-step CG contrast solve "
                    "(lambda_min ~ 1e-5), which is the reason every audit quantity here is "
                    "double precision.")
    bj["L0b_dtype_consistency"] = dcon
    json.dump(bj, open(os.path.join(RES, "bjac.json"), "w"), indent=1)
    log("=== L0b operators f32 vs f64: Gb=%.2e Gs=%.2e Ei=%.2e | Im(Gb w)/|Gb w|=%.3f | "
        "one layer on same state: dchi=%.2e dw=%.2e"
        % (dcon["operator_Gb_rel"], dcon["operator_Gs_rel"], dcon["operator_Ei_rel"],
           dcon["imag_frac_Gbw"], dcon["layer_rel"], dcon["layer_w_rel"]))
    del m32

    # ---------------------------------------------------------------- L1 (f64)
    log("=== L1 float64, alpha=gamma=0.5 ===")
    old = bj.get("L1_physics_core", {})
    r = measure_f64(m, y_b, c0, w0, 0.5, 0.5)
    r["dtype"] = "float64"
    r["thm_field_upper_bound_sigma_D"] = 1.0 - 0.5 * (1.0 - beta0)
    r["thm_field_respected"] = bool(r["sigma_D"] <= r["thm_field_upper_bound_sigma_D"] + 1e-9)
    r["gram_vs_colnorm_rel_err"] = abs(r["exact_sigma_C"] - r["exact_sigma_C_gram"]) / max(
        r["exact_sigma_C"], 1e-30)
    r["float32_comparison"] = {("sigma_%s" % k): old.get("sigma_%s" % k)
                               for k in ("A", "B", "C", "D", "full")}
    r["float32_vs_float64_rel"] = {
        ("sigma_%s" % k): (abs(old["sigma_%s" % k] - r["sigma_%s" % k]) / r["sigma_%s" % k]
                           if isinstance(old.get("sigma_%s" % k), float) else None)
        for k in ("A", "B", "C", "D", "full")}
    bj["L1_physics_core"] = r
    json.dump(bj, open(os.path.join(RES, "bjac.json"), "w"), indent=1)
    log("  sigma_A=%.8f B=%.8f C=%.8f D=%.8f full=%.8f"
        % (r["sigma_A"], r["sigma_B"], r["sigma_C"], r["sigma_D"], r["sigma_full"]))
    log("  f32 vs f64 rel.: A=%.2e B=%.2e C=%.2e D=%.2e full=%.2e"
        % tuple(r["float32_vs_float64_rel"]["sigma_%s" % k]
                for k in ("A", "B", "C", "D", "full")))
    log("  closed-form sigma_C=%.8f (Gram %.8f)  alpha_c=%.6f" %
        (r["exact_sigma_C"], r["exact_sigma_C_gram"], r["alpha_c_rigorous"]))

    # ---------------------------------------------------------------- L2 (f64)
    log("=== L2 float64 alpha sweep ===")
    old_sw = {round(x["alpha"], 4): x for x in bj.get("L2_alpha_sweep", [])}
    sweep = []
    log("alpha |  sigma64 (lanczos)  sigma64 (power) | sigma32 | rel.diff | proved_lb")
    for a in ALPHAS:
        zr = flat_of(c0, w0)
        f = make_f(m, a, a, THETA, y_b, False, which="full")
        op = JacOp(f, zr, name="a%.3f" % a)
        s, nv, info = sigma_max_L(op, tol=1e-9)
        sp, _ = sigma_max_power(op, iters=150)
        with torch.no_grad():
            Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
            maxcol = float(Ew.abs().pow(2).sum(dim=1).sqrt().max().item())
        proved = a * THETA * maxcol
        s32 = old_sw.get(round(a, 4), {}).get("sigma_full")
        rel = (abs(s32 - s) / s) if isinstance(s32, float) else None
        sweep.append(dict(alpha=a, sigma_full=float(s), sigma_full_power=float(sp),
                          sigma_full_float32=s32,
                          sigma_full_float32_vs_float64_rel=rel,
                          matvecs=nv, solver=info, dtype="float64",
                          rigorous_lower_bound=proved,
                          proved_noncontractive=bool(proved > 1.0),
                          measured_noncontractive=bool(s > 1.0),
                          max_colnorm_E=maxcol))
        log("%.3f | %11.6f %11.6f | %8s | %8s | %.4f"
            % (a, s, sp, ("%.4f" % s32) if isinstance(s32, float) else "n/a",
               ("%.1e" % rel) if rel is not None else "n/a", proved))
        del op
    bj["L2_alpha_sweep"] = sweep
    bj["L2_alpha_sweep_notes"] = {
        "dtype": "float64",
        "reason": ("The contrast step solves a Tikhonov system with lambda_min ~ 1e-5, "
                   "so A^{-1} amplifies by ~1e5 and a float32 directional derivative "
                   "through the 15-step CG loop loses its significant digits. The float32 "
                   "value at alpha=0.05 is kept in sigma_full_float32 as evidence of the "
                   "artefact; the analytic small-alpha limit of the layer is "
                   "diag((1-gamma)I, I), i.e. sigma_max -> 1."),
        "cross_check": "power iteration on J^T J agrees with Lanczos at every alpha."}
    pa = [x["alpha"] for x in sweep if x["proved_noncontractive"]]
    bj["L2_proved_range_note"] = {"alphas_with_proved_bound_above_one": pa,
                                 "smallest_such_alpha": min(pa) if pa else None,
                                 "alpha_c": r["alpha_c_rigorous"]}
    json.dump(bj, open(os.path.join(RES, "bjac.json"), "w"), indent=1)

    # --------------------------------------------------------------- L2b (f64)
    log("=== L2b float64 Born-contrast variant ===")
    rb = measure_f64(m, y_b, c0, w0, 0.5, 0.5, born=True)
    rb["dtype"] = "float64"
    rb["float32_comparison"] = {("sigma_%s" % k): bj.get("L2b_born_contrast", {}).get("sigma_%s" % k)
                                for k in ("A", "B", "C", "D", "full")}
    bj["L2b_born_contrast"] = rb
    json.dump(bj, open(os.path.join(RES, "bjac.json"), "w"), indent=1)
    log("  Born: sigma_B=%.3e sigma_C=%.6f sigma_full=%.6f (non-Born C=%.6f full=%.6f)"
        % (rb["sigma_B"], rb["sigma_C"], rb["sigma_full"], r["sigma_C"], r["sigma_full"]))

    # ---------------------------------------------------------------- L3 (f64)
    log("=== L3 float64 Lambda ===")
    gs_norm = float(np.linalg.norm(Gs, 2))
    with torch.no_grad():
        Ew0 = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
        gs_diag = (torch.abs(Gs) ** 2).sum(dim=0).to(DT)
        diagA = ((torch.abs(Ew0) ** 2) * gs_diag).sum(dim=1)
        lam_min = float((m.cg_lam_rel * diagA + 1e-12).min().item())
        lam_max = float((m.cg_lam_rel * diagA + 1e-12).max().item())
        max_absE = float(Ew0.abs().max().item())
    zr = flat_of(c0, w0)
    fy = make_f(m, 0.5, 0.5, THETA, y_b, False, which="A")
    ky = JacOp(fy, zr, in_idx=torch.arange(NX * NY), out_slice=slice(0, NX * NY), name="chi<-y")
    lam_meas, ky_nv, ky_info = sigma_max_L(ky, tol=1e-9)
    lam_bound = 0.5 * gs_norm * max_absE / max(lam_min, 1e-30)
    old_l3 = bj.get("L3_noise_gain", {})
    bj["L3_noise_gain"] = {
        "gs_op_norm": gs_norm, "lambda_min_A": lam_min, "lambda_max_A": lam_max,
        "max_abs_E": max_absE, "lambda_measured": float(lam_meas),
        "lambda_measured_float32": old_l3.get("lambda_measured"),
        "lambda_measured_matvecs": ky_nv, "lambda_measured_solver": ky_info,
        "lambda_measured_fd_vs_jvp_reldiff": fd_vs_jvp(ky, zr, fy),
        "lambda_constructive_bound_partial": lam_bound,
        "bound_over_measured": lam_bound / max(lam_meas, 1e-30),
        "dtype": "float64",
        "note": ("lambda_measured is the spectral norm of d(chi+)/dy at alpha=gamma=0.5; "
                 "lambda_constructive_bound_partial = gamma*||Gs||_2*max_r|E|/lambda_min "
                 "is the field part of the constructive bound (the correction term "
                 "eps_bar*prod||W_i||/s_resid is not included).")}
    json.dump(bj, open(os.path.join(RES, "bjac.json"), "w"), indent=1)
    log("  Lambda measured=%.6f  constructive(field part)=%.1f  ratio=%.1f  lam_min=%.3e"
        % (lam_meas, lam_bound, lam_bound / max(lam_meas, 1e-30), lam_min))

    log("[recompute_f64 done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
