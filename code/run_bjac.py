# -*- coding: utf-8 -*-
"""run_bjac.py — certification audit for the impossibility / noise / fidelity results.

State ordering (paper convention): z = (chi, w_real, w_imag), so the realified
layer Jacobian is
        J = [[ A, B ],
             [ C, D ]],   A = d chi+/d chi, B = d chi+/d w,
                          C = d w+/d chi,   D = d w+/d w.

Exact structural facts used (theta = global relaxation, alpha/gamma = damping):
  * C = theta*alpha * diag(E)   [E = E^i + G_b w^t = total field at the operating point]
        C maps the SPATIAL chi-block into the INCIDENCE-STACKED w-block and is a direct
        sum over cells r of the rank-1 map delta_chi_r -> theta*alpha*E_{.,r} * delta_chi_r.
        Hence EXACTLY
            sigma_max(C) = theta*alpha * max_r || E_{.,r} ||_2 ,
        with || . ||_2 the ell^2 norm ACROSS INCIDENCES (largest per-cell field energy).
        The tempting alpha*max_r|E| is the ell^infinity version and is WRONG by a factor
        up to sqrt(n_inc).  Both are evaluated here and their ratio is reported.
  * D = (1-theta*alpha) I + theta*alpha * diag(chi) G_b
        => sigma_max(D) <= 1 - theta*alpha*(1-beta0)   [Thm. field]
  * Blockwise rigorous lower bounds: sigma_max(J) >= sigma_max(any block), hence for
        every alpha,gamma in (0,1]
            sigma_max(J) >= theta*alpha*max_r ||E_{.,r}||_2 ,
        which is > 1 whenever alpha > alpha_c := 1/(theta*max_r||E_{.,r}||_2).

INSTRUMENT.  See spec_tools.py.  The Jacobian is never formed: forward products are
exact (torch.func.jvp, no finite differences) and adjoints are exact (reverse-mode AD).
sigma_max is obtained by Lanczos/IRAM on J^T J, and the spectral radius by IRAM on J.
An earlier revision of this script used a finite-difference power iteration, which on
these strongly non-normal maps had not converged and produced a full-Jacobian estimate
BELOW one of its own blocks.  Both estimators are kept as mutually independent
cross-checks and are cross-reported below (a Lanczos Ritz value and a power-iteration
value are both rigorous lower bounds; their spread is an honest dispersion).

CPU-only by design, so that it never contends with a GPU training job.
Writes results/bjac.json.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(True)
from isp_core import UnrolledISP
from spec_tools import (JacOp, sigma_max_L, sigma_max_power, spectral_radius,
                        transient_M, as_float64_model)
from run_experiments import load_data, init_state, NX, NY, N_INC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cpu"                      # never contend with the GPU training chain
ALPHA0 = GAMMA0 = 0.5
THETA = 1.0
CHI_MAX = 0.5
BATCH = 16                       # "16 test states", as quoted in Tab. cert
SIGTOL = 1e-6
COMP_TOL = 1e-4          # looser tolerance for the 8x-cost K=8 composite
COMP_POWER = 15          # power-iteration cross-check budget on the composite
COMP_MS = (1, 2)         # transient exponents for M_J


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def log(*a):
    """Print AND append to results/bjac_progress.txt: `conda run` buffers the child's
    stdout, so the log file is the only real-time progress channel."""
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(RES, "bjac_progress.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# --------------------------------------------------------------------------- #
def core_parts(m, zr, alpha, gamma, theta, y, use_corr=False, k=0, born=False, learned=False):
    """One layer step with explicit alpha/gamma/theta, or the model's learned values
    when learned=True."""
    B = y.shape[0]
    n, I = m.nx * m.ny, m.n_inc
    if learned:
        alpha = m.a_max * torch.sigmoid(m.a[k])
        gamma = m.g_max * torch.sigmoid(m.g[k])
        theta = m.theta
    chi = zr[:, :n]
    wr = zr[:, n:n + n * I].reshape(B, I, n)
    wi = zr[:, n + n * I:].reshape(B, I, n)
    w = wr + 1j * wi

    Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w)                     # total field E(w)
    w_t = (1 - alpha) * w + alpha * chi.unsqueeze(1) * Ew
    E = m.Ei.unsqueeze(0) + m._Gb_apply(w_t)
    E_ls = m.Ei.unsqueeze(0).expand_as(E) if born else E
    chi_hat = m._cg_ls(E_ls, y)
    chi_t = (1 - gamma) * chi + gamma * chi_hat

    if use_corr:
        resid = m._Gs_apply(w_t) - y
        resid_dom = torch.matmul(m.Gs.conj().T,
                                 resid.reshape(-1, resid.shape[-1]).T).T.reshape(B, I, -1)
        feat = torch.cat([
            resid_dom.real.reshape(B, -1), resid_dom.imag.reshape(B, -1),
            E.real.reshape(B, -1), E.imag.reshape(B, -1),
            w_t.real.reshape(B, -1), w_t.imag.reshape(B, -1), chi], dim=1)
        feat = feat.reshape(B, -1, m.nx, m.ny)
        feat = torch.tanh(feat / m.feat_scale) if m.featbound else feat / m.feat_scale
        dchi = m.corrs[k](feat).flatten(1)
    else:
        dchi = 0.0

    chi_out = chi + theta * (chi_t - chi) + dchi
    w_out = w + theta * (w_t - w)
    if m.proj:
        chi_out = chi_out.clamp(0.0, m.chi_max)
    return chi_out, w_out, Ew, E


def flat_of(zr_chi, zr_w):
    B = zr_chi.shape[0]
    return torch.cat([zr_chi, zr_w.real.reshape(B, -1), zr_w.imag.reshape(B, -1)],
                     dim=1).detach()


def make_f(m, alpha, gamma, theta, y, use_corr, k=0, born=False, which="full", K=None,
           learned=False):
    """Flat real map. which in {A (=chi+), w (=w+), full}."""
    B = y.shape[0]
    n, I = m.nx * m.ny, m.n_inc

    def f(zr):
        if K is None:
            c, w, _, _ = core_parts(m, zr, alpha, gamma, theta, y, use_corr, k, born, learned)
        else:                                     # cyclic composite over K layers
            z = zr
            for kk in range(K):
                c, w, _, _ = core_parts(m, z, alpha, gamma, theta, y, use_corr,
                                        kk, born, learned)
                z = torch.cat([c, w.real.reshape(B, -1), w.imag.reshape(B, -1)], dim=1)
            c, w = z[:, :n], z[:, n:n + n * I] + 1j * z[:, n + n * I:]
        if which == "A":
            return c
        if which == "w":
            return torch.cat([w.real.reshape(B, -1), w.imag.reshape(B, -1)], dim=1)
        return torch.cat([c, w.real.reshape(B, -1), w.imag.reshape(B, -1)], dim=1)
    return f


def exact_C(m, w0, alpha, theta=THETA):
    """Closed-form sigma_max(C) = theta*alpha*max_r||E_{.,r}||_2, evaluated two ways."""
    with torch.no_grad():
        Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w0)              # E at the w0 operating point
        maxabs = float(Ew.abs().max().item())
        colnorm = Ew.abs().pow(2).sum(dim=1).sqrt()           # (B,n)
        maxcol = float(colnorm.max().item())
    exact = theta * alpha * maxcol                            # closed form
    gram = float(torch.sqrt(((theta * alpha) ** 2 * colnorm ** 2).max()).item())
    return dict(max_abs_E=maxabs, max_colnorm_E=maxcol, exact=exact, gram=gram,
                E_inf_ratio=(maxcol / maxabs if maxabs > 0 else float("inf")))


# --------------------------------------------------------------------------- #
def measure_layer(m, y_b, chi0, w0, alpha, gamma, theta=THETA, use_corr=False,
                  born=False, power_iters=120):
    """All four blocks + full Jacobian of ONE layer, by Lanczos, with power-iteration
    cross-checks."""
    n, I = m.nx * m.ny, m.n_inc
    zr = flat_of(chi0, w0)
    idx_chi = torch.arange(0, n)
    idx_w = torch.arange(n, n * (1 + 2 * I))
    fA = make_f(m, alpha, gamma, theta, y_b, use_corr, born=born, which="A")
    fw = make_f(m, alpha, gamma, theta, y_b, use_corr, born=born, which="w")
    ff = make_f(m, alpha, gamma, theta, y_b, use_corr, born=born, which="full")
    Cf = exact_C(m, w0, alpha, theta)

    res = {}
    for tag, f, kw in (
            ("A", fA, dict(in_idx=idx_chi, out_slice=slice(0, n))),
            ("B", fA, dict(in_idx=idx_w, out_slice=slice(0, n))),
            ("C", fw, dict(in_idx=idx_chi, out_slice=slice(0, 2 * I * n))),
            ("D", fw, dict(in_idx=idx_w, out_slice=slice(0, 2 * I * n))),
            ("full", ff, dict())):
        op = JacOp(f, zr, name=tag, **kw)
        s, nv, info = sigma_max_L(op, tol=SIGTOL)
        sp, npv = sigma_max_power(op, iters=power_iters)
        res["sigma_%s" % tag] = s
        res["sigma_%s_matvecs" % tag] = nv
        res["sigma_%s_solver" % tag] = info
        res["sigma_%s_power" % tag] = sp
        res["sigma_%s_power_matvecs" % tag] = npv
        log("    block %-4s sigma=%.6f (power %.6f)  matvecs=%d  %s"
            % (tag, s, sp, nv, info))

    # ---- internal structural consistency: sigma_max(J) >= sigma_max(any block)
    blk = max(res["sigma_A"], res["sigma_B"], res["sigma_C"], res["sigma_D"])
    res["block_max"] = blk
    res["full_minus_block_max"] = res["sigma_full"] - blk
    res["consistent_full_ge_blocks"] = bool(res["sigma_full"] >= blk - 1e-6)

    res.update(dict(max_abs_E=Cf["max_abs_E"], max_colnorm_E=Cf["max_colnorm_E"],
                    E_inf_ratio=Cf["E_inf_ratio"],
                    exact_sigma_C=Cf["exact"], exact_sigma_C_gram=Cf["gram"],
                    sigma_C_ratio_meas_over_formula=res["sigma_C"] / max(Cf["exact"], 1e-30),
                    rigorous_lower_bound=Cf["exact"],
                    alpha_c_rigorous=(1.0 / (theta * Cf["max_colnorm_E"])
                                      if Cf["max_colnorm_E"] > 0 else float("inf"))))
    return res


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    # complex data must go to complex128; .to(torch.float64) would discard the field.
    # The pre-fix version kept float32/complex64 here, which is exactly the dtype bug
    # that made the archived bjac composite disagree with run_gain_decomp by ~46%
    # (see results/tol_probe.json: on matched tolerance the two operator
    # constructions are bit-identical once both run in f64).
    chi_te_d = chi_te.to(DEV).to(torch.float64)
    y_te_d = y_te.to(DEV).to(torch.complex128)
    assert y_te_d.is_complex(), 'incident field lost its imaginary part'
    n = NX * NY
    gbn = float(np.linalg.norm(Gb, 2))
    beta0 = CHI_MAX * gbn
    out = {"config": {"nx": NX, "ny": NY, "n_inc": N_INC, "chi_max": CHI_MAX,
                      "theta": THETA, "batch": BATCH, "gb_norm2": gbn, "beta0": beta0,
                      "device": DEV, "dtype": "float64 (complex128 fields)",
                      "sigma_tol": SIGTOL, "instrument": "exact JVP/VJP + Lanczos"}}

    # as_float64_model re-installs the complex buffers in complex128 (a plain
    # .to(torch.float64) would silently destroy the complex Gb/Gs/Ei buffers).
    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
        Gb, Gs, Ei)
    assert next(m.parameters()).dtype == torch.float64, "audit model is not float64"
    assert m.Gb.is_complex() and m.Gb.dtype == torch.complex128, "Gb is not complex128"
    with torch.no_grad():                      # pre-warm the lazily cached diag(Gs^H Gs)
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
    chi_b, y_b = chi_te_d[:BATCH], y_te_d[:BATCH]
    chi0, w0 = init_state(m, y_b, BATCH)
    assert chi0.dtype == torch.float64 and w0.is_complex() and w0.dtype == torch.complex128, (
        "init_state produced dtype %s / %s" % (chi0.dtype, w0.dtype))
    zr = flat_of(chi0, w0)
    n_chi, n_w = n, n * (1 + 2 * N_INC)

    # ------------------------------------------------- L0: instrument validation
    log("=== [L0] instrument validation ===")
    torch.manual_seed(0)
    Dv, Rv = 400, 260
    L = torch.randn(Rv, Dv) / np.sqrt(Dv)
    op = JacOp(lambda z: z @ L.T, torch.randn(1, Dv), name="rand")
    s_lin, nv_lin, info_lin = sigma_max_L(op)
    s_lin_svd = float(torch.linalg.matrix_norm(L, 2))
    s_lin_pow, _ = sigma_max_power(op, iters=200)
    rel_lin = abs(s_lin - s_lin_svd) / s_lin_svd
    out["L0_instrument"] = {
        "linear_map_sigma_lanczos": s_lin, "linear_map_sigma_svd": s_lin_svd,
        "linear_map_sigma_power": s_lin_pow,
        "rel_error": rel_lin, "matvecs": nv_lin, "solver": info_lin,
        "shapes_ok": bool(op.Nin == Dv and op.Nout == Rv),
        "note": ("check 1: Lanczos sigma_max of a random linear map vs the exact value "
                 "from torch.linalg.matrix_norm (and a power-iteration cross-check). "
                 "check 2: the closed form sigma_max(C)=theta*alpha*max_r||E_.,r||_2 "
                 "reproduced two independent ways (direct column norms and the Gram "
                 "matrix) and matched by Lanczos on the layer itself. "
                 "check 3: sigma_max(J) >= max of its blocks.")}
    log("  [check1] linear map: lanczos=%.10f svd=%.10f power=%.10f rel.err=%.2e"
        % (s_lin, s_lin_svd, s_lin_pow, rel_lin))
    del L, op

    # ---------------------------------------------------------------- L1: core
    log("=== [L1] physics core blocks, alpha=gamma=%.2f ===" % ALPHA0)
    r = measure_layer(m, y_b, chi0, w0, ALPHA0, GAMMA0)
    r["thm_field_upper_bound_sigma_D"] = 1.0 - ALPHA0 * (1.0 - beta0)
    r["gram_vs_colnorm_rel_err"] = (abs(r["exact_sigma_C"] - r["exact_sigma_C_gram"])
                                    / max(r["exact_sigma_C"], 1e-30))
    r["thm_field_respected"] = bool(r["sigma_D"] <= r["thm_field_upper_bound_sigma_D"] + 1e-9)
    out["L1_physics_core"] = r
    save("bjac.json", out)
    log("  [check2] sigma_max(C): closed form=%.6f  Gram=%.6f  rel.err=%.2e  lanczos=%.6f"
        % (r["exact_sigma_C"], r["exact_sigma_C_gram"], r["gram_vs_colnorm_rel_err"],
           r["sigma_C"]))
    log("  [check3] sigma_full=%.4f >= max block=%.4f  -> %s"
        % (r["sigma_full"], r["block_max"], r["consistent_full_ge_blocks"]))
    log("  max|E|=%.4f  max_r||E_.r||2=%.4f  ratio=%.4f  alpha_c=%.4f"
        % (r["max_abs_E"], r["max_colnorm_E"], r["E_inf_ratio"], r["alpha_c_rigorous"]))
    log("  sigma_D=%.4f <= thm bound %.4f -> %s"
        % (r["sigma_D"], r["thm_field_upper_bound_sigma_D"], r["thm_field_respected"]))

    # ------------------------------------------------------ L2: alpha sweep
    log("=== [L2] alpha=gamma sweep (proved threshold) ===")
    sweep = []
    for a in (0.05, 0.10, 0.20, 0.317, 0.40, 0.50, 0.75, 1.00):
        Cf = exact_C(m, w0, a, THETA)
        f = make_f(m, a, a, THETA, y_b, False, which="full")
        o = JacOp(f, zr, name="a=%.3f" % a)
        s, nv, info = sigma_max_L(o, tol=1e-5)
        sp, _ = sigma_max_power(o, iters=100)
        row = dict(alpha=a, sigma_full=s, sigma_full_power=sp, matvecs=nv, solver=info,
                   rigorous_lower_bound=Cf["exact"],
                   proved_noncontractive=bool(Cf["exact"] > 1.0),
                   measured_noncontractive=bool(s > 1.0),
                   max_colnorm_E=Cf["max_colnorm_E"])
        sweep.append(row)
        log("  alpha=%.3f  sigma=%.4f  proved_lb=%.4f  %s"
            % (a, s, Cf["exact"], "PROVED>1" if Cf["exact"] > 1 else ""))
    out["L2_alpha_sweep"] = sweep
    alphas_proved = [r["alpha"] for r in sweep if r["proved_noncontractive"]]
    out["L2_proved_range_note"] = {
        "alphas_with_proved_bound_above_one": alphas_proved,
        "smallest_such_alpha": min(alphas_proved) if alphas_proved else None}
    save("bjac.json", out)

    # ------------------------------------------------ L2b: Born-contrast variant
    log("=== [L2b] Born-contrast variant (forces B=0 by linearising the data misfit) ===")
    rb = measure_layer(m, y_b, chi0, w0, ALPHA0, GAMMA0, born=True, power_iters=80)
    out["L2b_born_contrast"] = rb
    log("  sigma_B=%.3e  sigma_C=%.4f  sigma_full=%.4f  (non-Born C=%.4f, full=%.4f)"
        % (rb["sigma_B"], rb["sigma_C"], rb["sigma_full"], r["sigma_C"], r["sigma_full"]))
    save("bjac.json", out)

    # ------------------------------------------------- L3: noise gain / Lambda
    log("=== [L3] noise gain Lambda ===")
    gs_norm = float(np.linalg.norm(Gs, 2))
    with torch.no_grad():
        Ew0 = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
        gs_diag = torch.as_tensor(np.abs(Gs) ** 2, dtype=Ew0.real.dtype,
                                  device=Ew0.device).sum(dim=0)
        diagA = ((torch.abs(Ew0) ** 2) * gs_diag).sum(dim=1)      # (B,n)
        lam_min = float((m.cg_lam_rel * diagA + 1e-12).min().item())
        lam_max = float((m.cg_lam_rel * diagA + 1e-12).max().item())
        max_absE = float(Ew0.abs().max().item())
    # measured Lambda = spectral norm of d(chi^+)/dy of one layer at alpha=gamma=0.5
    fw_ = make_f(m, ALPHA0, GAMMA0, THETA, y_b, False, which="A")
    sA_y, nv_y, info_y = sigma_max_L(JacOp(fw_, zr, in_idx=torch.arange(n_chi),
                                           out_slice=slice(0, n), name="chi<-y"))
    # constructive bound, composed from the layer's own constants:
    #   chi+ = (1-g) chi + g * (A)^{-1} b + theta*(...)      (theta=1 in all experiments)
    #   d chi+/dy = gamma * A^{-1} d b/dy,  ||d b/dy|| <= ||Gs||_2 * max_r|E| ,
    #   hence  Lambda <= gamma * ||Gs||_2 * max_r|E| / lambda_min.
    lam_bound = GAMMA0 * gs_norm * max_absE / max(lam_min, 1e-30)
    out["L3_noise_gain"] = {
        "gs_op_norm": gs_norm, "lambda_min_A": lam_min, "lambda_max_A": lam_max,
        "max_abs_E": max_absE, "lambda_measured": sA_y, "lambda_measured_matvecs": nv_y,
        "lambda_measured_solver": info_y,
        "lambda_constructive_bound_partial": lam_bound,
        "bound_over_measured": lam_bound / max(sA_y, 1e-30),
        "note": ("lambda_measured is the spectral norm of d(chi+)/dy at alpha=gamma=0.5; "
                 "lambda_constructive_bound_partial = gamma*||Gs||_2*max_r|E|/lambda_min "
                 "is the field-less part of the constructive bound (the correction term "
                 "eps_bar*prod||W_i||/s_resid is not included).")}
    log("  ||Gs||2=%.4f  lambda_min=%.3e  max|E|=%.4f" % (gs_norm, lam_min, max_absE))
    log("  Lambda: measured=%.4f  constructive(field part)=%.1f  ratio=%.1f"
        % (sA_y, lam_bound, lam_bound / max(sA_y, 1e-30)))
    save("bjac.json", out)

    # ------------------------------------- L4/L5: composite at the trained K=8
    log("=== [L4/L5] K=8 cyclic composite, trained checkpoints ===")
    for tag, ck, proj in (("constrained", "ckpt_constrained_K8.pt", True),
                          ("unconstrained", "ckpt_unconstrained_K8.pt", False)):
        path = os.path.join(RES, ck)
        if not os.path.exists(path):
            log("  [skip] %s not found" % path)
            continue
        K8 = 8
        mk = as_float64_model(
            UnrolledISP(K8, Gb, Gs, Ei, NX, NY, N_INC,
                        mode=("constrained" if tag == "constrained" else "unconstrained"),
                        proj=proj).to(DEV),
            Gb, Gs, Ei)
        assert next(mk.parameters()).dtype == torch.float64, "composite model is not float64"
        mk.load_state_dict(torch.load(path, map_location=DEV))
        mk.eval()
        with torch.no_grad():
            mk._cg_ls(mk.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
            c0, ww0 = mk(chi0, w0, y_b)
            maxchi = float(c0.abs().max().item())
        fc = make_f(mk, ALPHA0, GAMMA0, THETA, y_b, True, K=K8, learned=True, which="full")
        Jc = JacOp(fc, zr, name="comp-" + tag)
        s_comp, nv_c, info_c = sigma_max_L(Jc, tol=COMP_TOL)
        sp_comp, _ = sigma_max_power(Jc, iters=COMP_POWER)
        rho_comp, nv_r, info_r = spectral_radius(Jc, tol=COMP_TOL)
        # chi -> chi block of the composite: x |-> (J_K (x,0))_chi
        def chi_only(z):
            return fc(z)[:, :n]
        Jx = JacOp(chi_only, zr, in_idx=torch.arange(n_chi), out_slice=slice(0, n),
                   name="chichi-" + tag)
        rho_chi, nv_rc, info_rc = spectral_radius(Jx, tol=COMP_TOL)
        tr = transient_M(Jc, ms=COMP_MS)
        MJ = max([v for v in tr.values() if v == v] or [float("nan")])
        rec = {
            "sigma_max_composite": s_comp, "sigma_max_composite_power": sp_comp,
            "sigma_matvecs": nv_c, "sigma_solver": info_c,
            "rho_composite": rho_comp, "rho_matvecs": nv_r, "rho_solver": info_r,
            "rho_chi_composite": rho_chi, "rho_chi_matvecs": nv_rc, "rho_chi_solver": info_rc,
            "transient_norms": tr, "M_J_approx": MJ,
            "gap_ratio_sigma_over_rho": (s_comp / rho_comp if rho_comp else float("nan")),
            "maxabs_chi_after_cycle": maxchi, "K": K8, "proj": bool(proj),
            "rho_chi_lt_one": bool(rho_chi < 1.0)}
        out["L4_%s_composite" % tag] = rec
        log("  [%s] sigma_comp=%.4f  rho=%.4f  rho_chi=%.4f  M_J~%.4f  gap=%.3f  (rho_chi<1: %s)"
            % (tag, s_comp, rho_comp, rho_chi, MJ, rec["gap_ratio_sigma_over_rho"],
               rec["rho_chi_lt_one"]))
        save("bjac.json", out)                   # checkpoint after each variant

    # --------------------------------------------- L5: non-normality summary
    ck = out.get("L4_constrained_composite")
    if ck:
        out["L5_non_normality"] = {
            "variant": "constrained K=8",
            "sigma_max_composite": ck["sigma_max_composite"],
            "rho_composite": ck["rho_composite"],
            "gap_ratio_sigma_over_rho": ck["gap_ratio_sigma_over_rho"],
            "M_J_approx": ck["M_J_approx"],
            "transient_norms": ck["transient_norms"],
            "note": ("rho^m <= ||J^m|| <= kappa_2(V) rho^m; the observed gap "
                     "sigma_max/rho is the price of reading a spectral radius as a "
                     "contraction rate on a non-normal layer.")}
        uk = out.get("L4_unconstrained_composite")
        if uk:
            out["L5_non_normality_unconstrained"] = {
                "sigma_max_composite": uk["sigma_max_composite"],
                "rho_composite": uk["rho_composite"],
                "gap_ratio_sigma_over_rho": uk["gap_ratio_sigma_over_rho"],
                "M_J_approx": uk["M_J_approx"]}

    # ------------------------------------------------------------- summary
    out["summary"] = {
        "alpha_c": out["L1_physics_core"]["alpha_c_rigorous"],
        "proved_noncontractivity_at_alpha_half":
            bool(out["L1_physics_core"]["exact_sigma_C"] > 1.0),
        "sigma_C_exact": out["L1_physics_core"]["exact_sigma_C"],
        "sigma_full_measured": out["L1_physics_core"]["sigma_full"],
        "block_max_measured": out["L1_physics_core"]["block_max"],
        "consistent_full_ge_blocks": out["L1_physics_core"]["consistent_full_ge_blocks"],
        "thm_field_respected": out["L1_physics_core"]["thm_field_respected"],
        "instrument_rel_error_on_linear_map": out["L0_instrument"]["rel_error"],
    }
    save("bjac.json", out)
    log("[audit done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
