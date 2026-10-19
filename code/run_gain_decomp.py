# -*- coding: utf-8 -*-
"""run_gain_decomp.py — confound-free spectral decomposition of the layer map.

The raw per-layer composite gain (gain.json) mixes three effects: the physics step
size (alpha,gamma, learned separately per run), the correction, and the projection.
Here we FIX alpha=gamma=0.5 for every variant so that all differences are
attributable to the correction / projection only.

Reported quantities (all at K=8 checkpoints, 16 test states):
  A) physics core only (no correction), alpha=gamma=0.5:
       - chi-block (A), w-block (D), composite sigma_max
       - => validates the a-priori field-subsystem bound beta0 = chi_max*||Gb||_2
  B) correction alone: Lip(H_k) = sigma_max of the Jacobian of H_k wrt (chi,w)
       - constrained  (expected <= eps_bar = 0.03)
       - unconstrained (expected >> eps_bar)
  C) composite sigma_max with correction, alpha=gamma=0.5
       - evidence for Prop. (impossibility of composite contraction)

INSTRUMENT (revised).  This driver now uses the SAME converged instrument as the
certification audit (spec_tools.py: exact torch.func.jvp forward products, exact
reverse-mode adjoints, Lanczos/IRAM on J^T J), so that Tables cert/abl do not mix two
different spectral estimators.  The previous version used a finite-difference power
iteration, which on these strongly non-normal maps had not converged.

Writes results/gain_decomp.json.

PROVENANCE (added after round-3 review R1 t3/Q3.2).  A reviewer could not reproduce the
composite row C_constrained.sigma_max_composite_max = 13.0317: re-running this file's own
frozen functions at what he took to be the operating point gave 11.75, his own
re-implementation gave 14.55.  The reason was an unrecorded detail of *this* file: the
state is the Born-LS INITIAL state,

    chi0, w0 = init_state(...);  c0, ww0 = mk(chi0, w0, y_b);  zr = flat_state(chi0, w0)

where the post-rollout state (c0, ww0) is computed and then NOT used (it is used only for
`maxabs_chi_after_cycle`).  Everything in A/B/C is therefore evaluated at the initial
state, with proj ON, alpha = gamma = 0.5 forcibly fixed, float64, batch 16, Lanczos
tol 1e-5, at layer index k = argmax over k = 0..7.  The state-dependent quantities are NOT
comparable to the L45/K=8-cyclic rows, which sit at a different operating point.  This is
now written into the artifact as out["provenance"] so that it cannot go missing again.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(True)
from isp_core import UnrolledISP
from spec_tools import JacOp, sigma_max_L, sigma_max_power, as_float64_model
from run_experiments import load_data, init_state, NX, NY, N_INC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
# CPU-only by design, so that it never contends with a GPU training job AND so that
# every double-precision audit driver (this file, run_bjac.py, _ngc_audit.py,
# run_prec_audit.py) evaluates the same operator on the same arithmetic: the GPU and
# CPU evaluations of this operator differ by ~3% (3.2193 vs 3.1234) because the
# 15-step CG contrast solve is ill-conditioned enough to amplify device-level
# round-off (results/xcheck_operators.json).  Quoting a GPU value next to a CPU value
# as if they were the same measurement is exactly the confusion this pinning removes.
DEV = "cpu"
ALPHA = GAMMA = 0.5
BATCH = 16
TOL = 1e-5


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def log(*a):
    print(*a, flush=True)


def layer_fixed(m, k, alpha, gamma, use_corr, chi, w, y):
    """Layer map with prescribed (alpha,gamma) and optional correction."""
    B = chi.shape[0]
    w_t = (1 - alpha) * w + alpha * chi.unsqueeze(1) * (m.Ei.unsqueeze(0) + m._Gb_apply(w))
    E = m.Ei.unsqueeze(0) + m._Gb_apply(w_t)
    chi_t = (1 - gamma) * chi + gamma * m._cg_ls(E, y)
    n = m.nx * m.ny
    if use_corr:
        resid = m._Gs_apply(w_t) - y
        resid_dom = torch.matmul(m.Gs.conj().T,
                                 resid.reshape(-1, resid.shape[-1]).T).T.reshape(
            B, m.n_inc, -1)
        feat = torch.cat([
            resid_dom.real.reshape(B, -1), resid_dom.imag.reshape(B, -1),
            E.real.reshape(B, -1), E.imag.reshape(B, -1),
            w_t.real.reshape(B, -1), w_t.imag.reshape(B, -1), chi], dim=1)
        feat = feat.reshape(B, -1, m.nx, m.ny)
        feat = torch.tanh(feat / m.feat_scale) if m.featbound else feat / m.feat_scale
        dchi = m.corrs[k](feat).flatten(1)
    else:
        dchi = 0.0
    chi_out = chi + (chi_t - chi) + dchi
    w_out = w + (w_t - w)
    if m.proj:
        chi_out = chi_out.clamp(0.0, m.chi_max)
    return chi_out, w_out


def make_flatten(m, k, alpha, gamma, use_corr, y, out="full"):
    B = y.shape[0]
    n, I = m.nx * m.ny, m.n_inc

    def f(zr):
        chi = zr[:, :n]
        wr = zr[:, n:n + n * I].reshape(B, I, n)
        wi = zr[:, n + n * I:].reshape(B, I, n)
        c, ww = layer_fixed(m, k, alpha, gamma, use_corr, chi, wr + 1j * wi, y)
        if out == "chi":
            return c
        if out == "w":
            return torch.cat([ww.real.reshape(B, -1), ww.imag.reshape(B, -1)], dim=1)
        return torch.cat([c, ww.real.reshape(B, -1), ww.imag.reshape(B, -1)], dim=1)
    return f


def make_H(m, k, y):
    """The correction H_k ALONE (no physics step): its Jacobian is the pure
    Lipschitz constant the amplitude budget constrains."""
    B = y.shape[0]
    n, I = m.nx * m.ny, m.n_inc

    def f(zr):
        chi = zr[:, :n]
        wr = zr[:, n:n + n * I].reshape(B, I, n)
        wi = zr[:, n + n * I:].reshape(B, I, n)
        w = wr + 1j * wi
        w_t = (1 - ALPHA) * w + ALPHA * chi.unsqueeze(1) * (
            m.Ei.unsqueeze(0) + m._Gb_apply(w))
        E = m.Ei.unsqueeze(0) + m._Gb_apply(w_t)
        resid = m._Gs_apply(w_t) - y
        resid_dom = torch.matmul(m.Gs.conj().T,
                                 resid.reshape(-1, resid.shape[-1]).T).T.reshape(
            B, m.n_inc, -1)
        feat = torch.cat([
            resid_dom.real.reshape(B, -1), resid_dom.imag.reshape(B, -1),
            E.real.reshape(B, -1), E.imag.reshape(B, -1),
            w_t.real.reshape(B, -1), w_t.imag.reshape(B, -1), chi], dim=1)
        feat = feat.reshape(B, -1, m.nx, m.ny)
        feat = torch.tanh(feat / m.feat_scale) if m.featbound else feat / m.feat_scale
        return m.corrs[k](feat).flatten(1)
    return f


def flat_state(chi, w):
    return torch.cat([chi, w.real.reshape(chi.shape[0], -1),
                      w.imag.reshape(chi.shape[0], -1)], dim=1).detach()


def selfcheck_layer(m, chi, w, y, tol=1e-4):
    """Pin the AUDITED operator to the DEPLOYED one.

    layer_fixed() re-implements the layer so that (alpha,gamma) can be prescribed
    independently of the learned parameters; everything downstream in this driver is a
    Jacobian of THAT function.  If the two implementations ever diverge, this driver
    would certify a map that is not the map the network actually runs, and the numbers
    would still look plausible.  So we compare them on the same state at theta=1.

    alpha/gamma are set through the network's own parameterisation (alpha = a_max *
    sigmoid(a)); a[k]=g[k]=0 therefore gives alpha=a_max/2=0.25, gamma=g_max/2=0.30,
    which is the initialisation the trained runs start from -- reachable exactly, unlike
    the boundary value 0.5 that the sweep fixes.
    """
    k = 0
    with torch.no_grad():
        a_save, g_save, th_save = m.a.data.clone(), m.g.data.clone(), m.theta
        m.a.data[k] = 0.0
        m.g.data[k] = 0.0
        m.theta = 1.0
        al = float(m.a_max * torch.sigmoid(m.a[k]))
        ga = float(m.g_max * torch.sigmoid(m.g[k]))
        c_ref, w_ref = m.layer(k, chi, w, y)
        c_fix, w_fix = layer_fixed(m, k, al, ga, True, chi, w, y)
        m.a.data, m.g.data, m.theta = a_save, g_save, th_save
    d_chi = float((c_ref - c_fix).norm() / (c_ref.norm() + 1e-30))
    d_w = float((w_ref - w_fix).norm() / (w_ref.norm() + 1e-30))
    assert d_chi < tol and d_w < tol, (
        "layer_fixed diverges from UnrolledISP.layer: dchi=%.3e dw=%.3e" % (d_chi, d_w))
    return dict(alpha=al, gamma=ga, dchi_rel=d_chi, dw_rel=d_w)


def s_of(f, zr, inj=None, out=None, name=""):
    op = JacOp(f, zr, in_idx=inj, out_slice=out, name=name)
    s, nv, info = sigma_max_L(op, tol=TOL)
    sp, _ = sigma_max_power(op, iters=80)
    return s, sp, nv, info


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    # complex data must go to complex128; .to(torch.float64) would discard the field
    chi_te_d = chi_te.to(DEV).to(torch.float64)
    y_te_d = y_te.to(DEV).to(torch.complex128)
    DT = torch.float64
    assert y_te_d.is_complex(), 'incident field lost its imaginary part'
    n, I = NX * NY, N_INC
    out = {"instrument": "spec_tools: exact jvp/vjp + Lanczos (same as run_bjac)",
           "alpha_fixed": ALPHA, "gamma_fixed": GAMMA, "batch": BATCH,
           "device": DEV,
           "dtype": "float64 (complex128 fields)",
           "provenance": {
               "state": ("Born-LS initial state, flat_state(init_state(...)) on the first "
                         "BATCH test samples (chi_te seed 2, SNR 300 dB).  The post-rollout "
                         "state computed by c0, ww0 = mk(chi0, w0, y_b) is NOT the operating "
                         "point of A/B/C; it feeds only maxabs_chi_after_cycle."),
               "proj": "on (model default)",
               "step_sizes": "alpha = gamma = 0.5 forcibly fixed for every variant",
               "layer_index": "k = argmax over k = 0..7; each layer's own correction module",
               "checkpoint": "results/ckpt_constrained_K8.pt (and ckpt_unconstrained_K8.pt)",
               "driver_command": "python code/run_gain_decomp.py",
               "independent_cross_check": ("code/_gaindecomp_prov.py recomputes "
                                          "sigma_max_composite_max at both the init state and "
                                          "the post-cycle state and archives the pair in "
                                          "results/gaindecomp_prov.json"),
               "not_comparable_to": ("L45.json / the K=8 cyclic composite rows, which are at a "
                                     "different state and use the learned step sizes"),
           },
           "note": ("R2 m2.11: this driver previously ran in float32, which is why its "
                    "sigma_max_full disagreed with the float64 audit; the gap to the "
                    "converged-solve value was dominated by precision, not by n_cg")}

    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
        Gb, Gs, Ei)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
    chi_b, y_b = chi_te_d[:BATCH], y_te_d[:BATCH]
    chi0, w0 = init_state(m, y_b, BATCH)
    chi0 = chi0.to(DT)          # w0 stays complex128
    assert w0.is_complex(), 'float64 conversion lost the complex field state'
    zr = flat_state(chi0, w0)
    idx_chi = torch.arange(0, n)
    idx_w = torch.arange(n, n * (1 + 2 * I))

    # ---- L0) the audited operator must be the deployed operator ------------
    assert next(m.parameters()).dtype == torch.float64, "audit model is not float64"
    assert m.Gb.is_complex() and m.Gb.dtype == torch.complex128, "Gb is not complex128"
    out["L0_layer_equivalence"] = selfcheck_layer(m, chi_b, w0, y_b)
    log("=== L0) layer_fixed vs UnrolledISP.layer: dchi=%.2e dw=%.2e (alpha=%.3f gamma=%.3f) ==="
        % (out["L0_layer_equivalence"]["dchi_rel"], out["L0_layer_equivalence"]["dw_rel"],
           out["L0_layer_equivalence"]["alpha"], out["L0_layer_equivalence"]["gamma"]))

    # ---- A) physics core only, fixed alpha=gamma ---------------------------
    log("=== A) physics core only (alpha=gamma=%.2f, no correction) ===" % ALPHA)
    a_val, a_pow, a_nv, a_info = s_of(make_flatten(m, 0, ALPHA, GAMMA, False, y_b, "chi"),
                                      zr, inj=idx_chi, out=slice(0, n), name="A")
    d_val, d_pow, d_nv, d_info = s_of(make_flatten(m, 0, ALPHA, GAMMA, False, y_b, "w"),
                                      zr, inj=idx_w, out=slice(0, 2 * I * n), name="D")
    f_val, f_pow, f_nv, f_info = s_of(make_flatten(m, 0, ALPHA, GAMMA, False, y_b, "full"),
                                      zr, name="full")
    out["A_physics_core"] = {
        "alpha": ALPHA, "gamma": GAMMA,
        "gain_chi_block": a_val, "gain_w_block": d_val, "sigma_max_full": f_val,
        "gain_chi_block_power": a_pow, "gain_w_block_power": d_pow,
        "sigma_max_full_power": f_pow,
        "matvecs": {"A": a_nv, "D": d_nv, "full": f_nv},
        "solvers": {"A": a_info, "D": d_info, "full": f_info},
        "thm_field_upper_bound": 1.0 - ALPHA * (1.0 - 0.5 * float(np.linalg.norm(Gb, 2))),
        "beta_apriori": float(chi_tr.max().item() * np.linalg.norm(Gb, 2)),
        "beta_apriori_chimax_grid": float(0.5 * np.linalg.norm(Gb, 2)),
    }
    log("  A(chi-block)=%.6f (power %.6f)   D(w-block)=%.6f (power %.6f)   full=%.6f (power %.6f)"
        % (a_val, a_pow, d_val, d_pow, f_val, f_pow))
    log("  Thm field bound on D: %.6f  -> respected: %s"
        % (out["A_physics_core"]["thm_field_upper_bound"],
           d_val <= out["A_physics_core"]["thm_field_upper_bound"] + 1e-9))
    del m
    save("gain_decomp.json", out)

    # ---- B) correction alone + C) composite with correction ----------------
    for mode in ["constrained", "unconstrained"]:
        ck = os.path.join(RES, "ckpt_%s_K8.pt" % mode)
        if not os.path.exists(ck):
            log("  [skip] %s" % ck)
            continue
        log("=== B/C) %s (K=8 checkpoint) ===" % mode)
        mk = as_float64_model(
            UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV),
            Gb, Gs, Ei)
        mk.load_state_dict(torch.load(ck, map_location=DEV))
        mk.eval()
        with torch.no_grad():
            mk._cg_ls(mk.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
            c0, ww0 = mk(chi0, w0, y_b)
        zr = flat_state(chi0, w0)

        lip, lip_pow, comp, comp_pow, chi_upd = [], [], [], [], []
        for k in range(8):
            lv, lp, _, _ = s_of(make_H(mk, k, y_b), zr, name="H%d" % k)
            lip.append(lv)
            lip_pow.append(lp)
            cv, cp, _, _ = s_of(make_flatten(mk, k, ALPHA, GAMMA, True, y_b, "full"),
                                zr, name="comp%d" % k)
            comp.append(cv)
            comp_pow.append(cp)
            uv, _, _, _ = s_of(make_flatten(mk, k, ALPHA, GAMMA, True, y_b, "chi"),
                               zr, inj=idx_chi, out=slice(0, n), name="chiupd%d" % k)
            chi_upd.append(uv)
            log("   layer %d: Lip(H)=%.6f (power %.6f)  composite=%.6f (power %.6f)"
                % (k, lv, lp, cv, cp))
        alphas = (mk.a_max * torch.sigmoid(mk.a)).tolist()
        gammas = (mk.g_max * torch.sigmoid(mk.g)).tolist()
        out["B_%s" % mode] = {
            "Lip_H_pure_per_layer": lip,
            "Lip_H_pure_per_layer_power": lip_pow,
            "Lip_H_pure_max": float(max(lip)),
            "Lip_H_pure_max_power": float(max(lip_pow)),
            "eps_bar": mk.eps_bar,
            "alpha_learned": alphas, "gamma_learned": gammas,
            "maxabs_chi_after_cycle": float(c0.abs().max().item()),
        }
        out["C_%s" % mode] = {
            "sigma_max_composite_fixed_ag": comp,
            "sigma_max_composite_max": float(max(comp)),
            "sigma_max_composite_max_power": float(max(comp_pow)),
            "sigma_max_chi_update_fixed_ag": chi_upd,
        }
        log("  pure Lip(H_k) max = %.6f (power %.6f)  (eps_bar=%.3f, budget respected: %s)"
            % (max(lip), max(lip_pow), mk.eps_bar, max(lip) <= mk.eps_bar + 1e-9))
        log("  learned alpha=%s" % [round(a, 3) for a in alphas])
        log("  learned gamma=%s" % [round(g, 3) for g in gammas])
        del mk
        save("gain_decomp.json", out)

    log("[gain decomp done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
