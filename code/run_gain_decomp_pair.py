# -*- coding: utf-8 -*-
"""run_gain_decomp_pair.py -- M1.12 paired n_cg resolution measurement.

Reviewer M1.12 asked for a *resolution-matched* comparison: the physics-core
sigma_max(J) and the core+correction composite sigma_max, measured at BOTH the
implemented n_cg=15 and the converged solve, so the "correction lowers the gain"
direction cannot be an artefact of the CG truncation.

This driver reuses run_gain_decomp's OWN frozen instrument (JacOp = exact
torch.func.jvp/vjp + Lanczos, float64 audit model) at the SAME Born-LS initial
state / first-16 test samples / alpha=gamma=0.5 operating point, so the numbers
are internally consistent (no cross-driver mixing).

PROTOCOL (v2, after a confound was diagnosed in v1 -- see below):
the STATE is held fixed at the archived n_cg=15 Born-LS operating point (this is
the operating point of the deployed network, which uses n_cg=15 for the initial
solve as well), and ONLY the layer-internal contrast-solve resolution is varied
(n_cg=15 vs 200).  v1 varied both (init_state at the same ncg as the layers),
which confounded the state with the resolution: at n_cg=200 the Born-LS initial
state itself changes materially, so v1's n_cg=200 numbers (core 1.845 / composite
38.38) measured a DIFFERENT operating point, not a different resolution of the
same operator.  v1's records are preserved in results/log_gain_decomp_pair_full.txt
and are superseded, not deleted.

  python run_gain_decomp_pair.py selfcheck   # n_cg=15 only; ASSERTS reproduction of the
                                              # archived run_gain_decomp pair (same device,
                                              # same instrument, same state, same checkpoint)
  python run_gain_decomp_pair.py diag        # state-vs-resolution diagnostic (cheap)
  python run_gain_decomp_pair.py clean       # full: state fixed, layers n_cg in {15, 200}
Writes results/gain_decomp_pair.json.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_gain_decomp as R
from run_gain_decomp import (load_data, init_state, NX, NY, N_INC,
                             make_flatten, flat_state, s_of, ALPHA, GAMMA,
                             BATCH, as_float64_model, RES, log)
from isp_core import UnrolledISP

DEV = R.DEV


def set_ncg(m, ncg):
    """Shadow the instance _cg_ls so layer_fixed's m._cg_ls(E, y) uses ncg iters."""
    orig = type(m)._cg_ls.__get__(m)                 # true bound method (self bound)

    def patched(E, y, _o=orig, _n=ncg, _lr=1e-2):
        return _o(E, y, n_cg=_n, lam_rel=_lr)
    m._cg_ls = patched
    return orig


def build_audit(mode, K, Gb, Gs, Ei, ncg, load_ckpt=True):
    """K=1 physics-core model carries no learned weights (run_gain_decomp.py builds
    it the same way: the Born-LS init state and the fixed-step core operator are
    weight-free), so it is NOT loaded from the K=8 checkpoint; only the composite
    K=8 model is.  Loading would be a size mismatch by construction."""
    mk = as_float64_model(
        UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV),
        Gb, Gs, Ei)
    if load_ckpt:
        ck = os.path.join(RES, "ckpt_%s_K8.pt" % mode)
        mk.load_state_dict(torch.load(ck, map_location=DEV))
    mk.eval()
    set_ncg(mk, ncg)
    return mk


def born_ls_state(mode, Gb, Gs, Ei, y_b, ncg):
    """Born-LS initial state at resolution ncg (class-method call so the instance
    shadow cannot interfere): chi0 = clamp(cg(Ei, y), 0, chi_max), w0 = chi0 * Ei."""
    m = build_audit(mode, 1, Gb, Gs, Ei, ncg, load_ckpt=False)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
        Ei_e = m.Ei.unsqueeze(0).expand(y_b.shape[0], -1, -1)
        raw = UnrolledISP._cg_ls(m, Ei_e, y_b, n_cg=ncg)      # pre-clamp
        chi0 = raw.clamp(0.0, m.chi_max)
        w0 = chi0.unsqueeze(1) * m.Ei.unsqueeze(0)
    del m
    return raw, chi0, w0


def data_resid(m, chi, y):
    """||Gs diag(Ei) chi - y||_F per sample (data-fit term of the Born LS objective)."""
    with torch.no_grad():
        Ei_e = m.Ei.unsqueeze(0).expand(chi.shape[0], -1, -1)
        pred = m._Gs_apply(chi.unsqueeze(1) * Ei_e)
        return (pred - y).norm(dim=(1, 2))


def measure_fixed_state(mode, ncg, Gb, Gs, Ei, y_b):
    """Clean single-variable measurement: state fixed at the n_cg=15 Born-LS
    operating point (the deployed network's), layer-internal solve at ncg.
    Records BOTH estimators (Lanczos and power iteration) and the solver info:
    the n_cg=200 operator is ill-conditioned enough that eigsh can fail and fall
    back to a Rayleigh quotient (a lower bound), which must not be quoted alone."""
    _, chi0, w0 = born_ls_state(mode, Gb, Gs, Ei, y_b, 15)
    zr = flat_state(chi0, w0)
    m1 = build_audit(mode, 1, Gb, Gs, Ei, ncg, load_ckpt=False)
    core, core_pw, _, core_info = s_of(make_flatten(m1, 0, ALPHA, GAMMA, False, y_b, "full"),
                                       zr, name="A_core_full")
    log("  [state15/layers%d %s] core: lanczos=%.4f power=%.4f info=%s"
        % (ncg, mode, core, core_pw, core_info))
    del m1
    mk = build_audit(mode, 8, Gb, Gs, Ei, ncg, load_ckpt=True)
    comp, comp_l, comp_pw, comp_infos, comp_rel, comp_ok = [], [], [], [], [], []
    for k in range(8):
        cv, cp, _, ci = s_of(make_flatten(mk, k, ALPHA, GAMMA, True, y_b, "full"),
                             zr, name="comp%d" % k)
        hi = max(abs(cv), abs(cp))
        rel = abs(cv - cp) / hi if hi > 0 else 0.0
        ok = rel <= 1e-2
        comp.append(cv if ok else cp)
        comp_l.append(cv)
        comp_pw.append(cp)
        comp_rel.append(rel)
        comp_ok.append(bool(ok))
        comp_infos.append(ci)
        log("  [state15/layers%d %s] layer %d composite: lanczos=%.4f power=%.4f rel=%.1e%s"
            % (ncg, mode, k, cv, cp, rel, "" if ok else "  <-- SOLVER MISMATCH, power used"))
    del mk
    # The two estimators are BOTH archived and the reported value follows one fixed rule:
    # use Lanczos when it agrees with power to 1% (then the historical value is reproduced
    # exactly), else use power and say so.  Unconditional Lanczos reported sigma=0.0 for two
    # layers at n_cg=200 and a core value 10% low -- a failure that is visible only because
    # power is computed alongside.
    _chi = max(abs(core), abs(core_pw))
    core_rel = abs(core - core_pw) / _chi if _chi > 0 else 0.0
    core_ok = bool(core_rel <= 1e-2)
    return {"core_sigma_full": float(core if core_ok else core_pw),
            "core_sigma_lanczos": float(core), "core_sigma_power": float(core_pw),
            "core_sigma_rel": float(core_rel), "core_sigma_agrees": core_ok,
            "core_solver": core_info,
            "composite_per_layer": [float(x) for x in comp],
            "composite_lanczos_per_layer": [float(x) for x in comp_l],
            "composite_power_per_layer": [float(x) for x in comp_pw],
            "composite_rel_per_layer": [float(x) for x in comp_rel],
            "composite_agrees": comp_ok,
            "composite_solver": comp_infos,
            "composite_sigma_max": float(max(comp)),
            "composite_sigma_power_max": float(max(comp_pw)),
            "all_solvers_agree": bool(all(comp_ok) and core_ok)}


def diag(Gb, Gs, Ei, y_b):
    """Quantify the state-vs-resolution confound of the superseded v1 protocol."""
    m = build_audit("constrained", 1, Gb, Gs, Ei, 15, load_ckpt=False)
    out = {}
    for ncg in (15, 200):
        raw, chi0, w0 = born_ls_state("constrained", Gb, Gs, Ei, y_b, ncg)
        r = data_resid(m, raw, y_b)
        rc = data_resid(m, chi0, y_b)
        rec = {
            "raw_norm": float(raw.norm(dim=1).mean()),
            "raw_max": float(raw.max()),
            "raw_neg_frac": float((raw < 0).float().mean()),
            "clamped_frac": float((raw > m.chi_max).float().mean()),
            "chi0_norm": float(chi0.norm(dim=1).mean()),
            "chi0_max": float(chi0.max()),
            "chi0_support": float((chi0 > 1e-6).float().mean()),
            "w0_norm": float(w0.norm(dim=(1, 2)).mean()),
            "data_resid_raw": float(r.mean()),
            "data_resid_clamped": float(rc.mean()),
        }
        out[str(ncg)] = rec
        log("  [n_cg=%d] raw: norm=%.3f max=%.3f neg_frac=%.3f clamp_frac=%.3f | "
            "chi0: norm=%.3f max=%.3f support=%.3f | w0 norm=%.3f | resid raw=%.4f clamped=%.4f"
            % (ncg, rec["raw_norm"], rec["raw_max"], rec["raw_neg_frac"],
               rec["clamped_frac"], rec["chi0_norm"], rec["chi0_max"],
               rec["chi0_support"], rec["w0_norm"], rec["data_resid_raw"],
               rec["data_resid_clamped"]))
    del m
    # relative state change
    a, b = out["15"], out["200"]
    log("  [diag] ||chi0|| change: %+.1f%%   max|chi0|: %.3f -> %.3f   "
        "clamped pixels: %.1f%% -> %.1f%%   data resid (raw): %.4f -> %.4f (%+.1f%%)"
        % (100.0 * (b["chi0_norm"] / a["chi0_norm"] - 1.0), a["chi0_max"], b["chi0_max"],
           100 * a["clamped_frac"], 100 * b["clamped_frac"],
           a["data_resid_raw"], b["data_resid_raw"],
           100.0 * (b["data_resid_raw"] / a["data_resid_raw"] - 1.0)))
    return out


def run(ncgs, mode="constrained", fixed_state=True):
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    y_te_d = y_te.to(torch.complex128).to(DEV)
    y_b = y_te_d[:BATCH]
    log("[env] device=%s batch=%d alpha=gamma=%.2f fixed_state=%s"
        % (DEV, BATCH, ALPHA, fixed_state))
    out = {"mode": mode, "batch": BATCH, "alpha": ALPHA, "gamma": GAMMA,
           "protocol": ("v2: Born-LS initial state held at n_cg=15 (the deployed "
                        "operating point); only the layer-internal contrast-solve "
                        "resolution varies.  v1 (state and layers both at ncg) was "
                        "superseded: at n_cg=200 the Born-LS state itself changes, "
                        "so v1's n_cg=200 records measured a different operating "
                        "point (core 1.8451 / composite 38.3819); see "
                        "results/log_gain_decomp_pair_full.txt."),
           "instrument": "run_gain_decomp JacOp (exact jvp/vjp) + Lanczos, float64",
           "records": {}}
    for ncg in ncgs:
        log("=== n_cg=%d (state at 15) ===" % ncg)
        rec = measure_fixed_state(mode, ncg, Gb, Gs, Ei, y_b)
        out["records"][str(ncg)] = rec
        log("  [n_cg=%d] core=%.4f (power %.4f)  composite_max=%.4f (power %.4f)  "
            "(composite<core? %s / %s)"
            % (ncg, rec["core_sigma_full"], rec["core_sigma_power"],
               rec["composite_sigma_max"], rec["composite_sigma_power_max"],
               rec["composite_sigma_max"] < rec["core_sigma_full"],
               rec["composite_sigma_power_max"] < rec["core_sigma_power"]))
        with open(os.path.join(RES, "gain_decomp_pair.json"), "w") as f:
            json.dump(out, f, indent=1)
    log("[done] %.0fs -> results/gain_decomp_pair.json" % (time.time() - t0))


if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "constrained"
    cmd = sys.argv[1] if len(sys.argv) > 1 else "clean"
    if cmd == "selfcheck":
        run([15], mode)
        # The self-check used to be a docstring promise the code never kept: it only
        # re-measured and left the comparison to the operator's eyes.  It is now a real
        # assertion.  Both drivers run on the same device (CPU, pinned in
        # run_gain_decomp.py) with the same instrument, the same Born-LS state, the
        # same checkpoint and the same n_cg=15, so the two records must agree to
        # Lanczos tolerance; a failure here means the pair driver has drifted off the
        # gain_decomp operating point and its n_cg=200 numbers cannot be differenced
        # against the as-implemented ones.
        gd = json.load(open(os.path.join(RES, "gain_decomp.json")))
        rec = json.load(open(os.path.join(RES, "gain_decomp_pair.json")))["records"]["15"]
        ref_core = gd["A_physics_core"]["sigma_max_full"]
        ref_comp = gd["C_%s" % mode]["sigma_max_composite_max"]
        d_core = abs(rec["core_sigma_full"] - ref_core) / ref_core
        d_comp = abs(rec["composite_sigma_max"] - ref_comp) / ref_comp
        tol = 1e-6
        log("  [selfcheck] core %.6f vs archive %.6f (rel %.2e); "
            "composite %.6f vs archive %.6f (rel %.2e); tol %.0e"
            % (rec["core_sigma_full"], ref_core, d_core,
               rec["composite_sigma_max"], ref_comp, d_comp, tol))
        assert d_core < tol and d_comp < tol, (
            "pair driver does not reproduce the archived gain_decomp pair "
            "(device=%s): core rel %.2e, composite rel %.2e" % (DEV, d_core, d_comp))
        log("  [selfcheck] PASS")
    elif cmd == "diag":
        Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
        y_b = y_te.to(torch.complex128).to(DEV)[:BATCH]
        d = diag(Gb, Gs, Ei, y_b)
        with open(os.path.join(RES, "gain_decomp_pair_diag.json"), "w") as f:
            json.dump(d, f, indent=1)
        log("[diag] done -> results/gain_decomp_pair_diag.json")
    elif cmd == "conv":
        run([200], mode)          # converged resolution only (n_cg=15 is archived)
    else:
        run([15, 200], mode)
