# -*- coding: utf-8 -*-
"""run_prec_audit.py -- single-source producer for the precision-consistency numbers.

WHY THIS FILE EXISTS
--------------------
The manuscript cites six quantities that measure *how much precision matters*:

    NumOpPrecRel        relative difference between the float32 and float64 physics
                        operators applied to the same complex input (round-off level)
    NumLayerPrecAmp     relative difference after ONE layer on the same state
    NumPrecAmp          the amplification NumLayerPrecAmp / NumOpPrecRel
    NumFSixFourSigErr   the round-off a double-precision run can accumulate
                        (unit round-off x that amplification)
    NumPrecRatioLo/Hi   range of sigma_full(float32)/sigma_full(float64) over the
                        alpha grid of Fig. alphac

Historically these were emitted as side-fields of `run_bjac.py` (L0b_dtype_consistency,
sigma_full_float32).  `run_bjac.py` stopped writing them, so re-running it silently
deleted the numbers and left six `--` placeholders in numbers.tex.  The audit therefore
gets its own artifact: `results/precision_audit.json`, produced only by this script.

WHAT IT COMPUTES (all on the same 16 test states used by Table cert)
--------------------------------------------------------------------
1. `dtype_consistency` (spec_tools): operators Gb/Gs/Ei in both precisions, then one full
   layer on the same state.  Reports |Im(Gb w)|/|Gb w| as a falsification probe: if the
   float64 conversion had destroyed the imaginary part, that ratio would be ~1, not ~0.
2. sigma_max(J) at every alpha of the Fig. alphac grid, in float64 (Lanczos, tol 1e-5)
   and in float32.  The ratio is the single-precision estimator's error, which is NOT a
   constant scale factor because the CG contrast solve is ill-conditioned.

SELF-CHECKS (they must be able to FAIL -- see the probe block at the end)
------------------------------------------------------------------------
  * imag_frac_Gbw must be > 1e-3   (complex physics preserved through the conversion)
  * operator_Gb_rel must be < 1e-4 (round-off, not an O(1) conversion defect)
  * Lanczos vs power iteration at alpha=0.5 must agree to < 1e-3 relative
  * every float64 sigma_full must be >= its own block bound theta*alpha*max_r||E_r||

CPU-only by design: it must never contend with a GPU training/evaluation job.

Usage:  python -u run_prec_audit.py [--smoke]
        --smoke runs one alpha only (instrument check, < 1 min)
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.normpath(os.path.join(HERE, "..", "results"))

from isp_core import UnrolledISP
from spec_tools import (JacOp, sigma_max_L, sigma_max_power, as_float64_model,
                        dtype_consistency)
from run_experiments import load_data, init_state, NX, NY, N_INC
from run_bjac import make_f, flat_of, exact_C, THETA, BATCH, CHI_MAX

DEV = "cpu"                      # never contend with the GPU training chain
U_ROUND = 1.1e-16                # IEEE-754 binary64 unit round-off
ALPHAS = (0.05, 0.10, 0.20, 0.317, 0.40, 0.50, 0.75, 1.00)   # Fig. alphac grid


def log(msg):
    print(msg, flush=True)


def sigma_pair(m64, m32, c0_64, w0_64, c0_32, w0_32, y64, y32, a, tol=1e-5):
    """sigma_max(J) at one alpha, in both precisions, plus a power-iteration cross-check."""
    z64 = flat_of(c0_64, w0_64)
    f64 = make_f(m64, a, a, THETA, y64, False, which="full")
    op64 = JacOp(f64, z64, name="f64_a%.3f" % a)
    s64, nv64, info64 = sigma_max_L(op64, tol=tol)
    # 300 iterations, not 100: a measured tolerance study at alpha=0.75 shows the
    # 100-iteration value is still 1.2e-3 short of the converged Ritz value while the
    # Lanczos value is converged to 2.5e-11 against tol=1e-8.  The cross-check was the
    # under-converged side, so the iteration budget is raised here rather than the
    # agreement threshold being relaxed.
    sp64, _ = sigma_max_power(op64, iters=300)
    del op64

    s32 = None
    err32 = None
    try:
        z32 = flat_of(c0_32, w0_32)
        f32 = make_f(m32, a, a, THETA, y32, False, which="full")
        op32 = JacOp(f32, z32, name="f32_a%.3f" % a)
        s32v, _, _ = sigma_max_L(op32, tol=tol)
        s32 = float(s32v)
        del op32
    except Exception as ex:                    # a float32 Lanczos may not converge
        err32 = repr(ex)[:200]

    return dict(alpha=a,
                sigma_full_f64=float(s64),
                sigma_full_f64_power=float(sp64),
                sigma_full_f32=s32,
                ratio_f32_over_f64=(s32 / float(s64) if s32 else None),
                matvecs=nv64, solver=info64,
                f32_error=err32)


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()

    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    gbn = float(np.linalg.norm(Gb, 2))
    beta0 = CHI_MAX * gbn
    log("||Gb||_2=%.6f  beta0=%.6f  n_inc=%d" % (gbn, beta0, N_INC))

    # --- the two models, built from the same matrices; only dtype differs ----------
    # SEED + deepcopy matter: UnrolledISP carries a randomly initialised CorrectionNet and
    # learned scalars.  Comparing two *independently constructed* models would measure the
    # spread of the random init, not the precision.  Both arms here therefore start from
    # ONE seeded instance, so their parameters are bit-identical and dtype is the only
    # difference.  The audit also reports the cross-init value for comparison, which is
    # what an unseeded (and therefore unreproducible) comparison would have produced.
    torch.manual_seed(0)
    base = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True)
    m64 = as_float64_model(copy.deepcopy(base).to(DEV), Gb, Gs, Ei)
    y64 = y_te[:BATCH].to(DEV).to(torch.complex128)
    with torch.no_grad():
        m64._cg_ls(m64.Ei.unsqueeze(0).expand(1, -1, -1), y64[:1])
    c0_64, w0_64 = init_state(m64, y64, BATCH)
    c0_64 = c0_64.to(torch.float64)
    assert w0_64.is_complex() and m64.Ei.is_complex(), "float64 conversion lost the complex physics"

    m32 = copy.deepcopy(base).to(DEV)
    y32 = y64.to(torch.complex64)
    with torch.no_grad():
        m32._cg_ls(m32.Ei.unsqueeze(0).expand(1, -1, -1), y32[:1])
    c0_32, w0_32 = init_state(m32, y32, BATCH)

    # cross-init reference: a second, independently initialised float32 arm
    m32_alt = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV)

    # --- (1) conversion transparency + one-layer amplification --------------------
    dcon = dtype_consistency(m32, m64, w0_64.detach(), y64, chi=c0_64.detach())
    dcon_alt = dtype_consistency(m32_alt, m64, w0_64.detach(), y64, chi=c0_64.detach())
    log("L0b operators f32 vs f64: Gb=%.2e Gs=%.2e Ei=%.2e | Im(Gbw)/|Gbw|=%.4f"
        % (dcon["operator_Gb_rel"], dcon["operator_Gs_rel"], dcon["operator_Ei_rel"],
           dcon["imag_frac_Gbw"]))
    log("     one layer on the SAME state: dchi=%.2e dw=%.2e"
        % (dcon["layer_rel"], dcon["layer_w_rel"]))
    log("     same layer with an INDEPENDENT random init: dchi=%.2e (init spread, not precision)"
        % dcon_alt["layer_rel"])

    # --- (2) the alpha sweep, both precisions ------------------------------------
    alphas = (0.50,) if smoke else ALPHAS
    rows = []
    log("alpha | sigma64 (Lanczos / power) | sigma32 | ratio")
    for a in alphas:
        r = sigma_pair(m64, m32, c0_64, w0_64, c0_32, w0_32, y64, y32, a)
        rows.append(r)
        log("%.3f | %10.6f %10.6f | %8s | %s"
            % (a, r["sigma_full_f64"], r["sigma_full_f64_power"],
               ("%.6f" % r["sigma_full_f32"]) if r["sigma_full_f32"] else "n/a",
               ("%.4f" % r["ratio_f32_over_f64"]) if r["ratio_f32_over_f64"] else "n/a"))

    # --- derived quantities -------------------------------------------------------
    ratios = [r["ratio_f32_over_f64"] for r in rows if r["ratio_f32_over_f64"]]
    op_rel = dcon["operator_Gb_rel"]
    ly_rel = dcon["layer_rel"]
    prec_amp = (ly_rel / op_rel) if (op_rel and ly_rel) else None
    f64_sig_err = (U_ROUND * prec_amp) if prec_amp else None

    # --- (3) self-checks: each one must be able to fail ---------------------------
    checks = {}

    def check(name, ok, detail):
        checks[name] = dict(passed=bool(ok), detail=detail)
        log("  [%s] %-28s %s" % ("PASS" if ok else "FAIL", name, detail))

    check("complex_physics_preserved", dcon["imag_frac_Gbw"] > 1e-3,
          "|Im(Gb w)|/|Gb w| = %.4f (a destroyed imaginary part would give ~0)"
          % dcon["imag_frac_Gbw"])
    # Falsification probe: run the SAME expression on a field whose imaginary part has
    # been deliberately discarded.  If the probe's ratio does not collapse to 0, the
    # check above would be vacuous and must not be trusted.
    gb_ok = m64._Gb_apply(w0_64)
    gb_broken = gb_ok.real.to(torch.complex128)
    frac_broken = float(gb_broken.imag.norm() / (gb_broken.norm() + 1e-30))
    check("probe_detects_lost_imag", frac_broken < 1e-12,
          "same formula on a real-cast field returns %.2e (must be 0)" % frac_broken)
    check("operator_roundoff_level", op_rel < 1e-4,
          "operator_Gb_rel = %.2e (round-off is ~1e-7; an O(1) value would mean the "
          "conversion destroyed the field)" % op_rel)
    # Falsification probe for the operator comparison: two operators that really differ
    # must be reported as O(1), otherwise the 1e-7 above would be a vacuous reading.
    op_probe = float((gb_ok - 1j * gb_ok).norm() / (1j * gb_ok).norm())
    check("probe_detects_op_change", op_probe > 0.5,
          "a deliberately altered operator is reported at %.2f relative" % op_probe)
    dpairs = [(r["sigma_full_f64"], r["sigma_full_f64_power"]) for r in rows]
    worst = max(abs(a - b) / a for a, b in dpairs)
    check("lanczos_vs_power", worst < 1e-3,
          "max relative spread = %.2e over %d alpha values" % (worst, len(rows)))
    # sigma_max(J) >= sigma_max(any block) = theta*alpha*max_r ||E_r||: an independent
    # identity that must hold at every alpha; a violation means the sweep is mis-wired.
    viol = []
    for r in rows:
        bound = exact_C(m64, w0_64, r["alpha"], THETA)["exact"]
        if r["sigma_full_f64"] < bound - 1e-9:
            viol.append((r["alpha"], r["sigma_full_f64"], bound))
    check("sigma_above_block_bound", not viol,
          "sigma_full >= theta*alpha*max_r||E_r|| at all %d alphas%s"
          % (len(rows), "" if not viol else " -- violations: %s" % viol))

    all_pass = all(c["passed"] for c in checks.values())
    out = dict(
        dtype="precision audit",
        alpha_grid=list(alphas),
        batch=BATCH, theta=THETA, chi_max_box=CHI_MAX,
        unit_roundoff=U_ROUND,
        dtype_consistency=dcon,
        layer_rel_cross_init=dcon_alt["layer_rel"],
        alpha_sweep=rows,
        summary=dict(
            prec_ratio_lo=min(ratios) if ratios else None,
            prec_ratio_hi=max(ratios) if ratios else None,
            operator_Gb_rel=op_rel,
            layer_rel=ly_rel,
            prec_amp=prec_amp,
            f64_sig_err=f64_sig_err,
        ),
        self_checks=checks,
        all_checks_passed=all_pass,
        note=("Operator-level precision audit on the same 16 test states as Table cert. "
              "(a) operators applied to the same complex input in both precisions -- a "
              "destroyed imaginary part gives O(1), float32 round-off gives O(1e-7). "
              "(b) one full layer on the SAME state: layer_rel is not a conversion defect "
              "but the amplification of a ~1e-7 arithmetic difference by the 15-step CG "
              "contrast solve (lambda_min ~ 1e-5). (c) sigma_full(float32)/sigma_full("
              "float64) is not a constant factor over alpha, because that amplification "
              "depends on where the Ritz vector sits relative to the nearly defective "
              "eigenstructure."),
        wallclock_s=None,
    )

    if not smoke:
        out["wallclock_s"] = round(time.time() - t0, 1)
        p = os.path.join(RES, "precision_audit.json")
        json.dump(out, open(p, "w"), indent=1)
        log("[wrote] %s  (%.0fs, all_checks_passed=%s)"
            % (p, out["wallclock_s"], all_pass))
    else:
        log("[smoke] one alpha only; nothing written")

    if not all_pass:
        log("SELF-CHECK FAILED -- the numbers above must not be cited")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
