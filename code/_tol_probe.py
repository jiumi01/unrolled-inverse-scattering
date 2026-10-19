# -*- coding: utf-8 -*-
"""_tol_probe.py -- what actually separates the two archived physics-core gains (R2 Q3.1b).

R2 asked us to confirm that the 1.2% gap between

    results/gain_decomp.json : A_physics_core.sigma_max_full = 14.4186   (run_gain_decomp)
    results/bjac.json        : L1_physics_core.sigma_full    = 14.5936   (run_bjac)

comes *only* from the Lanczos stopping criterion, and not from some other difference in setup.
That is a claim about a cause, so it needs a measurement, not an assertion.  Reading the two
drivers shows the same operating point (Born-LS init state, batch 16, alpha=gamma=0.5 forced,
float64, no correction, K=1 constrained model with proj on) but TWO differences:

  (i)  tolerance: run_gain_decomp uses TOL = 1e-5; run_bjac uses SIGTOL = 1e-6;
  (ii) operator construction: run_gain_decomp's layer_fixed() vs run_bjac's make_f() -- two
       independent re-implementations of the same prescribed-step-size layer.

This probe separates them by evaluating BOTH operators at BOTH tolerances (and one tighter), on
one state, in one process:

  * if operator A at 1e-6 rises to ~14.59 -> the gap is the tolerance;
  * if operator B at 1e-5 stays at ~14.59 -> the gap is the operator construction;
  * if neither moves -> neither explanation holds and the provenance claim must be withdrawn
    rather than argued (the honest failure mode).

Writes results/tol_probe.json.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_bjac as RB            # noqa: E402
import run_gain_decomp as RD     # noqa: E402
from run_experiments import load_data, init_state, NX, NY, N_INC   # noqa: E402
from spec_tools import as_float64_model, sigma_max_L               # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"
BATCH = RD.BATCH
ALPHA = GAMMA = RD.ALPHA
TOLS = [1e-5, 1e-6, 1e-8]


def main():
    if DEV == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[fatal] cuda requested but unavailable")
    torch.set_grad_enabled(True)
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d = chi_te.to(DEV).to(torch.float64)
    y_te_d = y_te.to(DEV).to(torch.complex128)

    m = as_float64_model(
        RD.UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
        Gb, Gs, Ei)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
    chi_b, y_b = chi_te_d[:BATCH], y_te_d[:BATCH]
    chi0, w0 = init_state(m, y_b, BATCH)
    zr = RD.flat_state(chi0, w0)

    # operator A: run_gain_decomp's construction (the one that produced 14.4186)
    opA = RD.make_flatten(m, 0, ALPHA, GAMMA, False, y_b, "full")
    # operator B: run_bjac's construction (the one that produced 14.5936)
    opB = RB.make_f(m, ALPHA, GAMMA, RB.THETA, y_b, False, which="full")
    # cross-check on one vector: the two constructions must agree to numerical precision,
    # otherwise the tolerance question is moot and this becomes an operator discrepancy.
    with torch.no_grad():
        va, vb = opA(zr), opB(zr)
        op_rel = float((va - vb).norm() / (va.norm() + 1e-30))

    rows = {}
    for tag, f in (("gain_decomp_operator", opA), ("bjac_operator", opB)):
        rows[tag] = {}
        for tol in TOLS:
            s, nv, info = sigma_max_L(RD.JacOp(f, zr, name="%s_tol%.0e" % (tag, tol)), tol=tol)
            rows[tag]["tol_%.0e" % tol] = {"sigma": s, "matvecs": nv, "solver": info}
            print("  %-22s tol=%.0e  sigma=%.6f  matvecs=%d" % (tag, tol, s, nv), flush=True)

    gd = json.load(open(os.path.join(RES, "gain_decomp.json"), encoding="utf-8"))
    bj = json.load(open(os.path.join(RES, "bjac.json"), encoding="utf-8"))
    arch_gd = gd.get("A_physics_core", {}).get("sigma_max_full")
    arch_bj = bj.get("L1_physics_core", {}).get("sigma_full")

    a5 = rows["gain_decomp_operator"]["tol_1e-05"]["sigma"]
    a6 = rows["gain_decomp_operator"]["tol_1e-06"]["sigma"]
    b5 = rows["bjac_operator"]["tol_1e-05"]["sigma"]
    b6 = rows["bjac_operator"]["tol_1e-06"]["sigma"]

    # Symmetric, falsifiable attribution.  The archived gap is
    #   A@1e-5 (run_gain_decomp, 14.4186)  vs  B@1e-6 (run_bjac, 14.5936).
    # Two candidate causes, each with its OWN diagnostic measured at matched conditions:
    #   * operator: |A - B| at a MATCHED tolerance (1e-5 and 1e-6) -- if the constructions
    #     differ, this is non-zero at every tolerance;
    #   * tolerance: |X@1e-6 - X@1e-5| for each X -- if the stopping rule moves the estimate,
    #     this is non-zero for whichever operator is sensitive.
    # The previous version of this probe tested A@1e-5 against B@1e-5 in one branch and
    # A@1e-6 against B@1e-6 in another but never *divided* by the archived gap, so a 0.5%
    # threshold could pass on a 1.2% effect.  Instruments first.
    def rel(x, y):
        return abs(x - y) / y if y else None

    op_at_matched = {  # is the operator the cause? compare at the SAME tolerance
        "tol_1e-05": rel(a5, b5),
        "tol_1e-06": rel(a6, b6),
    }
    tol_sens = {       # is the stopping rule the cause? compare the SAME operator
        "gain_decomp_operator": rel(a6, a5),
        "bjac_operator": rel(b6, b5),
    }
    arch_gap = rel(arch_bj, arch_gd) if (arch_gd and arch_bj) else None
    half = 0.5 * arch_gap if arch_gap else None

    max_op = max(v for v in op_at_matched.values() if v is not None)
    max_tol = max(v for v in tol_sens.values() if v is not None)
    op_is = bool(half is not None and max_op > half and max_op > 0.005)
    tol_is = bool(half is not None and max_tol > half and max_op < 0.005)
    # The third outcome is not "inconclusive" but a result in its own right: if BOTH diagnostics
    # are at machine zero while the operators still reproduce one archive value and not the
    # other, then both candidate causes are FALSIFIED and the remaining explanation is a
    # difference in the state/model fed to the archived driver.  (Measured: 14.418558 for both
    # constructions at 1e-5, 1e-6 and 1e-8, i.e. exactly the run_gain_decomp archive value, while
    # run_bjac's archive says 14.5936.)  Reporting that as "unresolved" would be weaker than the
    # evidence allows.
    both_excluded = (max_op <= 1e-9 and max_tol <= 1e-9)
    if both_excluded:
        statement = ("BOTH candidate causes are excluded. The two operator constructions agree "
                     "to %.1e at matched tolerance, and neither moves when the stopping "
                     "tolerance is tightened by three orders of magnitude, yet both reproduce the "
                     "run_gain_decomp archive value (%.6f) and neither reproduces the run_bjac "
                     "archive value (%.6f, off by %.4f). The 1.2%% inter-driver gap is therefore "
                     "NOT the Lanczos stopping criterion and NOT the operator construction; the "
                     "attribution to the stopping criterion is WITHDRAWN, and the difference must "
                     "be a different operating point or model in the archived bjac run."
                     % (max_op, a5, arch_bj if arch_bj else float("nan"),
                        abs(b6 - arch_bj) if arch_bj else float("nan")))
    elif op_is:
        statement = ("the two constructions disagree by %.3f%% at MATCHED tolerance, i.e. the "
                     "archived 1.2%% gap is at least partly a property of the operator "
                     "construction, not only of the Lanczos stopping criterion -- the provenance "
                     "claim must be narrowed" % (100 * max_op))
    elif tol_is:
        statement = ("the constructions coincide at matched tolerance (<=0.5%%) while the "
                     "stopping rule moves the estimate by %.3f%%, i.e. the 1.2%% gap is the "
                     "Lanczos stopping criterion" % (100 * max_tol))
    else:
        statement = ("neither diagnostic exceeds half the archived gap (operator %.4f%%, "
                     "tolerance %.4f%% vs gap %.4f%%): the attribution is UNRESOLVED and must be "
                     "reported as such rather than argued"
                     % (100 * max_op, 100 * max_tol, 100 * (arch_gap or 0)))
    out = {
        "device": DEV,
        "protocol": {"state": "Born-LS init state, batch 16", "alpha": ALPHA, "gamma": GAMMA,
                     "theta": RB.THETA, "dtype": "float64 / complex128", "proj": True,
                     "use_corr": False, "model": "UnrolledISP(1, constrained, proj=True)"},
        "grid": rows,
        "operator_agreement_rel": op_rel,
        "archived": {"gain_decomp_A_physics_core_sigma_max_full": arch_gd,
                     "bjac_L1_physics_core_sigma_full": arch_bj,
                     "cross_driver_rel_gap": arch_gap,
                     "cross_driver_rel_gap_pct": (100 * arch_gap) if arch_gap else None},
        "diagnostics": {"operator_at_matched_tolerance": op_at_matched,
                        "tolerance_sensitivity_of_each_operator": tol_sens},
        "reproduces_archive": {
            "gain_decomp_operator_at_1e-05": a5,
            "bjac_operator_at_1e-06": b6,
            "abs_diff_vs_archive_gain_decomp": abs(a5 - arch_gd) if arch_gd else None,
            "abs_diff_vs_archive_bjac": abs(b6 - arch_bj) if arch_bj else None},
        "verdict": {
            "tolerance_is_the_cause": tol_is,
            "operator_construction_is_the_cause": op_is,
            "both_candidate_causes_excluded": bool(both_excluded),
            "statement": statement,
        },
        "driver_commands": ["python code/run_gain_decomp.py", "python code/run_bjac.py",
                            "python code/_tol_probe.py %s" % DEV],
    }
    with open(os.path.join(RES, "tol_probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["reproduces_archive"], indent=1))
    print(json.dumps(out["diagnostics"], indent=1))
    print(json.dumps(out["verdict"], indent=1))
    print("\nTOL PROBE DONE (one-vector operator agreement %.2e)" % op_rel)


if __name__ == "__main__":
    main()
