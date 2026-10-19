# -*- coding: utf-8 -*-
"""_prov_state_probe.py -- is the provenance probe's initial state the driver's initial state?

Context.  results/gaindecomp_prov.json reports

    sigma_max_composite_max_init_state        = 14.2373
    archived (gain_decomp.json C_constrained) = 13.0317

i.e. the probe built to *close* R1 t3 does not reproduce the archived composite gain, although
it calls the driver's own layer_fixed()/s_of()/Lanczos path.  The per-layer values differ by an
almost constant +1.21 across all eight layers, which is the signature of a different operating
point rather than of a mis-indexed correction module.

The two scripts differ in exactly one visible respect:

    run_gain_decomp.py  :  chi0, w0 = init_state(m_K1, ...)   m_K1 = UnrolledISP(1, ..., proj=True)
    _gaindecomp_prov.py :  chi0, w0 = init_state(mk_K8, ...)  mk_K8 = UnrolledISP(8, ...)

init_state() reads only model.Ei, model._cg_ls and model.chi_max, none of which obviously depends
on K -- so this file measures it instead of arguing about it.  It builds BOTH states, reports the
relative difference, and evaluates the composite at ONE layer under each state: if the states are
equal and the layer values differ, the archived value is stale; if the states differ, the probe
(not the archive) was wrong.

Writes results/prov_state.json.  Usage: _prov_state_probe.py [cuda|cpu]
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_gain_decomp as RD            # noqa: E402
from run_experiments import load_data, init_state, NX, NY, N_INC   # noqa: E402
from spec_tools import as_float64_model                            # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
CKPT = os.path.join(RES, "ckpt_constrained_K8.pt")
BATCH = RD.BATCH
ALPHA = GAMMA = RD.ALPHA
LAYER = 3          # the layer that attained the max in both archives


def build(mode_k, **kw):
    return as_float64_model(
        RD.UnrolledISP(mode_k, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", **kw).to(DEV),
        Gb, Gs, Ei)


def main():
    global Gb, Gs, Ei
    if DEV == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[fatal] cuda requested but unavailable")
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d = chi_te.to(DEV).to(torch.float64)
    y_te_d = y_te.to(DEV).to(torch.complex128)
    chi_b, y_b = chi_te_d[:BATCH], y_te_d[:BATCH]

    # --- the driver's model: K=1, proj=True (run_gain_decomp.py, section A) ------------
    m1 = build(1, proj=True)
    with torch.no_grad():
        m1._cg_ls(m1.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
        chi_a, w_a = init_state(m1, y_b, BATCH)
        chi_a = chi_a.to(torch.float64)

    # --- the probe's model: K=8 constrained, checkpoint-loaded ------------------------
    m8 = build(8, eps_bar=0.03)
    m8.load_state_dict(torch.load(CKPT, map_location=DEV))
    m8.eval()
    with torch.no_grad():
        m8._cg_ls(m8.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
        chi_b8, w8 = init_state(m8, y_b, BATCH)
        chi_b8 = chi_b8.to(torch.float64)

    def rel(x, y):
        return float((x - y).norm() / (y.norm() + 1e-30))

    out = {
        "device": DEV,
        "layer": LAYER,
        "state_diff": {
            "chi_rel": rel(chi_a, chi_b8),
            "w_rel": rel(w_a, w8),
            "chi_max_abs_a": float(chi_a.abs().max()), "chi_max_abs_b": float(chi_b8.abs().max()),
            "proj_a": bool(getattr(m1, "proj", None)),
            "proj_b": bool(getattr(m8, "proj", None)),
            "chi_max_attr_a": float(getattr(m1, "chi_max", float("nan"))),
            "chi_max_attr_b": float(getattr(m8, "chi_max", float("nan"))),
            "featbound_a": bool(getattr(m1, "featbound", None)),
            "featbound_b": bool(getattr(m8, "featbound", None)),
            "ncorr_a": len(getattr(m1, "corrs", []) or []),
            "ncorr_b": len(getattr(m8, "corrs", []) or []),
        },
        "sigma_at_layer": {},
    }

    # --- evaluate the SAME composite layer under each state ----------------------------
    for tag, mk, zr in (("driver_state_K1_model", m8, RD.flat_state(chi_a, w_a)),
                        ("probe_state_K8_model", m8, RD.flat_state(chi_b8, w8))):
        v, pw, nv, info = RD.s_of(
            RD.make_flatten(mk, LAYER, ALPHA, GAMMA, True, y_b, "full"), zr,
            name="state_probe_%s" % tag)
        out["sigma_at_layer"][tag] = {"sigma": v, "power": pw, "matvecs": nv}
        print("  %-24s sigma=%.6f (power %.6f)" % (tag, v, pw), flush=True)

    arch = json.load(open(os.path.join(RES, "gain_decomp.json"), encoding="utf-8"))
    pv = json.load(open(os.path.join(RES, "gaindecomp_prov.json"), encoding="utf-8"))
    out["archive"] = {"gain_decomp_C_layer_value": arch["C_constrained"]
                      ["sigma_max_composite_fixed_ag"][LAYER],
                      "gain_decomp_C_max": arch["C_constrained"]["sigma_max_composite_max"],
                      "prov_init_layer_value": pv["per_layer_init_state"][LAYER],
                      "prov_init_max": pv["sigma_max_composite_max_init_state"]}
    same_state = out["state_diff"]["chi_rel"] < 1e-9 and out["state_diff"]["w_rel"] < 1e-9
    a = out["sigma_at_layer"]["driver_state_K1_model"]["sigma"]
    b = out["sigma_at_layer"]["probe_state_K8_model"]["sigma"]
    out["verdict"] = {
        "states_identical": bool(same_state),
        "sigma_agrees_across_states": bool(abs(a - b) <= 1e-6 * max(a, b)),
        "statement": (
            "the two init_state() calls give the SAME state and the SAME layer value: the probe "
            "was right and the archived C row is not reproducible by the current checkpoint"
            if same_state and abs(a - b) <= 1e-6 * max(a, b) else
            "the states DIFFER: the provenance probe (not the archive) used the wrong operating "
            "point and must be corrected before any provenance claim is made"
            if not same_state else
            "states identical but the two evaluations disagree: the difference is inside s_of, "
            "not the state -- escalate before writing anything"),
    }
    with open(os.path.join(RES, "prov_state.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
