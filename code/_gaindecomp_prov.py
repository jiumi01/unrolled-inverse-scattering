# -*- coding: utf-8 -*-
"""_gaindecomp_prov.py — provenance probe for the load-bearing composite gain (R1 t3).

Why this exists.  Round-3 R1 could not reproduce

    results/gain_decomp.json : C_constrained.sigma_max_composite_max = 13.0317

by any route: re-running the driver's own frozen functions at what he took to be the
operating point gave 11.75, and his own re-implementation gave 14.55.  He asked for the
*exact* operating point and the driving command.  The answer turns out to hinge on one
detail that the driver never recorded: the driver computes the post-rollout state

    c0, ww0 = mk(chi0, w0, y_b)          # computed ...
    zr = flat_state(chi0, w0)            # ... and then NOT used

so the archived numbers are evaluated at the **Born-LS initial state**, not at the state
after the K=8 cycle.  This probe tests exactly that hypothesis: it computes the same
quantity at both states with the driver's own functions and reports both.  If the
post-cycle state reproduces the reviewer's 11.75, the provenance is closed rather than
argued.

Device agreement is NOT established here (this file runs on one device per invocation, given
on the command line); the earlier version of this docstring promised a CPU/GPU cross-check that
the code did not perform, which is the same over-promise class this project keeps finding in
others' papers.  Backend sensitivity is instead measured where it actually matters, by
_tol_probe.py, which evaluates the two operator constructions on a tolerance grid in one process.
Writes results/gaindecomp_prov.json.  Usage: _gaindecomp_prov.py [cuda|cpu]
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_gain_decomp as RD          # noqa: E402  (module-level main() is guarded)
from run_experiments import load_data, NX, NY, N_INC   # noqa: E402
from spec_tools import as_float64_model                # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
CKPT = os.path.join(RES, "ckpt_constrained_K8.pt")
BATCH = RD.BATCH
ALPHA = GAMMA = RD.ALPHA


def composite_max(mk, zr, y_b, n, I):
    """max_k sigma_max of the fixed-(alpha,gamma) composite layer at state zr."""
    vals, per_layer = [], []
    for k in range(8):
        v, _, nv, info = RD.s_of(
            RD.make_flatten(mk, k, ALPHA, GAMMA, True, y_b, "full"), zr,
            name="prov_comp%d" % k)
        per_layer.append(v)
        vals.append(v)
    return float(max(vals)), per_layer


def _state_sensitivity():
    """Read the state-sensitivity measurements from _prov_state_probe.py's artefact.

    Hard-coding them here would recreate exactly the drift this project keeps finding: the
    value would be true on the day it was typed.  If the probe has not been run, record that
    fact rather than a number.
    """
    p = os.path.join(RES, "prov_state.json")
    if not os.path.exists(p):
        return {"source": "results/prov_state.json", "available": False,
                "note": "run code/_prov_state_probe.py to populate this"}
    d = json.load(open(p, encoding="utf-8"))
    sl = d.get("sigma_at_layer") or {}
    a = (sl.get("driver_state_K1_model") or {}).get("sigma")
    b = (sl.get("probe_state_K8_model") or {}).get("sigma")
    sd = d.get("state_diff") or {}
    return {
        "source": "results/prov_state.json",
        "available": True,
        "rel_state_diff_driver_vs_probe_model": sd.get("chi_rel"),
        "sigma_at_layer_driver_state": a,
        "sigma_at_layer_probe_state": b,
        "rel_sigma_change": (abs(a - b) / a) if (a and b) else None,
        "warning": ("mirroring the driver's state construction is not cosmetic: a ~1e-5 "
                    "relative state difference moves this quantity by ~9%, so the state "
                    "construction is load-bearing to five significant figures"),
    }


def main():
    if DEV == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[fatal] cuda requested but unavailable")
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d = chi_te.to(DEV).to(torch.float64)
    y_te_d = y_te.to(DEV).to(torch.complex128)
    n, I = NX * NY, N_INC

    mk = as_float64_model(
        RD.UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=0.03).to(DEV),
        Gb, Gs, Ei)
    mk.load_state_dict(torch.load(CKPT, map_location=DEV))
    mk.eval()
    chi_b, y_b = chi_te_d[:BATCH], y_te_d[:BATCH]

    # The state must be built EXACTLY as run_gain_decomp.py builds it: the driver takes it from
    # a *different* model instance -- K=1 with proj=True (its section A object) -- and only then
    # evaluates the composite with the K=8 corrector mk.  The first version of this probe called
    # init_state(mk) instead, which is a 1e-5 relative difference in the state and, because this
    # map amplifies (see below), a 9% difference in sigma_max: it reported 14.2373 against an
    # archive of 13.0317 and looked like a stale archive.  It was not.  Measured amplification:
    # results/prov_state.json, |dchi|/|chi| = 1.0e-5 -> |d sigma|/sigma = 9.2e-2 at this layer.
    m1 = as_float64_model(
        RD.UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
        Gb, Gs, Ei)
    with torch.no_grad():
        m1._cg_ls(m1.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
        chi0, w0 = RD.init_state(m1, y_b, BATCH)
        chi0 = chi0.to(torch.float64)          # w0 stays complex128, as in the driver
        mk._cg_ls(mk.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
        c_post, w_post = mk(chi0, w0, y_b)      # the state the driver computed
    if not (w0.is_complex() and w_post.is_complex()):
        raise SystemExit("[fatal] state lost its complex field (use complex128 moves)")

    zr_init = RD.flat_state(chi0, w0)
    zr_post = RD.flat_state(c_post, w_post)
    s_init, per_init = composite_max(mk, zr_init, y_b, n, I)
    s_post, per_post = composite_max(mk, zr_post, y_b, n, I)

    arch = json.load(open(os.path.join(RES, "gain_decomp.json"), encoding="utf-8"))
    arch_val = arch.get("C_constrained", {}).get("sigma_max_composite_max")

    d_init = abs(s_init - arch_val) if arch_val else None
    out = {
        "device": DEV,
        "checkpoint": os.path.relpath(CKPT, os.path.dirname(RES)),
        "checkpoint_md5": __import__("hashlib").md5(open(CKPT, "rb").read()).hexdigest(),
        "protocol": {
            "alpha": ALPHA, "gamma": GAMMA, "batch": BATCH,
            "dtype": "float64 / complex128 fields",
            "estimator": "spec_tools JacOp + Lanczos (sigma_max_L), same as run_gain_decomp",
            "layer_index": "max over k = 0..7, each layer's own correction module",
            "proj": True, "featbound": True, "eps_bar": 0.03,
            "state": "Born-LS init on the first 16 test samples (seed 2, SNR 300 dB)",
            "state_source": ("init_state() on a K=1, proj=True model instance -- NOT on the K=8 "
                             "corrector.  This mirrors run_gain_decomp.py exactly, which takes "
                             "the state from its section-A object and only the composite from "
                             "the K=8 checkpoint."),
        },
        # Measured by code/_prov_state_probe.py: calling init_state() on a different model
        # instance shifts the state by 1e-5 relative, which this map amplifies into a ~9%
        # shift of sigma_max.  Read from that probe's artefact rather than retyped, so the two
        # cannot drift apart.  Recorded so that the next person who tries to reproduce the
        # number knows the state construction is load-bearing to five significant figures.
        "state_sensitivity": _state_sensitivity(),
        "sigma_max_composite_max_init_state": s_init,
        "sigma_max_composite_max_post_cycle_state": s_post,
        "per_layer_init_state": per_init,
        "per_layer_post_cycle_state": per_post,
        "archived_value": arch_val,
        "abs_diff_init_vs_archived": d_init,
        "init_state_reproduces_archive": bool(d_init is not None and d_init < 1e-2),
        "post_cycle_state_explains_reviewer_value": bool(abs(s_post - 11.7526) < 2e-1),
        "driver_command": "python code/run_gain_decomp.py",
        "driver_provenance_gap": ("the driver computes the post-cycle state (c0, ww0) and then "
                                  "does not use it: zr = flat_state(chi0, w0) is the initial "
                                  "state.  Unrecorded, this made the archived number look "
                                  "unreproducible; recorded here."),
    }
    with open(os.path.join(RES, "gaindecomp_prov.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_layer_init_state",
                                                                "per_layer_post_cycle_state")},
                     indent=1))
    print("\nGAINDECOMP PROVENANCE PROBE DONE")


if __name__ == "__main__":
    main()
