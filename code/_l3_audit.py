# -*- coding: utf-8 -*-
"""_l3_audit.py — the Lambda (data-perturbation gain) section of the certification audit,
split out because the previous driver hung here.

Why it hung.  The chi <- y block is the sensitivity of the contrast step to the data, i.e.
d chi_hat / d y = A^{-1} d b / d y.  It is a CONTRAST-SOLVE path quantity, so by the same
mechanism documented for blocks A/B/full (see _ngc_audit.py) it is numerically non-smooth at
the implemented n_cg = 15, and ARPACK run to tol = 1e-9 on a non-converging operator does not
return.  Here both the Lanczos attempt and the power fallback are bounded (maxiter / fixed
iteration count), so the driver always terminates, and the value is computed at BOTH solve
settings.

What is and is not reportable:
  * gs_op_norm, lambda_min_A, lambda_max_A, max_abs_E, lambda_constructive_bound_partial are
    CG-free (they are built from Gs and from the diagonal of A) and are exact.
  * lambda_measured is the norm of the chi <- y block; it is reported at the converged solve
    as the primary value, with the implemented solve kept alongside.
All values are float64.
"""
import json, os, sys, time
import numpy as np
import torch
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_grad_enabled(True)

from isp_core import UnrolledISP
from spec_tools import JacOp, as_float64_model
from run_bjac import make_f, flat_of, THETA, BATCH
from run_experiments import load_data, init_state, NX, NY, N_INC

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DT = torch.float64
MAXITER = 400


def log(msg):
    print(msg, flush=True)


def bounded_sigma(op, seed=0, maxiter=MAXITER, tol=1e-6):
    """sigma_max with a hard iteration bound, so a non-smooth operator cannot hang the run."""
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(op.Nin)
    try:
        w = eigsh(op.MTM(), v0=v0, k=1, which="LM", tol=tol, maxiter=maxiter,
                  return_eigenvectors=False)
        return float(np.sqrt(max(float(np.atleast_1d(w).max()), 0.0))), "eigsh", op.matvec
    except Exception as e:
        it = 150
        X = torch.as_tensor(rng.standard_normal((op.B, op.m)), dtype=op.dtype, device=op.dev)
        X = X / (X.norm() + 1e-30)
        sig = 0.0
        for _ in range(it):
            Y = op.jvp(X.reshape(-1))
            nv = float(Y.norm())
            if nv < 1e-30:
                return 0.0, "zero(power)", op.matvec
            X = op.vjp((Y / nv).reshape(-1)).reshape(op.B, op.m)
            nu = float(X.norm())
            if nu < 1e-30:
                return 0.0, "zero(power)", op.matvec
            X = X / nu
            sig = nv
        return float(sig), "power(it=%d, eigsh=%s)" % (it, type(e).__name__), op.matvec


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    m = as_float64_model(
        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True), Gb, Gs, Ei)
    y_b = y_te[:BATCH].to(torch.complex128)
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_b[:1])
    c0, w0 = init_state(m, y_b, BATCH)
    c0 = c0.to(DT)
    assert m.Ei.is_complex() and w0.is_complex(), "float64 conversion lost the complex physics"

    n = NX * NY
    gs_norm = float(np.linalg.norm(Gs, 2))
    zr = flat_of(c0, w0)

    # ---- CG-free quantities --------------------------------------------------------
    with torch.no_grad():
        Ew0 = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
        gs_diag = (torch.abs(m.Gs) ** 2).sum(dim=0).to(DT)   # m.Gs is the complex128 tensor
        diagA = ((torch.abs(Ew0) ** 2) * gs_diag).sum(dim=1)
        lam = m.cg_lam_rel * diagA + 1e-12
        lam_min = float(lam.min().item())
        lam_max = float(lam.max().item())
        max_absE = float(Ew0.abs().max().item())
    lam_bound = 0.5 * gs_norm * max_absE / max(lam_min, 1e-30)
    log("gs_op_norm=%.6f  lam_min=%.3e  lam_max=%.3e  max|E|=%.4f" %
        (gs_norm, lam_min, lam_max, max_absE))
    log("lambda constructive bound (field part) = %.1f" % lam_bound)

    # ---- the ACTUAL data sensitivity d(chi+)/d y ------------------------------------
    # The previous version reused make_f(..., which="A") and therefore measured d chi+/d chi
    # again (its value equalled sigma_A exactly), which is not the Lambda of eq. (Lambda).
    # Here y is realified and used as the state, so the operator is genuinely d chi+ / d y.
    from run_bjac import core_parts
    nr = int(m.Gs.shape[0])
    I = int(m.n_inc)

    def make_gy(chi_f, w_f):
        B = chi_f.shape[0]

        def g(yr):
            yy = (yr[:, :I * nr].reshape(B, I, nr)
                  + 1j * yr[:, I * nr:].reshape(B, I, nr))
            c, _ = core_parts(m, flat_of(chi_f, w_f), 0.5, 0.5, THETA, yy,
                              False, 0, False)[:2]
            return c
        return g

    yzr = torch.cat([y_b.real.reshape(BATCH, -1), y_b.imag.reshape(BATCH, -1)], dim=1).detach()

    orig = m._cg_ls
    meas = {}
    for n_cg in (15, 200):
        m._cg_ls = (lambda nc: (lambda E, y, **kw: orig(E, y, n_cg=nc)))(n_cg)
        gy = make_gy(c0, w0)
        opy = JacOp(gy, yzr, name="chi<-y")
        sy, infoy, nvy = bounded_sigma(opy)
        log("  dchi+/dy    (n_cg=%d) = %.6f  [%s]  matvecs=%d  (Nin=%d)" %
            (n_cg, sy, infoy, nvy, opy.Nin))
        meas.setdefault("data", {})[str(n_cg)] = dict(value=sy, solver=infoy, matvecs=nvy)
        del opy, gy
    m._cg_ls = orig

    for n_cg in (15, 200):
        m._cg_ls = (lambda nc: (lambda E, y, **kw: orig(E, y, n_cg=nc)))(n_cg)
        fy = make_f(m, 0.5, 0.5, THETA, y_b, False, which="A")
        ky = JacOp(fy, zr, in_idx=torch.arange(0, n), out_slice=slice(0, n), name="chi<-y")
        s, info, nv = bounded_sigma(ky)
        meas[str(n_cg)] = dict(value=s, solver=info, matvecs=nv)
        log("  lambda_measured (n_cg=%d) = %.6f  [%s]  matvecs=%d" % (n_cg, s, info, nv))
        del ky
    m._cg_ls = orig

    path = os.path.join(RES, "bjac.json")
    bj = json.load(open(path))
    old = bj.get("L3_noise_gain", {})
    bj["L3_noise_gain"] = {
        "gs_op_norm": gs_norm, "lambda_min_A": lam_min, "lambda_max_A": lam_max,
        "max_abs_E": max_absE,
        "lambda_measured": meas["data"]["200"]["value"],
        "lambda_measured_data_ncg200": meas["data"]["200"]["value"],
        "lambda_measured_data_ncg15": meas["data"]["15"]["value"],
        "sigma_chi_block_ncg200": meas["200"]["value"],
        "sigma_chi_block_ncg15": meas["15"]["value"],
        "lambda_measured_float32": old.get("lambda_measured_float32", old.get("lambda_measured")),
        "lambda_measured_solver": meas["data"]["200"]["solver"],
        "lambda_measured_matvecs": meas["data"]["200"]["matvecs"],
        "lambda_measured_is_data_sensitivity": True,
        "deprecated_note": ("An earlier revision labelled the chi -> chi block as 'chi <- y'; "
                            "that value is now kept in sigma_chi_block_* and lambda_measured is "
                            "the genuine spectral norm of d(chi+)/d y."),
        "lambda_measured_maxiter_cap": MAXITER,
        "lambda_constructive_bound_partial": lam_bound,
        "bound_over_measured": lam_bound / max(meas["200"]["value"], 1e-30),
        "dtype": "float64",
        "note": ("lambda_measured is the spectral norm of d(chi+)/dy at alpha=gamma=0.5, quoted "
                 "at a CONVERGED contrast solve (n_cg=200) because it is a contrast-solve path "
                 "quantity; the n_cg=15 value is kept in lambda_measured_ncg15. "
                 "lambda_constructive_bound_partial = gamma*||Gs||_2*max_r|E|/lambda_min is the "
                 "field part of the constructive bound (the correction term "
                 "eps_bar*prod||W_i||/s_resid is not included). ARPACK is iteration-capped so a "
                 "non-smooth operator cannot hang the driver; the solver field records which "
                 "estimator answered."),
    }
    json.dump(bj, open(path, "w"), indent=1)
    log("[L3 done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
