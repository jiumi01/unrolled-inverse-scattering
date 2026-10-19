# -*- coding: utf-8 -*-
"""_ngc_audit.py — the contrast solve's iteration count is a CONFOUND in the layer spectrum.

Finding that motivates this file (probes _probe_fdvj*.py, all reproduced here):

  The layer map passes the contrast step through `_cg_ls`, a FIXED n_cg = 15 step CG solve.
  Its relative residual is 1.65e-3 -- not converged.  Consequences, measured:

    * block D (w+ = ... , no iterative solve) : analytic JVP vs central difference agree to
      2.7e-12.  The instrument is exact.
    * block A (through the CG solve)          : ||f(z+eps*V) - f(z-eps*V)|| stays at
      ~0.02-0.04 for EVERY eps in 1e-3 .. 1e-10, so the difference quotient does not
      converge: the executed map has no derivative at those scales.
    * converging the solve (n_cg -> 600) restores both: residual ~4e-10 and FD-vs-JVP
      agreement to 8.9e-4.

  Therefore sigma_max measured on the CG-path blocks at n_cg = 15 is a property of one
  under-converged arithmetic path, not of a well-defined linearisation.  Blocks C and D --
  which carry every CERTIFIED statement of the paper -- are exact and unaffected.

This driver reports the layer spectrum TWICE, at n_cg = 15 (as implemented) and at
n_cg = 200 (converged), plus
  * the CG relative residual at each setting, and
  * the FD-vs-JVP control on a CG-free block (D) and on a CG-path block (A),

so that the confound is visible in the record instead of being folded into a single number.

Writes results/ngc.json (float64, B = 16, one layer at alpha = gamma = 0.5).
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_grad_enabled(True)

from isp_core import UnrolledISP
from spec_tools import JacOp, sigma_max_L, as_float64_model
from run_bjac import make_f, flat_of, THETA, BATCH
from run_experiments import load_data, init_state, NX, NY, N_INC

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DT = torch.float64
N_CG_IMPL, N_CG_CONV = 15, 200
ALPHAS = (0.05, 0.10, 0.20, 0.317, 0.50, 1.00)
BLOCKS = (("A", "A"), ("B", "A"), ("C", "w"), ("D", "w"), ("full", "f"))


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(RES, "ngc_progress.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def cg_residual(m, E, y, n_cg):
    """relative residual of the regularised least-squares system, same A and b as _cg_ls."""
    with torch.no_grad():
        x = m._cg_ls(E, y, n_cg=n_cg)
        B, I, n = E.shape
        gsd = (torch.abs(m.Gs) ** 2).sum(dim=0)

        def apply_A(v):
            Ex = E * v.unsqueeze(1)
            mm = torch.matmul(m.Gs, Ex.reshape(-1, n).T).T.reshape(B, I, -1)
            Gm = torch.matmul(m.Gs.conj().T, mm.reshape(-1, mm.shape[-1]).T
                              ).T.reshape(B, I, n)
            return (E.conj() * Gm).real.sum(dim=1)

        b = (E.conj() * torch.matmul(
            m.Gs.conj().T, y.reshape(-1, y.shape[-1]).T).T.reshape(B, I, -1)).real.sum(dim=1)
        lam = m.cg_lam_rel * ((torch.abs(E) ** 2) * gsd).sum(dim=1) + 1e-12
        r = b - (apply_A(x) + lam * x)
        return float(r.norm() / (b.norm() + 1e-30))


def fd_control(op, eps=(1e-2, 1e-3, 1e-4)):
    """best (smallest) relative disagreement between the analytic JVP and a central
    difference, over the eps ladder -- 'best' because round-off grows as eps shrinks."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(op.Nin, generator=g, dtype=DT, device=op.dev)
    x = x / x.norm()
    V = op._to_states(x)
    Ja = op.jvp(x)
    na = float(Ja.norm())
    out = {"sigma_free_jvp_norm": na}
    best, best_eps = float("inf"), None
    for e in eps:
        with torch.no_grad():
            Df = (op._raw(op.zr + e * V).reshape(-1) - op._raw(op.zr - e * V).reshape(-1))
        nf = float(Df.norm())
        if nf < 1e-20:
            continue
        rel = float((Ja - Df / (2 * e)).norm() / (nf / (2 * e)))
        out["rel_eps_%.0e" % e] = rel
        out["dfnorm_eps_%.0e" % e] = nf
        if rel < best:
            best, best_eps = rel, e
    out["best_rel"] = best if np.isfinite(best) else None
    out["best_eps"] = best_eps
    return out


def measure(m, y_b, c0, w0, n_cg, eps_control=True):
    n, I = m.nx * m.ny, m.n_inc
    zr = flat_of(c0, w0)
    idx_chi = torch.arange(0, n)
    idx_w = torch.arange(n, n * (1 + 2 * I))
    funcs = {tag: make_f(m, 0.5, 0.5, THETA, y_b, False, which=w)
             for tag, w in (("A", "A"), ("w", "w"), ("f", "full"))}
    kw = {"A": dict(in_idx=idx_chi, out_slice=slice(0, n)),
          "B": dict(in_idx=idx_w, out_slice=slice(0, n)),
          "C": dict(in_idx=idx_chi, out_slice=slice(0, 2 * I * n)),
          "D": dict(in_idx=idx_w, out_slice=slice(0, 2 * I * n)),
          "full": dict()}
    res = {"n_cg": n_cg}
    for tag, fkey in BLOCKS:
        op = JacOp(funcs[fkey], zr, name=tag, **kw[tag])
        s, nv, info = sigma_max_L(op, tol=1e-9)
        res["sigma_%s" % tag] = float(s)
        res["matvecs_%s" % tag] = nv
        res["solver_%s" % tag] = info
        if eps_control and tag in ("A", "D"):
            res["fd_%s" % tag] = fd_control(op)
        log("    n_cg=%d  block %-4s sigma=%.8f  matvecs=%d  fd.best=%s"
            % (n_cg, tag, s, nv,
               ("%.2e" % res["fd_%s" % tag]["best_rel"]) if ("fd_%s" % tag) in res else "-"))
        del op
    with torch.no_grad():
        Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
        maxcol = float(Ew.abs().pow(2).sum(dim=1).sqrt().max().item())
    res["max_colnorm_E"] = maxcol
    res["exact_sigma_C"] = THETA * 0.5 * maxcol
    res["alpha_c"] = 1.0 / (THETA * maxcol)
    blk = max(res["sigma_%s" % t] for t in ("A", "B", "C", "D"))
    res["block_max"] = blk
    res["sigma_full_minus_block_max"] = res["sigma_full"] - blk
    res["consistent_full_ge_blocks"] = bool(res["sigma_full"] >= blk - 1e-9)
    return res


def alpha_sweep(m, y_b, c0, w0, n_cg):
    n = m.nx * m.ny
    zr = flat_of(c0, w0)
    rows = []
    for a in ALPHAS:
        f = make_f(m, a, a, THETA, y_b, False, which="full")
        op = JacOp(f, zr, name="a%.3f" % a)
        s, nv, info = sigma_max_L(op, tol=1e-7)
        with torch.no_grad():
            Ew = m.Ei.unsqueeze(0) + m._Gb_apply(w0)
            maxcol = float(Ew.abs().pow(2).sum(dim=1).sqrt().max().item())
        proved = a * THETA * maxcol
        rows.append(dict(alpha=a, sigma_full=float(s), matvecs=nv, solver=info,
                         n_cg=n_cg, rigorous_lower_bound=proved,
                         proved_noncontractive=bool(proved > 1.0),
                         measured_noncontractive=bool(s > 1.0)))
        log("    alpha=%.3f sigma=%.6f proved_lb=%.4f" % (a, s, proved))
        del op
    return rows


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

    out = {"config": dict(alpha=0.5, gamma=0.5, batch=BATCH, dtype="float64",
                          device="cpu",   # implicit (no .to(DEV) anywhere); pinned with the
                                          # other float64 audit drivers so every evaluation
                                          # of this operator sits on the same arithmetic
                          theta=THETA, n_cg_implemented=N_CG_IMPL,
                          n_cg_converged=N_CG_CONV,
                          note=("The contrast step is a fixed-iteration CG solve. At the "
                                "implemented n_cg=15 it is not converged (rel. residual "
                                "~1.6e-3) and the executed layer has no derivative at "
                                "sub-1e-2 scales, so the CG-path blocks (A, B, full) are "
                                "measured twice: as implemented and at a converged solve. "
                                "Blocks C and D carry every certified statement and are "
                                "exact (closed form) and CG-free."))}

    orig = m._cg_ls
    # CG residual at the operating point
    with torch.no_grad():
        w_t = (1 - 0.5) * w0 + 0.5 * c0.unsqueeze(1) * (m.Ei.unsqueeze(0) + m._Gb_apply(w0))
        E2 = m.Ei.unsqueeze(0) + m._Gb_apply(w_t)
    out["cg_residual"] = {str(nc): cg_residual(m, E2, y_b, nc)
                          for nc in (15, 50, 150, 200, 400)}
    log("=== CG relative residual of the contrast solve ===")
    for k, v in out["cg_residual"].items():
        log("    n_cg=%-4s ||b-Ax||/||b|| = %.4e" % (k, v))

    for n_cg in (N_CG_IMPL, N_CG_CONV):
        m._cg_ls = (lambda nc: (lambda E, y, **kw: orig(E, y, n_cg=nc)))(n_cg)
        log("=== L1 block spectrum, n_cg=%d ===" % n_cg)
        out["as_implemented" if n_cg == N_CG_IMPL else "converged"] = measure(
            m, y_b, c0, w0, n_cg)

    m._cg_ls = (lambda nc: (lambda E, y, **kw: orig(E, y, n_cg=nc)))(N_CG_CONV)
    log("=== alpha sweep at converged n_cg=%d ===" % N_CG_CONV)
    out["alpha_sweep_converged"] = alpha_sweep(m, y_b, c0, w0, N_CG_CONV)
    m._cg_ls = orig

    with open(os.path.join(RES, "ngc.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("[ngc done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
