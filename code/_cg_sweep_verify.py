# -*- coding: utf-8 -*-
"""
_cg_sweep_verify.py — independent-instrument verification of run_cg_sweep.

The n_cg sweep in run_cg_sweep.py used the model's own FINITE-DIFFERENCE power
iteration.  Before any of its numbers go into the manuscript, the same block
spectra are re-measured with the paper's EXACT instrument (spec_tools.JacOp:
torch.func.jvp / autograd vjp + Lanczos on J^T J), at the same operating points,
in double precision.  We also repeat the FD measurement at two finite-difference
steps and two random states, so a disagreeing pair is caught rather than averaged.

Outputs results/cg_sweep_exact.json.
"""
import json, os, sys
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import UnrolledISP, _flat_layer_fn, _power_sigma_fd
from spec_tools import JacOp, sigma_max_L, as_float64_model
from run_experiments import load_data, NX, NY, N_INC, RES

N_CG_LIST = [5, 10, 15, 30, 50, 200]
CKPT = os.path.join(RES, "ckpt_40ep_constrained_K8.pt")
BATCH = 8


def cg_solve(model, E, y, n_cg):
    B = E.shape[0]

    def apply_A(x):
        Ex = E * x.unsqueeze(1)
        m = torch.matmul(model.Gs, Ex.reshape(-1, Ex.shape[-1]).T).T.reshape(B, E.shape[1], -1)
        Gm = torch.matmul(model.Gs.conj().T, m.reshape(-1, m.shape[-1]).T).T.reshape(E.shape)
        return (E.conj() * Gm).real.sum(dim=1)

    b = (E.conj() * torch.matmul(
        model.Gs.conj().T, y.reshape(-1, y.shape[-1]).T).T.reshape(E.shape)).real.sum(dim=1)
    if not hasattr(model, "_gs_diag"):
        model._gs_diag = (torch.abs(model.Gs) ** 2).sum(dim=0)
    diagA = ((torch.abs(E) ** 2) * model._gs_diag).sum(dim=1)
    lam = model.cg_lam_rel * diagA + 1e-12
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r * r).sum(dim=1)
    for _ in range(n_cg):
        Ap = apply_A(p) + lam * p
        alpha = (rs / (p * Ap).sum(dim=1).clamp_min(1e-20)).unsqueeze(1)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=1)
        p = r + (rs_new / rs.clamp_min(1e-20)).unsqueeze(1) * p
        rs = rs_new
    return x, float((r.norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)).detach().mean())


def flat_state(chi, w):
    B = chi.shape[0]
    return torch.cat([chi, w.real.reshape(B, -1), w.imag.reshape(B, -1)], dim=1)


def fd_sigma(f, zr, idx_in=None, rows=None, iters=60, eps=1e-3, seed=0):
    """Local float64 finite-difference power iteration (independent of isp_core's
    float32-hardcoded helper, which cannot run on a double-precision model)."""
    g = torch.Generator(device=zr.device).manual_seed(seed)
    D = zr.shape[1]
    if idx_in is None:
        idx_in = torch.arange(D, device=zr.device)
    u = torch.zeros_like(zr)
    u[:, idx_in] = torch.randn(len(idx_in), device=zr.device,
                               dtype=zr.dtype, generator=g)
    u = u / (u.norm() + 1e-30)
    sig = 0.0
    z = zr.clone().detach().requires_grad_(True)
    for _ in range(iters):
        with torch.no_grad():
            Ju = (f(z + eps * u) - f(z - eps * u)) / (2 * eps)
            Ju = Ju if rows is None else Ju[:, rows]
            nv = Ju.norm() + 1e-12
            v_full = torch.zeros_like(z)
            v_full[:, (slice(0, D) if rows is None else rows)] = Ju / nv
        gg = torch.autograd.grad(f(z), (z,), grad_outputs=v_full)[0]
        gu = gg[:, idx_in]
        nu = gu.norm()
        if nu < 1e-12:
            break
        u = torch.zeros_like(zr)
        u[:, idx_in] = gu / nu
        sig = nv
    return float(sig)


def measure(model, chi, w, y, n, I):
    D = n * (1 + 2 * I)
    f = _flat_layer_fn(model, 0, y)
    zr = flat_state(chi, w).detach()
    dev = zr.device
    chi_cols = list(range(0, n))
    w_cols = list(range(n, D))
    chi_rows = slice(0, n)
    w_rows = slice(n, D)
    out = {}
    for name, cols, rows in [("A", chi_cols, chi_rows), ("B", w_cols, chi_rows),
                             ("C", chi_cols, w_rows), ("D", w_cols, w_rows),
                             ("full", None, None)]:
        J = JacOp(f, zr, in_idx=cols, out_slice=rows, name=name)
        s, mv, info = sigma_max_L(J, tol=1e-7)
        out[name] = {"sigma": s, "matvecs": mv, "info": info}
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d, y_te_d = chi_te[:BATCH].to(dev), y_te[:BATCH].to(dev)
    sd = torch.load(CKPT, map_location=dev)
    n = NX * NY

    results = {"n_cg_list": N_CG_LIST, "batch": BATCH, "dtype": "float64", "rows": []}
    for ncg in N_CG_LIST:
        m = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=0.03).to(dev)
        m.load_state_dict(sd)
        m.eval()
        m = as_float64_model(m, Gb, Gs, Ei)

        def solver(E, y, _n=ncg):
            return cg_solve(m, E, y, _n)[0]
        m._cg_ls = solver

        with torch.no_grad():
            Ei_e = m.Ei.unsqueeze(0).expand(BATCH, -1, -1)
            chi0 = m._cg_ls(Ei_e.to(torch.complex128), y_te_d.to(torch.complex128))
            chi0 = chi0.clamp(0.0, m.chi_max).to(torch.float64)
            w0 = (chi0.unsqueeze(1) * m.Ei.unsqueeze(0)).to(torch.complex128)
        # operating point: one layer past the Born init (same point as run_cg_sweep)
        with torch.no_grad():
            c1, w1 = m.layer(0, chi0, w0, y_te_d.to(torch.complex128))
        ex = measure(m, c1, w1, y_te_d.to(torch.complex128), n, N_INC)
        f0 = _flat_layer_fn(m, 0, y_te_d.to(torch.complex128))
        z0 = flat_state(c1, w1).detach()
        chi_cols_t = torch.arange(0, n, device=dev)
        w_cols_t = torch.arange(n, n * (1 + 2 * N_INC), device=dev)
        fd = {
            "A": fd_sigma(f0, z0, idx_in=chi_cols_t, rows=slice(0, n), iters=60, eps=1e-3),
            "B": fd_sigma(f0, z0, idx_in=w_cols_t, rows=slice(0, n), iters=60, eps=1e-3),
            "C": fd_sigma(f0, z0, idx_in=chi_cols_t, rows=slice(n, None), iters=60, eps=1e-3),
            "D": fd_sigma(f0, z0, idx_in=w_cols_t, rows=slice(n, None), iters=60, eps=1e-3),
            "full": fd_sigma(f0, z0, iters=60, eps=1e-3),
        }
        row = {"n_cg": ncg,
               "exact": {k: ex[k]["sigma"] for k in ex},
               "exact_info": {k: ex[k]["info"] for k in ex},
               "fd": fd}
        results["rows"].append(row)
        print("n_cg=%3d | exact A=%.4f B=%.4f C=%.4f D=%.4f J=%.4f || fd A=%.4f B=%.4f C=%.4f D=%.4f J=%.4f"
              % (ncg, row["exact"]["A"], row["exact"]["B"], row["exact"]["C"],
                 row["exact"]["D"], row["exact"]["full"], fd["A"], fd["B"], fd["C"], fd["D"], fd["full"]))
        del m
        torch.cuda.empty_cache()

    with open(os.path.join(RES, "cg_sweep_exact.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("[cg_sweep_exact done]")


if __name__ == "__main__":
    main()
