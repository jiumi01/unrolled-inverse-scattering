# -*- coding: utf-8 -*-
"""
run_grid_study.py — Grid refinement study.

Question addressed
------------------
The study uses a single discretisation (32x32 @ 0.02 lambda); this script recomputes
||Gb||2, beta0, the non-contractivity threshold and the measured Jacobian indices
at 32/64/128 grids.

Design (physically matched refinement)
--------------------------------------
The physical domain is held FIXED at L = 32 x 0.02 = 0.64 lambda.  The grid is
refined by shrinking the cell:  nx in {32,64,128}  ->  dx = L/nx.  The scatterer
generator (random_contrast) works in physical coordinates, so the targets are
geometrically identical across grids -- this isolates the *discretisation*, which
is what the certificate's grid-robustness claim is about.

Exactness
---------
On a uniform grid the MoM matrix Gb is 2-level Toeplitz, so its matvec is an exact
2-D linear convolution with the sampled Green kernel (with the self-term replaced
by the MoM g_self, which is exactly what build_matrices does on the diagonal).
We therefore get an *exact*, O(N log N) matvec at any resolution and validate it
against the dense matrix at nx=32 before trusting 64/128.

Reported per grid: ||Gb||2, ||Gs||2, beta0 = chi_max ||Gb||2, and -- via a
matrix-free forward solve (GMRES) -- the peak field column norm E and the
threshold alpha_c = (theta E)^-1.
"""
import json, os, sys, time
import numpy as np
from scipy.special import hankel2
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator, gmres

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import build_matrices, make_dirs, incident_fields, random_contrast, green_2d
from run_experiments import RES

L_PHYS = 32 * 0.02          # = 0.64 lambda, fixed physical side
GRIDS = [32, 64, 128]
N_RX = 32
N_INC = 8
CHI_MAX = 0.5
LAM = 1.0
KB = 2 * np.pi / LAM
TEST_TARGETS = 6


def kernel(nx, ny, dx):
    """Sampled Green kernel of the 2-level Toeplitz operator Gb, shape (2nx-1,2ny-1)."""
    dA = dx * dx
    a_eq = dx / np.sqrt(np.pi)
    g_self = -1j / 4.0 * (1 + (2j / np.pi) *
                          (np.log(KB * a_eq / 2) + 0.5772156649 - 1j * np.pi / 2))
    aa = np.arange(-(nx - 1), nx)
    bb = np.arange(-(ny - 1), ny)
    D = np.sqrt((aa[:, None] * dx) ** 2 + (bb[None, :] * dx) ** 2)
    K = KB ** 2 * dA * (-1j / 4.0 * hankel2(0, KB * D))
    K[nx - 1, ny - 1] = KB ** 2 * dA * g_self          # MoM self-term
    return K


def gb_apply(K, x2d, nx, ny, conj=False):
    Kc = np.conj(K) if conj else K
    full = fftconvolve(x2d, Kc, mode="full")
    return full[nx - 1:nx - 1 + nx, ny - 1:ny - 1 + ny]


def sigma_max_gb(K, nx, ny, iters=300, tol=1e-10, seed=0):
    """||Gb||2 by power iteration on Gb^H Gb (Gb complex-symmetric -> Gb^H=conj(Gb))."""
    rng = np.random.default_rng(seed)
    n = nx * ny
    u = rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))
    u = u.reshape(nx, ny) / np.linalg.norm(u.ravel())
    s_prev = 0.0
    for _ in range(iters):
        v = gb_apply(K, u, nx, ny)
        w = gb_apply(K, v, nx, ny, conj=True)      # Gb^H v
        nw = np.linalg.norm(w.ravel())
        if nw < 1e-300:
            break
        u = w / nw
        s = np.sqrt(nw)
        if abs(s - s_prev) <= tol * max(s, 1e-30):
            s_prev = s
            break
        s_prev = s
    v = gb_apply(K, u, nx, ny)
    return float(np.linalg.norm(v.ravel()))


def sigma_max_gs(Gs, n, iters=300, tol=1e-10, seed=0):
    rng = np.random.default_rng(seed + 5)
    u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    u /= np.linalg.norm(u)
    s_prev = 0.0
    for _ in range(iters):
        v = Gs @ u
        w = Gs.conj().T @ v
        nw = np.linalg.norm(w)
        if nw < 1e-300:
            break
        u = w / nw
        s = np.sqrt(nw)
        if abs(s - s_prev) <= tol * max(s, 1e-30):
            break
        s_prev = s
    return float(np.linalg.norm(Gs @ u))


def forward_solve(K, nx, ny, chi, Ei, tol=1e-10, maxiter=400):
    """Solve (I - diag(chi) Gb) E = Ei per incidence, matrix-free via FFT matvec."""
    n = nx * ny
    chi2 = chi.reshape(nx, ny)
    E = np.zeros_like(Ei)                      # (n_inc, n)
    for i in range(Ei.shape[0]):
        b = Ei[i].reshape(nx, ny)

        def mv(x):
            x2 = x.reshape(nx, ny)
            return (x2 - chi2 * gb_apply(K, x2, nx, ny)).ravel()
        A = LinearOperator((n, n), matvec=mv, dtype=complex)
        sol, info = gmres(A, b.ravel(), rtol=tol, maxiter=maxiter)
        E[i] = sol
    return E


def main():
    t0 = time.time()
    out = {"L_phys": L_PHYS, "n_rx": N_RX, "n_inc": N_INC, "chi_max": CHI_MAX,
           "grids": []}

    # ---- validation of the FFT matvec against the dense operator at nx=32 ----
    Gb32, Gs32, pts32, rx32 = build_matrices(32, 32, dx=0.02)
    K32 = kernel(32, 32, 0.02)
    rng = np.random.default_rng(3)
    xr = rng.standard_normal(32 * 32) + 1j * rng.standard_normal(32 * 32)
    dense = Gb32 @ xr
    fftv = gb_apply(K32, xr.reshape(32, 32), 32, 32).ravel()
    relerr = float(np.linalg.norm(dense - fftv) / np.linalg.norm(dense))
    out["fft_dense_relerr_nx32"] = relerr
    print("[validate] FFT vs dense matvec rel-err @ nx=32: %.3e" % relerr)
    assert relerr < 1e-9, "FFT matvec does not reproduce the dense operator"

    # ---- also cross-check ||Gb||2 dense vs power iteration at nx=32 ----
    dense_sig = float(np.linalg.norm(Gb32, 2))
    pi_sig = sigma_max_gb(K32, 32, 32)
    out["sigma_gb_nx32_dense"] = dense_sig
    out["sigma_gb_nx32_power"] = pi_sig
    print("[validate] ||Gb||2 @ nx=32  dense=%.6f  power=%.6f" % (dense_sig, pi_sig))

    dirs = make_dirs(N_INC)
    for nx in GRIDS:
        ny = nx
        dx = L_PHYS / nx
        n = nx * ny
        t1 = time.time()
        K = kernel(nx, ny, dx)

        # receiver ring and Gs at this grid
        x = (np.arange(nx) - (nx - 1) / 2) * dx
        y = (np.arange(ny) - (ny - 1) / 2) * dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        th = np.linspace(0, 2 * np.pi, N_RX, endpoint=False)
        rx = 2.0 * np.stack([np.cos(th), np.sin(th)], axis=1)
        Gs = KB ** 2 * dx * dx * green_2d(rx, pts, KB)
        Ei = incident_fields(pts, dirs, KB)

        sig_gb = sigma_max_gb(K, nx, ny)
        sig_gs = sigma_max_gs(Gs, n)
        beta0 = CHI_MAX * sig_gb

        # measured peak field column norm E and threshold alpha_c = (theta E)^-1
        rng2 = np.random.default_rng(1234)
        Epeak = 0.0
        ok = True
        try:
            for _ in range(TEST_TARGETS):
                chi = random_contrast(nx, ny, rng2, dx=dx, chi_max=CHI_MAX)
                E = forward_solve(K, nx, ny, chi, Ei)
                # per-cell L2 norm across incidence
                colnorm = np.sqrt((np.abs(E) ** 2).sum(axis=0))    # (n,)
                Epeak = max(Epeak, float(colnorm.max()))
        except Exception as e:                                     # keep the run alive
            ok = False
            Epeak = float("nan")
            print("   [warn] forward solve failed at nx=%d: %s" % (nx, e))
        alpha_c = 1.0 / Epeak if (ok and Epeak > 0) else float("nan")

        row = {"nx": nx, "ny": ny, "dx": dx, "n_cells": n,
               "sigma_gb": sig_gb, "sigma_gs": sig_gs, "beta0": beta0,
               "E_peak": Epeak, "alpha_c": alpha_c,
               "beta0_lt1": bool(beta0 < 1.0), "seconds": time.time() - t1}
        out["grids"].append(row)
        print("nx=%3d dx=%.4f  ||Gb||2=%.5f  ||Gs||2=%.5f  beta0=%.5f  "
              "E=%.5f  alpha_c=%.5f  (beta0<1: %s)  %.1fs"
              % (nx, dx, sig_gb, sig_gs, beta0, Epeak, alpha_c, beta0 < 1.0,
                 row["seconds"]))

    path = os.path.join(RES, "grid_study.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("[grid_study done] %s  %.0fs" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
