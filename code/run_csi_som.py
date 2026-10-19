# -*- coding: utf-8 -*-
"""
run_csi_som.py — Classical baselines: CSI and SOM.

Two STANDARD, stronger classical inverse-scattering baselines alongside the paper's
damped alternating iteration:
  * CSI  — contrast source inversion (van den Berg & Kleinman, IEEE TGRS 1997)
  * SOM  — subspace-based optimization method (Chen, PIER 2010)

Conventions follow UnrolledISP exactly:  chi (N,n), E (N,I,n), y (N,I,n_rx),
Gb (n,n), Gs (n_rx,n); the contrast solve sums over incidences, as in the model's
own _cg_ls.  Double precision, identical discretisation, identical Born-LS init.
All heavy products are BLAS matmuls on flattened (N*I, n) views.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import build_matrices, make_dirs, make_dataset
from run_experiments import RES, NX, NY, N_INC

CHECK_K = [1, 2, 4, 8, 16, 24, 50, 100]
_CHI_TRUE = None


def load():
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    kb = 2 * np.pi
    make_dataset(300, Gb, Gs, pts, dirs, kb, NX, NY, seed=1, snr_db=20.0)
    chi_te, y_te, Ei = make_dataset(100, Gb, Gs, pts, dirs, kb, NX, NY, seed=2, snr_db=300.0)
    return (Gb.astype(np.complex128), Gs.astype(np.complex128),
            Ei.astype(np.complex128), chi_te.astype(np.float64), y_te.astype(np.complex128))


def rre_batch(chi_hat, chi_true):
    num = np.sqrt(((chi_hat - chi_true) ** 2).sum(axis=1))
    den = np.sqrt((chi_true ** 2).sum(axis=1)) + 1e-12
    return float((num / den).mean())


def _cg(apply_A, b, x0, n_cg):
    x = x0.copy(); r = b - apply_A(x); p = r.copy()
    rs = np.sum(np.conj(r) * r, axis=-1).real
    for _ in range(n_cg):
        Ap = apply_A(p)
        a = rs / np.maximum(np.sum(np.conj(p) * Ap, axis=-1).real, 1e-30)
        x = x + a[..., None] * p
        r = r - a[..., None] * Ap
        rsn = np.sum(np.conj(r) * r, axis=-1).real
        p = r + (rsn / np.maximum(rs, 1e-30))[..., None] * p
        rs = rsn
    return x


def born_init(Gb, Gs, Ei, y, chi_max=0.5, n_cg=60, cg_lam_rel=0.05):
    """Identical to UnrolledISP's init_state: per-sample solve with incidence summation."""
    N, I, n = y.shape[0], Ei.shape[0], Gb.shape[0]
    M = N * I
    gsdiag = (np.abs(Gs) ** 2).sum(axis=0)
    E3 = np.broadcast_to(Ei, (N, I, n))
    Ef = E3.reshape(M, n)
    yf = y.reshape(M, -1)
    b = (np.conj(Ef) * (yf @ Gs.conj())).real.reshape(N, I, n).sum(axis=1)     # (N,n)

    def apply_A(x):                      # x (N,n)
        Ex = E3 * x[:, None, :]
        m = (Ex.reshape(M, n) @ Gs.T).reshape(N, I, -1)
        Gm = (m.reshape(M, -1) @ Gs.conj()).reshape(N, I, n)
        diagA = (gsdiag * (np.abs(E3) ** 2)).sum(axis=1)
        return ((np.conj(E3) * Gm).real).sum(axis=1) + (cg_lam_rel * diagA + 1e-12) * x
    x = _cg(apply_A, b, np.zeros_like(b), n_cg)
    return np.clip(x, 0.0, chi_max)


def _field(E0, Gb, w, N, I, n):
    return E0 + (w.reshape(N * I, n) @ Gb.T).reshape(N, I, n)


def run_csi(Gb, Gs, Ei, y, chi_init, Ks, chi_max=0.5, n_cg=40):
    """Contrast source inversion (van den Berg & Kleinman 1997), multiplicative-free form:

        F = ||Gs w - y||^2 / ||y||^2  +  ||chi E - w||^2 / ||chi E||^2

    Both denominators are GLOBAL scalars over the whole dataset, as in the original
    formulation.  (A per-sample denominator on one term only, with a global one on the
    other, silently re-weights the objective by ~N*I and starves the data term -- the
    first version of this script did exactly that and CSI barely moved off the Born
    initialisation.)  The w-step is the normal equation of F solved by CG; the contrast
    step is the standard pointwise minimiser chi = Re(sum_i conj(E_i) w_i)/sum_i |E_i|^2.
    """
    N, I, n = chi_init.shape[0], Ei.shape[0], Gb.shape[0]
    M = N * I
    ny = float((np.abs(y) ** 2).sum())
    E0 = np.broadcast_to(Ei, (N, I, n))
    chi = chi_init.copy()
    chi_ex = np.repeat(chi, I, axis=0).reshape(N, I, n)
    w = chi_ex * E0
    gsHy = (y.reshape(M, -1) @ Gs.conj()).reshape(N, I, n)                     # Gs^H y
    st = {"ne": 1.0}                       # ||chi E||^2, global, refreshed each iteration

    def A_w(x):                          # x (N,I,n)
        m = (x.reshape(M, n) @ Gs.T)
        back = (m @ Gs.conj()).reshape(N, I, n)
        return back / ny + x / st["ne"]
    out = {}
    t0 = time.time()
    for k in range(1, max(Ks) + 1):
        E = _field(E0, Gb, w, N, I, n)
        chi_ex = np.repeat(chi, I, axis=0).reshape(N, I, n)
        ce = chi_ex * E
        st["ne"] = max(float((np.abs(ce) ** 2).sum()), 1e-30)
        rhs = gsHy / ny + ce / st["ne"]
        w = _cg(A_w, rhs, w, n_cg)
        E = _field(E0, Gb, w, N, I, n)
        den = (np.abs(E) ** 2).sum(axis=1) + 1e-30
        chi = np.clip((np.conj(E) * w).sum(axis=1).real / den, 0.0, chi_max)
        if k in Ks:
            out[k] = (rre_batch(chi, _CHI_TRUE), time.time() - t0)
    return out


def run_som(Gb, Gs, Ei, y, chi_init, Ks, Msub=12, chi_max=0.5):
    N, I, n = chi_init.shape[0], Ei.shape[0], Gb.shape[0]
    Mx = N * I
    U, S, Vh = np.linalg.svd(Gs, full_matrices=True)
    V = Vh.conj().T
    Vm, Vn, Sm = V[:, :Msub], V[:, Msub:], S[:Msub]
    E0 = np.broadcast_to(Ei, (N, I, n))
    yf = y.reshape(Mx, -1)
    # w_det = Vm diag(1/Sm) U_m^H y  (Moore-Penrose pseudo-inverse truncated to the top
    # Msub singular directions of the data operator Gs).  Gs = U S Vh, so U_m^H y is the
    # *conjugated* projection -- the data equation here is y = Gs w (no conjugate on Gs).
    w_det = ((yf @ U[:, :Msub].conj()) / Sm[None, :]) @ Vm.T                   # (Mx,n)
    chi = chi_init.copy()
    w = np.repeat(chi, I, axis=0).reshape(N, I, n) * E0
    Vnc = Vn.conj()
    # sanity: the deterministic part must reproduce the top-Msub projection of the data,
    #   Gs w_det_i == U_m U_m^H y_i.  A wrong conjugation convention shows up here instantly.
    _p = U[:, :Msub] @ (U[:, :Msub].conj().T @ yf[:4].T)
    _q = (Gs @ w_det[:4].T)
    _chk = float(np.linalg.norm(_q - _p) / (np.linalg.norm(_p) + 1e-30))
    print("  [SOM M=%2d] pseudo-inverse check  rel.err=%.2e" % (Msub, _chk), flush=True)
    out = {}
    t0 = time.time()
    for k in range(1, max(Ks) + 1):
        E = _field(E0, Gb, w, N, I, n)
        resid = (np.repeat(chi, I, axis=0).reshape(N, I, n) * E
                 - w_det.reshape(N, I, n)).reshape(Mx, n)
        w = w_det.reshape(N, I, n) + (((resid @ Vnc) @ Vn.T)).reshape(N, I, n)
        E = _field(E0, Gb, w, N, I, n)
        den = (np.abs(E) ** 2).sum(axis=1) + 1e-30
        chi = np.clip((np.conj(E) * w).sum(axis=1).real / den, 0.0, chi_max)
        if k in Ks:
            out[k] = (rre_batch(chi, _CHI_TRUE), time.time() - t0)
    return out


def main():
    global _CHI_TRUE
    Gb, Gs, Ei, chi_te, y_te = load()
    _CHI_TRUE = chi_te
    out = {"check_K": CHECK_K, "chi_max": 0.5, "n_inc": N_INC, "n_test": chi_te.shape[0]}
    chi0 = born_init(Gb, Gs, Ei, y_te)
    out["born_init_rre"] = rre_batch(chi0, chi_te)
    print("[born init] RRE=%.4f" % out["born_init_rre"], flush=True)

    t0 = time.time()
    csi = run_csi(Gb, Gs, Ei, y_te, chi0.copy(), CHECK_K)
    out["csi"] = {str(k): {"rre": v[0], "wallclock_s": v[1]} for k, v in csi.items()}
    print("[CSI] " + "  ".join("K=%-3d RRE=%.4f(%.1fs)" % (k, v[0], v[1]) for k, v in csi.items()),
          flush=True)

    som = {}
    for Ms in [6, 12, 20]:
        s = run_som(Gb, Gs, Ei, y_te, chi0.copy(), CHECK_K, Msub=Ms)
        som[str(Ms)] = {str(k): {"rre": v[0], "wallclock_s": v[1]} for k, v in s.items()}
        bk = min(s, key=lambda k: s[k][0])
        print("[SOM M=%2d] best K=%-3d RRE=%.4f  |  " % (Ms, bk, s[bk][0])
              + "  ".join("%.3f" % s[k][0] for k in CHECK_K), flush=True)
    out["som"] = som
    out["totals_s"] = time.time() - t0
    with open(os.path.join(RES, "csi_som.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[csi_som done]  %.0fs" % out["totals_s"], flush=True)


if __name__ == "__main__":
    main()
