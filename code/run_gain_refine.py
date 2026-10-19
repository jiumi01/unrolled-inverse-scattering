# -*- coding: utf-8 -*-
"""
run_gain_refine.py — Refined per-layer / cycle gain estimates for Experiment B.

The first pass (run_contrast_sweep.py) used isp_core's _power_sigma_fd with its
hard-coded iters=12 and an unseeded starting vector; for these highly non-normal
Jacobians that is far from converged (the paper's own numbers use float64 +
Lanczos/eigsh on the same operator). This pass re-measures the
per-layer spectral norms and the cycle composite gain at each contrast level with
  * float64 model (as the paper's audit does),
  * 300-step seeded power iteration on J^T J (FD Jacobian-vector products,
    autograd for the adjoint),
so the growth with chi_max is reported on a defensible basis. Transient estimates
are not asymptotic spectral radii; we report cycle_gain (||J^K||-type norm) only.

CPU-only by design, like the other float64 audit drivers: never contend with the
GPU training chain, and keep every double-precision evaluation of this operator
on the same arithmetic (GPU vs CPU differ by ~3% through the ill-conditioned
contrast solve).

Output: results/contrast_sweep_gain_refined.json
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import (build_matrices, make_dirs, incident_fields, forward_exact,
                      add_noise, random_contrast, UnrolledISP)
import run_contrast_sweep as rcs
from run_contrast_sweep import init_state, gen_test, NX, NY, N_INC, N_TE, SEED_TE, SNR_TE

torch.manual_seed(1234)
np.random.seed(1234)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cpu"                      # never contend with the GPU training chain
CKPT = os.path.join(RES, "ckpt_constrained_K8.pt")
LEVELS = [0.5, 0.75, 1.0, 1.5]
ITERS = 100

def sigma_max_pow(f, zr, iters=ITERS, seed=0):
    """Seeded power iteration on J^T J via FD JVP + autograd adjoint (float64)."""
    z = zr.clone().detach()
    z.requires_grad_(True)
    g = torch.Generator(device=z.device).manual_seed(seed)
    u = torch.randn_like(z, generator=g)
    u = u / (u.norm() + 1e-12)
    sig = 0.0
    for _ in range(iters):
        with torch.no_grad():
            Ju = (f(z + 1e-3 * u) - f(z - 1e-3 * u)) / (2e-3)
            nv = Ju.norm() + 1e-12
            v = Ju / nv
        with torch.enable_grad():
            gv = torch.autograd.grad(f(z), (z,), grad_outputs=v)[0]
        nu = gv.norm()
        if nu < 1e-12:
            break
        u = gv / nu
        sig = nv
    return float(sig)

def _flat_layer_fn(model, k, y):
    B = y.shape[0]
    n = model.nx * model.ny
    I = model.n_inc
    def f(zr):
        chi = zr[:, :n]
        wr = zr[:, n:n + n * I].reshape(B, I, n)
        wi = zr[:, n + n * I:].reshape(B, I, n)
        chi2, w2 = model.layer(k, chi, wr + 1j * wi, y)
        return torch.cat([chi2, w2.real.reshape(B, -1), w2.imag.reshape(B, -1)], dim=1)
    return f

def _flat_cycle_fn(model, y):
    B = y.shape[0]
    n = model.nx * model.ny
    I = model.n_inc
    K = model.K
    def f(zr):
        chi = zr[:, :n]
        wr = zr[:, n:n + n * I].reshape(B, I, n)
        wi = zr[:, n + n * I:].reshape(B, I, n)
        c, ww = chi, wr + 1j * wi
        for k in range(K):
            c, ww = model.layer(k, c, ww, y)
        return torch.cat([c, ww.real.reshape(B, -1), ww.imag.reshape(B, -1)], dim=1)
    return f

def main():
    t0 = time.time()
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    kb = 2 * np.pi
    rcs.Gb, rcs.Gs, rcs.pts, rcs.dirs, rcs.kb = Gb, Gs, pts, dirs, kb
    Ei = np.asarray(incident_fields(pts, dirs, kb))
    rec = {}
    for L in LEVELS:
        chi_te, y_te, _ = gen_test(L, N_TE, SEED_TE, SNR_TE)
        chi_d = torch.from_numpy(chi_te).double().to(DEV)
        y_d = torch.from_numpy(y_te).to(torch.complex128).to(DEV)
        m = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained",
                        eps_bar=0.03, proj=True).double().to(DEV)
        for _b in ["Gb", "Gs", "Ei"]:
            getattr(m, _b).data = getattr(m, _b).to(torch.complex128)
        m.load_state_dict(torch.load(CKPT, map_location=DEV))
        m.eval()
        with torch.no_grad():
            chi0, w0 = init_state(m, y_d[:16], 16)
        zr = torch.cat([chi0, w0.real.reshape(16, -1),
                        w0.imag.reshape(16, -1)], dim=1).detach()
        gl = []
        c, ww = chi0.clone(), w0.clone()
        for k in range(8):
            gl.append(sigma_max_pow(_flat_layer_fn(m, k, y_d[:16]), zr))
            with torch.no_grad():
                c, ww = m.layer(k, c, ww, y_d[:16])
            zr = torch.cat([c, ww.real.reshape(16, -1),
                            ww.imag.reshape(16, -1)], dim=1).detach()
        cyc = sigma_max_pow(_flat_cycle_fn(m, y_d[:16]),
                            torch.cat([chi0, w0.real.reshape(16, -1),
                                       w0.imag.reshape(16, -1)], dim=1).detach())
        rec[str(L)] = {"sigma_max_Jk_refined": gl, "cycle_gain_refined": cyc,
                       "iters": ITERS, "dtype": "float64", "device": DEV,
                       "checkpoint": os.path.basename(CKPT)}
        print(f"[L={L}] refined sigma_max(Jk)="
              f"{[round(x, 3) for x in gl]} cycle_gain={cyc:.3f}", flush=True)
        with open(os.path.join(RES, "contrast_sweep_gain_refined.json"), "w") as f:
            json.dump(rec, f, indent=1)
        del m
    with open(os.path.join(RES, "contrast_sweep_gain_refined.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[done] {time.time() - t0:.0f}s -> results/contrast_sweep_gain_refined.json",
          flush=True)

if __name__ == "__main__":
    main()
