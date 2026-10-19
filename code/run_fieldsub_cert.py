# -*- coding: utf-8 -*-
"""run_fieldsub_cert.py -- M1.13 empirical field-subsystem certificate.

The damped Born field subsystem is  w+ = (1-a) w + a * chi (Ei + Gb w), whose
per-layer field gain is  (1-a) + a * sigma_max(diag(chi^k) G_b)  < 1  iff
sigma_max(diag(chi^k) G_b) < 1.  Hence

    sup_k sigma_max(diag(chi^k) G_b)

is the *measured* per-layer field contraction quantity that can stand in for the
a-priori  beta0 = chi_max * ||Gb||_2  exactly where beta0 > 1 (r9 reviewer M1.13).
It is tighter than beta0 because it uses the actual contrast support of the rollout
rather than the worst-case ||chi^k||_inf.

Protocol mirrors run_contrast_sweep.py (tab:contrast): constrained K=8 checkpoint,
projection ON, 100-layer cyclic rollout, seed-2 clean test set at ceilings
{0.5, 0.75, 1.0, 1.5}, first-16 + full-100 reporting.

Identity used:  sigma_max(diag(chi) Gb)^2 = lambda_max( Gb^H diag(chi^2) Gb )
(chi real).  Evaluated by single-vector power iteration warm-started across layers,
in the model's native complex64.

Usage:
    python run_fieldsub_cert.py            # full run on GPU/CPU
    python run_fieldsub_cert.py prescreen  # tiny CPU self-check
Writes results/fieldsub_cert.json.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import (build_matrices, make_dirs, incident_fields, forward_exact,
                      add_noise, random_contrast, UnrolledISP)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NX = NY = 32
N_INC = 8
N_RX = 32
KB = 2 * np.pi
CKPT = os.path.join(RES, "ckpt_constrained_K8.pt")
LEVELS = [0.5, 0.75, 1.0, 1.5]
DEPTH = 100
N_TE = 100
SEED_TE = 2
SNR_TE = 300.0
BATCH = 50
COLD_ITERS = 60
WARM_ITERS = 25
CONV_TOL = 1e-6

Gb = Gs = pts = dirs = kb = None


def gen_test(chi_max, n, seed, snr_db):
    """Identical to run_contrast_sweep.gen_test (same rng consumption per object)."""
    rng = np.random.default_rng(seed)
    Ei = incident_fields(pts, dirs, kb)
    chis, ys = [], []
    for _ in range(n):
        chi = random_contrast(NX, NY, rng, chi_max=chi_max)
        _, _, y = forward_exact(Gb, Gs, Ei, chi)
        ys.append(add_noise(y, snr_db, rng))
        chis.append(chi)
    return (np.stack(chis).astype(np.float32), np.stack(ys).astype(np.complex64), Ei)


def init_state(model, y, batch):
    with torch.no_grad():
        Ei = model.Ei.unsqueeze(0).expand(batch, -1, -1)
        chi0 = model._cg_ls(Ei, y)
        chi0 = chi0.clamp(0.0, model.chi_max)
        w0 = chi0.unsqueeze(1) * model.Ei.unsqueeze(0)
    return chi0, w0


@torch.no_grad()
def sigma_diagGb(Gb, GbH, chi, v=None, iters=WARM_ITERS):
    """Per-sample sigma_max(diag(chi) Gb) for a batch.

    chi : (B, n) real contrast states.
    Gb, GbH : (n, n) complex and its conjugate transpose.
    v  : optional warm-start power vectors (B, n) complex.
    Returns (sigma (B,), v (B, n))."""
    B, n = chi.shape
    chi2 = (chi * chi).to(Gb.dtype)                      # (B, n) real == |chi|^2
    if v is None:
        g = torch.Generator(device=Gb.device).manual_seed(1234)
        v = (torch.randn(B, n, generator=g, device=Gb.device) +
             1j * torch.randn(B, n, generator=g, device=Gb.device)).to(Gb.dtype)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-30)
    lam_prev = None
    for _ in range(iters):
        Gv = torch.matmul(Gb, v.unsqueeze(-1)).squeeze(-1)          # (B, n)
        w = chi2 * Gv                                                # (B, n)
        Gw = torch.matmul(GbH, w.unsqueeze(-1)).squeeze(-1)          # (B, n) = M v
        lam = Gw.norm(dim=1)                                         # ||M v||, ||v||=1
        v = Gw / (lam.unsqueeze(1) + 1e-30)
        if lam_prev is not None and bool(
                (torch.abs(lam - lam_prev) / (lam + 1e-30) < CONV_TOL).all()):
            break
        lam_prev = lam
    return lam.clamp_min(0.0).sqrt(), v                            # sqrt(lambda_max(M))


def rollout_sigma(model, chi_t, y_t, depth):
    """100-layer cyclic rollout, proj ON; return per-(sample,depth) sigma and max|chi|."""
    Gb, GbH = model.Gb, model.Gb.conj().T
    K = model.K
    n = model.nx * model.ny
    per_level = {"sup_sigma": 0.0, "sup_sigma_layer": -1, "max_chi": 0.0,
                 "max_chi_layer": -1, "n_samples": 0}
    v = None
    for i in range(0, chi_t.shape[0], BATCH):
        chi_b = chi_t[i:i + BATCH]
        y_b = y_t[i:i + BATCH]
        chi, w = init_state(model, y_b, chi_b.shape[0])
        per_level["n_samples"] += chi.shape[0]
        v = None
        for d in range(1, depth + 1):
            chi, w = model.layer((d - 1) % K, chi, w, y_b)
            sig, v = sigma_diagGb(Gb, GbH, chi.reshape(chi.shape[0], -1), v=v)
            s_max = float(sig.max().item())
            c_max = float(chi.abs().max().item())
            if s_max > per_level["sup_sigma"]:
                per_level["sup_sigma"] = s_max
                per_level["sup_sigma_layer"] = d
            if c_max > per_level["max_chi"]:
                per_level["max_chi"] = c_max
                per_level["max_chi_layer"] = d
    return per_level


def prescreen():
    global Gb, Gs, pts, dirs, kb
    print("[prescreen] CPU, 2 samples, 20 layers -- correctness check only")
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    kb = KB
    gb2 = float(np.linalg.norm(Gb, 2))
    model = UnrolledISP(8, Gb, Gs, incident_fields(pts, dirs, kb),
                        NX, NY, N_INC, mode="constrained", eps_bar=0.03,
                        proj=True).to("cpu")
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    chi_te, y_te, _ = gen_test(0.5, 2, SEED_TE, SNR_TE)
    chi_d = torch.from_numpy(chi_te).float()
    y_d = torch.from_numpy(y_te).to(torch.complex64)
    GbT, GbHT = model.Gb, model.Gb.conj().T
    chi, w = init_state(model, y_d, 2)
    v = None
    for d in range(1, 21):
        chi, w = model.layer((d - 1) % 8, chi, w, y_d)
        sig, v = sigma_diagGb(GbT, GbHT, chi.reshape(2, -1), v=v, iters=80)
        # independent numpy SVD cross-check on the first sample
        chi_np = chi[0].detach().numpy().astype(np.complex128)
        sv = float(np.linalg.norm(np.diag(chi_np.real) @ Gb, 2))
        print("  d=%2d  sigma_batch=%.5f  sigma_svd=%.5f  max|chi|=%.4f"
              % (d, float(sig[0]), sv, float(chi.abs().max())))
    # trivial upper bound check: sigma <= ||chi||_inf * ||Gb||_2
    print("  ||Gb||_2 = %.4f ; ||chi||_inf*||Gb||_2 upper bound sanity: OK if above"
          % gb2)
    print("[prescreen] done")


def main():
    global Gb, Gs, pts, dirs, kb
    if len(sys.argv) > 1 and sys.argv[1] == "prescreen":
        prescreen()
        return
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    kb = KB
    gb2 = float(np.linalg.norm(Gb, 2))
    print("[env] ||Gb||2=%.4f device=%s" % (gb2, DEV), flush=True)

    model = UnrolledISP(8, Gb, Gs, incident_fields(pts, dirs, kb),
                        NX, NY, N_INC, mode="constrained", eps_bar=0.03,
                        proj=True).to(DEV)
    model.load_state_dict(torch.load(CKPT, map_location=DEV))
    model.eval()
    print("[model] ckpt_constrained_K8.pt (proj ON)", flush=True)

    rec = {"gb_norm2": gb2, "depth": DEPTH, "n_te": N_TE, "seed_te": SEED_TE,
           "snr_te": SNR_TE, "device": DEV, "ckpt": "ckpt_constrained_K8.pt",
           "instrument": "power iteration on Gb^H diag(chi^2) Gb, warm-started, complex64",
           "levels": {}}
    for L in LEVELS:
        chi_te, y_te, _ = gen_test(L, N_TE, SEED_TE, SNR_TE)
        chi_d = torch.from_numpy(chi_te).float().to(DEV)
        y_d = torch.from_numpy(y_te).to(torch.complex64).to(DEV)
        r = rollout_sigma(model, chi_d, y_d, DEPTH)
        r["beta0_meas"] = float(chi_te.max().item() * gb2)
        r["chi_max_actual"] = float(chi_te.max().item())
        r["upper_bound_trivial"] = r["chi_max_actual"] * gb2
        rec["levels"][str(L)] = r
        print("[L=%.2f] sup_k sigma_max(diag chi^k Gb)=%.4f (layer %d)  "
              "max|chi|=%.4f (layer %d)  beta0_meas=%.4f  trivial_ub=%.4f  <1? %s"
              % (L, r["sup_sigma"], r["sup_sigma_layer"], r["max_chi"],
                 r["max_chi_layer"], r["beta0_meas"], r["upper_bound_trivial"],
                 r["sup_sigma"] < 1.0), flush=True)

    with open(os.path.join(RES, "fieldsub_cert.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("[done] %.0fs -> results/fieldsub_cert.json" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
