# -*- coding: utf-8 -*-
"""
run_contrast_sweep.py — Experiment B: certificate failure boundary at higher contrast.

Evaluates the EXISTING K=8 constrained checkpoint (trained on chi in [0.25, 0.5],
results/ckpt_constrained_K8.pt) on test data whose contrast range is raised to
chi_max in {0.75, 1.0, 1.5}, with chi_max = 0.5 as the in-distribution baseline.

Protocol mirrors run_experiments.py (same grid, incidences, snr, seeds, batching)
so the numbers are directly comparable with the paper's tables:

  * test set: 100 samples, seed=2, snr=300 (clean); noise added later with rng(7).
  * same seed across levels => identical target GEOMETRY, only contrast magnitude
    is scaled (validated by a mask-equality assertion).
  * certificate quantities: beta0_meas = max(chi_test) * ||Gb||_2  (vs certified
    beta0 = chi_max_box * ||Gb||_2 = 0.5 * 1.6606 = 0.8303).
  * fidelity: RRE at trained depth K=8; RRE vs depth (layers cycled) to D=100.
  * boundedness: max|chi|, max|w| over the depth sweep (projection ON).
  * per-layer exact gains sigma_max(J_k) and cycle composite gain/rho at the
    higher-contrast working point (first 16 samples, as in the paper's gain block).
  * noise sensitivity: snr in {30,20,10,5}, reporting RRE and relative gain (sens),
    matching the noise.json protocol.
  * projection ablation at chi_max = 1.0 and 1.5: depth sweep with proj=False to
    test whether the physics-prior projection alone keeps states bounded when
    beta0_meas > 1.

Output: results/contrast_sweep.json (per-level records) + printed summary.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import (build_matrices, make_dirs, incident_fields, forward_exact,
                      add_noise, random_contrast, UnrolledISP,
                      exact_gain, exact_cycle_gain, exact_cycle_rho)

torch.manual_seed(1234)
np.random.seed(1234)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NX = NY = 32
N_INC = 8
N_RX = 32
KB = 2 * np.pi
CKPT = os.path.join(RES, "ckpt_constrained_K8.pt")
LEVELS = [0.5, 0.75, 1.0, 1.5]          # chi_max per level
DEPTHS = [1, 2, 4, 8, 16, 24, 50, 100]
SNRS = [30, 20, 10, 5]
N_TE = 100
SEED_TE = 2
SNR_TE = 300.0
NOISE_SEED = 7

def init_state(model, y, batch):
    with torch.no_grad():
        Ei = model.Ei.unsqueeze(0).expand(batch, -1, -1)
        chi0 = model._cg_ls(Ei, y)
        chi0 = chi0.clamp(0.0, model.chi_max)
        w0 = chi0.unsqueeze(1) * model.Ei.unsqueeze(0)
    return chi0, w0

def rre(chi_hat, chi_true):
    num = ((chi_hat - chi_true) ** 2).sum(dim=1).sqrt()
    den = (chi_true ** 2).sum(dim=1).sqrt() + 1e-12
    return (num / den).mean().item()

@torch.no_grad()
def eval_rre(model, chi_t, y_t, batch=50):
    model.eval()
    errs = []
    for i in range(0, chi_t.shape[0], batch):
        chi_b = chi_t[i:i + batch]
        y_b = y_t[i:i + batch]
        chi0, w0 = init_state(model, y_b, chi_b.shape[0])
        chi_k, _ = model(chi0, w0, y_b)
        errs.append(rre(chi_k, chi_b))
    return float(np.mean(errs))

@torch.no_grad()
def depth_sweep(model, chi_t, y_t, depths, batch=50):
    """Cycle the K layers; record RRE and state max-abs at the listed depths."""
    model.eval()
    K = model.K
    out = {}
    for i in range(0, chi_t.shape[0], batch):
        chi_b = chi_t[i:i + batch]
        y_b = y_t[i:i + batch]
        chi, w = init_state(model, y_b, chi_b.shape[0])
        maxc = chi.abs().max().item()
        maxw = w.abs().max().item()
        dprev = 0
        for d in range(1, max(depths) + 1):
            chi, w = model.layer((d - 1) % K, chi, w, y_b)
            if d in depths:
                out.setdefault(d, {"rre": [], "max_chi": [], "max_w": []})
                out[d]["rre"].append(rre(chi, chi_b))
                out[d]["max_chi"].append(chi.abs().max().item())
                out[d]["max_w"].append(w.abs().max().item())
    return {str(d): {"rre": float(np.mean(v["rre"])),
                     "max_chi": float(np.max(v["max_chi"])),
                     "max_w": float(np.max(v["max_w"]))}
            for d, v in out.items()}

def gen_test(chi_max, n, seed, snr_db):
    """Test dataset at a given contrast ceiling. Same seed across levels => the
    rng consumption of random_contrast is chi_max-independent (one chi draw per
    object), so geometry masks are identical; verified by the caller."""
    rng = np.random.default_rng(seed)
    Ei = incident_fields(pts, dirs, kb)
    chis, ys = [], []
    for _ in range(n):
        chi = random_contrast(NX, NY, rng, chi_max=chi_max)
        _, _, y = forward_exact(Gb, Gs, Ei, chi)
        ys.append(add_noise(y, snr_db, rng))
        chis.append(chi)
    return np.stack(chis).astype(np.float32), np.stack(ys).astype(np.complex64), Ei

def main():
    global Gb, Gs, pts, dirs, kb
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    kb = KB
    gb2 = float(np.linalg.norm(Gb, 2))
    print(f"[env] ||Gb||2={gb2:.4f} device={DEV}", flush=True)

    # --- generate all levels first, assert geometry identity -----------------
    data = {}
    masks = None
    for L in LEVELS:
        chi_te, y_te, Ei = gen_test(L, N_TE, SEED_TE, SNR_TE)
        m = (chi_te > 0)
        if masks is None:
            masks = m
        else:
            assert np.array_equal(m, masks), f"geometry mismatch at chi_max={L}"
        data[L] = (torch.from_numpy(chi_te).float(),
                   torch.from_numpy(y_te).to(torch.complex64))
        print(f"[data] chi_max={L}: max_chi={chi_te.max():.3f} "
              f"mean_pos={chi_te[chi_te>0].mean():.3f} nz={m.sum()}/{chi_te.size}",
              flush=True)
    print("[data] geometry masks identical across all levels (same shapes, "
          "contrast scaled)", flush=True)

    # --- model (trained in-distribution, projection ON) -----------------------
    Ei = np.asarray(incident_fields(pts, dirs, kb))
    m_on = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained",
                       eps_bar=0.03, proj=True).to(DEV)
    m_on.load_state_dict(torch.load(CKPT, map_location=DEV))
    m_on.eval()
    print("[model] loaded ckpt_constrained_K8.pt (proj ON)", flush=True)

    rec = {}
    for L in LEVELS:
        chi_te, y_te = data[L]
        chi_d, y_d = chi_te.to(DEV), y_te.to(DEV)
        r = {}
        # certificate quantities ----------------------------------------------
        r["gb_norm2"] = gb2
        r["chi_max_actual"] = float(chi_te.max().item())
        r["beta0_meas"] = float(chi_te.max().item() * gb2)
        r["beta0_cert_box"] = 0.5 * gb2
        r["frac_gt_above_box"] = float((chi_te > 0.5).float().mean().item())
        r["mean_chi_pos"] = float(chi_te[chi_te > 0].mean().item())
        # fidelity at trained depth -------------------------------------------
        r["rre_K8"] = eval_rre(m_on, chi_d, y_d)
        print(f"[L={L}] beta0_meas={r['beta0_meas']:.4f} "
              f"(cert_box={r['beta0_cert_box']:.4f}) rre_K8={r['rre_K8']:.4f}",
              flush=True)
        # depth sweep (cycle), projection ON ----------------------------------
        r["depth_proj_on"] = depth_sweep(m_on, chi_d, y_d, DEPTHS)
        print(f"[L={L}] depth(proj on): D=8 rre={r['depth_proj_on']['8']['rre']:.4f} "
              f"maxchi={r['depth_proj_on']['100']['max_chi']:.3f} "
              f"maxw={r['depth_proj_on']['100']['max_w']:.3f}", flush=True)
        # per-layer exact gains + cycle composite at first 16 samples ----------
        chi_b = chi_d[:16]
        y_b = y_d[:16]
        with torch.no_grad():
            chi0, w0 = init_state(m_on, y_b, 16)
        gl = []
        with torch.no_grad():
            c, ww = chi0.clone(), w0.clone()
            for k in range(8):
                gl.append(exact_gain(m_on, k, c, ww, y_b))
                c, ww = m_on.layer(k, c, ww, y_b)
        r["sigma_max_Jk"] = gl
        r["cycle_gain"] = exact_cycle_gain(m_on, chi0, w0, y_b)
        r["cycle_rho"] = exact_cycle_rho(m_on, chi0, w0, y_b)
        print(f"[L={L}] sigma_max(Jk)={[round(x,3) for x in gl]} "
              f"cycle_gain={r['cycle_gain']:.3f} cycle_rho={r['cycle_rho']:.3f}",
              flush=True)
        # noise sensitivity (rng seed 7, snr order 30,20,10,5) -----------------
        rng = np.random.default_rng(NOISE_SEED)
        with torch.no_grad():
            chi0, w0 = init_state(m_on, y_d, chi_d.shape[0])
            chi_clean, _ = m_on(chi0, w0, y_d)
        r["noise"] = {}
        npy = y_te.cpu().numpy()
        for snr in SNRS:
            pn = np.mean(np.abs(npy) ** 2)
            s = pn / (10 ** (snr / 10))
            nn_ = np.sqrt(s / 2) * (rng.standard_normal(npy.shape)
                                    + 1j * rng.standard_normal(npy.shape))
            y_n = torch.from_numpy((npy + nn_).astype(np.complex64)).to(DEV)
            with torch.no_grad():
                chi0, w0 = init_state(m_on, y_n, chi_d.shape[0])
                chi_hat, _ = m_on(chi0, w0, y_n)
            r["noise"][str(snr)] = {
                "rre": rre(chi_hat, chi_d),
                "sens": (((chi_hat - chi_clean).abs() ** 2).sum(dim=1).sqrt() /
                         (chi_clean.abs() ** 2).sum(dim=1).sqrt().clamp_min(1e-12)).mean().item()}
        print(f"[L={L}] noise: " + json.dumps({k: round(v["rre"], 4) for k, v in
                                                r["noise"].items()}), flush=True)
        rec[str(L)] = r

    # --- projection ablation at high contrast --------------------------------
    for L in [1.0, 1.5]:
        chi_te, y_te = data[L]
        chi_d, y_d = chi_te.to(DEV), y_te.to(DEV)
        m_off = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained",
                            eps_bar=0.03, proj=False).to(DEV)
        m_off.load_state_dict(torch.load(CKPT, map_location=DEV))
        m_off.eval()
        rec[str(L)]["depth_proj_off"] = depth_sweep(m_off, chi_d, y_d, DEPTHS)
        d100 = rec[str(L)]["depth_proj_off"]["100"]
        print(f"[L={L}] depth(proj OFF): D=8 rre="
              f"{rec[str(L)]['depth_proj_off']['8']['rre']:.4f} "
              f"D=100 maxchi={d100['max_chi']:.3f} maxw={d100['max_w']:.3f}",
              flush=True)

    with open(os.path.join(RES, "contrast_sweep.json"), "w") as f:
        json.dump({"levels": LEVELS, "depths": DEPTHS, "snrs": SNRS,
                   "ckpt": "ckpt_constrained_K8.pt", "device": DEV,
                   "protocol": "mirrors run_experiments.py (seed2 test, snr300 clean, "
                               "rng7 noise)", "records": rec}, f, indent=1)
    print(f"[done] total {time.time() - t0:.0f}s -> results/contrast_sweep.json",
          flush=True)

if __name__ == "__main__":
    main()
