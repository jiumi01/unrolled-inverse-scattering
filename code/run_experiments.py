# -*- coding: utf-8 -*-
"""
run_experiments.py — training + evaluation for the unrolled ISP convergence study.
Outputs JSON/npz artifacts into ../results/ for the paper's figures and tables.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import (build_matrices, make_dirs, incident_fields, make_dataset,
                      forward_exact, add_noise, UnrolledISP, estimate_gain,
                      exact_gain, exact_cycle_gain)

torch.manual_seed(1234)
np.random.seed(1234)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NX = NY = 32
N_INC = 8
N_RX = 32
KS = [2, 4, 8, 16, 24]

def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)

def init_state(model, y, batch):
    """Born-LS init: chi0 = argmin ||Gs diag(Ei) chi - y||^2 + lam||chi||^2 (CG),
    w0 = chi0 * Ei."""
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
def eval_model(model, chi_t, y_t, batch=50):
    model.eval()
    errs = []
    for i in range(0, chi_t.shape[0], batch):
        chi_b = chi_t[i:i + batch]
        y_b = y_t[i:i + batch]
        chi0, w0 = init_state(model, y_b, chi_b.shape[0])
        chi_k, _ = model(chi0, w0, y_b)
        errs.append(rre(chi_k, chi_b))
    return float(np.mean(errs))

def train(model, chi_tr, y_tr, epochs=40, bs=16, lr=1e-3, lam_data=0.5, log=print):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = chi_tr.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            chi_b = chi_tr[idx].to(DEV)
            y_b = y_tr[idx].to(DEV)
            chi0, w0 = init_state(model, y_b, chi_b.shape[0])
            chi_k, w_k = model(chi0, w0, y_b)
            Gs_w = torch.matmul(model.Gs, w_k.reshape(-1, w_k.shape[-1]).T).T.reshape(y_b.shape)
            loss = ((chi_k - chi_b) ** 2).mean() + lam_data * (Gs_w - y_b).abs().pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
            opt.step()
            model.clamp_sn()
            tot += loss.item()
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"    ep{ep:03d} loss={tot / max(1, (n // bs)): .5f}")
    return model

def load_data():
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    kb = 2 * np.pi
    chi_tr, y_tr, Ei = make_dataset(300, Gb, Gs, pts, dirs, kb, NX, NY, seed=1, snr_db=20.0)
    chi_te, y_te, _ = make_dataset(100, Gb, Gs, pts, dirs, kb, NX, NY, seed=2, snr_db=300.0)
    chi_tr = torch.from_numpy(chi_tr).float()
    y_tr = torch.from_numpy(y_tr).to(torch.complex64)
    chi_te = torch.from_numpy(chi_te).float()
    y_te = torch.from_numpy(y_te).to(torch.complex64)
    return Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te

def gb_norm(Gb):
    return float(np.linalg.norm(Gb, 2))

def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    n = NX * NY
    meta = {"gb_op_norm": gb_norm(Gb), "chi_max_train": float(chi_tr.max()),
            "device": DEV, "chi_max_box": 0.5, "eps_bar_default": 0.03,
            "featbound": True, "proj_default": True}
    chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)
    meta["cert_beta_bound"] = float(0.5 * gb_norm(Gb))
    meta["cert_beta_bound_realized"] = float(chi_tr.max().item() * gb_norm(Gb))
    save("meta.json", meta)
    print(f"[meta] ||Gb||2={meta['gb_op_norm']:.4f}  chi_max={meta['chi_max_train']:.2f}  "
          f"cert beta<= {meta['cert_beta_bound']:.4f}")

    # ---------------- classical baseline curve ------------------------------
    print("[1/6] classical Born/CSI alternating (no learning)")
    cls_model = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="classical").to(DEV)
    curve = {}
    for K in [1, 2, 4, 8, 16, 24, 50, 100]:
        m = UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, mode="classical").to(DEV)
        curve[str(K)] = eval_model(m, chi_te_d, y_te_d)
        del m
    save("classical_curve.json", curve)
    print("   ", {k: round(v, 4) for k, v in curve.items()})

    # ---------------- training: depth sweep ---------------------------------
    print("[2/6] training depth sweep (constrained / unconstrained)")
    ckpts = {}
    for mode in ["constrained", "unconstrained"]:
        for K in KS:
            tag = f"{mode}_K{K}"
            m = UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, mode=mode,
                            eps_bar=0.03).to(DEV)
            print(f"  train {tag}")
            train(m, chi_tr, y_tr, epochs=40 if K <= 8 else 30)
            r = eval_model(m, chi_te_d, y_te_d)
            ckpts[tag] = r
            torch.save(m.state_dict(), os.path.join(RES, f"ckpt_{tag}.pt"))
            print(f"    test RRE={r:.4f}")
    save("depth_curve.json", ckpts)

    # ---------------- per-layer exact gain (power iteration on J^T J) -------
    print("[3/6] per-layer exact gain (power iteration on J^T J)")
    gains = {}
    for mode in ["constrained", "unconstrained"]:
        m = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV)
        m.load_state_dict(torch.load(os.path.join(RES, f"ckpt_{mode}_K8.pt")))
        m.eval()
        chi_b = chi_te_d[:16]
        y_b = y_te_d[:16]
        chi0, w0 = init_state(m, y_b, 16)
        gs = []
        for k in range(8):
            g = exact_gain(m, k, chi0, w0, y_b)
            gs.append(g)
            chi0, w0 = m.layer(k, chi0, w0, y_b)
        cyc = exact_cycle_gain(m, *init_state(m, y_b, 16), y_b) if mode != "classical" else None
        gains[mode] = gs
        gains[mode + "_cycle"] = cyc
        print(f"   {mode}: per-layer {[round(x,3) for x in gs]} cycle={cyc}")
    save("gain.json", gains)

    # ---------------- depth extrapolation -----------------------------------
    print("[4/6] depth extrapolation (cycle trained K=8 layers x100)")
    extrap = {}
    for mode in ["constrained", "unconstrained", "classical"]:
        K0 = 8
        m = UnrolledISP(K0, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV)
        if mode != "classical":
            m.load_state_dict(torch.load(os.path.join(RES, f"ckpt_{mode}_K8.pt")))
        m.eval()
        chi_b = chi_te_d[:50]
        y_b = y_te_d[:50]
        with torch.no_grad():
            chi0, w0 = init_state(m, y_b, 50)
            res, errs, maxc, maxw = [], [], [], []
            for k in range(100):
                kk = (k % K0) if mode != "classical" else k
                chi_prev = chi0.clone()
                chi0, w0 = m.layer(kk, chi0, w0, y_b)
                if k % 5 == 0:
                    res.append((((chi0 - chi_prev).norm(dim=1)) / (chi0.norm(dim=1) + 1e-12)).mean().item())
                    errs.append(rre(chi0, chi_b))
                    maxc.append(chi0.abs().max().item())
                    maxw.append(w0.abs().max().item())
        extrap[mode] = {"residual": res, "rre": errs, "maxabs_chi": maxc, "maxabs_w": maxw}
        print(f"   {mode}: final res={res[-1]:.2e} rre={errs[-1]:.4f} max|chi|={maxc[-1]:.3f}")
        del m
    save("extrap.json", extrap)

    # ---------------- noise sensitivity -------------------------------------
    print("[5/6] noise sensitivity (K=8 models)")
    rng = np.random.default_rng(7)
    noise = {}
    sens_cfg = {}
    for mode in ["constrained", "unconstrained"]:
        m = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV)
        m.load_state_dict(torch.load(os.path.join(RES, f"ckpt_{mode}_K8.pt")))
        m.eval()
        with torch.no_grad():
            chi0, w0 = init_state(m, y_te_d, chi_te_d.shape[0])
            chi_clean, _ = m(chi0, w0, y_te_d)
        noise[mode] = {}
        for snr in [30, 20, 10, 5]:
            y_n = y_te_d.clone()
            npy = y_te_d.cpu().numpy()
            pn = np.mean(np.abs(npy) ** 2)
            s = pn / (10 ** (snr / 10))
            nn_ = np.sqrt(s / 2) * (rng.standard_normal(npy.shape) + 1j * rng.standard_normal(npy.shape))
            y_n = torch.from_numpy((npy + nn_).astype(np.complex64)).to(DEV)
            with torch.no_grad():
                chi0, w0 = init_state(m, y_n, chi_te_d.shape[0])
                chi_hat, _ = m(chi0, w0, y_n)
            noise[mode][str(snr)] = {
                "rre": rre(chi_hat, chi_te_d),
                "sens": (((chi_hat - chi_clean).abs() ** 2).sum(dim=1).sqrt() /
                         (chi_clean.abs() ** 2).sum(dim=1).sqrt().clamp_min(1e-12)).mean().item()}
        print(f"   {mode}: {noise[mode]}")
        del m
    save("noise.json", noise)

    # ---------------- eps_bar Pareto sweep ----------------------------------
    print("[6/6] eps_bar sweep (constrained, K=8)")
    pareto = {}
    for eb in [0.01, 0.03, 0.06, 0.12]:
        m = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=eb).to(DEV)
        train(m, chi_tr, y_tr, epochs=40)
        r = eval_model(m, chi_te_d, y_te_d)
        rng2 = np.random.default_rng(11)
        npy = y_te_d.cpu().numpy()
        s = np.mean(np.abs(npy) ** 2) / 100.0   # 20 dB
        nn_ = np.sqrt(s / 2) * (rng2.standard_normal(npy.shape) + 1j * rng2.standard_normal(npy.shape))
        y_n = torch.from_numpy((npy + nn_).astype(np.complex64)).to(DEV)
        with torch.no_grad():
            c1, _ = m(*init_state(m, y_te_d, 100), y_te_d)
            c2, _ = m(*init_state(m, y_n, 100), y_n)
        sens = (((c2 - c1).abs() ** 2).sum(dim=1).sqrt() / (c1.abs() ** 2).sum(dim=1).sqrt().clamp_min(1e-12)).mean().item()
        pareto[str(eb)] = {"rre": r, "sens": sens}
        torch.save(m.state_dict(), os.path.join(RES, f"ckpt_cons_K8_eb{eb}.pt"))
        print(f"   eps_bar={eb}: RRE={r:.4f} sens={sens:.4f}")
        del m
    save("pareto.json", pareto)

    # ---------------- gradient norm vs depth --------------------------------
    print("[extra] gradient norm vs depth (K=24 models)")
    gradnorm = {}
    for mode in ["constrained", "unconstrained"]:
        m = UnrolledISP(24, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV)
        m.load_state_dict(torch.load(os.path.join(RES, f"ckpt_{mode}_K24.pt")))
        gradnorm[mode] = {}
        for kk in [1, 2, 4, 8, 16, 24]:
            m.train()
            chi_b = chi_te_d[:8]
            y_b = y_te_d[:8]
            chi0, w0 = init_state(m, y_b, 8)
            for k in range(kk):
                chi0, w0 = m.layer(k, chi0, w0, y_b)
            loss = ((chi0 - chi_b) ** 2).mean()
            m.zero_grad()
            loss.backward()
            gnorm = sum(p.grad.norm().item() for p in m.corrs[0].parameters() if p.grad is not None)
            gradnorm[mode][str(kk)] = gnorm
        print(f"   {mode}: {gradnorm[mode]}")
        del m
    save("gradnorm.json", gradnorm)

    # ---------------- reconstruction images ---------------------------------
    print("[extra] save reconstruction examples")
    ex = {}
    with torch.no_grad():
        gt = chi_te_d[0:4].cpu().numpy()
        mcls = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="classical").to(DEV)
        c_cls, _ = mcls(*init_state(mcls, y_te_d[:4], 4), y_te_d[:4])
        imgs = {"gt": gt, "classical8": c_cls.cpu().numpy()}
        for mode in ["constrained", "unconstrained"]:
            m = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV)
            m.load_state_dict(torch.load(os.path.join(RES, f"ckpt_{mode}_K8.pt")))
            c, _ = m(*init_state(m, y_te_d[:4], 4), y_te_d[:4])
            imgs[mode] = c.cpu().numpy()
            del m
        mcls50 = UnrolledISP(50, Gb, Gs, Ei, NX, NY, N_INC, mode="classical").to(DEV)
        c50, _ = mcls50(*init_state(mcls50, y_te_d[:4], 4), y_te_d[:4])
        imgs["classical50"] = c50.cpu().numpy()
    np.savez_compressed(os.path.join(RES, "images.npz"), **imgs)
    print(f"[done] total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        NX = 16
        print("SMOKE test")
    main()
