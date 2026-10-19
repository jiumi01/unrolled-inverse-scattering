# -*- coding: utf-8 -*-
"""run_fresnel_recon_v2.py -- improved measured-data reconstruction.

Changes vs v1:
  * chi_max = 2.5 (plastic cylinder chi~=2 inside the box, no truncation)
  * full 15-step CG least-squares contrast solve (matches the paper's
    n_cg=15), not a one-step matched filter
  * alpha/gamma grid search over {0.25, 0.5, 1.0}, pick the pair with the
    smallest label-free residual
  * 3x3 truth-vs-recon comparison figure written for the paper
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

SIBLING = r"D:\workbudd成果\全波电磁逆散射的置信度标定与不确定性量化研究\01_code"
sys.path.insert(0, SIBLING)
import fresnel_io as F
from fresnel_proj import FresnelGeom, mm2grid, N

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")

FREQS = [2.0, 3.0, 4.0]
TARGETS = ["FoamDielExt", "FoamDielInt", "FoamTwinDiel"]
TWIN = dict(p1=(-1.5, -8.5), p2=(2.5, -60.0), r=15.5, foam=40.0)
CC = np.arange(N) * (2.0 / N) - 1.0 + (2.0 / N) / 2.0
XX, YY = np.meshgrid(CC, CC, indexing="xy")
HC = 2.0 / N

CHI_MAX = 2.5
N_CG = 15


def disc(cx_mm, cy_mm, r_mm, v, f):
    cx, cy, r = mm2grid([cx_mm, cy_mm, r_mm], f)
    return v * np.clip((r - np.hypot(XX - cx, YY - cy)) / HC + 0.5, 0, 1)


def true_geom(name, f):
    if name == "FoamDielExt":
        img = disc(0, 0, 40.0, 0.225, f) + disc(0, -55.5, 15.5, 1.0, f)
    elif name == "FoamDielInt":
        img = np.maximum(disc(0, 0, 40.0, 0.225, f), disc(0, -10.0, 15.5, 1.0, f))
    else:
        img = disc(0, 0, TWIN["foam"], 0.225, f)
        for p in (TWIN["p1"], TWIN["p2"]):
            img = np.maximum(img, disc(p[0], p[1], TWIN["r"], 1.0, f))
    return np.clip(img, 0, 1).astype(np.float64)


def fidx(d, f):
    return int(np.argmin(np.abs(d["freq"] / 1e9 - f)))


def instr_const(t, f):
    d = F.load(t, "TM")
    fi = fidx(d, f)
    g = FresnelGeom(f, d["ridx"][fi])
    Em = g.forward((2.0 * true_geom(t, f)).ravel())
    a = complex(np.vdot(Em, d["escat"][fi]) / np.vdot(Em, Em))
    return a, d, fi, g


# ---------------------------------------------------------- CG contrast solve
def cg_contrast(g, E, y_meas, n_cg=N_CG, lam_rel=0.05):
    """Regularized LS contrast step, numpy port of isp_core._cg_ls.
    A x = b, A = Re sum_v diag(|E_v|^2) Gs^H Gs + lam I."""
    I = g.n_views
    n = g.N * g.N
    # Gs^H Gs diagonal proxy = sum_recv |Gs[recv, j]|^2
    gs_diag = np.zeros(n)
    for v in range(I):
        gs_diag += (np.abs(g.Grec[v]) ** 2).sum(0)
    gs_diag *= g.K ** 2

    def apply_A(x):
        out = np.zeros(n)
        for v in range(I):
            Ex = E[v] * x                       # (n,)
            GEx = g.K * (g.Grec[v] @ Ex)        # (n_recv,)
            GHGEx = g.K * (g.Grec[v].conj().T @ GEx)  # (n,)
            out += (E[v].conj() * GHGEx).real
        return out

    b = np.zeros(n)
    for v in range(I):
        Ghr = g.K * (g.Grec[v].conj().T @ y_meas[v])
        b += (E[v].conj() * Ghr).real

    diagA = np.zeros(n)
    for v in range(I):
        diagA += (np.abs(E[v]) ** 2) * gs_diag / I
    lam = lam_rel * diagA + 1e-12

    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    rs = (r * r).sum()
    for _ in range(n_cg):
        Ap = apply_A(p) + lam * p
        pAp = (p * Ap).sum() + 1e-20
        alpha = rs / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        beta = rs_new / (rs + 1e-20)
        p = r + beta * p
        rs = rs_new
    return x


def residual(g, chi, y_meas):
    """Label-free residual with one free positive scale s."""
    y_norm = np.linalg.norm(y_meas)
    if y_norm < 1e-30:
        return 1.0
    def misfit(s):
        Em = g.forward(s * chi)
        return float(np.linalg.norm(y_meas - Em) / y_norm)
    best = (misfit(1.0), 1.0)
    for s in [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 3.0]:
        m = misfit(s)
        if m < best[0]:
            best = (m, s)
    return best[0]


def unrolled_run(g, y_meas, K_max=100, alpha=0.25, gamma=0.30, theta=1.0,
                 proj=True, chi_max=CHI_MAX):
    I = g.n_views
    n = g.N * g.N
    chi = np.zeros(n)
    w = np.zeros((I, n), dtype=complex)
    for k in range(K_max):
        Gbw = np.array([g.K * (g.G @ w[v]) for v in range(I)])
        w_t = (1 - alpha) * w + alpha * chi[None, :] * (g.Einc + Gbw)
        E = g.Einc + np.array([g.K * (g.G @ w_t[v]) for v in range(I)])
        chi_hat = cg_contrast(g, E, y_meas)
        chi_t = (1 - gamma) * chi + gamma * chi_hat
        chi_out = chi + theta * (chi_t - chi)
        w_out = w + theta * (w_t - w)
        if proj:
            chi_out = np.clip(chi_out, 0.0, chi_max)
        chi, w = chi_out, w_out
    return chi, w


def main():
    t0 = time.time()
    cinfo = {}
    for t in TARGETS:
        cinfo[t] = {}
        for f in FREQS:
            a, d, fi, g = instr_const(t, f)
            cinfo[t][f] = (a, d, fi, g)

    rows = []
    recon_imgs = {}
    for f in FREQS:
        for t in TARGETS:
            others = [cinfo[o][f][0] for o in TARGETS if o != t]
            a = complex(np.mean(others))
            _, d, fi, g = cinfo[t][f]
            Es = d["escat"][fi] / a
            Y = true_geom(t, f).ravel()

            best = None
            for alpha, gamma in [(0.25, 0.30), (0.5, 0.5), (1.0, 1.0)]:
                chi, w = unrolled_run(g, Es, K_max=30, alpha=alpha,
                                      gamma=gamma, proj=True)
                r = residual(g, chi, Es)
                if best is None or r < best[0]:
                    best = (r, alpha, gamma, chi, w)
            r, alpha, gamma, chi_on, w_on = best
            # projection OFF for divergence check
            chi_off, w_off = unrolled_run(g, Es, K_max=30, proj=False)

            mse = float(np.mean((chi_on - Y) ** 2))
            r_true = residual(g, Y, Es)
            diverges = bool((not np.isfinite(chi_off).all()) or
                            np.abs(chi_off).max() > 10.0)
            print("%-13s f=%.0f | a=%.2f g=%.2f | r_net=%.3f r_true=%.3f "
                  "MSE=%.4f | max|chi|=%.3f max|w|=%.4f | OFF-div=%s"
                  % (t, f, alpha, gamma, r, r_true, mse,
                     float(np.abs(chi_on).max()),
                     float(np.abs(w_on).max()), diverges))
            rows.append(dict(target=t, freq=f, alpha=alpha, gamma=gamma,
                             r_net=r, r_true=r_true, mse=mse,
                             max_chi_on=float(np.abs(chi_on).max()),
                             max_w_on=float(np.abs(w_on).max()),
                             diverges_off=diverges))
            recon_imgs["%s_%.0f" % (t, f)] = (
                chi_on.reshape(N, N).astype(np.float32),
                Y.reshape(N, N).astype(np.float32))

    json.dump(rows, open(os.path.join(RES, "fresnel_recon_v2.json"), "w"),
              indent=1)

    # ---- 3x3 comparison figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for j, t in enumerate(TARGETS):
        Y = recon_imgs["%s_3" % t][1]
        axes[0, j].imshow(Y, vmin=0, vmax=1, cmap="gray")
        axes[0, j].set_title("truth %s" % t, fontsize=10)
        axes[0, j].axis("off")
        chi = recon_imgs["%s_3" % t][0]
        axes[1, j].imshow(chi, vmin=0, vmax=1.0, cmap="gray")
        axes[1, j].set_title("unrolled, f=3 GHz", fontsize=10)
        axes[1, j].axis("off")
    plt.tight_layout()
    figpath = os.path.join(RES, "fresnel_recon_v2.png")
    plt.savefig(figpath, dpi=110)
    print("figure ->", figpath)
    print("[done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
