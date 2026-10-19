# -*- coding: utf-8 -*-
"""run_fresnel_recon.py -- Experiment C: zero-shot reconstruction of the
measured Institut Fresnel 2-D opus-2 data with the paper's unrolled ISP
structure (damped Born field step + regularized contrast step + physics-prior
projection), on the REAL measured scattered fields.

What this is
------------
The paper's headline claim is NOT that the unrolled network converges; it is
that, even without layer-wise contraction, the physics-prior projection keeps
the state bounded at ANY depth.  Experiment C checks this on measured data:

  * load FoamDielExt/FoamDielInt/FoamTwinDiel TM .exp files
  * calibrate the target-independent instrument constant a (leave-one-out,
    exactly as the sibling project did in _fresnel_zeroshot.py)
  * run the paper's layer iteration (classical mode, no learned correction --
    the correction nets were trained on a different geometry and are NOT
    transferred; the structural claim is what is tested)
  * record, per layer: max |chi|, max |w|, residual
  * projection ON vs OFF contrast: does the projection keep the state bounded
    at depth, and does turning it off let it diverge?
  * compare the recovered chi against the published phantom geometry.

Geometry/IO are reused from the sibling project (fresnel_io, fresnel_proj)
which already verified the .exp column convention and the phase convention.
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

SIBLING = r"D:\workbudd成果\全波电磁逆散射的置信度标定与不确定性量化研究\01_code"
sys.path.insert(0, SIBLING)
import fresnel_io as F                      # noqa: E402
from fresnel_proj import FresnelGeom, mm2grid, N   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")

FREQS = [2.0, 3.0, 4.0]
TARGETS = ["FoamDielExt", "FoamDielInt", "FoamTwinDiel"]
TWIN = dict(p1=(-1.5, -8.5), p2=(2.5, -60.0), r=15.5, foam=40.0)
CC = np.arange(N) * (2.0 / N) - 1.0 + (2.0 / N) / 2.0
XX, YY = np.meshgrid(CC, CC, indexing="xy")
HC = 2.0 / N


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


def instrument_constant(t, f):
    """Complex instrument constant from the published phantom on measured data."""
    d = F.load(t, "TM")
    fi = fidx(d, f)
    g = FresnelGeom(f, d["ridx"][fi])
    Em = g.forward((2.0 * true_geom(t, f)).ravel())
    a = complex(np.vdot(Em, d["escat"][fi]) / np.vdot(Em, Em))
    return a, d, fi, g


# ------------------------------------------------------------------ layers ---
def gb_apply(g, w_v):
    """G_b w = K * G @ w_v, one view, (n,)."""
    return g.K * (g.G @ w_v)


def gs_apply(g, v, w_v):
    """G_s^{(v)} w = K * Grec[v] @ w_v, (n_recv,)."""
    return g.K * (g.Grec[v] @ w_v)


def matched_filter_chi(g, E, w_t, y_meas, lam_rel=0.05):
    """Regularized-LS contrast step: one Newton/CG step on
       A x = b, A = Re sum_v |E_v|^2 (Gs^H Gs)_diag + lam I."""
    I = g.n_views
    n = g.N * g.N
    b = np.zeros(n)
    diag = np.zeros(n)
    for v in range(I):
        r = y_meas[v] - gs_apply(g, v, w_t[v])
        Ghr = g.K * (g.Grec[v].conj().T @ r)           # (n,)
        b += (E[v].conj() * Ghr).real
        diag += (np.abs(E[v]) ** 2) * (np.abs(g.Grec[v]) ** 2).sum(0) * g.K ** 2
    lam = lam_rel * diag + 1e-12
    return b / (diag + lam)          # solve (diag A + lam) x = b, not b/lam


def residual(g, chi, y_meas):
    """r = ||y_meas - F(chi)||_F / ||y_meas||_F with one free scale s."""
    y_norm = np.linalg.norm(y_meas)
    if y_norm < 1e-30:
        return 1.0
    # solve for best scalar s: chi_hat = s*chi, minimize ||y - F(s*chi)||
    # (F is linear in chi via the Born approximation; exact MoM used by
    # FresnelGeom.forward with chi=s*chi)
    def misfit(s):
        Em = g.forward(s * chi)
        return float(np.linalg.norm(y_meas - Em) / y_norm)
    # coarse golden search
    best = (misfit(1.0), 1.0)
    for s in [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0]:
        m = misfit(s)
        if m < best[0]:
            best = (m, s)
    return best[0]


def unrolled_run(g, y_meas, K_max=100, a=1.0, gamma=1.0, theta=1.0,
                 proj=True, chi_max=0.5):
    """The paper's layer iteration, classical mode, on measured data."""
    I = g.n_views
    n = g.N * g.N
    chi = np.zeros(n)
    w = np.zeros((I, n), dtype=complex)
    trace = []
    for k in range(K_max):
        Gbw = np.array([gb_apply(g, w[v]) for v in range(I)])
        w_t = (1 - a) * w + a * chi[None, :] * (g.Einc + Gbw)
        E = g.Einc + np.array([gb_apply(g, w_t[v]) for v in range(I)])
        chi_hat = matched_filter_chi(g, E, w_t, y_meas)
        chi_t = (1 - gamma) * chi + gamma * chi_hat
        chi_out = chi + theta * (chi_t - chi)
        w_out = w + theta * (w_t - w)
        if proj:
            chi_out = np.clip(chi_out, 0.0, chi_max)
        chi, w = chi_out, w_out
        # record at select depths
        if k in (0, 1, 3, 7, 15, 31, 63, 99) or k == K_max - 1:
            r = residual(g, chi, y_meas)
            trace.append(dict(k=k + 1,
                              max_chi=float(np.abs(chi).max()),
                              max_w=float(np.abs(w).max()),
                              residual=r))
    return chi, w, trace


def main():
    t0 = time.time()
    out = {"meta": {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "measured Fresnel opus-2 reconstruction, paper's "
                            "unrolled structure, classical mode"}}

    # ---- instrument constants (per target, per freq) -----------------------
    cinfo = {}
    for t in TARGETS:
        cinfo[t] = {}
        for f in FREQS:
            a, d, fi, g = instrument_constant(t, f)
            cinfo[t][f] = (a, d, fi, g)
            print("a[%s@%.0fGHz] = %.4f %+.4fj  |a|=%.3f arg=%.3f"
                  % (t, f, a.real, a.imag, abs(a), np.angle(a)))

    instances = {}
    for f in FREQS:
        for t in TARGETS:
            others = [cinfo[o][f][0] for o in TARGETS if o != t]
            a = complex(np.mean(others))          # leave-one-out
            _, d, fi, g = cinfo[t][f]
            Es = d["escat"][fi] / a               # -> model units
            Y = true_geom(t, f).ravel()

            # projection ON -- training damping budgets (a_max=0.5,g_max=0.6,
            # a=g=0 -> sigmoid=0.5 -> alpha=gamma=0.25/0.30)
            chi_on, w_on, tr_on = unrolled_run(g, Es, K_max=100, proj=True,
                                                chi_max=0.5, a=0.25, gamma=0.30)
            # projection OFF
            chi_off, w_off, tr_off = unrolled_run(g, Es, K_max=100, proj=False,
                                                  a=0.25, gamma=0.30)

            mse_on = float(np.mean((chi_on - Y) ** 2))
            mse_off = float(np.mean((chi_off - Y) ** 2))
            r_true = residual(g, Y, Es)

            rec = dict(
                freq=f, target=t,
                a_loo=[float(a.real), float(a.imag)],
                r_true=float(r_true),
                mse_on=mse_on, mse_off=mse_off,
                max_chi_on_final=float(np.abs(chi_on).max()),
                max_w_on_final=float(np.abs(w_on).max()),
                max_chi_off_final=float(np.abs(chi_off).max()),
                max_w_off_final=float(np.abs(w_off).max()),
                diverges_off=bool((not np.isfinite(chi_off).all()) or
                                  (not np.isfinite(w_off).all()) or
                                  np.abs(chi_off).max() > 10.0 or
                                  np.abs(w_off).max() > 10.0),
                trace_on=tr_on, trace_off=tr_off,
            )
            key = "%s_%.0f" % (t, f)
            instances[key] = rec
            print("%-13s f=%.0f | r_true=%.3f | MSE on=%.4f off=%.4f | "
                  "max|chi| on=%.3f off=%.2f | max|w| on=%.3f off=%.2f | "
                  "OFF-div=%s" % (t, f, r_true, mse_on, mse_off,
                                   rec["max_chi_on_final"],
                                   rec["max_chi_off_final"],
                                   rec["max_w_on_final"],
                                   rec["max_w_off_final"],
                                   rec["diverges_off"]))
            # save reconstructed image for plotting
            np.savez_compressed(
                os.path.join(RES, "fresnel_recon_%s_%.0f.npz" % (t, f)),
                chi_on=chi_on.reshape(N, N).astype(np.float32),
                chi_off=chi_off.reshape(N, N).astype(np.float32),
                Y=Y.reshape(N, N).astype(np.float32),
                Es_aligned=np.stack([Es.real, Es.imag]).astype(np.float32))

    out["instances"] = instances
    path = os.path.join(RES, "fresnel_recon.json")
    with open(path, "w") as fp:
        json.dump(out, fp, indent=1)
    print("\n[done] %s  (%.0fs)" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
