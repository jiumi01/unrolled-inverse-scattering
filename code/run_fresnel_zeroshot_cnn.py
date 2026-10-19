# -*- coding: utf-8 -*-
"""run_fresnel_zeroshot_cnn.py -- zero-shot transfer of the TRAINED correction
networks (ckpt_constrained_K8.pt) onto the measured Fresnel opus-2 geometry.

We pick 8 of the 18 views and 32 of the 241 receivers per view so the
UnrolledISP input channels (n_inc=8, n_rx=32) match the trained checkpoint;
the per-view receiver arc is approximated by one representative Gs (the
geometry is rotationally symmetric to first order).  The trained a/g
relaxation parameters and the CorrectionNet convolution weights are loaded
strictly; the Gb/Gs/Ei buffers are replaced by the Fresnel geometry.
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

SIBLING = r"D:\workbudd成果\全波电磁逆散射的置信度标定与不确定性量化研究\01_code"
sys.path.insert(0, SIBLING)
import fresnel_io as F                      # noqa: E402
from fresnel_proj import FresnelGeom, mm2grid, N   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import torch
import isp_core as ic

RES = os.path.join(ROOT, "results")
FREQS = [2.0, 3.0, 4.0]
TARGETS = ["FoamDielExt", "FoamDielInt", "FoamTwinDiel"]
TWIN = dict(p1=(-1.5, -8.5), p2=(2.5, -60.0), r=15.5, foam=40.0)
CC = np.arange(N) * (2.0 / N) - 1.0 + (2.0 / N) / 2.0
XX, YY = np.meshgrid(CC, CC, indexing="xy")
HC = 2.0 / N

N_INC = 8
N_RX = 32


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
        for p in (TWIN["p1"], TWIN["r"], 1.0, f):
            pass
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


def build_torch_model(g, dev="cpu"):
    """Build UnrolledISP with Fresnel Gb/Gs/Ei, n_inc=8, n_rx=32.
    Pick 8 views uniformly; pick 32 receivers from view 0's arc as representative Gs."""
    n = g.N * g.N
    # pick 8 of 18 views
    sel_v = np.linspace(0, g.n_views - 1, N_INC).astype(int)
    # pick 32 of the receivers on view 0's arc
    sel_r = np.linspace(0, g.n_recv - 1, N_RX).astype(int)
    Gb = (g.K * g.G).astype(np.complex64)
    # representative Gs: view 0's selected receivers
    Gs = (g.K * g.Grec[0][sel_r, :]).astype(np.complex64)
    Ei = g.Einc[sel_v].astype(np.complex64)
    m = ic.UnrolledISP(8, Gb, Gs, Ei, g.N, g.N, N_INC,
                       mode="constrained", eps_bar=0.03, proj=True).to(dev)
    sd = torch.load(os.path.join(RES, "ckpt_constrained_K8.pt"),
                    map_location=dev)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print("  loaded: missing=%d unexpected=%d (a/g+conv weights should match)"
          % (len(missing), len(unexpected)))
    m.eval()
    return m, sel_v, sel_r


def residual(g, chi, y_full, sel_v, sel_r):
    """Residual on the selected sub-geometry."""
    chi_c = np.asarray(chi).reshape(-1)
    # forward on full geometry then compare selected
    Em = g.forward(chi_c)
    num = np.sqrt(np.sum(np.abs(y_full[sel_v][:, sel_r] - Em[sel_v][:, sel_r]) ** 2))
    den = np.sqrt(np.sum(np.abs(y_full[sel_v][:, sel_r]) ** 2))
    return float(num / max(den, 1e-30))


def main():
    t0 = time.time()
    cinfo = {}
    for t in TARGETS:
        cinfo[t] = {}
        for f in FREQS:
            a, d, fi, g = instr_const(t, f)
            cinfo[t][f] = (a, d, fi, g)

    rows = []
    for f in FREQS:
        for t in TARGETS:
            others = [cinfo[o][f][0] for o in TARGETS if o != t]
            a = complex(np.mean(others))
            _, d, fi, g = cinfo[t][f]
            Es = d["escat"][fi] / a
            Y = true_geom(t, f).ravel()
            m, sel_v, sel_r = build_torch_model(g)
            # y tensor: (1, N_INC, N_RX)
            y = torch.as_tensor(Es[sel_v][:, sel_r], dtype=torch.complex64)[None]
            chi0 = torch.zeros(1, g.N * g.N, dtype=torch.float32)
            w0 = torch.zeros(1, N_INC, g.N * g.N, dtype=torch.complex64)
            with torch.no_grad():
                chi_hat, w_hat = m(chi0, w0, y)
            chi_np = chi_hat[0].numpy()
            mse = float(np.mean((chi_np - Y) ** 2))
            r_net = residual(g, chi_np, Es, sel_v, sel_r)
            r_true = residual(g, Y, Es, sel_v, sel_r)
            print("%-13s f=%.0f | MSE_cnn=%.4f | r_net=%.3f r_true=%.3f | "
                  "max|chi|=%.3f" % (t, f, mse, r_net, r_true,
                                      float(np.abs(chi_np).max())))
            rows.append(dict(target=t, freq=f, mse_cnn=mse, r_net=r_net,
                             r_true=r_true,
                             max_chi=float(np.abs(chi_np).max())))
            np.savez_compressed(
                os.path.join(RES, "fresnel_cnn_%s_%.0f.npz" % (t, f)),
                chi=chi_np.reshape(N, N).astype(np.float32),
                Y=Y.reshape(N, N).astype(np.float32))

    json.dump(rows, open(os.path.join(RES, "fresnel_cnn.json"), "w"), indent=1)
    print("\n[done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
