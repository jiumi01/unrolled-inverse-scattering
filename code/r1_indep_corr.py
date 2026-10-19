# -*- coding: utf-8 -*-
"""R1 独立复核（续）：带学习修正的层 Jacobian。
自实现物理核心 + 修正网络（按 isp_core 架构），载入 ckpt_{mode}_K8.pt，
用 torch.func.jvp/autograd.grad + scipy eigsh 求
  (i)  Lip(H_k)=sigma_max(d dchi / d(chi,w))
  (ii) 复合层 sigma_max（物理核心 + 修正, alpha=gamma=0.5）。
不使用作者的测量/结论函数。"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import hankel2

torch.set_grad_enabled(True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "cache")
CKPT = os.path.join(ROOT, "results")


def green_2d(r1, r2, kb):
    d = np.linalg.norm(r1[:, None, :] - r2[None, :, :], axis=-1)
    return -1j / 4.0 * hankel2(0, kb * d)


def build(nx=32, ny=32, dx=0.02, lam=1.0, n_rx=32):
    kb = 2 * np.pi / lam
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dA = dx * dx
    a_eq = dx / np.sqrt(np.pi)
    g_self = -1j / 4.0 * (1 + (2j / np.pi) *
                          (np.log(kb * a_eq / 2) + 0.5772156649 - 1j * np.pi / 2))
    G = green_2d(pts, pts, kb)
    np.fill_diagonal(G, g_self)
    Gb = kb ** 2 * dA * G
    th = np.linspace(0, 2 * np.pi, n_rx, endpoint=False)
    rx = 2.0 * np.stack([np.cos(th), np.sin(th)], axis=1)
    Gs = kb ** 2 * dA * green_2d(rx, pts, kb)
    return Gb, Gs, pts


def make_dirs(n_inc=8):
    th = np.linspace(0, 2 * np.pi, n_inc, endpoint=False) + np.pi / n_inc
    return np.stack([np.cos(th), np.sin(th)], axis=1)


LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1_indep_corr.log"), "w", encoding="utf-8")


def say(*a):
    print(*a, flush=True)
    LOG.write(" ".join(str(x) for x in a) + "\n"); LOG.flush()


def sigma_max(jvp, vjp, D, ncv=1, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(D); x0 /= np.linalg.norm(x0)

    def MTM(x):
        return vjp(jvp(x))
    op = LinearOperator((D, D), matvec=MTM, rmatvec=MTM, dtype=np.float64)
    try:
        ev = eigsh(op, k=1, which="LM", tol=tol, maxiter=3000, v0=x0,
                   return_eigenvectors=False)
        s = float(np.sqrt(max(float(np.atleast_1d(ev).max()), 0.0)))
    except Exception as e:
        say("   [eigsh failed]", type(e).__name__); s = float("nan")
    # 幂迭代下界（少量步，仅作方向性核对）
    u = rng.standard_normal(D); u /= np.linalg.norm(u); sp = 0.0
    for _ in range(30):
        v = jvp(u); nv = np.linalg.norm(v)
        if nv < 1e-30: break
        w = vjp(v / nv); nu = np.linalg.norm(w)
        if nu < 1e-30: break
        u = w / nu; sp = nv
    return s, sp


def main():
    Gb, Gs, pts = build()
    dirs = make_dirs(8); kb = 2 * np.pi
    Ei = np.exp(-1j * kb * (pts @ dirs.T)).T.astype(np.complex128)
    z = np.load(os.path.join(CACHE, "ds_n100_nx32_ny32_seed2_snr300.0_c6cf983a78.npz"))
    chis, ys = z["chis"], z["ys"]
    BATCH = 16
    y_np = ys[:BATCH].astype(np.complex128)

    dev = "cpu"
    GbT = torch.as_tensor(Gb, dtype=torch.complex128, device=dev)
    GsT = torch.as_tensor(Gs, dtype=torch.complex128, device=dev)
    EiT = torch.as_tensor(Ei, dtype=torch.complex128, device=dev)
    yT = torch.as_tensor(y_np, dtype=torch.complex128, device=dev)
    n, I = 1024, 8
    gs_diag = (torch.abs(GsT) ** 2).sum(dim=0)
    ALPHA = GAMMA = 0.5
    EPS = 0.03

    def gb_apply(w):
        B = w.shape[0]
        return torch.matmul(GbT, w.reshape(-1, n).T).T.reshape(B, I, n)

    def gs_apply(w):
        B = w.shape[0]
        return torch.matmul(GsT, w.reshape(-1, n).T).T.reshape(B, I, -1)

    def cg_ls(E, y, cg_lam_rel=0.05, n_cg=15):
        B = E.shape[0]

        def apply_A(x):
            Ex = E * x.unsqueeze(1)
            m = torch.matmul(GsT, Ex.reshape(-1, n).T).T.reshape(B, I, -1)
            Gm = torch.matmul(GsT.conj().T, m.reshape(-1, m.shape[-1]).T).T.reshape(E.shape)
            return (E.conj() * Gm).real.sum(dim=1)
        b = (E.conj() * torch.matmul(
            GsT.conj().T, y.reshape(-1, y.shape[-1]).T).T.reshape(E.shape)).real.sum(dim=1)
        diagA = ((torch.abs(E) ** 2) * gs_diag).sum(dim=1)
        lam = cg_lam_rel * diagA + 1e-12
        x = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = (r * r).sum(dim=1)
        for _ in range(n_cg):
            Ap = apply_A(p) + lam * p
            pAp = (p * Ap).sum(dim=1).clamp_min(1e-20)
            al = (rs / pAp).unsqueeze(1)
            x = x + al * p
            r = r - al * Ap
            rs_new = (r * r).sum(dim=1)
            be = (rs_new / rs.clamp_min(1e-20)).unsqueeze(1)
            p = r + be * p
            rs = rs_new
        return x

    def init_state(y):
        EiB = EiT.unsqueeze(0).expand(y.shape[0], -1, -1)
        chi0 = cg_ls(EiB, y).clamp(0.0, 0.5)
        w0 = chi0.unsqueeze(1) * EiT.unsqueeze(0)
        return chi0, w0

    def flat(chi, w):
        return torch.cat([chi, w.real.reshape(chi.shape[0], -1),
                          w.imag.reshape(chi.shape[0], -1)], dim=1).detach()

    chi0, w0 = init_state(yT)
    zr = flat(chi0, w0)
    D = zr.numel()

    feat_scale = torch.full((49,), 2.0)
    feat_scale[16:32] = 3.0
    feat_scale = feat_scale.reshape(1, 49, 1, 1)

    def make_H_f(corr):
        def f(zrv):
            ch = zrv[:, :n]
            wr = zrv[:, n:n + n * I].reshape(zrv.shape[0], I, n)
            wi = zrv[:, n + n * I:].reshape(zrv.shape[0], I, n)
            w = wr + 1j * wi
            Ew = EiT.unsqueeze(0) + gb_apply(w)
            w_t = (1 - ALPHA) * w + ALPHA * ch.unsqueeze(1) * Ew
            E = EiT.unsqueeze(0) + gb_apply(w_t)
            resid = gs_apply(w_t) - yT
            resid_dom = torch.matmul(GsT.conj().T,
                                     resid.reshape(-1, resid.shape[-1]).T).T.reshape(
                zrv.shape[0], I, -1)
            feat = torch.cat([resid_dom.real.reshape(zrv.shape[0], -1),
                              resid_dom.imag.reshape(zrv.shape[0], -1),
                              E.real.reshape(zrv.shape[0], -1),
                              E.imag.reshape(zrv.shape[0], -1),
                              w_t.real.reshape(zrv.shape[0], -1),
                              w_t.imag.reshape(zrv.shape[0], -1), ch], dim=1)
            feat = feat.reshape(zrv.shape[0], -1, 32, 32)
            feat = torch.tanh(feat / feat_scale)
            h = F.relu(F.conv2d(feat, corr["c1w"], corr["c1b"], padding=1))
            h = F.relu(F.conv2d(h, corr["c2w"], corr["c2b"], padding=1))
            d = F.conv2d(h, corr["c3w"], corr["c3b"], padding=1)
            return (EPS * torch.tanh(d)).reshape(zrv.shape[0], -1)
        return f

    def make_comp_f(corr):
        fh = make_H_f(corr)

        def f(zrv):
            ch = zrv[:, :n]
            wr = zrv[:, n:n + n * I].reshape(zrv.shape[0], I, n)
            wi = zrv[:, n + n * I:].reshape(zrv.shape[0], I, n)
            w = wr + 1j * wi
            Ew = EiT.unsqueeze(0) + gb_apply(w)
            w_t = (1 - ALPHA) * w + ALPHA * ch.unsqueeze(1) * Ew
            E = EiT.unsqueeze(0) + gb_apply(w_t)
            chi_t = (1 - GAMMA) * ch + GAMMA * cg_ls(E, yT)
            dchi = fh(zrv)
            chi_out = (ch + (chi_t - ch) + dchi).clamp(0.0, 0.5)
            w_out = w + (w_t - w)
            return torch.cat([chi_out, w_out.real.reshape(zrv.shape[0], -1),
                              w_out.imag.reshape(zrv.shape[0], -1)], dim=1)
        return f

    def jvpf(f):
        def g(x):
            xt = torch.as_tensor(x.reshape(zr.shape), dtype=zr.dtype, device=dev)
            _, out = torch.func.jvp(f, (zr,), (xt,))
            return out.detach().cpu().numpy().reshape(-1)
        return g

    def vjpf(f):
        def g(yv):
            Bf = BATCH
            # 输出维度从函数自身推断
            with torch.no_grad():
                Rf = f(zr).shape[1]
            yt = torch.as_tensor(yv.reshape(Bf, Rf), dtype=zr.dtype, device=dev)
            zc = zr.clone().detach().requires_grad_(True)
            gr = torch.autograd.grad(f(zc), (zc,), grad_outputs=yt, allow_unused=True)[0]
            return gr.detach().cpu().numpy().reshape(-1)
        return g

    # 载入 checkpoints。只复算各模式下取到最大值的层：
    #   constrained:  Lip(H) max @ k=7,  composite max @ k=3
    #   unconstrained: Lip(H) max @ k=4, composite max @ k=2
    PLAN = {"constrained": {"lip_k": 7, "comp_k": 3},
            "unconstrained": {"lip_k": 4, "comp_k": 2}}
    for mode, spec in PLAN.items():
        sd = torch.load(os.path.join(CKPT, "ckpt_%s_K8.pt" % mode), map_location=dev)
        for quantity, k in (("Lip_H", spec["lip_k"]), ("composite", spec["comp_k"])):
            corr = {}
            for c in ("c1", "c2", "c3"):
                corr[c + "w"] = sd["corrs.%d.%s.conv.weight" % (k, c)].to(torch.float64)
                corr[c + "b"] = sd["corrs.%d.%s.conv.bias" % (k, c)].to(torch.float64)
            f = make_H_f(corr) if quantity == "Lip_H" else make_comp_f(corr)
            s, sp = sigma_max(jvpf(f), vjpf(f), D, tol=1e-5)
            say("  %-13s %-9s layer %d: sigma_max=%.6f (power %.6f)"
                % (mode, quantity, k, s, sp))
    say("paper: constrained Lip(H)=0.3192 composite=13.032 ; unconstrained Lip(H)=1.743 composite=13.682")


if __name__ == "__main__":
    main()
