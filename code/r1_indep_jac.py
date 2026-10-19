# -*- coding: utf-8 -*-
"""R1 独立复核（续）：物理核心层 Jacobian 的 sigma_max。
本脚本从零实现物理核心层（Algorithm 1，无修正项），用 torch.func.jvp /
torch.autograd.grad 得到精确 J 与 J^T 作用，再用 scipy eigsh 求 sigma_max(J^T J)。
不使用作者的任何测量/结论函数。"""
import os
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import hankel2

torch.set_grad_enabled(True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "cache")


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

    def gb_apply(w):
        B = w.shape[0]
        return torch.matmul(GbT, w.reshape(-1, n).T).T.reshape(B, I, n)

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

    ALPHA = GAMMA = 0.5

    def layer(chi, w):
        Ew = EiT.unsqueeze(0) + gb_apply(w)
        w_t = (1 - ALPHA) * w + ALPHA * chi.unsqueeze(1) * Ew
        E = EiT.unsqueeze(0) + gb_apply(w_t)
        chi_t = (1 - GAMMA) * chi + GAMMA * cg_ls(E, yT)
        chi_out = (chi + (chi_t - chi)).clamp(0.0, 0.5)
        w_out = w + (w_t - w)
        return chi_out, w_out

    def f(zr):
        ch = zr[:, :n]
        wr = zr[:, n:n + n * I].reshape(zr.shape[0], I, n)
        wi = zr[:, n + n * I:].reshape(zr.shape[0], I, n)
        c, ww = layer(ch, wr + 1j * wi)
        return torch.cat([c, ww.real.reshape(zr.shape[0], -1),
                          ww.imag.reshape(zr.shape[0], -1)], dim=1)

    chi0, w0 = init_state(yT)
    zr = flat(chi0, w0)
    D = zr.numel()
    print("state dim =", D, " (expect 16*1024*(1+16) =", BATCH * n * (1 + 2 * I), ")")

    def jvp(x):
        xt = torch.as_tensor(x.reshape(zr.shape), dtype=zr.dtype, device=dev)
        _, out = torch.func.jvp(f, (zr,), (xt,))
        return out.detach().cpu().numpy().reshape(-1)

    def vjp(yv):
        yt = torch.as_tensor(yv.reshape(BATCH, -1), dtype=zr.dtype, device=dev)
        zc = zr.clone().detach().requires_grad_(True)
        g = torch.autograd.grad(f(zc), (zc,), grad_outputs=yt, allow_unused=True)[0]
        return g.detach().cpu().numpy().reshape(-1)

    def MTM(x):
        return vjp(jvp(x))

    def power(x):
        return jvp(x)

    # 交叉核对：随机向量 Rayleigh 商 与 Lanczos
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(D)
    x0 /= np.linalg.norm(x0)
    op = LinearOperator((D, D), matvec=MTM, rmatvec=MTM, dtype=np.float64)
    ev = eigsh(op, k=1, which="LM", tol=1e-6, maxiter=3000, v0=x0,
               return_eigenvectors=False)
    sig = float(np.sqrt(max(float(np.atleast_1d(ev).max()), 0.0)))
    print("sigma_max(J_physics_core) [my Lanczos] = %.10f" % sig)
    print("   paper A_physics_core.sigma_max_full  = 14.4185579370")

    # 幂迭代独立交叉核对（下界）
    u = rng.standard_normal(D); u /= np.linalg.norm(u)
    s = 0.0
    for _ in range(200):
        v = power(u); nv = np.linalg.norm(v)
        if nv < 1e-30: break
        w_ = vjp(v / nv); nu = np.linalg.norm(w_)
        if nu < 1e-30: break
        u = w_ / nu; s = nv
    print("sigma_max(J_physics_core) [my power-iter] = %.10f" % s)

    # 附：w-块（D）与 chi-块（A）的 sigma_max 供 tab:cert 对照
    def A_map(chi_zr):
        zz = torch.as_tensor(chi_zr.reshape(BATCH, n), dtype=zr.dtype, device=dev)
        _, ww = layer(zz, w0)
        return torch.cat([ww.real.reshape(BATCH, -1), ww.imag.reshape(BATCH, -1)], dim=1)

    print("done")


if __name__ == "__main__":
    main()
