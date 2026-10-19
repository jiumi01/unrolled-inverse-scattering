# -*- coding: utf-8 -*-
"""R1 独立数值复核（第三轮）。不调用作者任何结论函数；
物理算子、初值求解、闭式耦合块的奇异值均由本脚本独立实现。
结果打印到 stdout。"""
import hashlib, os, json
import numpy as np
from scipy.special import hankel2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "cache")


# ----------------------------------------------------------------- 物理算子
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
    return Gb, Gs, pts, rx


def make_dirs(n_inc=8):
    th = np.linspace(0, 2 * np.pi, n_inc, endpoint=False) + np.pi / n_inc
    return np.stack([np.cos(th), np.sin(th)], axis=1)


def incident_fields(pts, dirs, kb):
    return np.exp(-1j * kb * (pts @ dirs.T)).T.astype(np.complex128)


def ds_key(n, Gb, Gs, Ei, nx, ny, seed, snr):
    h = hashlib.md5()
    h.update(np.ascontiguousarray(Gb, dtype=np.complex128).tobytes())
    h.update(np.ascontiguousarray(Gs, dtype=np.complex128).tobytes())
    h.update(np.ascontiguousarray(Ei, dtype=np.complex128).tobytes())
    return "ds_n%d_nx%d_ny%d_seed%d_snr%s_%s" % (n, nx, ny, seed, snr, h.hexdigest()[:10])


# ------------------------------------------------------------- Born-LS 初值
def cg_ls(Gs, E, y, cg_lam_rel=0.05, n_cg=15):
    """独立实现：阻尼 Tikhonov 正规方程 CG（按 Algorithm 1 描述）。"""
    B = E.shape[0]
    gs_diag = (np.abs(Gs) ** 2).sum(axis=0)

    def apply_A(x):
        Ex = E * x[:, None, :]
        m = (Gs @ Ex.reshape(-1, Ex.shape[-1]).T).T.reshape(B, E.shape[1], -1)
        Gm = (Gs.conj().T @ m.reshape(-1, m.shape[-1]).T).T.reshape(E.shape)
        return (E.conj() * Gm).real.sum(axis=1)

    b = (E.conj() * (Gs.conj().T @ y.reshape(-1, y.shape[-1]).T)
         .T.reshape(E.shape)).real.sum(axis=1)
    diagA = ((np.abs(E) ** 2) * gs_diag).sum(axis=1)
    lam = cg_lam_rel * diagA + 1e-12
    x = np.zeros_like(b); r = b.copy(); p = r.copy()
    rs = (r * r).sum(axis=1)
    for _ in range(n_cg):
        Ap = apply_A(p) + lam * p
        pAp = np.maximum((p * Ap).sum(axis=1), 1e-20)
        alpha = (rs / pAp)[:, None]
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(axis=1)
        beta = (rs_new / np.maximum(rs, 1e-20))[:, None]
        p = r + beta * p
        rs = rs_new
    return x


def direct_ls(Gs, E, y, cg_lam_rel=0.05):
    """收敛解：逐样本解正规方程 (A + diag(lam)) x = b（numpy 稠密）。"""
    B = E.shape[0]
    n = E.shape[2]
    gs_diag = (np.abs(Gs) ** 2).sum(axis=0)
    out = np.zeros((B, n))
    GsHGs = Gs.conj().T @ Gs          # (n,n)
    for b in range(B):
        Eb = E[b]                      # (I,n)
        A = np.zeros((n, n), dtype=complex)
        for i in range(Eb.shape[0]):
            A += np.conj(Eb[i])[:, None] * GsHGs * Eb[i][None, :]
        bb = np.zeros(n, dtype=complex)
        for i in range(Eb.shape[0]):
            bb += np.conj(Eb[i]) * (Gs.conj().T @ y[b, i])
        diagA = ((np.abs(Eb) ** 2) * gs_diag[None, :]).sum(axis=0)
        lam = cg_lam_rel * diagA + 1e-12
        out[b] = np.linalg.solve(A + np.diag(lam), bb).real
    return out


def gb_apply(Gb, w):
    B, I, n = w.shape
    return np.matmul(Gb, w.reshape(-1, n).T).T.reshape(B, I, n)


def main():
    print("=" * 72)
    Gb, Gs, pts, rx = build()
    dirs = make_dirs(8)
    kb = 2 * np.pi
    Ei = incident_fields(pts, dirs, kb)

    # --- (a) ||Gb||_2 与 beta0
    gbn = float(np.linalg.norm(Gb, 2))
    beta0 = 0.5 * gbn
    print("[a] ||Gb||_2 (my SVD)        = %.10f   (paper gb_norm2 = 1.6606348724)" % gbn)
    print("    beta0 = chi_max*||Gb||_2   = %.10f   (paper beta0       = 0.8303174362)" % beta0)
    print("    gamma_w = 1 - 0.5*(1-beta0)= %.10f   (paper NumGammaW   = 0.915)" % (1 - 0.5 * (1 - beta0)))

    # --- digest 校验（证明我的算子与作者逐位一致）
    key = ds_key(100, Gb, Gs, Ei, 32, 32, 2, 300.0)
    print("    cache key (my build)      = %s" % key)
    cpath = os.path.join(CACHE, key + ".npz")
    print("    cache file exists         = %s" % os.path.exists(cpath))

    # --- (b) 操作点与 E 的列范数
    z = np.load(cpath)
    chis, ys = z["chis"], z["ys"]          # (100,1024) float32, (100,8,32) c64
    BATCH = 16
    chi_t = chis[:BATCH].astype(np.float64)
    y_t = ys[:BATCH].astype(np.complex128)
    EiB = np.broadcast_to(Ei[None], (BATCH, 8, 1024))

    for tag, solver in (("15-step CG", cg_ls), ("converged LS", direct_ls)):
        chi0 = np.clip(solver(Gs, EiB, y_t), 0.0, 0.5)
        w0 = chi0[:, None, :] * Ei[None]
        E0 = Ei[None] + gb_apply(Gb, w0)      # (B,8,n)
        colnorm = np.sqrt((np.abs(E0) ** 2).sum(axis=1))   # (B,n)
        Emax_col = float(colnorm.max())
        Emax_abs = float(np.abs(E0).max())
        print("-" * 72)
        print("[b:%s] operating point" % tag)
        print("    E_max colnorm  = %.10f   (paper max_colnorm_E = 3.1550936212)" % Emax_col)
        print("    E_max |E|      = %.10f   (paper max_abs_E      = 1.4088302173)" % Emax_abs)
        print("    linf ratio     = %.10f   (paper E_inf_ratio    = 2.2395130247)"
              % (Emax_col / Emax_abs))
        print("    alpha_c = 1/E  = %.10f   (paper alpha_c        = 0.3169478057)" % (1.0 / Emax_col))
        # --- 闭式 sigma_max(C) 用 numpy SVD 独立核对（批内取最大）
        # w 的展平序为 (I,n) 行优先：索引 = i*n + r；C 的第 r 列在行 {i*n+r} 上取值 θα E[i,r]
        I, n = 8, 1024
        theta_alpha = 0.5
        best = 0.0
        for b in range(BATCH):
            rows = (np.arange(I)[:, None] * n + np.arange(n)[None, :]).reshape(-1)
            Cmat = np.zeros((I * n, n), dtype=complex)
            for r in range(n):
                Cmat[rows[r::n], r] = theta_alpha * E0[b, :, r]
            s = float(np.linalg.svd(Cmat, compute_uv=False)[0])
            best = max(best, s)
        print("    sigma_max(C) SVD (batch max) = %.10f   (paper exact_sigma_C  = 1.5775468106)"
              % best)
        print("    check theta_alpha*Emax_col   = %.10f" % (0.5 * Emax_col))
    print("=" * 72)


if __name__ == "__main__":
    main()
