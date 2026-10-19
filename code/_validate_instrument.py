# -*- coding: utf-8 -*-
"""Instrument validation: FD power iteration vs exact SVD on a known linear map."""
import numpy as np, torch
from isp_core import build_matrices
import run_experiments as R

Gb, Gs, pts, rx = R.build_matrices(32, 32, dx=0.02)
Gb_t = torch.as_tensor(Gb, dtype=torch.complex64).to(R.DEV)
n = 1024; B = 4; I = 8
# known linear map: f(zr) = [(0.5*chi + 0.3*sum_i Re w_i), (A_w wr block...)]
# simpler: f(zr) = M zr with random M, sigma known
D = n * (1 + 2 * I)
M = torch.randn(D, D, device=R.DEV) / np.sqrt(D)
print("sigma_max(M) numpy-ish (torch svd):", torch.linalg.matrix_norm(M, 2).item())

def f(zr):
    return zr @ M.T

zr = torch.randn(B, D, device=R.DEV)
from isp_core import _power_sigma_fd
est = _power_sigma_fd(f, zr, iters=20)
print("FD power iteration estimate:", round(est, 4))

# also validate on the w->w physics block: (1-a)I + a*diag(chi)Gb
chi = torch.rand(B, n, device=R.DEV) * 0.5
a = 0.26
def f_w(zr):
    wr = zr[:, :n * I].reshape(B, I, n); wi = zr[:, n * I:].reshape(B, I, n)
    w = wr + 1j * wi
    Gw = torch.matmul(Gb_t, w.reshape(-1, n).T).T.reshape(B, I, n)
    w2 = (1 - a) * w + a * chi.unsqueeze(1) * Gw
    return torch.cat([w2.real.reshape(B, -1), w2.imag.reshape(B, -1)], dim=1)
zr2 = torch.randn(B, 2 * n * I, device=R.DEV)
est2 = _power_sigma_fd(f_w, zr2, iters=20)
# exact: block diagonal over i of (1-a)I + a*diag(chi)Gb
sig_exact = 0.0
with torch.no_grad():
    Aw = (1 - a) * torch.eye(n, device=R.DEV) + a * (chi[0].unsqueeze(1) * Gb_t)
    sig_exact = torch.linalg.matrix_norm(torch.cat([torch.cat([Aw.real, -Aw.imag], 1),
                                                    torch.cat([Aw.imag, Aw.real], 1)], 0), 2).item()
print("w-block: FD=%.4f  exact=%.4f" % (est2, sig_exact))
