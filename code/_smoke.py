# -*- coding: utf-8 -*-
import numpy as np, torch, isp_core as ic
Gb, Gs, pts, rx = ic.build_matrices(16, 16, dx=0.1)
dirs = ic.make_dirs(8); kb = 2*np.pi
Ei = ic.incident_fields(pts, dirs, kb)
chi = ic.random_contrast(16, 16, np.random.default_rng(0))
E, w, y = ic.forward_exact(Gb, Gs, Ei, chi)
print("y shape", y.shape, "|y| mean", float(np.abs(y).mean()))
m = ic.UnrolledISP(3, Gb, Gs, Ei, 16, 16, 8, mode="constrained", eps_bar=0.05).cuda()
chi_b = torch.tensor(chi, dtype=torch.float32).unsqueeze(0).cuda().repeat(4,1)
y_b = torch.tensor(y, dtype=torch.complex64).unsqueeze(0).cuda().repeat(4,1,1)
GsH_y = torch.matmul(m.Gs.conj().T, y_b.reshape(-1, y_b.shape[-1]).T).T.reshape(y_b.shape[0], y_b.shape[1], -1)
num = (m.Ei.conj().unsqueeze(0) * GsH_y).real.sum(1)
den = (torch.abs(m.Ei)**2).sum(0).unsqueeze(0) + 1e-4
chi0 = num/den; w0 = chi0.unsqueeze(1)*m.Ei.unsqueeze(0)
chi_k, w_k = m(chi0, w0, y_b)
print("forward ok, chi range", chi_k.min().item(), chi_k.max().item())
loss = ((chi_k-chi_b)**2).mean()
loss.backward(); m.clamp_sn()
print("backward ok, loss", float(loss.item()))
g = ic.estimate_gain(m, 0, chi0, w0, y_b)
print("gain layer0:", g)
print("SMOKE OK")
