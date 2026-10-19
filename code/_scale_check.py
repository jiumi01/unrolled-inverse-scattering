# -*- coding: utf-8 -*-
import numpy as np, torch
from isp_core import build_matrices, make_dirs, make_dataset, incident_fields, random_contrast, forward_exact
Gb, Gs, pts, rx = build_matrices(32, 32, dx=0.02)
dirs = make_dirs(8); kb = 2*np.pi
Ei = incident_fields(pts, dirs, kb)
gs_norm2 = np.linalg.norm(Gs, 2) ** 2
print("||Gs||^2 =", gs_norm2)
rng = np.random.default_rng(0)
for t in range(3):
    chi = random_contrast(32, 32, rng)
    E, w, y = forward_exact(Gb, Gs, Ei, chi)
    num = np.sum(np.real(np.conj(Ei) * (Gs.conj().T @ y.T).T), axis=0)
    den = np.sum(np.abs(Ei)**2, axis=0) + 1e-4
    chi_born_raw = num / den
    chi_scaled = num / (den * gs_norm2)
    corr = np.corrcoef(chi_born_raw, chi)[0,1]
    print("raw chi range %.4g  scaled %.4g  true %.4g  corr=%.3f  RRE_scaled=%.3f"
          % (chi_born_raw.max(), chi_scaled.max(), chi.max(), corr,
             np.linalg.norm(chi_scaled-chi)/np.linalg.norm(chi)))
