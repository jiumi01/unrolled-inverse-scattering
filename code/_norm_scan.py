# -*- coding: utf-8 -*-
import numpy as np
from isp_core import build_matrices
for dx in [0.02, 0.025, 0.03, 0.04]:
    Gb, Gs, pts, rx = build_matrices(32, 32, dx=dx)
    print("dx=%.3f  ||Gb||2=%.4f  beta_cert(0.5)=%.4f  beta_cert(0.6)=%.4f"
          % (dx, np.linalg.norm(Gb, 2), 0.5 * np.linalg.norm(Gb, 2), 0.6 * np.linalg.norm(Gb, 2)))
