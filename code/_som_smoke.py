# -*- coding: utf-8 -*-
"""Fast smoke test for the corrected SOM core in run_csi_som.py.

Validates: (i) the pseudo-inverse convention check, (ii) that the SOM RRE series
decreases monotonically-ish and lands below the Born-LS init.  Reduced K only.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_csi_som as R

Ks = [1, 2, 4, 8, 12]
t0 = time.time()
Gb, Gs, Ei, chi_te, y_te = R.load()
R._CHI_TRUE = chi_te
print("load %.1fs  Gs%s Gb%s" % (time.time() - t0, Gs.shape, Gb.shape), flush=True)

chi0 = R.born_init(Gb, Gs, Ei, y_te)
print("born init RRE = %.4f" % R.rre_batch(chi0, chi_te), flush=True)

for Ms in [6, 12]:
    s = R.run_som(Gb, Gs, Ei, y_te, chi0.copy(), Ks, Msub=Ms)
    print("[SOM M=%2d] " % Ms + "  ".join("K=%-3d RRE=%.4f(%.0fs)" % (k, s[k][0], s[k][1])
                                          for k in Ks), flush=True)
print("total %.0fs" % (time.time() - t0), flush=True)
