# -*- coding: utf-8 -*-
"""_warm_cache.py — generate the deterministic datasets ONCE and store them under
results/cache/, so that the audit / L4-L5 / seed jobs (and any replay of the main
suite) load in milliseconds instead of each regenerating 400 dense MoM solves.

Cost: 1000 samples x ~0.25 s single-threaded ~ 4 min.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isp_core import CACHE, build_matrices, make_dirs, make_dataset

NX = NY = 32
N_INC = 8
JOBS = [(300, 1, 20.0), (100, 2, 300.0),      # load_data() of every audit script
        (300, 11, 20.0), (300, 12, 20.0)]     # run_seeds.py repeats r=1,2

def main():
    t0 = time.time()
    Gb, Gs, pts, rx = build_matrices(NX, NY, dx=0.02)
    dirs = make_dirs(N_INC)
    before = set(os.listdir(CACHE)) if os.path.isdir(CACHE) else set()
    for n, seed, snr in JOBS:
        t = time.time()
        chis, ys, Ei = make_dataset(n, Gb, Gs, pts, dirs, 2 * np.pi, NX, NY,
                                    seed=seed, snr_db=snr)
        # independent recheck: a cache that is never written would otherwise pass silently
        # (np.savez_compressed appends ".npz" to a string path, so the atomic rename can
        #  miss its own temporary file -- that mistake is exactly what this guards against)
        t2 = time.time()
        _, _, _ = make_dataset(n, Gb, Gs, pts, dirs, 2 * np.pi, NX, NY,
                               seed=seed, snr_db=snr)
        hit = time.time() - t2
        after = set(os.listdir(CACHE))
        print("n=%-4d seed=%-3d snr=%-6s -> chis%s ys%s Ei%s  gen %.1fs  reload %.2fs  new-file=%s"
              % (n, seed, snr, chis.shape, ys.shape, Ei.shape, time.time() - t, hit,
                 sorted(after - before)), flush=True)
        before = after
        assert hit < 5.0, "cache miss after generation (%.2fs) -- caching is broken" % hit
    print("[warm] %.0fs total" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
