# -*- coding: utf-8 -*-
"""Make placeholder PDFs so the LaTeX pipeline can be smoke-tested before the run finishes."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
os.makedirs(FIG, exist_ok=True)
names = ["fig_depth", "fig_extrap", "fig_noise", "fig_gain", "fig_pareto",
         "fig_gradnorm", "fig_ablation", "fig_frozen", "fig_images", "fig_alphac"]
for n in names:
    p = os.path.join(FIG, n + ".pdf")
    if os.path.exists(p):
        print("keep", n)
        continue
    fig = plt.figure(figsize=(3.5, 2.2))
    plt.plot([0, 1], [0, 1]); plt.title(n)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("placeholder", n)
