# -*- coding: utf-8 -*-
"""make_figs.py — generate all paper figures from results/*.json|npz.

SIZING DISCIPLINE (added 2026-10-13, MDPI single column)
--------------------------------------------------------
The MDPI (Mathematics) text block is ``\\textwidth = 394.35522 pt = 5.4566 in``.
Every figure is created with ``figsize[0] = TEXTW`` and inserted in LaTeX with
``width=\\linewidth``, so the LaTeX scaling factor is exactly 1.0 and the
rendered glyph size equals the value set in rcParams.

Previously this script sized for an IEEE two-column layout (3.5 in single /
7.16 in double) while the manuscript is single-column, and saved with
``bbox_inches="tight"``.  Because tight-cropping shrinks some figures and
(combined with ``clip_on=False``) inflates others, the effective LaTeX scale
ranged from 0.70x to 1.77x, i.e. the *same* 9 pt setting rendered anywhere
between 6.3 pt and 15.9 pt -- a 2.5x spread.  That is the root cause of the
"figures are inconsistently sized" defect.

Fixes applied here:
  1. figsize width is pinned to the text block (TEXTW) for every figure;
  2. ``bbox_inches`` is NOT used, so the PDF box equals figsize exactly;
  3. spacing is handled by ``layout="constrained"`` instead of tight-cropping;
  4. ``clip_on=False`` is removed (it was inflating the bounding box);
  5. one rcParams block, so every panel has the same glyph sizes.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- constants --
PT_PER_IN = 72.27
TEXTW = 394.35522 / PT_PER_IN          # 5.4566 in == \textwidth of mdpi.cls
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
NUM = os.path.normpath(os.path.join(HERE, "..", "paper_mdpi", "numbers.tex"))
os.makedirs(FIG, exist_ok=True)


def set_style():
    """One style for every panel. Rendered sizes == these values (scale 1.0)."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 4.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150,
    })


def save(fig, name):
    """Save with the PDF box exactly equal to figsize (no tight cropping)."""
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)


def panel(w=TEXTW, ratio=0.60, ncols=2, **kw):
    """Figure + axes whose width is pinned to the text block.

    Two-panel figures use explicit margins instead of `layout="constrained"`:
    the shared legend sits in the reserved strip ABOVE the panels and the
    (a)/(b) tags in the strip BELOW them.  With constrained layout the figure
    legend was drawn on top of the axes, because a figure-level legend is not
    part of the layout engine's rect calculation.
    """
    if ncols == 2:
        fig, ax = plt.subplots(1, 2, figsize=(w, w * ratio * 1.34))
        fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.20,
                            wspace=0.30)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(w, w * ratio),
                               layout="constrained")
    return fig, ax


def shared_legend(fig, ax, ncol):
    """Legend strip above a two-panel figure, clear of the data region."""
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.995))


def tag_below(a, tag):
    """Sub-panel tag centred under the panel, below the x-axis label.

    The tag used to be an axes title, which placed "(a)" above the data where it
    read as a figure-level heading.  Journals set sub-panel tags underneath, so the
    tag is now drawn in axes coordinates below the axis.
    """
    a.text(0.5, -0.215, tag, transform=a.transAxes, ha="center", va="top",
           fontsize=10)


def style_ax(a, tag=None):
    a.spines[["top", "right"]].set_visible(False)
    if tag:
        tag_below(a, tag)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def num(macro, default=None):
    """Read a numeric macro from the single source paper_mdpi/numbers.tex."""
    try:
        with open(NUM, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return default
    tag = "\\newcommand{\\%s}{" % macro
    i = src.find(tag)
    if i < 0:
        return default
    j = src.find("}", i + len(tag))
    raw = src[i + len(tag):j].replace("\\,", "").replace("$", "")
    try:
        return float(raw)
    except ValueError:
        return default


# ============================================================ fig_structure ==
def fig_structure():
    """Block diagram of one unrolled layer and the four Jacobian blocks.

    This is the visual anchor for Lemma 4.1 (exact block structure) and for the
    whole theory section, which previously ran for ~14 pages without a figure.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(TEXTW, TEXTW * 0.56), layout="constrained")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, s, fc, fs=8.2, ec="0.30"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.008,rounding_size=0.018",
                                    linewidth=0.9, edgecolor=ec, facecolor=fc,
                                    zorder=2))
        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center",
                fontsize=fs, zorder=4)

    def arrow(p0, p1, col="0.25", ls="-", lw=1.1, rad=0.0, ms=9):
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=%f" % rad,
                                     arrowstyle="-|>", mutation_scale=ms,
                                     linewidth=lw, linestyle=ls, color=col,
                                     zorder=3, shrinkA=0, shrinkB=0))

    C_GREY, C_BLUE, C_ORANGE = "#F2F2F2", "#DCE9F5", "#FBE7CE"
    # ---- input states -------------------------------------------------------
    ax.text(0.035, 0.78, r"$\chi^{k}$", ha="center", va="center", fontsize=10)
    ax.text(0.035, 0.22, r"$w^{k}$", ha="center", va="center", fontsize=10)

    # ---- contrast path (row 1) ----------------------------------------------
    box(0.22, 0.62, 0.28, 0.28, r"contrast step" "\n" r"$\hat\chi^{\mathrm{LS}}(\chi,w)$",
        C_GREY, fs=8.0)
    ax.add_patch(plt.Circle((0.585, 0.76), 0.032, facecolor="white",
                            edgecolor="0.30", linewidth=0.9, zorder=3))
    ax.text(0.585, 0.76, "+", ha="center", va="center", fontsize=11, zorder=4)
    box(0.665, 0.62, 0.125, 0.28, r"$\Pi$", C_ORANGE, fs=11)
    # ---- field path (row 2) -------------------------------------------------
    box(0.22, 0.10, 0.34, 0.28,
        r"field step" "\n" r"$w+\theta\alpha\,[\chi\odot(E^{i}+\mathbf{G}_b w)-w]$",
        C_BLUE, fs=8.0)

    # ---- outputs ------------------------------------------------------------
    ax.text(0.945, 0.78, r"$\chi^{k+1}$", ha="center", va="center", fontsize=10)
    ax.text(0.945, 0.24, r"$w^{k+1}$", ha="center", va="center", fontsize=10)

    # ---- data flow (solid) --------------------------------------------------
    arrow((0.075, 0.78), (0.213, 0.77))
    arrow((0.505, 0.76), (0.548, 0.76))
    arrow((0.622, 0.76), (0.658, 0.76))
    arrow((0.795, 0.76), (0.878, 0.78))
    arrow((0.075, 0.24), (0.213, 0.24))
    arrow((0.565, 0.24), (0.878, 0.24))
    # learned correction enters the sum node from above
    ax.text(0.585, 0.925, r"$H_k$", ha="center", va="center", fontsize=8.4)
    arrow((0.585, 0.895), (0.585, 0.80), lw=0.9, ms=7)

    # ---- Jacobian blocks (dashed; they are derivatives, not data) -----------
    arrow((0.045, 0.835), (0.28, 0.905), col="0.35", ls=(0, (3, 2)),
          lw=0.8, rad=0.18, ms=6)
    ax.text(0.155, 0.945, r"$A$", fontsize=8.2, ha="center", color="0.15")
    arrow((0.045, 0.185), (0.30, 0.098), col="0.35", ls=(0, (3, 2)),
          lw=0.8, rad=-0.16, ms=6)
    ax.text(0.155, 0.055, r"$D$", fontsize=8.2, ha="center", color="0.15")
    # B: w -> contrast (crosses upward); C: chi -> field (crosses downward)
    arrow((0.075, 0.26), (0.225, 0.615), col=OKABE_ITO[5], ls=(0, (3, 2)),
          lw=0.9, rad=0.30, ms=6)
    ax.text(0.030, 0.455, r"$B$", fontsize=8.2, ha="center", color=OKABE_ITO[5])
    arrow((0.075, 0.74), (0.245, 0.385), col=OKABE_ITO[6], ls=(0, (3, 2)),
          lw=0.9, rad=0.30, ms=6)
    ax.text(0.175, 0.545, r"$C=\theta\alpha\,\mathrm{diag}(E)$", fontsize=8.0,
            ha="center", color=OKABE_ITO[6])

    save(fig, "fig_structure")


# =============================================================== fig_regime ==
def fig_regime():
    """Where each theorem applies, in the (beta_0, alpha) plane.

    The two regimes are not exclusive and that is the point of the paper: for
    the synthetic certified box one has beta_0 < 1 (so the *field subsystem* is
    contractive, Thm. field) while simultaneously alpha > alpha_c (so the
    *composite layer* is provably non-contractive, Prop. imposs).  Stability has
    to be layered because these live on different objects.
    """
    set_style()
    ac = num("NumAlphaC", 0.3169)
    b_cert = num("betacert", 0.8303)
    b_meas = num("NumBetaMeas", 2.015)
    b_fres = num("NumFresnelBeta", 14.3987)
    a_op = 0.5

    fig, ax = plt.subplots(figsize=(TEXTW, TEXTW * 0.62), layout="constrained")
    xlo, xhi = 0.05, 30.0
    # field-subsystem contraction: beta_0 < 1, all alpha in (0,1]
    ax.axvspan(xlo, 1.0, color=OKABE_ITO[3], alpha=0.16, lw=0, zorder=0)
    # proved non-contractivity of the composite: alpha > alpha_c
    ax.axhspan(ac, 1.0, color=OKABE_ITO[6], alpha=0.14, lw=0, zorder=0)
    ax.axvline(1.0, color=OKABE_ITO[3], ls="-", lw=1.1, zorder=2)
    ax.axhline(ac, color=OKABE_ITO[6], ls="-", lw=1.1, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(0, 1)
    ax.set_xlabel(r"weak-scattering constant $\beta_0=\chi_{\max}\|\mathbf{G}_b\|_2$")
    ax.set_ylabel(r"step size $\alpha=\gamma$")

    ax.text(0.30, 0.90, r"composite provably" "\n" r"non-contractive:" "\n" r"$\sigma_{\max}(J)\geq\theta\alpha\mathcal{E}$",
            fontsize=8.2, color=OKABE_ITO[6], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))
    ax.text(0.135, 0.16, r"field subsystem" "\n" r"contractive" "\n" r"for $\beta_0<1$",
            fontsize=8.2, color=OKABE_ITO[3], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))
    ax.text(xhi * 0.42, 0.185, r"neither certificate applies;" "\n" r"the rollout stays bounded",
            fontsize=8.2, color="0.30", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))

    pts = [(b_cert, a_op, "synthetic, certified box", OKABE_ITO[0], "o"),
           (b_meas, a_op, "no-projection arm", OKABE_ITO[6], "s"),
           (b_fres, a_op, "Fresnel 2-D", OKABE_ITO[5], "^")]
    for b, a, lab, col, mk in pts:
        ax.plot([b], [a], marker=mk, ms=7.0, color=col, zorder=5,
                mec="white", mew=0.8)
    ax.annotate("synthetic, certified box",
                xy=(b_cert, a_op), xytext=(b_cert * 1.45, a_op + 0.20),
                fontsize=7.4, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.35"))
    ax.annotate("no-projection arm",
                xy=(b_meas, a_op), xytext=(b_meas * 1.30, a_op - 0.26),
                fontsize=7.4, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.35"))
    ax.annotate("Fresnel 2-D",
                xy=(b_fres, a_op), xytext=(b_fres * 0.42, a_op + 0.27),
                fontsize=7.4, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.35"))
    style_ax(ax)
    save(fig, "fig_regime")


# ================================================================ fig_depth ==
def fig_depth():
    cls = load("classical_curve.json"); dep = load("depth_curve.json")
    d40 = (load("depth_curve_40ep.json") or {}).get("rre") or {}
    set_style()
    fig, ax = panel(ncols=1, ratio=0.62)
    ks_c = sorted(int(k) for k in cls)
    ax.plot(ks_c, [cls[str(k)] for k in ks_c], color=OKABE_ITO[0], marker=MARKERS[0],
            label="Classical alternating")
    for idx, (mode, lab) in enumerate([("unconstrained", "Unconstrained unrolling"),
                                       ("constrained", "Constrained (ours)")]):
        pts = sorted(((int(k.rsplit("_", 1)[-1][1:]), v) for k, v in dep.items()
                      if k.startswith(mode + "_")))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=OKABE_ITO[2 + idx * 4],
                marker=MARKERS[1 + idx], label=lab)
        p40 = sorted(((int(k.rsplit("_", 1)[-1][1:]), v) for k, v in d40.items()
                      if k.startswith(mode + "_")))
        ax.plot([p[0] for p in p40], [p[1] for p in p40], color=OKABE_ITO[2 + idx * 4],
                marker=MARKERS[1 + idx], linestyle="--", mfc="none",
                label=lab.split(" (")[0] + " (uniform 40-epoch)")
    ax.set_xlabel("Depth $K$"); ax.set_ylabel("Relative reconstruction error")
    # linear axis: the data span 0.36..0.56 (factor < 2), so a log axis produces
    # no decade ticks and the y tick labels vanish entirely (observed in print).
    lin_ticks = [0.36, 0.40, 0.44, 0.48, 0.52, 0.56]
    ax.set_yticks(lin_ticks)
    ax.set_yticklabels(["%.2f" % v for v in lin_ticks])
    ax.set_ylim(0.355, 0.575)
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16, 24, 50, 100])
    ax.set_xticklabels(["2", "4", "8", "16", "24", "50", "100"])
    ax.minorticks_off()
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=2.2)
    style_ax(ax)
    save(fig, "fig_depth")


# =============================================================== fig_extrap ==
def fig_extrap():
    extrap = load("extrap.json")
    set_style()
    fig, axes = panel(ratio=0.44, ncols=2)
    for idx, (mode, lab) in enumerate([("classical", "Classical"),
                                       ("unconstrained", "Unconstrained"),
                                       ("constrained", "Constrained (ours)")]):
        e = extrap[mode]; x = np.arange(len(e["residual"])) * 5
        axes[0].semilogy(x, np.maximum(np.array(e["residual"]), 1e-16),
                         color=OKABE_ITO[[0, 2, 6][idx]], marker=MARKERS[idx],
                         markevery=2, label=lab)
        axes[1].plot(x, e["rre"], color=OKABE_ITO[[0, 2, 6][idx]], marker=MARKERS[idx],
                     markevery=2, label=lab)
    axes[0].set_xlabel(r"Extrapolation step $k'-K$")
    axes[0].set_ylabel("Fixed-point residual")
    axes[1].set_xlabel(r"Extrapolation step $k'-K$"); axes[1].set_ylabel("RRE")
    for a, tag in zip(axes, ["(a)", "(b)"]):
        style_ax(a, tag)
    # Shared legend above both panels: inside the axes it collided with the
    # curves in (a) and was absent from (b).
    shared_legend(fig, axes, 3)
    save(fig, "fig_extrap")


# ================================================================ fig_noise ==
def fig_noise():
    noise = load("noise.json")
    set_style()
    fig, axes = panel(ratio=0.44, ncols=2)
    snrs = [30, 20, 10, 5]
    for idx, (mode, lab) in enumerate([("unconstrained", "Unconstrained"),
                                       ("constrained", "Constrained (ours)")]):
        axes[0].plot(snrs, [noise[mode][str(s)]["rre"] for s in snrs],
                     color=OKABE_ITO[[2, 6][idx]], marker=MARKERS[idx], label=lab)
        axes[1].plot(snrs, [noise[mode][str(s)]["sens"] for s in snrs],
                     color=OKABE_ITO[[2, 6][idx]], marker=MARKERS[idx], label=lab)
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("RRE")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel(r"Sensitivity $\|\Delta\hat\chi\|/\|\hat\chi\|$")
    axes[1].set_yscale("log")
    for a, tag in zip(axes, ["(a)", "(b)"]):
        style_ax(a, tag)
    shared_legend(fig, axes, 2)
    save(fig, "fig_noise")


# ================================================================= fig_gain ==
def fig_gain():
    gain = load("gain.json")
    set_style()
    fig, ax = panel(ncols=1, ratio=0.60)
    x = np.arange(1, len(gain["constrained"]) + 1); bw = 0.38
    ax.bar(x - bw / 2, gain["constrained"], bw, color=OKABE_ITO[3], edgecolor="k",
           linewidth=0.6, label="Constrained")
    ax.bar(x + bw / 2, gain["unconstrained"], bw, color=OKABE_ITO[6], edgecolor="k",
           linewidth=0.6, label="Unconstrained")
    ax.set_xticks(x); ax.set_xlabel("Layer index $k$")
    ax.set_ylabel("Measured layer gain")
    # Headroom so the legend sits above every bar (it previously overlapped the
    # layer-1 bars, which are the tallest in the panel).
    ax.set_ylim(0, max(max(gain["constrained"]), max(gain["unconstrained"])) * 1.28)
    ax.legend(loc="upper center", ncol=2)
    style_ax(ax)
    save(fig, "fig_gain")


# =============================================================== fig_pareto ==
def fig_pareto():
    par = load("pareto.json")
    ebs = sorted(float(k) for k in par)
    set_style()
    fig, axes = panel(ratio=0.44, ncols=2)
    axes[0].plot(ebs, [par[str(e)]["rre"] for e in ebs], color=OKABE_ITO[5], marker="o")
    axes[1].plot(ebs, [par[str(e)]["sens"] for e in ebs], color=OKABE_ITO[5], marker="s")
    axes[0].set_xlabel(r"Amplitude budget $\bar\epsilon$")
    axes[0].set_ylabel("RRE (test, 20 dB)")
    axes[1].set_xlabel(r"Amplitude budget $\bar\epsilon$")
    axes[1].set_ylabel("Sensitivity (20 dB)")
    for a, tag in zip(axes, ["(a)", "(b)"]):
        style_ax(a, tag)
    save(fig, "fig_pareto")


# ============================================================ fig_gradnorm ===
def fig_gradnorm():
    gn = load("gradnorm.json")
    set_style()
    fig, ax = panel(ncols=1, ratio=0.60)
    for idx, (mode, lab) in enumerate([("unconstrained", "Unconstrained"),
                                       ("constrained", "Constrained")]):
        ks = sorted(int(k) for k in gn[mode])
        ax.semilogy(ks, [gn[mode][str(k)] for k in ks], color=OKABE_ITO[[2, 6][idx]],
                    marker=MARKERS[idx], label=lab)
    ax.set_xlabel("Truncated depth used in backward pass")
    ax.set_ylabel("Gradient norm (first-layer correction)")
    ax.legend(loc="best")
    style_ax(ax)
    save(fig, "fig_gradnorm")


# ============================================================= fig_ablation ==
def fig_ablation():
    abl = load("ablation.json")
    set_style()
    fig, axes = panel(ratio=0.50, ncols=2)
    tags = [("unc_pOn_fbOn_s1", "Unc.\n$\\Pi$ fb", OKABE_ITO[2]),
            ("unc_pOn_fbOff_s1", "Unc.\n$\\Pi$", OKABE_ITO[1]),
            ("unc_pOff_fbOn_s1", "Unc.\nfb", OKABE_ITO[0]),
            ("unc_pOff_fbOff_s1", "Unc.\nnone", OKABE_ITO[6]),
            ("cons_pOn_fbOn_s1", "Cons.\n$\\Pi$ fb", OKABE_ITO[3]),
            ("cons_pOn_fbOff_s1", "Cons.\n$\\Pi$", OKABE_ITO[4]),
            ("cons_pOff_fbOn_s1", "Cons.\nfb", OKABE_ITO[5]),
            ("cons_pOff_fbOff_s1", "Cons.\nnone", OKABE_ITO[7])]
    xs, ys, cols, labs, div = [], [], [], [], []
    for i, (tag, lab, c) in enumerate(tags):
        r = abl.get(tag, {})
        fin = r.get("finite_extrap", False)
        val = r.get("rre_extrap") if (fin and r.get("rre_extrap") is not None) else None
        xs.append(i); ys.append(val); cols.append(c); labs.append(lab); div.append(not fin)
    ax = axes[0]
    ax.bar(xs, [y if y is not None else 0.0 for y in ys], 0.62, color=cols,
           edgecolor="k", linewidth=0.6)
    for i, (y, isdiv) in enumerate(zip(ys, div)):
        if isdiv:
            ax.text(i, 0.02, "diverged", ha="center", va="bottom", fontsize=7,
                    rotation=90, color="white", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("Extrapolation RRE (800 layers)")
    style_ax(ax, "(a)")
    ax = axes[1]
    chis = [abl.get(t, {}).get("maxabs_chi_over") for t, _, _ in tags]
    ax.bar(xs, [c if c is not None else 0.0 for c in chis], 0.62, color=cols,
           edgecolor="k", linewidth=0.6)
    ax.axhline(0.5, color="k", ls="--", lw=1.0)
    ax.text(len(tags) - 0.45, 0.52, r"$\chi_{\max}=0.5$", fontsize=7.5, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel(r"$\max_k\|\chi^k\|_\infty$")
    style_ax(ax, "(b)")
    save(fig, "fig_ablation")


# =============================================================== fig_frozen ==
def fig_frozen():
    abl = load("ablation.json")
    set_style()
    fig, ax = panel(ncols=1, ratio=0.60)
    frz = [("unconstrained_frozen_proj1", "Unc., $\\Pi$ on", OKABE_ITO[2]),
           ("unconstrained_frozen_proj0", "Unc., $\\Pi$ off", OKABE_ITO[6]),
           ("constrained_frozen_proj1", "Cons., $\\Pi$ on", OKABE_ITO[3]),
           ("constrained_frozen_proj0", "Cons., $\\Pi$ off", OKABE_ITO[7])]
    xs, ys, cols, labs = [], [], [], []
    fin = [r["rre_extrap"] for _, t, _ in frz
           for r in [abl.get(t, {})] if isinstance(r.get("rre_extrap"), (int, float))]
    # the diverged arm is drawn to the top of the frame: it has no finite error, and a
    # short bar would read as a small error
    ytop = max(fin) * 3.2 if fin else 1.0
    div = []
    for i, (tag, lab, c) in enumerate(frz):
        r = abl.get(tag, {})
        v = r.get("rre_extrap")
        if isinstance(v, (int, float)):
            ys.append(max(v, 1e-3))
        else:                      # the rollout left the finite range before the horizon
            ys.append(ytop)
            div.append(i)
        xs.append(i)
        cols.append(c); labs.append(lab)
    ax.bar(xs, ys, 0.6, color=cols, edgecolor="k", linewidth=0.6)
    for i in div:                  # hatched bar = diverged, not a measured error
        ax.bar([i], [ytop], 0.6, color="none", edgecolor="k",
               linewidth=0.6, hatch="///", zorder=3)
    ax.set_yscale("log")
    for i, y in zip(xs, ys):
        txt = "div." if i in div else f"{y:.2f}"
        ax.text(i, y * 1.12, txt, ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Extrapolation RRE (800 layers)")
    ax.set_ylim(min(ys) * 0.5, ytop * 1.45)
    style_ax(ax)
    save(fig, "fig_frozen")


# =============================================================== fig_images ==
def fig_images():
    data = np.load(os.path.join(RES, "images.npz"))
    nx = ny = 32
    rows = ["gt", "classical8", "classical50", "unconstrained", "constrained"]
    titles = ["Ground truth", "Classical, $K{=}8$", "Classical, $K{=}50$",
              "Unconstrained, $K{=}8$", "Constrained (ours), $K{=}8$"]
    n_ex = min(4, data["gt"].shape[0])
    set_style()
    fig, axes = plt.subplots(n_ex, len(rows),
                             figsize=(TEXTW, TEXTW * 0.80), layout="constrained")
    vmax = max(float(data["gt"].max()), 1e-6)
    for r, (row, ttl) in enumerate(zip(rows, titles)):
        for c in range(n_ex):
            a = axes[c, r]
            im = a.imshow(data[row][c].reshape(nx, ny), cmap="viridis",
                          vmin=0, vmax=vmax, origin="lower")
            a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_title(ttl, fontsize=8, pad=3)
            if r == 0:
                a.set_ylabel(f"ex.{c+1}", fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(r"reconstructed contrast $|\chi|$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)
    save(fig, "fig_images")


# ============================================================== fig_alphac ===
def fig_alphac():
    bj = load("bjac.json"); ngc = load("ngc.json")
    sw = None
    if ngc and ngc.get("alpha_sweep_converged"):
        sw = sorted(ngc["alpha_sweep_converged"], key=lambda r: r["alpha"])
    elif bj and bj.get("L2_alpha_sweep"):
        sw = sorted(bj["L2_alpha_sweep"], key=lambda r: r["alpha"])
    art = sorted(bj["L2_alpha_sweep"], key=lambda r: r["alpha"]) if (bj and bj.get("L2_alpha_sweep")) else None
    if not sw:
        return
    a = np.array([r["alpha"] for r in sw], dtype=float)
    meas = np.array([r["sigma_full"] for r in sw], dtype=float)
    bnd = np.array([r["rigorous_lower_bound"] for r in sw], dtype=float)
    ac = None
    if ngc:
        ac = (ngc.get("converged") or {}).get("alpha_c")
    if ac is None and bj:
        ac = bj.get("L1_physics_core", {}).get("alpha_c_rigorous")
    set_style()
    fig, ax = panel(ncols=1, ratio=0.62)
    if ac:
        ax.axvspan(ac, 1.0, color=OKABE_ITO[2], alpha=0.14, lw=0, zorder=0)
        ax.axvline(ac, color=OKABE_ITO[5], ls=":", lw=1.0)
    if art:
        aa = np.array([r["alpha"] for r in art], dtype=float)
        am = np.array([r["sigma_full"] for r in art], dtype=float)
        ax.plot(aa, am, color="0.55", ls="-", lw=0.8, marker="x", ms=3.0, alpha=0.75,
                label=r"as-implemented $n_{\mathrm{cg}}{=}15$ (artefact)", zorder=2)
        ax.set_xlim(0.0, max(a.max(), aa.max()) * 1.05)
    ax.plot(a, meas, color=OKABE_ITO[0], marker=MARKERS[0], lw=1.0, ms=4.0,
            label=r"measured $\sigma_{\max}(J)$ (converged)", zorder=3)
    ax.plot(a, bnd, color=OKABE_ITO[6], ls="--", lw=1.0, marker=MARKERS[2], ms=3.6,
            label=r"proved bound $\alpha\mathcal{E}$", zorder=3)
    ax.axhline(1.0, color="0.4", lw=0.8, ls="-")
    ax.set_yscale("log")
    ax.set_xlabel(r"step size $\alpha=\gamma$")
    ax.set_ylabel(r"$\sigma_{\max}$ of the layer Jacobian")
    if ac:
        ax.annotate(r"$\alpha_c$", xy=(ac, 1.0), xytext=(ac * 1.04, 1.15),
                    fontsize=8, color=OKABE_ITO[5])
    # Legend above the axes: inside, the three entries sat on top of the curves.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style_ax(ax)
    save(fig, "fig_alphac")


def main():
    fig_structure()
    fig_regime()
    fig_depth()
    fig_extrap()
    fig_noise()
    fig_gain()
    fig_pareto()
    fig_gradnorm()
    fig_ablation()
    fig_frozen()
    fig_images()
    fig_alphac()


if __name__ == "__main__":
    main()
