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
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

# ---------------------------------------------------------------- constants --
PT_PER_IN = 72.27
# Text block of the target class, in points, set per submission tree:
#   mdpi.cls   (paper_mdpi)  -> 394.35522 pt
#   tsp.cls    (CMES)        -> 469.75539 pt
# Override with FIG_TEXTW_PT so the LaTeX scale stays exactly 1.0 in every tree.
TEXTW = float(os.environ.get("FIG_TEXTW_PT", "394.35522")) / PT_PER_IN
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
# Sky blue (#56B4E9) is part of the Okabe-Ito palette but as a *thin line* on
# white it is the faintest ink in every figure; curves therefore use the deep
# variant.  Fills keep the palette colour (they carry area, not strokes).
LINE_BLUE = "#0072B2"
GREY_NOTE = "0.25"      # darkest grey allowed for annotation text
GREY_LINE = "0.30"      # leader lines / reference lines

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG = os.path.normpath(os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figures")))
NUM = os.path.normpath(os.path.join(HERE, "..", "paper_mdpi", "numbers.tex"))
os.makedirs(FIG, exist_ok=True)


def set_style():
    """One style for every panel. Rendered sizes == these values (scale 1.0)."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 5,
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


def panel(w=TEXTW, ratio=0.52, ncols=2, **kw):
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
    """One unrolled layer, its two state channels, and its Jacobian block map.

    Layout v4 (2026-10-20).  One canvas, no (a)/(b) panel tags: the cascade, the
    layer itself and the Jacobian sit on the same drawing and are read in one
    pass (top band, then the layer, then the block map).

    Two defects of v3 are fixed here.

    1. SIZE / DENSITY.  v3 put 6.2-7.2 pt text inside boxes about 1 in tall, so
       the figure was large and looked empty, and the grey annotations were too
       faint to read at print size.  v4 drops the height from 5.2 in to 3.06 in,
       raises the smallest type from 6.2 pt to 6.9 pt, darkens the annotations
       (0.44 -> 0.30 grey), and lets each box fit its text rather than the other
       way round (``tfit`` shrinks a label only if it would overflow).

    2. ROW / COLUMN SEMANTICS.  v3 labelled the block map with the *inputs* on
       the left and the *outputs* on top.  For a block matrix that convention is
       the transpose of the one used in Lemma (blocks), where
       J = [[A, B], [C, D]] acts on (dchi, dw) and returns (dchi+, dw+).  Under
       v3's labels the off-diagonal cells therefore named the wrong blocks:
       B = dchi+/dw appeared where dw+/dchi belongs.  v4 puts the outputs on the
       left (rows) and the inputs on top (columns), which matches the Lemma and
       the caption.

    NOTE (cross-tree consistency): the caption of Fig. 1 in paper_mdpi/main.tex
    and paper_en/main.tex still describes the v2 layout ("dashed paths are
    derivatives rather than signals").  Neither tree regenerates its figures from
    this function at the moment, so they stay self-consistent; if either is ever
    rebuilt, its Fig. 1 caption must be updated to the wording used in
    cmes_submission/main.tex at the same time.
    """
    set_style()
    # The drawing is laid out on a 6.50-in coordinate grid (COORD_W) and the
    # canvas is the target text block.  When COORD_W > W the grid is compressed
    # horizontally while the font sizes stay at their authored values, which is
    # what keeps the type legible at \linewidth = 1.0; tfit() below compensates
    # for the compression when it shrinks a string that would not fit.
    COORD_W = 6.50
    W, H = TEXTW, 3.06
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, COORD_W)
    ax.set_ylim(0.09, 0.09 + H)
    ax.axis("off")

    C_STEP, C_FIELD = "#EFEFEF", "#D7E6F4"
    C_LEARN, C_PROJ = "#DDEEE4", "#FAE4C8"
    C_REC, C_CLOSED = "#E6E6E6", "#F0E0CC"
    INK, MUT = "0.08", "0.26"

    def box(x0, y0, w, h, fc, lw=0.9, ec="0.38", z=2):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.035",
                                    linewidth=lw, edgecolor=ec, facecolor=fc,
                                    zorder=z))

    def arrow(p0, p1, col=INK, lw=1.0, ms=8):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                     linewidth=lw, color=col, shrinkA=0,
                                     shrinkB=0, zorder=5))

    def tfit(x, y, s, maxw, fs, **kw):
        """Text that is shrunk only if it would overflow ``maxw`` (inches)."""
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        t = ax.text(x, y, s, fontsize=fs, **kw)
        fig.canvas.draw()
        wpt = (t.get_window_extent(fig.canvas.get_renderer()).width
               / fig.dpi * 72.0)
        lim = maxw * (W / 6.50) * 72.0
        if wpt > lim > 0:
            t.set_fontsize(fs * lim / wpt * 0.97)
        return t

    # ---- top band: the unrolled cascade --------------------------------
    ax.text(0.06, 2.80, r"$z^{0}$", fontsize=11.3, ha="left", va="center")
    for x0, lab in [(0.34, r"$T_{1}$"), (0.90, r"$T_{2}$"), (1.78, r"$T_{K}$")]:
        box(x0, 2.64, 0.44, 0.32, C_STEP)
        ax.text(x0 + 0.22, 2.80, lab, fontsize=10.4, ha="center", va="center",
                zorder=4)
    arrow((0.185, 2.80), (0.335, 2.80))
    arrow((0.785, 2.80), (0.895, 2.80))
    arrow((1.345, 2.80), (1.470, 2.80))
    ax.text(1.560, 2.80, r"$\cdots$", fontsize=10.8, ha="center", va="center")
    arrow((1.650, 2.80), (1.775, 2.80))
    arrow((2.225, 2.80), (2.350, 2.80))
    ax.text(2.385, 2.80, r"$z^{K}$", fontsize=11.3, ha="left", va="center")
    ax.text(2.72, 2.80, r"fixed point $z^{\star}=T(z^{\star})$", fontsize=8.4,
            ha="left", va="center", color=MUT)

    # ---- the layer being expanded ---------------------------------------
    ax.text(0.46, 2.44, r"detail of one layer $T_{k}$", fontsize=8.2, ha="left",
            va="center", color=MUT)

    # contrast channel
    box(0.46, 1.46, 1.86, 0.84, C_STEP)
    ax.text(1.39, 2.17, "contrast step", fontsize=9.7, ha="center",
            va="center", zorder=4, color=INK)
    tfit(1.39, 1.94,
         r"$\min_{\chi}\,\|y-\mathbf{G}_{s}\,\mathrm{diag}(E)\,\chi\|^{2}"
         r"+\lambda\|\chi\|^{2}$", 1.74, 8.5, zorder=4)
    ax.text(1.39, 1.63, r"CG, $n_{\mathrm{cg}}$ steps", fontsize=8.1,
            ha="center", va="center", zorder=4, color=MUT)

    box(2.44, 1.56, 0.58, 0.64, C_LEARN)
    tfit(2.73, 1.97, r"$+\,H_{k}(f_{k})$", 0.52, 9.3, zorder=4)
    ax.text(2.73, 1.72, "learned", fontsize=8.0, ha="center", va="center",
            zorder=4, color=MUT)

    box(3.14, 1.56, 0.46, 0.64, C_PROJ)
    ax.text(3.37, 1.97, r"$\Pi$", fontsize=12.4, ha="center", va="center",
            zorder=4)
    ax.text(3.37, 1.72, "clamp", fontsize=8.0, ha="center", va="center",
            zorder=4, color=MUT)

    # field channel
    box(0.46, 0.30, 1.86, 0.64, C_FIELD)
    ax.text(1.39, 0.80, "field step", fontsize=9.7, ha="center", va="center",
            zorder=4, color=INK)
    tfit(1.39, 0.55,
         r"$w+\theta\alpha\,[\,\chi\odot(E^{i}+\mathbf{G}_{b}w)-w\,]$",
         1.74, 7.9, zorder=4)

    # states and data flow
    # The channel names are given once, on the input side.  Repeating them under
    # the output states collided with the block-map row labels once the drawing
    # was compressed to the text block, and the row labels already name those
    # channels.
    for x, y, s, ch in [(0.30, 1.88, r"$\chi^{k}$", "contrast"),
                        (0.30, 0.62, r"$w^{k}$", "field"),
                        (3.94, 1.88, r"$\chi^{k+1}$", None),
                        (3.94, 0.62, r"$w^{k+1}$", None)]:
        ax.text(x, y, s, fontsize=11.9, ha="center", va="center")
        if ch:
            ax.text(x, y - 0.22, ch, fontsize=8.0, ha="center", va="center",
                    color=MUT)

    arrow((0.355, 1.88), (0.455, 1.88))
    arrow((2.325, 1.88), (2.435, 1.88))
    arrow((3.025, 1.88), (3.135, 1.88))
    arrow((3.605, 1.88), (3.715, 1.88))
    arrow((0.355, 0.62), (0.455, 0.62))
    arrow((2.325, 0.62), (3.715, 0.62))
    ax.text(3.02, 0.73, r"neither $\Pi$ nor $H_{k}$", fontsize=8.4,
            ha="center", va="center", color=MUT)

    # the only coupling between the two channels
    arrow((1.20, 0.96), (1.20, 1.44))
    ax.text(1.30, 1.20, r"$E=E^{i}+\mathbf{G}_{b}w$", fontsize=8.9,
            ha="left", va="center")

    # measured data
    arrow((0.310, 2.13), (0.450, 2.13), col=OKABE_ITO[5], lw=0.9, ms=7)
    ax.text(0.295, 2.13, r"$y$", fontsize=10.3, ha="right", va="center",
            color=OKABE_ITO[5])

    # ---- block map of the layer Jacobian --------------------------------
    ax.text(4.30, 2.96, r"layer Jacobian $DT_{k}$", fontsize=9.9, ha="left",
            va="top")
    xs = [4.60, 5.44, 6.28]
    ys = [1.36, 1.90, 2.44]
    cells = [(0, 1, r"$A$", C_REC, "recorded"),
             (1, 1, r"$B$", C_REC, "recorded"),
             (0, 0, r"$C$", C_CLOSED, "closed form"),
             (1, 0, r"$D$", C_CLOSED, "closed form")]
    for (ci, ri, lab, fc, tag) in cells:
        x0, x1 = xs[ci], xs[ci + 1]
        y0, y1 = ys[ri], ys[ri + 1]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc,
                                   edgecolor="0.42", linewidth=0.8, zorder=2))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.075, lab, fontsize=14.0,
                ha="center", va="center", zorder=4)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 0.120, tag, fontsize=8.1,
                ha="center", va="center", color=MUT, zorder=4)

    # rows = channels leaving the layer, columns = channels entering it
    for yc, s in [(2.17, r"$\delta\chi^{k+1}$"), (1.63, r"$\delta w^{k+1}$")]:
        ax.text(4.54, yc, s, fontsize=10.8, ha="right", va="center")
    for xc, s in [(5.02, r"$\delta\chi^{k}$"), (5.86, r"$\delta w^{k}$")]:
        ax.text(xc, 2.52, s, fontsize=10.8, ha="center", va="bottom")

    ax.text(4.30, 1.14, r"$C=\theta\alpha\,\mathrm{diag}(E)$", fontsize=8.4,
            ha="left", va="center")
    ax.text(4.30, 0.92,
            r"$D=(1-\theta\alpha)I+\theta\alpha\,\mathrm{diag}(\chi)"
            r"\mathbf{G}_{b}$", fontsize=8.4, ha="left", va="center")
    ax.text(4.30, 0.68, r"$A$, $B$: the regularised normal matrix",
            fontsize=8.2, ha="left", va="center", color=MUT)
    ax.text(4.30, 0.53, r"is inverted here, so both blocks are",
            fontsize=8.2, ha="left", va="center", color=MUT)
    ax.text(4.30, 0.38, r"recorded rather than derived",
            fontsize=8.2, ha="left", va="center", color=MUT)

    save(fig, "fig_structure")


# =============================================================== fig_regime ==
def fig_regime():
    """Where each theorem applies, in the (beta_0, alpha) plane.

    The two regimes are not exclusive and that is the point of the paper: for
    the synthetic certified box one has beta_0 < 1 (so the *field subsystem* is
    contractive, Thm. field) while simultaneously alpha > alpha_c (so the
    *composite layer* is provably non-contractive, Prop. imposs).  Stability has
    to be layered because these live on different objects.

    Layout: the two half-planes are drawn as bands, so the reader sees three
    populated quadrants rather than a collage of arrows.  Each quadrant is named
    by what it certifies, the two boundaries carry their values on the boundary
    itself, and the three operating points are labelled above their markers.
    The previous version put four white boxes inside the plane and drew three
    leader lines, which left the shading unexplained and made the labels look
    like data.
    """
    set_style()
    ac = num("NumAlphaC", 0.3169)
    b_cert = num("betacert", 0.8303)
    b_meas = num("NumBetaMeas", 2.015)
    b_fres = num("NumFresnelBeta", 14.3987)
    a_op = 0.5

    fig, ax = plt.subplots(figsize=(TEXTW, TEXTW * 0.56))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.940, bottom=0.150)
    xlo, xhi = 0.05, 30.0
    G, O = OKABE_ITO[3], OKABE_ITO[6]

    # the two half-planes; their intersection is the region where both hold
    ax.axvspan(xlo, 1.0, color=G, alpha=0.20, lw=0, zorder=0)
    ax.axhspan(ac, 1.0, color=O, alpha=0.16, lw=0, zorder=0)
    ax.axvline(1.0, color=G, lw=1.2, zorder=3)
    ax.axhline(ac, color=O, lw=1.2, zorder=3)
    ax.axhline(a_op, color=GREY_NOTE, lw=0.8, ls=(0, (1.6, 2.2)), zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(0, 1)
    ax.set_xlabel(r"weak-scattering constant $\beta_0=\chi_{\max}\|\mathbf{G}_b\|_2$")
    ax.set_ylabel(r"step size $\alpha=\gamma$")

    # what each quadrant certifies
    ax.text(0.21, 0.155, "field subsystem\ncontractive", fontsize=8.4, color=G,
            ha="center", va="center", linespacing=1.2)
    ax.text(0.15, 0.805, "field contractive,\nlayer provably\nnon-contractive",
            fontsize=8.4, color="0.22", ha="center", va="center", linespacing=1.2)
    ax.text(6.2, 0.865, "layer non-contractive;\nno field certificate",
            fontsize=8.4, color=O, ha="center", va="center", linespacing=1.2)
    ax.text(6.2, 0.135, "no certificate", fontsize=8.4, color=GREY_NOTE,
            ha="center", va="center")

    # values written on the boundaries themselves
    ax.text(0.955, 0.02, r"$\beta_0=1$", fontsize=8.0, rotation=90, ha="right",
            va="bottom", color=G)
    ax.text(xhi * 0.97, ac + 0.018, r"$\alpha_c=%.3f$" % ac, fontsize=8.0,
            ha="right", va="bottom", color=O)
    ax.text(xhi * 0.97, a_op - 0.022, r"operating point $\alpha=\gamma=0.5$",
            fontsize=8.0, ha="right", va="top", color=GREY_NOTE)

    # operating points, each labelled above its own marker
    # operating points, each labelled above its own marker; the leftmost label
    # is right-aligned so that it stops short of the beta_0 = 1 boundary
    pts = [(b_cert, "certified (synthetic)", OKABE_ITO[0], "o", "right", 0.98),
           (b_meas, r"$\Pi$-off arm", O, "s", "center", 1.0),
           (b_fres, "Fresnel 2-D", OKABE_ITO[5], "^", "center", 1.0)]
    for b, lab, col, mk, ha, fx in pts:
        ax.plot([b], [a_op], marker=mk, ms=7.5, color=col, mec="white", mew=0.9,
                zorder=6)
        ax.text(b * fx, a_op + 0.085, lab, fontsize=8.4, ha=ha, va="bottom")
    style_ax(ax)
    save(fig, "fig_regime")
# ================================================================ fig_depth ==
def fig_depth():
    cls = load("classical_curve.json"); dep = load("depth_curve.json")
    d40 = (load("depth_curve_40ep.json") or {}).get("rre") or {}
    set_style()
    fig, ax = panel(ncols=1, ratio=0.68)
    ks_c = sorted(int(k) for k in cls)
    curves = [("classical", OKABE_ITO[0], MARKERS[0], "-", True, +1,
               ks_c, [cls[str(k)] for k in ks_c])]
    for idx, (mode, lab) in enumerate([("unconstrained", "Unconstrained unrolling"),
                                       ("constrained", "Constrained (ours)")]):
        col = LINE_BLUE if idx == 0 else OKABE_ITO[6]
        mk = MARKERS[1 + idx]
        pts = sorted(((int(k.rsplit("_", 1)[-1][1:]), v) for k, v in dep.items()
                      if k.startswith(mode + "_")))
        curves.append((mode, col, mk, "-", True, +1,
                       [p[0] for p in pts], [p[1] for p in pts]))
        p40 = sorted(((int(k.rsplit("_", 1)[-1][1:]), v) for k, v in d40.items()
                      if k.startswith(mode + "_")))
        curves.append((mode + "_40", col, mk, "--", False, -1,
                       [p[0] for p in p40], [p[1] for p in p40]))

    leg = {"classical": "Classical alternating",
           "unconstrained": "Unconstrained unrolling",
           "constrained": "Constrained (ours)",
           "unconstrained_40": "Unconstrained (uniform 40-epoch)",
           "constrained_40": "Constrained (uniform 40-epoch)"}
    for key, col, mk, ls, fill, side, xs, ys in curves:
        ax.plot(xs, ys, color=col, marker=mk, linestyle=ls,
                mfc=(col if fill else "none"),
                label=leg[key])

    # Every measured point is labelled, not just the last one.  Two rules make 26
    # four-decimal labels fit in a band 0.08 wide: (i) the label goes on the side the
    # curve is NOT moving towards -- above a falling segment, below a rising one -- so
    # it never sits on its own line; (ii) the archived arms (``side``=+1) and their
    # 40-epoch replays (``side``=-1) take mirrored offsets, because those two curves
    # coincide to within 0.003 at K<=8 and would otherwise print over each other.
    for key, col, mk, ls, fill, side, xs, ys in curves:
        for j, (x, y) in enumerate(zip(xs, ys)):
            if j == 0:
                slope = ys[1] - ys[0]
            elif j == len(xs) - 1:
                slope = ys[-1] - ys[-2]
            else:
                slope = ys[j + 1] - ys[j - 1]
            dy = side * 6.5 * (-1.0 if slope > 0 else 1.0)
            last = (x > 60.0)          # only K=100 sits too close to the frame edge
            ax.annotate("%.4f" % y, xy=(x, y),
                        xytext=(-5 if last else 5, dy), textcoords="offset points",
                        ha="right" if last else "left", va="center", fontsize=6.3,
                        color=col, annotation_clip=False,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white",
                                  ec="none", alpha=0.80)
                        ).set_clip_on(False)
    ax.set_xlabel("Depth $K$"); ax.set_ylabel("Relative reconstruction error")
    # The frame must contain the whole measured range: the unconstrained arms sit at
    # 0.335-0.367.  The top tick stops at 0.44 so that the legend, which occupies the
    # top band of the axes, cannot collide with the classical arm's labels.
    lin_ticks = [0.34, 0.36, 0.38, 0.40, 0.42]
    ax.set_yticks(lin_ticks)
    ax.set_yticklabels(["%.2f" % v for v in lin_ticks])
    ax.set_ylim(0.328, 0.452)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 24, 50, 100])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "24", "50", "100"])
    ax.set_xlim(0.85, 260.0)
    ax.minorticks_off()
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=2.2)
    style_ax(ax)
    save(fig, "fig_depth")


# =============================================================== fig_extrap ==
def fig_extrap():
    extrap = load("extrap.json")
    set_style()
    fig, axes = panel(ratio=0.40, ncols=2)
    for idx, (mode, lab) in enumerate([("classical", "Classical"),
                                       ("unconstrained", "Unconstrained"),
                                       ("constrained", "Constrained (ours)")]):
        e = extrap[mode]; x = np.arange(len(e["residual"])) * 5
        c = (OKABE_ITO[0], LINE_BLUE, OKABE_ITO[6])[idx]
        axes[0].semilogy(x, np.maximum(np.array(e["residual"]), 1e-16),
                         color=c, marker=MARKERS[idx],
                         markevery=2, label=lab)
        axes[1].plot(x, e["rre"], color=c, marker=MARKERS[idx],
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
    fig, axes = panel(ratio=0.42, ncols=2)
    snrs = [30, 20, 10, 5]
    modes = [("unconstrained", "Unconstrained", LINE_BLUE),
             ("constrained", "Constrained (ours)", OKABE_ITO[6])]
    keys = ("rre", "sens")
    for a, key in zip(axes, keys):
        for idx, (mode, lab, c) in enumerate(modes):
            a.plot(snrs, [noise[mode][str(s)][key] for s in snrs], color=c,
                   marker=MARKERS[idx], label=lab)
        # Values on every marker.  The two series are within a few 1e-3 of each
        # other at every SNR, so at each abscissa the higher value is labelled
        # above the pair and the lower one below it, in the colour of its series.
        for s in snrs:
            pair = sorted(((noise[m][str(s)][key], m) for m, _, _ in modes),
                          reverse=True)
            for rank, (v, m) in enumerate(pair):
                col = dict((mm, cc) for mm, _, cc in modes)[m]
                # the upper value is put above *and to the right* of its marker:
                # centred above it, the label of the 10 dB point landed on the
                # descending blue segment
                a.annotate("%.3f" % v, xy=(s, v),
                           xytext=(5, 7) if rank == 0 else (-4, -11),
                           textcoords="offset points",
                           ha="left" if rank == 0 else "right",
                           va="bottom" if rank == 0 else "top",
                           fontsize=7.9, color=col,
                           annotation_clip=False).set_clip_on(False)
        a.set_xlabel("SNR (dB)")
        a.set_ylabel("RRE" if key == "rre" else
                     r"Sensitivity $\|\Delta\hat\chi\|/\|\hat\chi\|$")
        if key == "sens":
            a.set_yscale("log")
        a.margins(y=0.16)
        # left/right margin wide enough for the values printed on the end markers
        # (margins() must be called first: it re-autoscales x as well)
        a.set_xticks(snrs)
        a.set_xlim(2.0, 34.0)
    for a, tag in zip(axes, ["(a)", "(b)"]):
        style_ax(a, tag)
    shared_legend(fig, axes, 2)
    save(fig, "fig_noise")
# ================================================================= fig_gain ==
def fig_gain():
    gain = load("gain.json")
    set_style()
    fig, ax = panel(ncols=1, ratio=0.50)
    x = np.arange(1, len(gain["constrained"]) + 1); bw = 0.38
    ax.bar(x - bw / 2, gain["constrained"], bw, color=OKABE_ITO[3], edgecolor="k",
           linewidth=0.6, label="Constrained")
    ax.bar(x + bw / 2, gain["unconstrained"], bw, color=OKABE_ITO[6], edgecolor="k",
           linewidth=0.6, label="Unconstrained")
    # The text asserts that both variants exceed one at every layer, so the unit
    # level is drawn to make that assertion checkable by eye.  It sits above the
    # bars in zorder: a reference line drawn behind the fill is visible only in
    # the gaps between bars.
    ax.axhline(1.0, color=GREY_LINE, lw=0.7, ls=(0, (5, 3)), zorder=3,
               label="Unity gain")
    # Value on every bar: with two series per layer the reader cannot recover the
    # numbers from the axis, and the text quotes individual layers.
    for xi, v in zip(x - bw / 2, gain["constrained"]):
        ax.text(xi, v + 0.12, "%.2f" % v, ha="center", va="bottom", fontsize=6.8)
    for xi, v in zip(x + bw / 2, gain["unconstrained"]):
        ax.text(xi, v + 0.12, "%.2f" % v, ha="center", va="bottom", fontsize=6.8,
                color=OKABE_ITO[6])
    ax.set_xticks(x); ax.set_xlabel("Layer index $k$")
    ax.set_ylabel("Measured layer gain")
    # Headroom so the legend sits above every bar and the labels stay inside the frame.
    ax.set_ylim(0, max(max(gain["constrained"]), max(gain["unconstrained"])) * 1.34)
    ax.legend(loc="upper center", ncol=3)
    style_ax(ax)
    save(fig, "fig_gain")


# =============================================================== fig_pareto ==
def fig_pareto():
    par = load("pareto.json")
    ebs = sorted(float(k) for k in par)
    set_style()
    fig, axes = panel(ratio=0.42, ncols=2)
    series = [("rre", "RRE (test, 20 dB)", "o"),
              ("sens", "Sensitivity (20 dB)", "s")]
    for a, (key, ylab, mk) in zip(axes, series):
        vals = [par[str(e)][key] for e in ebs]
        a.plot(ebs, vals, color=LINE_BLUE, marker=mk)
        # The four budgets span a factor of twelve and are swept geometrically,
        # so they are drawn on a log axis at their true positions rather than at
        # four equal spacings, which would misstate the sweep.
        a.set_xscale("log")
        a.set_xticks(ebs)
        a.set_xticklabels(["%.2f" % e for e in ebs])
        a.minorticks_off()
        a.set_xlim(ebs[0] / 1.45, ebs[-1] * 1.45)
        # Each value sits on the side the curve leaves free.  On a descending
        # segment that is above-RIGHT (to the left the same segment is higher
        # and would run through the digits); at the minimum it is below, where
        # both arms rise away from the marker.
        pos = {"rre": [("left", 6, 5), ("left", 6, 5), ("left", 6, 5),
                       ("left", 6, -6)],
               "sens": [("left", 6, 5), ("left", 6, 5), ("center", 0, -11),
                        ("right", -6, 5)]}[key]
        for (e, v), (ha, dx, dy) in zip(zip(ebs, vals), pos):
            a.annotate("%.4f" % v, xy=(e, v), xytext=(dx, dy),
                       textcoords="offset points", ha=ha,
                       va="top" if dy < 0 else "bottom", fontsize=7.8)
        a.set_xlabel(r"Amplitude budget $\bar\epsilon$")
        a.set_ylabel(ylab)
        a.margins(y=0.20)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.965, bottom=0.245,
                        wspace=0.34)
    for a, tag in zip(axes, ["(a)", "(b)"]):
        style_ax(a, tag)
    save(fig, "fig_pareto")
# ============================================================ fig_gradnorm ===
def fig_gradnorm():
    gn = load("gradnorm.json")
    set_style()
    fig, ax = panel(ncols=1, ratio=0.58)
    for idx, (mode, lab) in enumerate([("unconstrained", "Unconstrained"),
                                       ("constrained", "Constrained")]):
        ks = sorted(int(k) for k in gn[mode])
        vs = [gn[mode][str(k)] for k in ks]
        ax.semilogy(ks, vs, color=(LINE_BLUE, OKABE_ITO[6])[idx],
                    marker=MARKERS[idx], label=lab)
        # the two series are a decade apart at every abscissa, so both labels
        # can sit above their own marker without colliding
        for k, v in zip(ks, vs):
            e = int(np.floor(np.log10(abs(v))))
            m = v / 10.0 ** e
            # the last label goes below-left: the segment into it descends from
            # the upper left, so above-the-marker would sit on the line
            last = (k == ks[-1])
            ax.annotate(r"$%.1f{\times}10^{%d}$" % (m, e), xy=(k, v),
                        xytext=(-5, -7) if last else (0, 6),
                        textcoords="offset points",
                        ha="right" if last else "center",
                        va="top" if last else "bottom",
                        fontsize=7.2, color=(LINE_BLUE, OKABE_ITO[6])[idx],
                        annotation_clip=False).set_clip_on(False)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 24])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "24"])
    ax.set_xlim(0.82, 30.0)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Truncated depth used in backward pass")
    ax.set_ylabel("Gradient norm (first-layer correction)")
    ax.margins(y=0.22)
    ax.legend(loc="best")
    style_ax(ax)
    save(fig, "fig_gradnorm")


# ============================================================= fig_ablation ==
def fig_ablation():
    """Ablation I: complete 2x2x2 factorial over the training-time flags.

    Encoding (replaces the previous eight-colour-per-arm palette, whose yellow
    entry #F0E442 was nearly invisible as a fill on white and whose eight
    adjacent hues imposed an unfair memory load):

      * hue   = family        (blue = unconstrained, orange-red = constrained)
      * fill  = projection    (filled = Pi on, open = Pi off)
      * hatch = feature bound (plain = on, hatched = off)

    The text labels under each bar repeat the family and the flag combination,
    so the figure also reads correctly in greyscale print.

    Every bar now carries its value.  That matters in panel (a) in particular: it
    is plotted from zero because all eight arms sit in 0.377-0.406, so the heights
    are deliberately uninformative -- which is the finding, the error does not
    separate the arms -- and only the printed numbers let a reader see how close
    they actually are.  The 0.5 ceiling of panel (b) is identified by a legend
    entry in the empty upper-right corner; the earlier free-standing note sat on
    the dashed line itself and collided with the first hatched bar.
    """
    abl = load("ablation.json")
    set_style()
    fig, axes = panel(ratio=0.46, ncols=2)
    C_UNC, C_CONS = LINE_BLUE, OKABE_ITO[6]
    tags = [("unc_pOn_fbOn_s1",   "Unc.\n$\\Pi$ fb",  C_UNC,  True,  True),
            ("unc_pOn_fbOff_s1",  "Unc.\n$\\Pi$",     C_UNC,  True,  False),
            ("unc_pOff_fbOn_s1",  "Unc.\nfb",          C_UNC,  False, True),
            ("unc_pOff_fbOff_s1", "Unc.\nnone",        C_UNC,  False, False),
            ("cons_pOn_fbOn_s1",   "Cons.\n$\\Pi$ fb", C_CONS, True,  True),
            ("cons_pOn_fbOff_s1",  "Cons.\n$\\Pi$",    C_CONS, True,  False),
            ("cons_pOff_fbOn_s1",  "Cons.\nfb",         C_CONS, False, True),
            ("cons_pOff_fbOff_s1", "Cons.\nnone",       C_CONS, False, False)]
    keys = [(tag, col, p_on, fb_on) for tag, _, col, p_on, fb_on in tags]
    labs = [lab for _, lab, _, _, _ in tags]
    xs = list(range(len(tags)))

    def arm_bar(ax, i, h, col, p_on, fb_on):
        if p_on:
            ax.bar(i, h, 0.62, facecolor=col, edgecolor="white", linewidth=0.9,
                   hatch=None if fb_on else "///", zorder=2)
        else:
            ax.bar(i, h, 0.62, facecolor="white", edgecolor=col, linewidth=1.1,
                   hatch=None if fb_on else "///", zorder=2)

    ax = axes[0]
    top = 0.0
    for i, (tag, col, p_on, fb_on) in enumerate(keys):
        r = abl.get(tag, {})
        fin = r.get("finite_extrap", False)
        v = r.get("rre_extrap") if (fin and r.get("rre_extrap") is not None) else None
        arm_bar(ax, i, v if v is not None else 0.0, col, p_on, fb_on)
        if v is not None:
            # three decimals: two would print 0.41/0.40/0.38/0.38 and hide the
            # 0.008 spread that the whole panel is about
            ax.text(i, v + 0.008, "%.3f" % v, ha="center", va="bottom",
                    fontsize=7.8)
            top = max(top, v)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8.4)
    ax.set_ylabel("Extrapolation RRE (800 layers)")
    ax.set_ylim(0, top * 1.13)
    style_ax(ax, "(a)")

    ax = axes[1]
    top = 0.0
    for i, (tag, col, p_on, fb_on) in enumerate(keys):
        c = abl.get(tag, {}).get("maxabs_chi_over")
        arm_bar(ax, i, c if c is not None else 0.0, col, p_on, fb_on)
        if c is not None:
            ax.text(i, c + 0.055, "%.2f" % c, ha="center", va="bottom",
                    fontsize=7.8)
            top = max(top, c)
    ax.axhline(0.5, color="k", ls="--", lw=1.0,
               label=r"training ceiling $\chi_{\max}=0.5$")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8.4)
    ax.set_ylabel(r"$\max_k\|\chi^k\|_\infty$")
    ax.set_ylim(0, top * 1.22)
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
    style_ax(ax, "(b)")
    save(fig, "fig_ablation")
# =============================================================== fig_frozen ==
def fig_frozen():
    """Ablation II: what removing the projection at test time costs.

    The diverged arm has no finite extrapolation error, so any column height
    drawn for it would be arbitrary.  It is therefore drawn as a hatched column
    that stops short and continues as an arrow, which is the standard glyph for
    "beyond the frame", and the annotation states the layer at which the rollout
    left the box and the state it had reached.  The axis is linear: with the one
    diverging arm represented by an arrow, the three finite bars (0.38, 0.40,
    0.53) can be read on a linear scale, which the previous log axis did not
    allow.
    """
    abl = load("ablation.json")
    set_style()
    fig, ax = panel(ncols=1, ratio=0.44)
    frz = [("unconstrained_frozen_proj1", "Unc., $\\Pi$ on", LINE_BLUE),
           ("unconstrained_frozen_proj0", "Unc., $\\Pi$ off", OKABE_ITO[6]),
           ("constrained_frozen_proj1", "Cons., $\\Pi$ on", OKABE_ITO[3]),
           ("constrained_frozen_proj0", "Cons., $\\Pi$ off", OKABE_ITO[7])]
    xs, ys, cols, labs, div = [], [], [], [], []
    for i, (tag, lab, c) in enumerate(frz):
        r = abl.get(tag, {})
        v = r.get("rre_extrap")
        finite = isinstance(v, (int, float)) and r.get("finite_extrap", True)
        xs.append(i); cols.append(c); labs.append(lab)
        ys.append(max(v, 1e-3) if finite else None)
        div.append(None if finite else r)
    fin = [y for y in ys if y is not None]
    ytop = max(fin) * 1.55          # headroom for the off-scale annotation
    kf = [k for k in xs if ys[k] is not None]
    ax.bar(kf, [ys[k] for k in kf], 0.6, color=[cols[k] for k in kf],
           edgecolor="k", linewidth=0.6, zorder=2)
    for i, r in enumerate(div):
        if r is None:
            continue
        # hatched shaft that stops short, then an arrow continuing upwards: the
        # column top is not a value, so no number is printed on it
        ycut = max(fin) * 0.90
        ax.bar([i], [ycut], 0.6, color="none", edgecolor="k", linewidth=0.6,
               hatch="/////", zorder=3)
        ax.add_patch(FancyArrowPatch((i, ycut * 1.01), (i, ytop * 0.71),
                                     arrowstyle="-|>,head_length=5,head_width=3",
                                     linewidth=1.0, color="k", zorder=4))
        ax.text(i, ytop * 0.745,
                "no finite RRE: the rollout\ndiverges at layer %d" % r.get("broke_at", 0),
                fontsize=8.6, ha="center", va="bottom", linespacing=1.35,
                color="0.15")
    for i, y in zip(xs, ys):
        if y is not None:
            ax.text(i, y + 0.012, "%.3f" % y, ha="center", va="bottom",
                    fontsize=8.4)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8.8)
    ax.set_ylabel("Extrapolation RRE (800 layers)")
    ax.set_ylim(0, ytop)
    ax.set_xlim(-0.6, len(frz) - 0.4)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"])
    style_ax(ax)
    save(fig, "fig_frozen")
# =============================================================== fig_images ==
def fig_images():
    """Reconstruction examples, quantified per panel.

    Two changes with respect to the previous version: every panel carries its own
    relative error, so the visual comparison is also a measurement, and the column
    headings are split over two lines with a fixed 2 mm gutter, which is what removes
    the overlap between the last two headings.
    """
    data = np.load(os.path.join(RES, "images.npz"))
    nx = ny = 32
    rows = ["gt", "classical8", "classical50", "unconstrained", "constrained"]
    titles = ["Ground\ntruth", "Classical\n$K{=}8$", "Classical\n$K{=}50$",
              "Unconstrained\n$K{=}8$", "Constrained\n(ours), $K{=}8$"]
    n_ex = min(4, data["gt"].shape[0])
    set_style()
    fig, axes = plt.subplots(n_ex, len(rows),
                             figsize=(TEXTW, TEXTW * 0.70))
    # 2 mm gutters: wspace is a fraction of the axes width, so solve for it.
    ncol = len(rows)
    total = 0.86 * TEXTW                      # usable width in inches
    gap_in = 2.0 / 25.4
    axw = (total - (ncol - 1) * gap_in) / ncol
    wspace = gap_in / axw
    fig.subplots_adjust(left=0.055, right=0.905, top=0.905, bottom=0.015,
                        wspace=wspace, hspace=0.075)
    vmax = max(float(data["gt"].max()), 1e-6)
    gt = data["gt"].reshape(n_ex, -1)
    for r, (row, ttl) in enumerate(zip(rows, titles)):
        for c in range(n_ex):
            a = axes[c, r]
            img = data[row][c].reshape(nx, ny)
            im = a.imshow(img, cmap="viridis", vmin=0, vmax=vmax, origin="lower")
            a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_title(ttl, fontsize=8.6, pad=3.0, linespacing=1.05)
            if r == 0:
                a.set_ylabel("ex.%d" % (c + 1), fontsize=8.6)
            if row != "gt":
                err = np.linalg.norm(img.ravel() - gt[c]) / np.linalg.norm(gt[c])
                a.text(0.04, 0.06, "%.2f" % err, transform=a.transAxes,
                       fontsize=7.6, color="white", ha="left", va="bottom",
                       bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none",
                                 alpha=0.70))
    cax = fig.add_axes([0.918, 0.12, 0.012, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"reconstructed contrast $|\chi|$", fontsize=8.6)
    cb.ax.tick_params(labelsize=7.6)
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
    fig, ax = panel(ncols=1, ratio=0.54)
    if ac:
        ax.axvspan(ac, 1.0, color=LINE_BLUE, alpha=0.16, lw=0, zorder=0)
        ax.axvline(ac, color=OKABE_ITO[5], ls=":", lw=1.0)
    if art:
        aa = np.array([r["alpha"] for r in art], dtype=float)
        am = np.array([r["sigma_full"] for r in art], dtype=float)
        ax.plot(aa, am, color=GREY_LINE, ls="-", lw=0.8, marker="x", ms=3.0,
                label=r"as-implemented $n_{\mathrm{cg}}{=}15$ (artefact)", zorder=2)
        ax.set_xlim(0.0, max(a.max(), aa.max()) * 1.05)
    ax.plot(a, meas, color=OKABE_ITO[0], marker=MARKERS[0], lw=1.0, ms=4.0,
            label=r"measured $\sigma_{\max}(J)$ (converged)", zorder=3)
    ax.plot(a, bnd, color=OKABE_ITO[6], ls="--", lw=1.0, marker=MARKERS[2], ms=3.6,
            label=r"proved bound $\alpha\mathcal{E}$", zorder=3)
    ax.axhline(1.0, color=GREY_NOTE, lw=0.8, ls="-")
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


# ==================================================== fig_fresnel_recon (new) ==
def fig_fresnel_recon():
    """Measured-data (Fresnel 2-D) reconstruction at 3 GHz, against three algorithms.

    The earlier version showed the published phantom above one reconstruction, so the
    reader could not tell what the 30-layer kernel bought over the standard alternatives
    on the same measured field.  The columns are now an algorithmic ladder computed on
    the identical data, calibration constant, projection box and CG solve:

        published phantom -> adjoint (1 CG step) -> Born (15 CG steps)
                          -> CSI kernel, 30 layers (the archived reconstruction)

    GUTTER.  ``wspace`` alone does not fix the white space between panels when the panels
    are square: ``imshow`` keeps the aspect and shrinks the image inside whatever box the
    layout hands it, so the visible gap is ``wspace * axw + (axw - image width)``.  The
    boxes are therefore placed explicitly, each exactly as wide as the image it holds.
    """
    targets = ["FoamDielExt", "FoamDielInt", "FoamTwinDiel"]
    cols = [("phantom", "Published\nphantom"),
            ("dsm", "Adjoint\n1 CG step"),
            ("born", "Born\n15 CG steps"),
            ("ours", "CSI kernel\n30 layers")]
    set_style()
    base = np.load(os.path.join(RES, "fresnel_baselines.npz"))
    d = np.load(os.path.join(RES, "fresnel_recon_v2_recons.npz"))
    rows = load("fresnel_baselines.json")
    stat = dict(((r["target"], r["arm"]), r) for r in rows if r["freq"] == 3.0)
    # the archived arm's residual lives in the v2 record, not in the baseline dump
    for r in load("fresnel_recon_v2.json"):
        if r["freq"] == 3.0:
            stat[(r["target"], "ours")] = {
                "rre": stat[(r["target"], "ours")]["rre"], "r_net": r["r_net"]}

    gap = 5.0 / 25.4                    # requested column gutter, inches
    vgap = 0.085                        # vertical gutter between rows
    title_h = 0.36                      # strip reserved above the top row (two-line heads)
    left_in = 0.86                      # room for the horizontal row labels
    axw = (TEXTW - 0.04 - left_in - 3 * gap) / 4.0
    H = 3 * axw + 2 * vgap + title_h + 0.05
    fig = plt.figure(figsize=(TEXTW, H))
    x0 = left_in / TEXTW
    for jj, (key, head) in enumerate(cols):
        for rr, tgt in enumerate(targets):
            y_in = 0.02 + (2 - rr) * (axw + vgap)
            a = fig.add_axes([x0 + jj * (axw + gap) / TEXTW, y_in / H,
                              axw / TEXTW, axw / H])
            if key == "phantom":
                pair = d["%s_3" % tgt]
                img = pair[1]
            elif key == "ours":
                pair = d["%s_3" % tgt]
                img = pair[0]
            else:
                img = base["%s_%s" % (tgt, key)]
            a.imshow(img, vmin=0, vmax=1, cmap="gray", aspect="equal")
            a.set_xticks([]); a.set_yticks([])
            for s_ in a.spines.values():
                s_.set_visible(False)
            if rr == 0:
                a.set_title(head, fontsize=8.2, pad=4, linespacing=1.1)
            if jj == 0:
                a.set_ylabel(tgt, fontsize=8.4, rotation=0, labelpad=8,
                             va="center", ha="right")
            if key != "phantom":
                r = stat[(tgt, key)]
                a.text(0.04, 0.05,
                       "RRE %.2f\nres %.2f" % (r["rre"], r["r_net"]),
                       transform=a.transAxes, fontsize=6.3, color="white",
                       ha="left", va="bottom", linespacing=1.15,
                       bbox=dict(boxstyle="square,pad=0.18", fc="black",
                                 ec="none", alpha=0.72))
    save(fig, "fig_fresnel_recon")
