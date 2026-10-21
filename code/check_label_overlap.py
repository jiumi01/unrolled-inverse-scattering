# -*- coding: utf-8 -*-
"""check_label_overlap.py -- find colliding annotations in a generated figure.

Eyeballing "does 0.3737 touch 0.3716" is exactly the kind of judgement that fails at
6 pt type, so the check is geometric: draw the figure, take every Text artist's window
extent, and report every intersecting pair (plus any label overlapping the legend box).

Usage:  FIG_TEXTW_PT=... python code/check_label_overlap.py fig_depth
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figs as M


def audit(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                items.append((t.get_text(), t.get_window_extent(r), ax))
        leg = ax.get_legend()
        if leg is not None:
            items.append(("<LEGEND>", leg.get_window_extent(r), ax))
        # axis tick labels and axis labels are candidates for collisions too
        for lab in (ax.xaxis.label, ax.yaxis.label):
            items.append(("<%s>" % lab.get_text(), lab.get_window_extent(r), ax))
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][2] is not items[j][2]:
                continue
            # 0.8 pt of air on every side: boxes that merely touch still read as one
            # smudge at 6 pt, so a bare intersection test is too permissive.
            if items[i][1].padded(0.8).overlaps(items[j][1].padded(0.8)):
                bad.append((items[i][0], items[j][0]))
    print("%-16s text items=%d  collisions=%d" % (name, len(items), len(bad)))
    for a, b in bad:
        print("   COLLIDE  %-28r  %r" % (a, b))
    return len(bad)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "fig_depth"
    fn = getattr(M, target)
    hits = {}

    def hook(fig, name):
        hits[name] = audit(fig, name)
        fig.savefig(os.path.join(M.FIG, name + "_check.png"), dpi=170)
        import matplotlib.pyplot as plt
        plt.close(fig)

    M.save = hook
    fn()
    print("TOTAL collisions:", sum(hits.values()))


if __name__ == "__main__":
    main()
