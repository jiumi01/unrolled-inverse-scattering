# -*- coding: utf-8 -*-
"""_macro_diff.py — changed / added / removed numeric macros between two numbers.tex files.

Used to build the per-round changelog: "what numbers moved, from what to what, and why".
Run it as part of every freeze so the changelog is generated rather than remembered.

    python _macro_diff.py <old numbers.tex> <new numbers.tex>
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")


def load(p):
    d = {}
    for line in io.open(p, encoding="utf-8"):
        m = DEF.match(line.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


def main():
    if len(sys.argv) >= 3:
        old_p, new_p = sys.argv[1], sys.argv[2]
    else:
        old_p = os.path.join(ROOT, "reviews", "round_r2", "snapshot", "paper_en", "numbers.tex")
        new_p = os.path.join(ROOT, "paper_en", "numbers.tex")
    a, b = load(old_p), load(new_p)
    print("old : %s  (%d macros)" % (os.path.relpath(old_p, ROOT), len(a)))
    print("new : %s  (%d macros)" % (os.path.relpath(new_p, ROOT), len(b)))
    keys = sorted(set(a) | set(b))
    changed = [k for k in keys if a.get(k) != b.get(k)]
    print("\nchanged / added / removed: %d" % len(changed))
    for k in changed:
        print("  %-28s %-16s -> %s" % (k, a.get(k, "(absent)"), b.get(k, "(removed)")))
    same = len(keys) - len(changed)
    print("\nunchanged: %d" % same)
    # a reminder of what must NOT move for a text-only revision
    guard = ["NumSigCompCons", "NumSigCompUnc", "NumPhysSigma", "NumRhoComp", "NumMJCons",
             "NumGapRatio", "NumSigC", "NumPhysWBlock", "NumAlphaC"]
    moved = [g for g in guard if g in a and g in b and a[g] != b[g]]
    print("certified/macro guard moved: %s" % (moved if moved else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
