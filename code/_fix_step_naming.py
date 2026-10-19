# -*- coding: utf-8 -*-
"""_fix_step_naming.py -- step vs layer naming in the extrapolation paragraph.

R3 n3.3 flagged that NumResConsEnd was called "the last step" when it is index k=95.  The fix
added an explanation, but that explanation says "layer 96 of 100" and "sampled every five
layers" -- and a step here is EIGHT layers (the trained K=8 block applied cyclically), so
"layer 96 of 100" is wrong on its face: there is no layer 96 of 100.  What is 96 of 100 is the
STEP.  Same defect class as n3.3 (naming a quantity by the wrong unit), caught by re-reading
the sentence the reviewer attached to.
"""
import io, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

EN_OLD = ("""horizon ($\\numv{\\NumResConsHead}$ at a representative step, $\\numv{\\NumResConsEnd}$ at the last
\\emph{recorded} step, i.e.\\ layer $96$ of $100$: the residual record is sampled every five
layers, so the last recorded step is $k=95$, and we label it as such rather than calling it
``the last step'')""")
EN_NEW = ("""horizon ($\\numv{\\NumResConsHead}$ at a representative step, $\\numv{\\NumResConsEnd}$ at the last
\\emph{recorded} step, which is the $96$th of the $100$ extrapolation \\emph{steps}: the record is
sampled every five steps, so the last recorded index is $k=95$ (zero-based) and we label it as
such rather than calling it ``the last step'')""")

CN_OLD = ("（取一个代表步时 $\\numv{\\NumResConsHead}$，末个\\emph{记录}步 $\\numv{\\NumResConsEnd}$，即 $100$ 层中的第 $96$ 层——残差记录每 $5$ 层采样一次，故最后被记录的是 $k=95$；我们按此标注，而不称其为“末步”）")
CN_NEW = ("（取一个代表步时 $\\numv{\\NumResConsHead}$，末个\\emph{记录}步 $\\numv{\\NumResConsEnd}$，即 $100$ 个外推\\emph{步}中的第 $96$ 步——残差记录每 $5$ \\emph{步}采样一次，故最后被记录的下标是 $k=95$（从 $0$ 起）；我们按此标注，而不称其为“末步”）")


def patch(path, old, new, tag):
    s = io.open(path, encoding="utf-8").read()
    if old not in s:
        print("[MISS] %-8s sentence not found" % tag)
        return False
    io.open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
    print("[ok]   %-8s step/layer naming corrected" % tag)
    return True


def main():
    ok = patch(os.path.join(ROOT, "paper_en", "main.tex"), EN_OLD, EN_NEW, "EN")
    ok &= patch(os.path.join(ROOT, "paper_cn", "main.tex"), CN_OLD, CN_NEW, "CN")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
