# -*- coding: utf-8 -*-
"""_patch_overfull2.py — clear the remaining overfull hboxes.

After the first overfull pass, the CN build still reported:
  line 95   27.69pt  (eq:contraststep objective on one line)
  line 121  11.04pt  (eq:budgets, \qquad between the two budgets)
  line 136  21.78pt  (eq:blocks, \qquad between C and D)
  line 225   7.30pt  (eq:noiseexact, \;\le\;)
  374--389  52.44pt  (tab:abl, five numeric columns)
  394--404  40.58pt  (tab:abl2)
and the EN build:
  1163--1180 4.47pt  (tab:trace p-columns one notch too wide)

Each edit is assert-guarded (exactly one match).  All edits are typographic;
no number, claim or equation content changes.
"""
import io, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CN = os.path.join(ROOT, "paper_cn", "main.tex")
EN = os.path.join(ROOT, "paper_en", "main.tex")


def patch(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for i, (old, new, tag) in enumerate(pairs, 1):
        n = s.count(old)
        assert n == 1, "%s [%s]: found %d occurrences" % (os.path.basename(path), tag, n)
        s = s.replace(old, new)
        print("  ok  %-22s %s" % (os.path.basename(path), tag))
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)


CN_PAIRS = [
    # -- (a) eq:contraststep: put the Tikhonov term on its own line ------------
    (
        r"""\hat\chi^{\mathrm{LS}}(E;y) &= \arg\min_{\chi\in\R^n}\;
\big\|\mathbf{G}_s\,\mathrm{diag}(E)\chi-y\big\|_F^{2}
\;+\;\sum_{j=1}^{n}\lambda_j\chi_j^{2},
\label{eq:contraststep}\\""",
        r"""\hat\chi^{\mathrm{LS}}(E;y) &= \arg\min_{\chi\in\R^n}\;
\big\|\mathbf{G}_s\,\mathrm{diag}(E)\chi-y\big\|_F^{2}\nonumber\\
&\qquad\;+\;\sum_{j=1}^{n}\lambda_j\chi_j^{2},
\label{eq:contraststep}\\""",
        "eq:contraststep split",
    ),
    # -- (b) eq:budgets: stacked aligned ---------------------------------------
    (
        r"""\begin{equation}\label{eq:budgets}
\bar\epsilon := \sup_{k}\sup_{z\in B}\norm{H_k(z;y)},\qquad
\ell := \sup_k \operatorname{Lip}\big(H_k(\cdot;y)\big),
\end{equation}""",
        r"""\begin{equation}\label{eq:budgets}
\begin{aligned}
\bar\epsilon &:= \sup_{k}\sup_{z\in B}\norm{H_k(z;y)},\\
\ell &:= \sup_k \operatorname{Lip}\big(H_k(\cdot;y)\big),
\end{aligned}
\end{equation}""",
        "eq:budgets aligned",
    ),
    # -- (c) eq:blocks: stacked aligned ---------------------------------------
    (
        r"""\begin{equation}\label{eq:blocks}
C \;=\; \theta\,\alpha\,\mathrm{diag}(E),\qquad
D \;=\; (1-\theta\alpha)I+\theta\alpha\,\mathrm{diag}(\chi)\,\mathbf{G}_b ,
\end{equation}""",
        r"""\begin{equation}\label{eq:blocks}
\begin{aligned}
C &\;=\; \theta\,\alpha\,\mathrm{diag}(E),\\
D &\;=\; (1-\theta\alpha)I+\theta\alpha\,\mathrm{diag}(\chi)\,\mathbf{G}_b ,
\end{aligned}
\end{equation}""",
        "eq:blocks aligned",
    ),
    # -- (d) eq:noiseexact: drop the wide relation spacing ---------------------
    (
        r"""\norm{z^{K}(y+\Delta y)-z^{K}(y)}
\;\le\;\Big[\sum_{k=0}^{K-1}\Lambda_k\!\!\prod_{j=k+1}^{K-1}\!\!L_j\Big]\norm{\Delta y}.""",
        r"""\norm{z^{K}(y+\Delta y)-z^{K}(y)}
\le\Big[\sum_{k=0}^{K-1}\Lambda_k\!\!\prod_{j=k+1}^{K-1}\!\!L_j\Big]\norm{\Delta y}.""",
        "eq:noiseexact spacing",
    ),
    # -- (e) tab:abl: shrink to footnotesize, tighter gutters ------------------
    (
        r"""\label{tab:abl}
\begin{tabular}{llcccc}""",
        r"""\label{tab:abl}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}""",
        "tab:abl footnotesize",
    ),
    # -- (f) tab:abl2 ----------------------------------------------------------
    (
        r"""\label{tab:abl2}
\begin{tabular}{llccc}""",
        r"""\label{tab:abl2}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccc}""",
        "tab:abl2 footnotesize",
    ),
]

EN_PAIRS = [
    (
        r"""\begin{tabular}{p{0.42\linewidth}lp{0.36\linewidth}}""",
        r"""\begin{tabular}{p{0.40\linewidth}lp{0.34\linewidth}}""",
        "tab:trace columns",
    ),
]

if __name__ == "__main__":
    print("== CN ==")
    patch(CN, CN_PAIRS)
    print("== EN ==")
    patch(EN, EN_PAIRS)
    print("done")
