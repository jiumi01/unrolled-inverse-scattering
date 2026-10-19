# -*- coding: utf-8 -*-
"""Measure the natural width of the tab:cert tabular at several font sizes.

`\small` made the reported overfull *larger* (146.8pt -> 197.1pt), which cannot
happen if the tabular were simply scaled down.  Rather than guess, extract the
tabular verbatim from the manuscript and \sbox it at each size, printing \wd.
"""
import io
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EN = os.path.join(ROOT, "paper_en")

src = io.open(os.path.join(EN, "main.tex"), encoding="utf-8").read()

# preamble: everything up to \begin{document}, minus the title/author machinery
pre = src.split("\\begin{document}")[0]
pre = re.sub(r"\\title\{.*?\n", "", pre, flags=re.S)
pre = pre.replace("\\input{numbers}", "")

# the tabular of tab:cert
m = re.search(r"(\\begin\{tabular\}\{llcc\}.*?\\end\{tabular\})", src, re.S)
assert m, "tab:cert tabular not found"
body = m.group(1)

numbers = io.open(os.path.join(EN, "numbers.tex"), encoding="utf-8").read()

probe = r"""\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{siunitx}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\newcommand{\infn}[1]{\left\lVert #1 \right\rVert_{\infty}}
\newcommand{\mean}{\operatorname{mean}}
\newcommand{\numv}[1]{#1}
%(numbers)s
\begin{document}
\typeout{TEXTWIDTH=\the\textwidth}
\newbox\probe
%(sweeps)s
x
\end{document}
"""

BODY = body.replace(r"\begin{tabular}{llcc}", r"\begin{tabular}{@{}%s@{}}")

sweeps = []
cands = [
    (r"p{0.40\textwidth}cc c", None),
    (r"p{0.40\textwidth}lcc", None),
    (r"p{0.42\textwidth}lcc", r"\small"),
    (r"p{0.40\textwidth}lcc", r"\footnotesize"),
    (r"p{0.36\textwidth}lcc", None),
    (r"p{0.34\textwidth}lcc", None),
]
for spec, size in cands:
    b = BODY % spec
    if size:
        b = size + " " + b
    sweeps.append(r"\sbox{\probe}{%s}\typeout{WIDTH [%s]%s = \the\wd\probe}"
                  % (b, spec, ("" if not size else "+" + size)))

probe = probe % {"numbers": numbers, "sweeps": "\n".join(sweeps)}

d = tempfile.mkdtemp(prefix="meas")
tex = os.path.join(d, "meas.tex")
io.open(tex, "w", encoding="utf-8").write(probe)
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "meas.tex"],
                   cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
out = (r.stdout or "") + (r.stderr or "")
for line in out.splitlines():
    if "WIDTH" in line or "TEXTWIDTH" in line:
        print(line.strip())
if "WIDTH" not in out:
    print("PROBE FAILED")
    print("\n".join(out.splitlines()[-25:]))
