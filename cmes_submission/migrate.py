#!/usr/bin/env python3
"""Migrate MDPI main.tex to TSP/CMES template."""
import re, pathlib

root = pathlib.Path(r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究")
src = (root / "paper_mdpi" / "main.tex").read_text(encoding="utf-8")

# --- Extract abstract ---
m = re.search(r"\\abstract\{(.*?)\n\}\n", src, re.S)
abstract = m.group(1).strip() if m else ""

# --- Extract keywords ---
m = re.search(r"\\keyword\{(.*?)\}", src)
keywords = m.group(1).strip() if m else ""

# --- Extract body: from \begin{document}+1 to \begin{thebibliography} ---
body_start = src.index("\\begin{document}") + len("\\begin{document}")
bib_start = src.index("\\begin{thebibliography}")
body = src[body_start:bib_start]

# --- Extract bibliography ---
bib_end = src.index("\\end{thebibliography}") + len("\\end{thebibliography}")
bib = src[bib_start:bib_end]

# --- Theorem env name mapping ---
env_map = [
    ("theorem", "Theorem"),
    ("lemma", "Lemma"),
    ("corollary", "Corollary"),
    ("proposition", "Proposition"),
    ("remark", "Remark"),
    ("definition", "Definition"),
]
for old, new in env_map:
    body = body.replace(f"\\begin{{{old}}}", f"\\begin{{{new}}}")
    body = body.replace(f"\\end{{{old}}}", f"\\end{{{new}}}")
    bib = bib.replace(f"\\begin{{{old}}}", f"\\begin{{{new}}}")
    bib = bib.replace(f"\\end{{{old}}}", f"\\end{{{new}}}")

# --- Remove \label references that MDPI adds for theorem-like environments ---
# TSP numbers them automatically, MDPI labels like \label{thm:bounded} still work as \ref

# --- Strip MDPI-specific commands that TSP doesn't know ---
body = re.sub(r"\\section\*?\{Supplementary\}", r"\\section{Supplementary}", body)

# --- Build new file ---
header = r"""%Tech Science Press (CMES)
\documentclass[journal,article,submit,moreauthors,pdftex]{Definitions/tsp}
\input{Definitions/package}
\include{Definitions/unicode}

\continuouspages{}
\firstpage{1}
\makeatletter
\setcounter{page}{\@firstpage}
\makeatother
\pubvolume{0}
\issuenum{0}
\elocationid{0}
\articlenumber{0}
\pubyear{2026}
\doiyear{2026}
\copyrightyear{2026}
\datereceived{}
\dateaccepted{}
\dateonlinefirst{}
\datepublished{}

\usepackage{mathtools}
\usepackage{siunitx}

\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\norm}[1]{\lVert #1 \rVert}
\newcommand{\infn}[1]{\lVert #1 \rVert_{\infty}}
\newcommand{\mean}{\operatorname{mean}}
\newcommand{\numv}[1]{#1}
\input{numbers}

\graphicspath{{figures/}}

\Title{Fixed Points and Certified Boundedness of Unrolled Born/Contrast-Source Inverse Scattering Networks}

\Author{Longgang Wang\textsuperscript{1,*}, Guoqiang Wang\textsuperscript{1}, and Nianfeng Shi\textsuperscript{1}}
\AuthorNames{Longgang Wang, Guoqiang Wang and Nianfeng Shi}

\address{%
\textsuperscript{1}Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China
}

\corres{Corresponding Author: Longgang Wang. Email: longgangwang@163.com}

\abstract{"""

novelty = r"""

\noindent\textbf{Statement of Novelty.} (1) We prove that the Born/contrast-source unrolled layer is \emph{provably non-contractive} above an explicit step-size threshold, because the bilinear coupling block is exactly $\theta\alpha\,\mathrm{diag}(E)$; this obstruction has not been identified before. (2) We supply a \emph{layer-wise} certification framework: a field-subsystem contraction condition $\beta_0<1$, an invariant compact set under a nonexpansive projection, and an exact finite-depth noise accounting that exposes the non-normal factor a naive geometric bound omits. (3) We delimit what is and is not certified: the geometric radii are conditional and the $\rho_\chi$ contraction hypothesis is measured to fail, so we report measured bounds rather than advertising guarantees. (4) The framework is validated on synthetic 2-D TM scattering with four baselines and on the published Institut Fresnel measured data.

"""

footer = r"""

\acknowledgement{Not applicable.}

\funding{The author(s) received no specific funding for this study.}

\authorcontributions{The author confirms sole responsibility for the following: conceptualization, methodology, software, validation, formal analysis, investigation, resources, data curation, writing---original draft preparation, writing---review and editing, visualization, supervision, project administration, and funding acquisition. The author has read and agreed to the published version of the manuscript.}

\availabilityofdataandmaterials{The code and numerical logs that support the findings of this study are openly available in the GitHub repository at \url{https://github.com/jiumi01/unrolled-inverse-scattering} (tag v1.1). The measured Institut Fresnel data set is available from the Institut Fresnel website.}

\ethicsapproval{Not applicable.}

\conflictsofinterest{The author declares no conflict of interest.}

"""

# keyword line
kw_line = f"\n\n\\keyword{{{keywords}}}\n"

# assemble
out = header + abstract + "}\n" + kw_line + "\n\\begin{document}\n" + body + "\n" + novelty + footer + bib + "\n\\end{document}\n"

dst = root / "cmes_submission" / "main.tex"
dst.write_text(out, encoding="utf-8")
print(f"wrote {dst}, {len(out)} chars")
