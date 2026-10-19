# -*- coding: utf-8 -*-
"""_patch_abstract.py — compress the English abstract (review R3 m3.5: it expanded to ~446
words against the IEEE norm of roughly 250), and trim the Chinese abstract to match.

Only length is changed: no claim, number, or hedging qualifier is added or strengthened.
The sentences removed are the frozen-arm figures and the price delta, both of which are
stated in the body (Sec. IV) and are unaffected here.
"""
import io, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

EN_NEW = r"""Deep unrolling of classical iterative solvers is a leading design principle for electromagnetic
inverse scattering: SOM-Net, unrolled contrast-source architectures and the Neural Born Iterative
Method replace classical update rules by learned operators and report consistent empirical gains.
To our knowledge none of them supplies, at the level of the individual layer, a sufficient condition
for convergence, a fixed-point characterisation, or an error bound. This paper supplies them, and
identifies the structural obstruction that explains the gap. Casting each unrolled layer as a physics
iteration perturbed by a learned correction, we prove three results. (i) \emph{Impossibility}: for
the faithful Born/contrast-source composite the bilinear coupling $\chi\odot E$ makes a block of the
layer Jacobian \emph{exactly} $\theta\alpha\,\mathrm{diag}(E)$, whence
$\sigma_{\max}(J)\ge\theta\alpha\mathcal{E}$ with $\mathcal{E}$ the peak per-cell field energy across
incidences; the layer is \emph{provably} non-contractive for every step size above the explicit
threshold $\alpha_c=(\theta\mathcal{E})^{-1}$, so global Euclidean contraction is unattainable on the
range a network actually uses and any stability guarantee must be layered. (ii) \emph{Certified
boundedness}: the weak-scattering condition $\beta_0=\chi_{\max}\norm{\mathbf{G}_b}_2<1$ makes the
damped field subsystem a contraction in the contrast source for \emph{every} $\alpha\in(0,1]$, and
combined with a nonexpansive physics-prior projection it confines the rollout to an \emph{explicit
compact set for arbitrary depth and arbitrary trained parameters}, so extrapolation cannot diverge.
(iii) \emph{Fidelity and stability, delimited honestly}: we give an exact finite-depth noise
accounting together with the non-normal correction $\kappa_2(V)\sup_m\norm{J^m}$ that a naive
$\Lambda/(1-\rho)$ omits, and show that the geometric bounds $\Lambda/(1-\rho)$ and
$\bar\epsilon/(1-\rho_\chi)$ hold only \emph{conditionally} --- their hypotheses are contradicted by
our own measurements --- reporting both so that nothing is claimed beyond what is proved. Ablations
over $800$ unrolled layers identify what each ingredient actually buys, and the framework is verified
on 2-D TM MoM-discretized scattering, where a certified $K{=}8$ network matches the classical
iteration at $K{=}100$.
"""

EN_TAIL_OLD = r"""$800$ 层展开的消融实验厘清了各要素的真实作用"""

def main():
    # ---------------- English ----------------
    p = os.path.join(ROOT, "paper_en", "main.tex")
    s = io.open(p, encoding="utf-8").read()
    A = "\\begin{abstract}"
    B = "\\end{abstract}"
    i = s.index(A) + len(A)
    j = s.index(B)
    old = s[i:j]
    assert "$K{=}100$" in old and "Impossibility" in old, "unexpected abstract body"
    s = s[:i] + "\n" + EN_NEW + s[j:]
    io.open(p, "w", encoding="utf-8").write(s)
    print("EN abstract: %d -> %d chars" % (len(old), len(EN_NEW)))

    # ---------------- Chinese ----------------
    p = os.path.join(ROOT, "paper_cn", "main.tex")
    s = io.open(p, encoding="utf-8").read()
    A = "\\begin{abstract}"
    B = "\\end{abstract}"
    i = s.index(A) + len(A)
    j = s.index(B)
    old = s[i:j]
    assert EN_TAIL_OLD in old, "CN abstract tail anchor not found"
    # cut the frozen-arm figures + price-delta sentence, which the body states in full
    cut_from = old.index(EN_TAIL_OLD)
    cut_to = old.index("框架在二维 TM MoM", cut_from)
    keep = old[:cut_from] + ("$800$ 层展开的消融实验厘清了各要素的真实作用，"
                             "而框架在二维 TM MoM 离散散射上得到验证：")
    keep = keep + old[old.index("一个认证的 $K{=}8$ 网络达到经典迭代 $K{=}100$ 的精度。", cut_to):]
    s = s[:i] + keep + s[j:]
    io.open(p, "w", encoding="utf-8").write(s)
    print("CN abstract: %d -> %d chars" % (len(old), len(keep)))


if __name__ == "__main__":
    main()
