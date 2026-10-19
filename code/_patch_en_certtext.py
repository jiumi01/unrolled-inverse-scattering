# -*- coding: utf-8 -*-
"""EN: rewrite the Tab. cert narrative so that it is true for the CORRECTED
measurements (the old text asserted the numerically evaluated channels were 'much
larger still', which was an artefact of a non-converged power iteration)."""
import io

P = r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\paper_en\main.tex"
s = io.open(P, encoding="utf-8").read()

OLD = r"""(3) The two numerically evaluated channels are much larger still
($\sigma_{\max}(A)=\numv{\NumSigA}$, $\sigma_{\max}(B)=\numv{\NumSigB}$), giving the strongest
rigorous lower bound $\numv{\NumSigLowBound}$; these are rigorous \emph{inequalities} whose
right-hand sides are computed by the calibrated instrument. (4) The composite gain exceeds one in"""

NEW = r"""(3) The remaining numerically evaluated channels give $\sigma_{\max}(A)=\numv{\NumSigA}$ and
$\sigma_{\max}(B)=\numv{\NumSigB}$, and the full-Jacobian measurement is
$\sigma_{\max}(J)=\numv{\NumPhysSigma}$. These satisfy the structural inequality the instrument
checks internally, $\sigma_{\max}(J)\ge\max_k\sigma_{\max}(\text{block}_k)$, and the strongest
rigorous lower bound over all channels is $\numv{\NumSigLowBound}$. Each such value is a Rayleigh
quotient (a Ritz value of $J^{\!\top}\!J$), hence a rigorous lower bound on the true norm, obtained
with the same calibrated instrument and reported together with its power-iteration cross-check
(Sec.~\ref{sec:exp}, reproducibility paragraph). (4) The composite gain exceeds one in"""

assert s.count(OLD) == 1, "EN cert narrative anchor count=%d" % s.count(OLD)
s = s.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(s)
print("EN cert narrative patched")
