# -*- coding: utf-8 -*-
"""Patch the EN reproducibility paragraph: the claimed closed form sigma_max(A)=1-gamma
is FALSE (A depends on chi through the field chain). Replace with the two real,
reported self-checks."""
import io

P = r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\paper_en\main.tex"
s = io.open(P, encoding="utf-8").read()

OLD = r"""\emph{Reproducibility.} Every quantity in Table~\ref{tab:cert} is produced by a single audit
driver that evaluates the block Jacobians of Lemma~\ref{lem:blocks} by finite-difference power
iteration with an analytic reverse-mode adjoint, and whose correctness is self-checked against the
closed form $\sigma_{\max}(A)=1-\gamma$ (Table~\ref{tab:cert}, row~2). The mapping between the
derivations, the numbered statements of this paper and the functions implementing them is given in
Table~\ref{tab:trace}."""

NEW = r"""\emph{Reproducibility.} Every quantity in Table~\ref{tab:cert} is produced by a single audit
driver that evaluates the block Jacobians of Lemma~\ref{lem:blocks} by finite-difference power
iteration with an analytic reverse-mode adjoint. The instrument is self-checked twice and both
checks are reported: (i) on a random linear test map the estimate is compared with the exact
singular value returned by \texttt{torch.linalg.matrix\_norm} (relative error
$\numv{\NumInstrErr}$), which validates the power iteration on a problem with a known answer; and
(ii) on the layer itself the measured coupling block is compared with the closed form
$\sigma_{\max}(C)=\theta\alpha\mathcal{E}$ of Lemma~\ref{lem:blocks} (measured-to-proved ratio
$\numv{\NumSigCRatio}$), which validates the adjoint and the state ordering against the structure
the theory predicts. We stress that $\sigma_{\max}(A)$ has \emph{no} closed form --- the contrast
step depends on $\chi$ through the field chain $\chi\to w^+\to E$ --- so $A$ is reported as a
measured quantity and never used as a certificate. The mapping between the derivations, the
numbered statements of this paper and the functions implementing them is given in
Table~\ref{tab:trace}."""

assert s.count(OLD) == 1, "EN setup anchor not unique: %d" % s.count(OLD)
s = s.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(s)
print("EN setup patched ok")
