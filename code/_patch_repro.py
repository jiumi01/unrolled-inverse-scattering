# -*- coding: utf-8 -*-
"""_patch_repro.py — rewrite the reproducibility paragraph in both manuscripts.

The paragraph still described the instrument as "finite-difference power iteration", which
is the estimator R1's blocking complaint was aimed at: on these strongly non-normal maps the
finite-difference power iteration had not converged and returned sigma_full = 48.5 BELOW its
own block sigma_B = 76.6, which is structurally impossible.  The instrument has since been
replaced (exact forward-mode JVP + exact reverse-mode adjoints, Lanczos/IRAM for sigma_max,
IRAM for the spectral radius, an eigenproblem on (J^T)^m J^m for the transient factor).
The new paragraph also reports the third exact self-check (Born blocks) and the
double-precision requirement, and it cites two macros rather than hard-coding numbers.
"""
import io, os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

EN_OLD = r"""\emph{Reproducibility.} Every quantity in Table~\ref{tab:cert} is produced by a single audit
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

EN_NEW = r"""\emph{Reproducibility.} Every quantity in Table~\ref{tab:cert} is produced by a single audit
driver. Its forward operator products are exact (\texttt{torch.func.jvp}) and its adjoints are
exact reverse-mode derivatives, both wrapped in a matrix-free operator, so no spectral estimate
is contaminated by finite-difference truncation: $\sigma_{\max}$ is obtained by Lanczos/IRAM on
$J^{\!\top}\!J$, the spectral radius by IRAM on $J$, and the transient factor $\sup_m\|J^m\|$ by
an eigenproblem on $(J^{\!\top})^mJ^m$. Power iteration is retained only as an independent
cross-check; since both estimators are rigorous lower bounds, their disagreement measures the
honest discretisation spread rather than the value. Three \emph{exact} self-checks are reported.
(i) On a random linear test map the estimate is compared with the exact singular value returned
by \texttt{torch.linalg.matrix\_norm} (relative error $\numv{\NumInstrErr}$). (ii) On the layer
itself the measured coupling block is compared with the closed form
$\sigma_{\max}(C)=\theta\alpha\mathcal{E}$ of Lemma~\ref{lem:blocks} (measured-to-proved ratio
$\numv{\NumSigCRatio}$). (iii) In the Born variant, whose contrast step does not depend on the
field state, two blocks are analytically transparent --- $\sigma_{\max}(B)\equiv0$ and
$\sigma_{\max}(A)=1-\gamma$ --- and the instrument reproduces both
($\numv{\NumBornSigmaB}$ and $\numv{\NumBornSigmaA}$). We stress that in the faithful variant
$\sigma_{\max}(A)$ has \emph{no} closed form --- the contrast step depends on $\chi$ through the
field chain $\chi\to w^+\to E$ --- so there $A$ is reported as a measured quantity and never used
as a certificate. All audit quantities are computed in double precision, because the contrast
step inverts a Tikhonov system whose smallest eigenvalue is
$\lambda_{\min}=\numv{\NumLamMin}$, so a single-precision directional derivative through the
$n_{\mathrm{cg}}=\numv{\NumNCg}$-step CG loop loses significant digits; as a recorded
illustration, the single-precision sweep returns $\sigma_{\max}=\numv{\NumAlphaLowArtifact}$ at
$\alpha=0.05$, whereas the analytic small-$\alpha$ limit of \eqref{eq:layer-code} is
$\mathrm{diag}((1-\gamma)I,I)$, i.e.\ $\sigma_{\max}\to1$. That single-precision value is kept in
the archived results as evidence of the artefact, not used. The mapping between the derivations,
the numbered statements of this paper and the functions implementing them is given in
Table~\ref{tab:trace}."""

CN_OLD = r"""\emph{可复现性}：表~\ref{tab:cert} 中每个量均由同一个审计驱动脚本产生，它以有限差分幂迭代配合解析反向模式伴随计算引理~\ref{lem:blocks} 的各分块 Jacobian。仪器做两次自校验，且两次结果均在文中报告：(i) 在随机线性测试映射上把估计值与 \texttt{torch.linalg.matrix\_norm} 给出的精确奇异值比较（相对误差 $\numv{\NumInstrErr}$），这验证了幂迭代在已知答案问题上的正确性；(ii) 在层本身把实测耦合块与引理~\ref{lem:blocks} 的闭式 $\sigma_{\max}(C)=\theta\alpha\mathcal{E}$ 比较（实测/已证之比 $\numv{\NumSigCRatio}$），这验证了伴随与状态排序与理论所预测的结构一致。需强调 $\sigma_{\max}(A)$ \emph{没有}闭式——对比度步经场链 $\chi\to w^{+}\to E$ 依赖 $\chi$——因此 $A$ 作为实测量报告，绝不作为证书使用。推导、本文编号陈述与实现函数之间的对应见表~\ref{tab:trace}。"""

CN_NEW = r"""\emph{可复现性}：表~\ref{tab:cert} 中每个量均由同一个审计驱动脚本产生。该仪器的正向算子乘积分用精确的 \texttt{torch.func.jvp}、伴随用精确反向模式导数，二者装入无矩阵算子，故任何谱估计都不受有限差分截断污染：$\sigma_{\max}$ 由 $J^{\!\top}\!J$ 上的 Lanczos/IRAM 给出，谱半径由 $J$ 上的 IRAM 给出，瞬态因子 $\sup_m\|J^m\|$ 由 $(J^{\!\top})^mJ^m$ 上的特征问题给出。幂迭代只保留作独立交叉校验；由于两种估计都是严格下界，二者之差度量的是诚实的离散化散布而不是取值本身。文中报告\emph{三条精确}自校验：(i) 在随机线性测试映射上把估计值与 \texttt{torch.linalg.matrix\_norm} 给出的精确奇异值比较（相对误差 $\numv{\NumInstrErr}$）；(ii) 在层本身把实测耦合块与引理~\ref{lem:blocks} 的闭式 $\sigma_{\max}(C)=\theta\alpha\mathcal{E}$ 比较（实测/已证之比 $\numv{\NumSigCRatio}$）；(iii) 在 Born 变体中，因对比度步不依赖场状态而有两个解析透明的分块——$\sigma_{\max}(B)\equiv0$ 与 $\sigma_{\max}(A)=1-\gamma$——仪器把二者都复现出来（$\numv{\NumBornSigmaB}$ 与 $\numv{\NumBornSigmaA}$）。需强调在忠实变体中 $\sigma_{\max}(A)$ \emph{没有}闭式——对比度步经场链 $\chi\to w^{+}\to E$ 依赖 $\chi$——因此该处 $A$ 作为实测量报告，绝不作为证书使用。全部审计量都在双精度下计算，因为对比度步求解的 Tikhonov 系统最小特征值为 $\lambda_{\min}=\numv{\NumLamMin}$，单精度方向导数经 $n_{\mathrm{cg}}=\numv{\NumNCg}$ 步 CG 循环会丢失有效位；作为记录在案的证据，单精度扫描在 $\alpha=0.05$ 处返回 $\sigma_{\max}=\numv{\NumAlphaLowArtifact}$，而 \eqref{eq:layer-code} 的解析小 $\alpha$ 极限为 $\mathrm{diag}((1-\gamma)I,I)$，即 $\sigma_{\max}\to1$。该单精度值保留在归档结果中作为伪影证据，不予使用。推导、本文编号陈述与实现函数之间的对应见表~\ref{tab:trace}。"""

for path, old, new, tag in ((os.path.join(ROOT, "paper_en", "main.tex"), EN_OLD, EN_NEW, "EN"),
                            (os.path.join(ROOT, "paper_cn", "main.tex"), CN_OLD, CN_NEW, "CN")):
    s = io.open(path, encoding="utf-8").read()
    assert s.count(old) == 1, "%s: %d matches" % (tag, s.count(old))
    io.open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("%s reproducibility paragraph rewritten" % tag)
