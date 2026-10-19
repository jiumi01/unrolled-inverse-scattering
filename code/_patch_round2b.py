# -*- coding: utf-8 -*-
"""_patch_round2b.py — round-1 review R2/R3, second batch of text corrections.

Covers, in the reviewers' numbering:
  M2.2/m3.4  "the per-layer composite gain is dominated by the physics core" and the
             1.869 vs 2.793 discrepancy.  Both are the as-implemented (n_cg=15) contrast
             path; the converged physics core is NumPhysSigma and the per-layer composite
             gain is governed by the coupling block, not by H.
  M2.1       Table cert protocol statement (which rows are at learned step sizes) and the
             over-broad "all audit quantities are computed in double precision".
  M2.4       contribution (iv): the retrained no-projection arm does NOT leave the
             floating-point range; the honest statement is about the state.
  M2.6/Q2.4  rho_chi attribution: the hypothesis of Cor. fidelity is on F (H = 0); the
             chi->chi block understates the chi-map Lipschitz constant.  New rows quote
             both objects and both quantities.
  m2.1       Table abl2 contrast column now uses the horizon maximum (gen_numbers change).
  m2.2       Lambda row labelled as the contrast-channel part.
  m2.6       SNR definition.
  m2.7/m3.8  noiseless-evaluation protocol stated in the depth caption without the
             misleading 300 dB placeholder.
  m3.5/M3.5  seed replication is now reported (paragraph (d)) instead of being archived
             and unused.
  m3.6       fig:abl is now cited; the related-work "no convergence theory" claim is
             softened and the neighbouring literatures are positioned explicitly.
  m3.9       row 1 of Table cert flagged as one quantity, not two independent checks.
Every replacement is assert-guarded; a failure leaves the file untouched.
"""
import io, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))


def patch(path, edits):
    p = os.path.join(ROOT, path)
    s = io.open(p, encoding="utf-8").read()
    for i, (old, new) in enumerate(edits, 1):
        assert old in s, "anchor %d NOT FOUND in %s:\n%r" % (i, path, old[:170])
        assert s.count(old) == 1, "anchor %d ambiguous (%d) in %s" % (i, s.count(old), path)
        s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched %s (%d edits)" % (path, len(edits)))


EN = [
    # ---------------- abstract: soften the universal negative ----------------
    (r"""empirical gains. None of these works provides a convergence theory for the resulting learned
iteration. This paper supplies one, and in doing so identifies a structural obstruction that""",
     r"""empirical gains. To our knowledge none of these works supplies, at the level of the
individual layer, a sufficient condition for convergence, a fixed-point characterisation, or an
error bound for the learned iteration. This paper supplies them, and in doing so identifies a
structural obstruction that"""),

    # ---------------- related work: per-work gap + neighbouring literatures ----------------
    (r"""constraints. All of these works validate performance empirically. The word ``contraction'' in""",
     r"""constraints. All of these works validate performance empirically; to our knowledge none of them
supplies a layer-level sufficient condition, a fixed-point characterisation or an error bound for
the trained iteration --- and the omission matters, because the property that would have to be
assumed is exactly the one the learned modules are free to break. The word ``contraction'' in"""),
    (r"""Sec.~\ref{sec:theory} shows that this is not a technical inconvenience but an
impossibility.""",
     r"""Sec.~\ref{sec:theory} shows that this is not a technical inconvenience but an
impossibility. Two neighbouring literatures are deliberately \emph{not} invoked, because their
hypotheses do not transfer: convergence analyses of non-convex unrolled proximal/ADMM schemes
assume a convex data term and read the layer as a proximal step, and plug-and-play /
regularization-by-denoising analyses assume the proximal operator of an explicit regularizer.
Neither reading is available for the Born/contrast-source composite, which is not the gradient of
any functional the scheme minimizes."""),

    # ---------------- contribution (iv): the honest ablation reading ----------------
    (r"""survives}. The amplitude budget buys the a-priori constants at a measurable accuracy cost of
$\numv{\NumPriceDelta}$; the spectral clamp alone produces no observed stability gain, because the
composite gain is dominated by the physics core whose step sizes are learned. A third ablation shows
that bounding the correction trunk's input is what makes the correction magnitude controllable in
the first place.""",
     r"""survives}. The amplitude budget buys the a-priori constants at a measurable accuracy cost of
$\numv{\NumPriceDelta}$; the spectral clamp alone produces no observed stability gain, because the
per-layer composite gain is governed by the coupling block $\theta\alpha\,\mathrm{diag}(E)$ rather
than by the correction, and the physics core is non-contractive on its own
($\numv{\NumPhysSigma}>1$ at the converged contrast solve). A third ablation shows that bounding the
correction trunk's input is what keeps the \emph{state} inside the admissible box: without it the
contrast leaves the box ($\infn{\chi}=\numv{\NumAblUncPOffFbOffChi}$) while the reconstruction error
stays finite --- so a finite error is not evidence that the certificate holds."""),

    # ---------------- precision claim qualification ----------------
    (r"""as a certificate. All audit quantities are computed in double precision, because the contrast""",
     r"""as a certificate. The quantities reported in Table~\ref{tab:cert} --- the blocks $A$--$D$, the
full layer Jacobian and the $\alpha$-sweep --- are computed in double precision, because the contrast"""),

    # ---------------- Table cert: protocol + row-1 note ----------------
    (r"""as-implemented $n_{\mathrm{cg}}=\numv{\NumNCg}$ values listed separately (Sec.~\ref{sec:exp},
convergence control).}""",
     r"""as-implemented $n_{\mathrm{cg}}=\numv{\NumNCg}$ values listed separately (Sec.~\ref{sec:exp},
convergence control). The composite rows (composite $\rho_\chi$, $\rho$, $M_J$, the gap, composite
$\sigma_{\max}$) are evaluated at the \emph{trained per-layer} step sizes, not at
$\alpha=\gamma=1/2$, and come from the double-precision composite audit; the per-layer composite row
is measured on the as-implemented contrast path and is a different operator. Row~1 is a single
quantity shown in two columns (the design value and the certificate are the same expression), not
two independent checks.}"""),

    # ---------------- m3.9: row 1 is one quantity ----------------
    (r"""$D$: field block $\sigma_{\max}$ (Thm.~\ref{thm:field}) & $1-\alpha(1-\beta_0)=\numv{\NumGammaW}$ & $\numv{\NumThmFieldD}$ & $\numv{\NumPhysWBlock}$ \\""",
     r"""$D$: field block $\sigma_{\max}$ (Thm.~\ref{thm:field}) & $1-\alpha(1-\beta_0)=\numv{\NumGammaW}$ & same & $\numv{\NumPhysWBlock}$ \\"""),

    # ---------------- m2.2: Lambda row provenance ----------------
    (r"""Noise gain $\Lambda$: bound \eqref{eq:Lambda} / measured & $\le\numv{\NumLambdaBound}$ & --- & $\numv{\NumLambdaMeas}$ \\""",
     r"""Noise gain $\Lambda$ (data$\to\chi$ channel, converged solve): bound \eqref{eq:Lambda} / measured & $\le\numv{\NumLambdaBound}$ & --- & $\numv{\NumLambdaMeas}$ \\"""),

    # ---------------- M2.6: rho_chi row relabel + F/row rows ----------------
    (r"""Composite $\rho_\chi$ = Lipschitz constant (Cor.~\ref{cor:fidelity}), constrained / unconstrained & hypothesis $<1$ & fails & $\numv{\NumRhoChiCons}$ / $\numv{\NumRhoChiUnc}$ \\""",
     r"""Composite $\rho_\chi$ as measured = $\sigma_{\max}$ of the $\chi\!\to\!\chi$ block, a \emph{lower} bound on the $\chi$-map Lipschitz constant (Cor.~\ref{cor:fidelity}) & hypothesis $<1$ & fails & $\numv{\NumRhoChiCons}$ / $\numv{\NumRhoChiUnc}$ \\
\quad\;\; $\chi$-map Lipschitz constant of $\hat G$: full row block $[A\,B]$ & --- & --- & $\numv{\NumChiLipRowCons}$ / $\numv{\NumChiLipRowUnc}$ \\
\quad\;\; physical map $F$ ($H\equiv0$): row block $[A\,B]$ / $\chi\!\to\!\chi$ block & --- & --- & $\numv{\NumChiLipRowFCons}$ / $\numv{\NumChiFBlockCons}$ \\
\quad\;\; physical map $F$: spectral radius of the $\chi\!\to\!\chi$ block & --- & --- & $\numv{\NumChiRhoFCons}$ / $\numv{\NumChiRhoFUnc}$ \\"""),

    # ---------------- M2.6: attribute the hypothesis to F in the text ----------------
    (r"""therefore unjustified. We report the measured $\rho_\chi$ instead
($\numv{\NumRhoChiCons}$ constrained, $\numv{\NumRhoChiUnc}$ unconstrained,
Table~\ref{tab:cert}); it exceeds one, so in this regime""",
     r"""therefore unjustified. Note which object the hypothesis is on: it is stated for the
\emph{physical} map $F$ ($H\equiv0$), whereas the quantity we can measure on the trained network is
that of $\hat G$. We therefore report both, and for both we report two different numbers --- the
$\sigma_{\max}$ of the $\chi\!\to\!\chi$ block and the norm of the whole $\chi$-row $[A\,B]$, the
latter being the Lipschitz constant of the $\chi$-map and the former only a lower bound on it
(Table~\ref{tab:cert}). On $\hat G$ the two are $\numv{\NumRhoChiCons}$ versus
$\numv{\NumChiLipRowCons}$; on $F$ they are $\numv{\NumChiFBlockCons}$ versus
$\numv{\NumChiLipRowFCons}$. Every one of these exceeds one, so the attribution does not depend on
the object or on the norm used: the measured $\chi$-map is expansive in this regime, and in this
regime"""),

    # ---------------- m3.4: the 1.869 vs 2.793 discrepancy ----------------
    (r"""$\numv{\NumSigAImpl}$, $\numv{\NumSigBImpl}$ and $\numv{\NumSigmaFullImpl}$ as implemented. These
satisfy the structural inequality the instrument checks internally,""",
     r"""$\numv{\NumSigAImpl}$, $\numv{\NumSigBImpl}$ and $\numv{\NumSigmaFullImpl}$ as implemented. A
separately archived per-layer record on the same operating point reports
$\numv{\NumGdPhysSigma}$ for the physics-core full Jacobian; the difference from
$\numv{\NumPhysSigma}$ is the same contrast-solve artefact (that record uses the implemented
$n_{\mathrm{cg}}$), not a different operator. These
satisfy the structural inequality the instrument checks internally,"""),

    # ---------------- m2.7 / m3.8: noiseless caption ----------------
    (r"""(The evaluation set is generated at $\SI{300}{\decibel}$ so that this and the
extrapolation study measure reconstruction error and not measurement noise; measurement noise is
injected separately in Fig.~\ref{fig:noise}.)""",
     r"""(The evaluation set is \emph{effectively noiseless} --- measurement noise is switched off at
synthesis --- so that this and the extrapolation study measure reconstruction error and not
measurement noise; measurement noise is injected separately in Fig.~\ref{fig:noise}.)"""),

    # ---------------- m2.6: define the SNR ----------------
    (r"""Gaussian noise at $\SI{20}{\decibel}$ SNR; the evaluation set ($100$ unseen samples) is""",
     r"""Gaussian noise at $\SI{20}{\decibel}$ SNR, defined as
$10\log_{10}(\mean|y^{0}|^{2}/\mean|n|^{2})$ with $y^{0}$ the noiseless scattered field, i.e.\ a
ratio averaged over all receiver and incidence channels; the evaluation set ($100$ unseen samples) is"""),

    # ---------------- m3.6: cite fig:abl ----------------
    (r"""\emph{(b) Retraining can hide the problem, but only at the cost of the certificate.} Ablation~I
shows that a network \emph{trained} without $\Pi$ re-learns an operating point that happens to""",
     r"""\emph{(b) Retraining can hide the problem, but only at the cost of the certificate.} Ablation~I
(Table~\ref{tab:abl}, Fig.~\ref{fig:abl}) shows that a network \emph{trained} without $\Pi$
re-learns an operating point that happens to"""),

    # ---------------- M3.5 / m2.5: seed replication paragraph ----------------
    (r"""makes the amplitude budget $\bar\epsilon$ attainable by construction
(Remark~\ref{rem:featscale}).

The practical recommendation is consequently stronger than ``project if convenient'':""",
     r"""makes the amplitude budget $\bar\epsilon$ attainable by construction
(Remark~\ref{rem:featscale}).

\emph{(d) The boundedness reading is not a single-run artefact.} The arms above are reported for one
training seed, so we replicated the contrast they rest on across
$\numv{\NumSeedN}$ independent training runs ($\numv{\NumSeedNCons}$ constrained,
$\numv{\NumSeedNUnc}$ unconstrained, $\numv{\NumSeedNProjOff}$ unconstrained with the projection
removed). Every run stays numerically finite ($\numv{\NumSeedFinite}$). The constrained arms sit at
$\infn{\chi}=\numv{\NumSeedChiConsMax}$ --- pinned at the box --- in \emph{every} seed, while the
arms trained without the projection leave it in every seed
($\infn{\chi}=\numv{\NumSeedProjOffBox}$); the separation is therefore a property of the design
flag, not of one initialization. The depth conclusion is equally stable: the seed-to-seed standard
deviation of the $K{=}8$ error is $\numv{\NumSeedRreSdCons}$ (constrained) and
$\numv{\NumSeedRreSdUnc}$ (unconstrained), well below the differences discussed in the text.

The practical recommendation is consequently stronger than ``project if convenient'':"""),
]

CN = [
    # abstract
    (r"""然而，这些工作均未给出所得学习迭代的收敛性理论。""",
     r"""然而据我们所知，这些工作均未在\emph{层级别}上给出收敛的充分条件、不动点刻画或误差界。"""),
    # related work
    (r"""上述工作均以经验性能验证为主。""",
     r"""上述工作均以经验性能验证为主，据我们所知未给出层级别上的收敛充分条件、不动点刻画或误差界——这一缺失是实质性的，因为需要被假设成立的性质，恰恰是可学习模块可以自由破坏的那一个。"""),
    (r"""第\ref{sec:theory}节将表明这不是技术性不便，而是不可能性。""",
     r"""第\ref{sec:theory}节将表明这不是技术性不便，而是不可能性。另有两类邻近文献我们\emph{刻意}不援引，因为其前提不可迁移：非凸展开近端/ADMM 格式的收敛分析要求凸数据项并把层读作近端步；即插即用（PnP）/去噪正则化分析要求某个显式正则项的近端算子。对 Born/对比源复合算子二者皆不可用——它不是该格式所极小化任何泛函的梯度。"""),
    # contribution (iv)
    (r"""幅度预算买到的是先验常数，精度代价为 $\numv{\NumPriceDelta}$；单独的谱钳位未产生可观测稳定性增益，因为复合增益由物理核心主导，而其步长本身是被学习的。第三组消融表明：使修正幅度可控的前提是对主干输入做有界化。""",
     r"""幅度预算买到的是先验常数，精度代价为 $\numv{\NumPriceDelta}$；单独的谱钳位未产生可观测稳定性增益，因为逐层复合增益由耦合块 $\theta\alpha\,\mathrm{diag}(E)$ 主导而非由修正主导，且物理核自身非压缩（收敛解下 $\numv{\NumPhysSigma}>1$）。第三组消融表明：对主干输入做有界化所保住的是\emph{状态}留在可行盒内——去掉它，对比度逸出盒外（$\infn{\chi}=\numv{\NumAblUncPOffFbOffChi}$）而重构误差仍有限，故“误差有限”不构成证书成立的证据。"""),
    # precision claim
    (r"""全部审计量都在双精度下计算，因为对比度步求解的 Tikhonov 系统最小特征值为""",
     r"""表~\ref{tab:cert} 所报的量——分块 $A$--$D$、全层 Jacobian 与 $\alpha$ 扫描——都在双精度下计算，因为对比度步求解的 Tikhonov 系统最小特征值为"""),
    # Table cert caption
    (r"""并按实现 $n_{\mathrm{cg}}=\numv{\NumNCg}$ 的值另列（见第\ref{sec:exp}节收敛性对照）。}""",
     r"""并按实现 $n_{\mathrm{cg}}=\numv{\NumNCg}$ 的值另列（见第\ref{sec:exp}节收敛性对照）。复合行（复合 $\rho_\chi$、$\rho$、$M_J$、间隙、复合 $\sigma_{\max}$）取\emph{训练所得逐层步长}，而非 $\alpha=\gamma=1/2$，来自双精度复合审计；逐层复合行则测于按实现的对比度路径，属于另一个算子。第 1 行是同一个量在两列中的显示（设计值与证书是同一表达式），并非两次独立校验。}"""),
    # row 1
    (r"""$D$：场块 $\sigma_{\max}$（定理~\ref{thm:field}） & $1-\alpha(1-\beta_0)=\numv{\NumGammaW}$ & $\numv{\NumThmFieldD}$ & $\numv{\NumPhysWBlock}$ \\""",
     r"""$D$：场块 $\sigma_{\max}$（定理~\ref{thm:field}） & $1-\alpha(1-\beta_0)=\numv{\NumGammaW}$ & 同左 & $\numv{\NumPhysWBlock}$ \\"""),
    # Lambda row
    (r"""噪声增益 $\Lambda$：界 \eqref{eq:Lambda} / 实测 & $\le\numv{\NumLambdaBound}$ & --- & $\numv{\NumLambdaMeas}$ \\""",
     r"""噪声增益 $\Lambda$（数据$\to\chi$ 通道，收敛解）：界 \eqref{eq:Lambda} / 实测 & $\le\numv{\NumLambdaBound}$ & --- & $\numv{\NumLambdaMeas}$ \\"""),
    # rho_chi rows
    (r"""复合 $\rho_\chi$ = Lipschitz 常数（推论~\ref{cor:fidelity}） & 假设 $<1$ & 不成立 & $\numv{\NumRhoChiCons}$ / $\numv{\NumRhoChiUnc}$ \\""",
     r"""复合 $\rho_\chi$ 实测量 = $\chi\!\to\!\chi$ 分块的 $\sigma_{\max}$，即 $\chi$ 映射 Lipschitz 常数的\emph{下}界（推论~\ref{cor:fidelity}） & 假设 $<1$ & 不成立 & $\numv{\NumRhoChiCons}$ / $\numv{\NumRhoChiUnc}$ \\
\quad\;\; $\hat G$ 的 $\chi$ 映射 Lipschitz 常数：整行分块 $[A\,B]$ & --- & --- & $\numv{\NumChiLipRowCons}$ / $\numv{\NumChiLipRowUnc}$ \\
\quad\;\; 物理映射 $F$（$H\equiv0$）：整行 $[A\,B]$ / $\chi\!\to\!\chi$ 分块 & --- & --- & $\numv{\NumChiLipRowFCons}$ / $\numv{\NumChiFBlockCons}$ \\
\quad\;\; 物理映射 $F$：$\chi\!\to\!\chi$ 分块的谱半径 & --- & --- & $\numv{\NumChiRhoFCons}$ / $\numv{\NumChiRhoFUnc}$ \\"""),
    # text attribution
    (r"""我们改为报告实测的 $\rho_\chi$（约束 $\numv{\NumRhoChiCons}$、无约束 $\numv{\NumRhoChiUnc}$，表~\ref{tab:cert}）；它大于 1，故该区域内推论~\ref{cor:fidelity} \emph{不}提供任何有限先验半径""",
     r"""需注意该假设作用的对象：它是对\emph{物理}映射 $F$（$H\equiv0$）陈述的，而我们在训练网络上所能测的是 $\hat G$。因此我们把两者都报告，且对每一者都报两个不同的数——$\chi\!\to\!\chi$ 分块的 $\sigma_{\max}$，与整条 $\chi$ 行 $[A\,B]$ 的范数；后者才是 $\chi$ 映射的 Lipschitz 常数，前者只是它的下界（表~\ref{tab:cert}）。在 $\hat G$ 上二者为 $\numv{\NumRhoChiCons}$ 与 $\numv{\NumChiLipRowCons}$，在 $F$ 上为 $\numv{\NumChiFBlockCons}$ 与 $\numv{\NumChiLipRowFCons}$。它们全部大于 1，故该结论既不取决于所测对象、也不取决于所用范数：本区域内实测的 $\chi$ 映射是扩张的。我们因此不报告任何有限先验半径"""),
    # m3.4
    (r"""三者均为\emph{收敛}对比度求解下的值，按实现则分别为 $\numv{\NumSigAImpl}$、$\numv{\NumSigBImpl}$、$\numv{\NumSigmaFullImpl}$。""",
     r"""三者均为\emph{收敛}对比度求解下的值，按实现则分别为 $\numv{\NumSigAImpl}$、$\numv{\NumSigBImpl}$、$\numv{\NumSigmaFullImpl}$。另一份独立归档的逐层记录在同一工作点上给出物理核全 Jacobian 为 $\numv{\NumGdPhysSigma}$；它与 $\numv{\NumPhysSigma}$ 的差别来自同一对比度求解伪影（该记录用按实现的 $n_{\mathrm{cg}}$），而非不同的算子。"""),
    # noiseless caption
    (r"""此处评估集按 $\SI{300}{\decibel}$ 生成，以便该研究与后续外推研究度量的是重构误差而非测量噪声；""",
     r"""此处评估集\emph{实际无噪}（合成时即关闭测量噪声），以便该研究与后续外推研究度量的是重构误差而非测量噪声；"""),
    # SNR definition
    (r"""\emph{噪声口径}：训练样本加入复高斯噪声（$\SI{20}{\decibel}$ SNR）；""",
     r"""\emph{噪声口径}：训练样本加入复高斯噪声（$\SI{20}{\decibel}$ SNR，定义为 $10\log_{10}(\mean|y^{0}|^{2}/\mean|n|^{2})$，其中 $y^{0}$ 为无噪散射场，即对所有接收与入射通道取平均的信噪比）；"""),
    # fig:abl cite
    (r"""\emph{(b) 重训会掩盖问题，但代价是证书。} 消融 I 表明，""",
     r"""\emph{(b) 重训会掩盖问题，但代价是证书。} 消融 I（表~\ref{tab:abl}、图~\ref{fig:abl}）表明，"""),
    # seeds paragraph
    (r"""因此实用建议比“方便就投影”更强：投影与有界特征映射是把展开式逆散射从经验稳定的启发式转变为具备闭式不变集的迭代的两个要素；幅度预算则提供推论~\ref{cor:fidelity} 的保真半径。""",
     r"""\emph{(d) 有界性结论不是单次运行的巧合。} 上述消融臂只报一个训练种子，故我们对其所依赖的对照做了多种子重复：共 $\numv{\NumSeedN}$ 次独立训练（约束 $\numv{\NumSeedNCons}$ 次、无约束 $\numv{\NumSeedNUnc}$ 次、无约束且去掉投影 $\numv{\NumSeedNProjOff}$ 次），全部保持数值有界（$\numv{\NumSeedFinite}$）。约束臂\emph{每一次}都停在 $\infn{\chi}=\numv{\NumSeedChiConsMax}$（钉在盒上），而去掉投影训练的臂每一次都越盒（$\infn{\chi}=\numv{\NumSeedProjOffBox}$）——该分离因此是设计开关的性质，而非某一次初始化的巧合。深度结论同样稳定：$K=8$ 误差的种子间标准差为 $\numv{\NumSeedRreSdCons}$（约束）与 $\numv{\NumSeedRreSdUnc}$（无约束），远小于正文所讨论的差异。

因此实用建议比“方便就投影”更强：投影与有界特征映射是把展开式逆散射从经验稳定的启发式转变为具备闭式不变集的迭代的两个要素；幅度预算则提供推论~\ref{cor:fidelity} 的保真半径。"""),
]

if __name__ == "__main__":
    patch("paper_en/main.tex", EN)
    patch("paper_cn/main.tex", CN)
    print("OK")
