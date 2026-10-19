# -*- coding: utf-8 -*-
"""_patch_cgcontrol.py — report the contrast-solve convergence control in both papers.

Why this patch exists.  The layer's contrast step is a FIXED n_cg = 15 step CG solve whose
relative residual is 1.4e-3.  Measured consequences (results/ngc.json, probes _probe_fdvj*):

  * a CG-free block (D) reproduces the analytic JVP to 7.9e-13 -- the instrument is exact;
  * on a CG-path block (A) ||f(z+eps V) - f(z-eps V)|| is pinned near 0.1 for every
    eps in 1e-4..1e-2, so the difference quotient does not converge there;
  * converging the solve (n_cg = 200) restores FD-vs-JVP agreement and changes
    sigma_A from 3.253 to 1.501 and sigma_full from 14.594 to 2.793, while the CG-free
    blocks C and D are bit-identical.

So the CG-path block values are reported twice, and the fact that every certified statement
rests on C and D (exact, CG-free) is stated explicitly.

Every replacement is assert-guarded: a missing anchor aborts instead of silently skipping.
"""
import io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))


def patch(path, edits):
    p = os.path.join(ROOT, path)
    s = io.open(p, encoding="utf-8").read()
    for i, (old, new) in enumerate(edits, 1):
        assert old in s, "anchor %d NOT FOUND in %s:\n%r" % (i, path, old[:160])
        assert s.count(old) == 1, "anchor %d ambiguous (%d hits) in %s" % (i, s.count(old), path)
        s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched %s (%d edits)" % (path, len(edits)))


# --------------------------------------------------------------------------- English
EN_PARAGRAPH = r"""\emph{Contrast-solve convergence control.} The contrast step is a fixed-iteration CG solve
(\eqref{eq:contraststep}), so the layer map is only as well defined as that solve. Reports of
its residual are therefore part of the record: with the implemented
$n_{\mathrm{cg}}=\numv{\NumNCg}$ iterations it is $\numv{\NumCgResidImpl}$ relative, and it
falls to $\numv{\NumCgResidConv}$ at $n_{\mathrm{cg}}=\numv{\NumNCgConv}$. The consequence is
measurable, and we separate it with a control block rather than assert it. The field block $D$
contains no iterative solve, and there the analytic directional derivative and a central
difference agree to $\numv{\NumFdBlockD}$; on the contrast-path block $A$ at
$n_{\mathrm{cg}}=\numv{\NumNCg}$ they disagree by $\numv{\NumFdBlockAImpl}$ instead, because
$\norm{f(z+\epsilon V)-f(z-\epsilon V)}$ is pinned near
$\numv{\NumDfCgBlockImpl}$ for \emph{every} step in $\epsilon\in[10^{-4},10^{-2}]$ --- the
difference quotient does not converge, so that map admits no local linearisation at those
scales. Converging the solve restores it: at $n_{\mathrm{cg}}=\numv{\NumNCgConv}$ the same
comparison returns $\numv{\NumFdBlockAConv}$. We therefore report every contrast-path block
\emph{twice}: $\sigma_{\max}(A)$ is $\numv{\NumSigAImpl}$ as implemented and
$\numv{\NumSigA}$ at a converged solve, $\sigma_{\max}(B)$ is $\numv{\NumSigBImpl}$ and
$\numv{\NumSigB}$, and the layer norm is $\numv{\NumSigmaFullImpl}$ and
$\numv{\NumPhysSigma}$; the tables and the $\alpha$-sweep below quote the converged values,
and the implemented ones are retained as the record of the artefact. Two checks fix the
interpretation. First, the two CG-free blocks are \emph{identical} at both settings
($\sigma_{\max}(C)=\numv{\NumSigC}$ and $\sigma_{\max}(D)=\numv{\NumPhysWBlock}$), which is
what one expects of quantities that never touch the solve. Second, they are also the only
blocks that are precision-insensitive: in single precision the CG-free blocks move by
$\sim10^{-3}$ while the contrast-path blocks move by order one --- the single-precision
artefact recorded before is a symptom of the same non-smoothness, not an independent
phenomenon. Every \emph{certified} statement of this paper rests on $C$ (closed form,
Lem.~\ref{lem:blocks}) and on $D$ (Thm.~\ref{thm:field}); both are exact and CG-free, so the
certificates are unaffected by the confound, and we say so rather than leave it implicit.

"""

EN_EDITS = [
    # 1) the convergence-control paragraph
    ("\\subsection{Depth convergence}",
     EN_PARAGRAPH + "\\subsection{Depth convergence}"),

    # 2) table caption: say which solve the contrast-path rows refer to
    (r"""the measured operating point; ``measured'' entries are numerical and are not used as hypotheses of
any theorem. The block notation is that of Lem.~\ref{lem:blocks}.}""",
     r"""the measured operating point; ``measured'' entries are numerical and are not used as hypotheses of
any theorem. The block notation is that of Lem.~\ref{lem:blocks}. Contrast-path rows
($A$, $B$, layer norm) are quoted at a \emph{converged} contrast solve, with the
as-implemented $n_{\mathrm{cg}}=\numv{\NumNCg}$ values listed separately (Sec.~\ref{sec:exp},
convergence control).}"""),

    # 3) rigorous-lower-bound row: it is the CLOSED-FORM coupling bound, not a max over blocks
    (r"""Strongest rigorous lower bound $\max(\sigma_C,\sigma_A,\sigma_B)$ & (from the three bounds) & $\numv{\NumSigLowBound}$ & --- \\""",
     r"""Coupling bound $\sigma_{\max}(C)$, closed form: rigorous lower bound on $\sigma_{\max}(J)$ & $\alpha\mathcal{E}=\numv{\NumSigCLower}$ & $\numv{\NumSigLowBound}$ & --- \\
Contrast-path blocks at the implemented solve ($A$, $B$, $J$) & --- & --- & $\numv{\NumSigAImpl}$ / $\numv{\NumSigBImpl}$ / $\numv{\NumSigmaFullImpl}$ \\"""),

    # 4) add the spectral radius next to the Lipschitz constant of the composite chi-map
    (r"""Composite $\rho_\chi$ (Cor.~\ref{cor:fidelity}), constrained & hypothesis $<1$ & fails & $\numv{\NumRhoChiCons}$ \\""",
     r"""Composite $\rho_\chi$ = Lipschitz constant (Cor.~\ref{cor:fidelity}), constrained / unconstrained & hypothesis $<1$ & fails & $\numv{\NumRhoChiCons}$ / $\numv{\NumRhoChiUnc}$ \\
\quad\;\; same block, \emph{spectral radius} (not the Lipschitz constant) & --- & --- & $\numv{\NumRhoChiSpecCons}$ / $\numv{\NumRhoChiSpecUnc}$ \\"""),

    # 5) observation (2): add the CG-free control next to the other two calibrations
    (r"""closed form (measured/proved ratio $\numv{\NumSigCRatio}$, row~2) and against the exact singular
value of a random linear map (relative error $\numv{\NumInstrErr}$), and both checks are the reason
we report the block values at all.""",
     r"""closed form (measured/proved ratio $\numv{\NumSigCRatio}$, row~2) and against the exact singular
value of a random linear map (relative error $\numv{\NumInstrErr}$), and both checks are the reason
we report the block values at all. A third control, on the CG-free field block, matches a central
difference to $\numv{\NumFdBlockD}$."""),

    # 6) observation (3): name the two solves instead of quoting a single number
    (r"""(3) The remaining numerically evaluated channels give $\sigma_{\max}(A)=\numv{\NumSigA}$ and
$\sigma_{\max}(B)=\numv{\NumSigB}$, and the full-Jacobian measurement is
$\sigma_{\max}(J)=\numv{\NumPhysSigma}$. These satisfy the structural inequality the instrument
checks internally, $\sigma_{\max}(J)\ge\max_k\sigma_{\max}(\text{block}_k)$, and the strongest
rigorous lower bound over all channels is $\numv{\NumSigLowBound}$.""",
     r"""(3) The remaining numerically evaluated channels give $\sigma_{\max}(A)=\numv{\NumSigA}$ and
$\sigma_{\max}(B)=\numv{\NumSigB}$, and the full-Jacobian measurement is
$\sigma_{\max}(J)=\numv{\NumPhysSigma}$ --- all three at a converged contrast solve, against
$\numv{\NumSigAImpl}$, $\numv{\NumSigBImpl}$ and $\numv{\NumSigmaFullImpl}$ as implemented. These
satisfy the structural inequality the instrument checks internally,
$\sigma_{\max}(J)\ge\max_k\sigma_{\max}(\text{block}_k)$, and the rigorous lower bound on
$\sigma_{\max}(J)$ is the closed-form coupling value $\numv{\NumSigLowBound}$, which never
touches the iterative solve."""),
]


# --------------------------------------------------------------------------- Chinese
CN_PARAGRAPH = r"""\emph{对比度求解收敛性对照}：对比度步是固定迭代步数的 CG 求解（式~\eqref{eq:contraststep}），因此层映射的良定义性取决于该求解。我们把它的残差作为记录的一部分：按实现的 $n_{\mathrm{cg}}=\numv{\NumNCg}$ 步，相对残差为 $\numv{\NumCgResidImpl}$；取 $n_{\mathrm{cg}}=\numv{\NumNCgConv}$ 时降至 $\numv{\NumCgResidConv}$。其后果是可测的，我们用一个\emph{对照分块}把它分离出来，而不是口头断言：场步分块 $D$ 不含任何迭代求解，在其上解析方向导数与中心差分一致到 $\numv{\NumFdBlockD}$；而在经对比度步的分块 $A$ 上、$n_{\mathrm{cg}}=\numv{\NumNCg}$ 时二者相差 $\numv{\NumFdBlockAImpl}$——因为 $\norm{f(z+\epsilon V)-f(z-\epsilon V)}$ 在 $\epsilon\in[10^{-4},10^{-2}]$ 的\emph{每一步}上都停在 $\numv{\NumDfCgBlockImpl}$ 附近，差分商不收敛，故该映射在这些尺度上不存在局部线性化。把求解收敛后即可恢复：在 $n_{\mathrm{cg}}=\numv{\NumNCgConv}$ 下同一比较给出 $\numv{\NumFdBlockAConv}$。因此我们把每个经对比度步的分块\emph{报告两次}：$\sigma_{\max}(A)$ 按实现为 $\numv{\NumSigAImpl}$、在收敛解下为 $\numv{\NumSigA}$；$\sigma_{\max}(B)$ 为 $\numv{\NumSigBImpl}$ 与 $\numv{\NumSigB}$；层范数为 $\numv{\NumSigmaFullImpl}$ 与 $\numv{\NumPhysSigma}$。下文各表与 $\alpha$ 扫描一律引用收敛解的值，实现值作为伪影记录保留。两点固定其解释：其一，两个\emph{不经} CG 的分块在两种设置下完全相同（$\sigma_{\max}(C)=\numv{\NumSigC}$、$\sigma_{\max}(D)=\numv{\NumPhysWBlock}$），这正是从不触碰求解的量应有的性质；其二，它们也是仅有的\emph{对精度不敏感}的分块——单精度下它们只移动约 $10^{-3}$，而经 CG 的分块移动一个数量级，故此前记录的“单精度伪影”是同一非光滑性的症状，而非独立现象。本文所有\emph{认证性}结论都依托 $C$（闭式，引理~\ref{lem:blocks}）与 $D$（定理~\ref{thm:field}），二者精确且不经 CG，因此证书不受该混杂影响——我们明确写出这一点，而不把它留作读者自明。

"""

CN_EDITS = [
    ("\\subsection{深度收敛}",
     CN_PARAGRAPH + "\\subsection{深度收敛}"),

    (r"""分块记号沿用引理~\ref{lem:blocks}。}""",
     r"""分块记号沿用引理~\ref{lem:blocks}。经对比度步的行（$A$、$B$、层范数）均在\emph{收敛}的对比度求解下引用，并按实现 $n_{\mathrm{cg}}=\numv{\NumNCg}$ 的值另列（见第\ref{sec:exp}节收敛性对照）。}"""),

    (r"""最强严格下界 $\max(\sigma_C,\sigma_A,\sigma_B)$ & （由上述三条界） & $\numv{\NumSigLowBound}$ & --- \\""",
     r"""耦合界 $\sigma_{\max}(C)$（闭式）：$\sigma_{\max}(J)$ 的严格下界 & $\alpha\mathcal{E}=\numv{\NumSigCLower}$ & $\numv{\NumSigLowBound}$ & --- \\
按实现的求解下经对比度步的分块（$A$、$B$、$J$） & --- & --- & $\numv{\NumSigAImpl}$ / $\numv{\NumSigBImpl}$ / $\numv{\NumSigmaFullImpl}$ \\"""),

    (r"""复合 $\rho_\chi$（推论~\ref{cor:fidelity}），约束 & 假设 $<1$ & 不成立 & $\numv{\NumRhoChiCons}$ \\""",
     r"""复合 $\rho_\chi$ = Lipschitz 常数（推论~\ref{cor:fidelity}），约束 / 无约束 & 假设 $<1$ & 不成立 & $\numv{\NumRhoChiCons}$ / $\numv{\NumRhoChiUnc}$ \\
\quad\;\; 同一分块的\emph{谱半径}（非 Lipschitz 常数） & --- & --- & $\numv{\NumRhoChiSpecCons}$ / $\numv{\NumRhoChiSpecUnc}$ \\"""),

    (r"""（相对误差 $\numv{\NumInstrErr}$）；两次校验也是我们敢于报告分块数值的原因。""",
     r"""（相对误差 $\numv{\NumInstrErr}$）；两次校验也是我们敢于报告分块数值的原因。第三条对照在\emph{不经} CG 的场步分块上完成，与中心差分一致到 $\numv{\NumFdBlockD}$。"""),

    (r"""(3) 其余两个数值求值的通道给出 $\sigma_{\max}(A)=\numv{\NumSigA}$、$\sigma_{\max}(B)=\numv{\NumSigB}$，全 Jacobian 的实测值为 $\sigma_{\max}(J)=\numv{\NumPhysSigma}$。它们满足仪器内部校验的结构不等式 $\sigma_{\max}(J)\ge\max_k\sigma_{\max}(\text{第}k\text{块})$，而全部通道上的最强严格下界为 $\numv{\NumSigLowBound}$。""",
     r"""(3) 其余两个数值求值的通道给出 $\sigma_{\max}(A)=\numv{\NumSigA}$、$\sigma_{\max}(B)=\numv{\NumSigB}$，全 Jacobian 的实测值为 $\sigma_{\max}(J)=\numv{\NumPhysSigma}$——三者均为\emph{收敛}对比度求解下的值，按实现则分别为 $\numv{\NumSigAImpl}$、$\numv{\NumSigBImpl}$、$\numv{\NumSigmaFullImpl}$。它们满足仪器内部校验的结构不等式 $\sigma_{\max}(J)\ge\max_k\sigma_{\max}(\text{第}k\text{块})$，而 $\sigma_{\max}(J)$ 的严格下界是不触碰迭代求解的闭式耦合值 $\numv{\NumSigLowBound}$。"""),
]


if __name__ == "__main__":
    patch("paper_en/main.tex", EN_EDITS)
    patch("paper_cn/main.tex", CN_EDITS)
    print("OK")
