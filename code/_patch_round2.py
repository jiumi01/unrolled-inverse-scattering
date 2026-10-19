# -*- coding: utf-8 -*-
"""_patch_round2.py — fix the four blocking data/claim inconsistencies raised in round 1
reviews R2/R3.

M2.1/M3.4  Table cert paired the per-layer composite (gain_decomp, K=1, n_cg=15,
           fixed alpha=gamma=1/2) with the K=8 composite quantities rho / M_J / gap
           from L45.json.  With sigma=1.890 and M_J=76.172 the inequality
           ||J^2|| <= ||J||^2 is violated, so the two cannot describe one operator.
           NumSigComp* now points at the K=8 cyclic composite and the per-layer value
           moves to its own row.
M2.3       "$L in [NumSigMin,NumSigMax] at the learned step sizes" mis-attributed a
           fixed-step converged grid to the learned step sizes.
M2.4/M3.1  "the third arm leaves the floating-point range" is false: every ablation arm
           has finite_extrap=true, and Table abl itself prints a finite RRE for that
           arm.  The honest statement is about the STATE escaping the admissible box,
           not about divergence -- and the no-projection arms are in fact slightly
           more accurate, which is itself the interesting negative result.
M2.5/M3.3  "the unconstrained residual grows (0.6024 -> 0.4393)" quotes a DECREASE.
           The real behaviour is a bounded oscillation with a large spike; and the
           constrained arm cycles rather than contracting monotonically.
M2.2       the per-layer gain figure is measured on the as-implemented contrast path;
           it must say so instead of being read as a property of the layer.

Every replacement is assert-guarded.
"""
import io, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))


def patch(path, edits):
    p = os.path.join(ROOT, path)
    s = io.open(p, encoding="utf-8").read()
    for i, (old, new) in enumerate(edits, 1):
        assert old in s, "anchor %d NOT FOUND in %s:\n%r" % (i, path, old[:160])
        assert s.count(old) == 1, "anchor %d ambiguous (%d) in %s" % (i, s.count(old), path)
        s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched %s (%d edits)" % (path, len(edits)))


EN = [
    # M2.3 -- attribute the alpha grid correctly, and quote the K=8 composite
    (r"""$L\in[\numv{\NumSigMin},\numv{\NumSigMax}]$ at the learned step sizes
(Table~\ref{tab:cert}).""",
     r"""$L\in[\numv{\NumSigMin},\numv{\NumSigMax}]$ over the converged fixed-step grid
$\alpha=\gamma\in[0.05,1]$ --- \emph{every} point of which is non-contractive, not only those
above $\alpha_c$ (Table~\ref{tab:cert}); at the learned step sizes the $K{=}8$ cyclic composite
is $\numv{\NumSigCompCons}$."""),
    # M2.1 / M3.4 -- one operator per row
    (r"""measure them: for the composite layer $M_J=\numv{\NumMJCons}$ against
$\rho=\numv{\NumRhoComp}$ and $\sigma_{\max}=\numv{\NumSigCompCons}$, giving a non-normality gap
$\sigma_{\max}/\rho=\numv{\NumGapRatio}$ (Table~\ref{tab:cert}).""",
     r"""measure them: for the $K{=}8$ cyclic composite $M_J=\numv{\NumMJCons}$ against
$\rho=\numv{\NumRhoComp}$ and $\sigma_{\max}=\numv{\NumSigCompCons}$, giving a non-normality gap
$\sigma_{\max}/\rho=\numv{\NumGapRatio}$ (Table~\ref{tab:cert}). These three belong to one and
the same operator --- the composite Jacobian of the $K{=}8$ rollout --- and are reported
together for that reason; the per-layer value at fixed $\alpha=\gamma=\tfrac12$,
$\numv{\NumSigCompLayerCons}$, is a different operator and is listed separately."""),
    (r"""Composite $\sigma_{\max}$, constrained / unconstrained & --- & --- & $\numv{\NumSigCompCons}$ / $\numv{\NumSigCompUnc}$ \\""",
     r"""Composite $\sigma_{\max}$ ($K{=}8$ cyclic; pairs with $\rho$, $M_J$) & --- & --- & $\numv{\NumSigCompCons}$ / $\numv{\NumSigCompUnc}$ \\
\quad\;\; per-layer composite at fixed $\alpha=\gamma=\tfrac12$ (different operator) & --- & --- & $\numv{\NumSigCompLayerCons}$ / $\numv{\NumSigCompLayerUnc}$ \\"""),
    # M2.4 / M3.1 -- the ablation caption claimed a divergence that did not happen
    (r"""\caption{Ablation~I: reconstruction error after $800$ unrolled layers; ``div.'' marks the arm that
left the floating-point range.}""",
     r"""\caption{Ablation~I: reconstruction error after $800$ unrolled layers. \emph{All eight arms
remain numerically finite} over the whole horizon --- none leaves the floating-point range, so
no ``div.'' marker appears; what separates them is the state, not the error
(Table~\ref{tab:abl}).}"""),
    # M2.5 / M3.3 -- the extrapolation caption
    (r"""network contracts; the unconstrained network drifts but remains bounded by the projection; the
classical iteration converges to its own fixed point.}""",
     r"""network contracts; the unconstrained network drifts but remains bounded by the projection; the
classical iteration converges to its own fixed point. Neither unrolled network contracts
monotonically: the constrained residual cycles in a narrow band with $\infn{\chi}$ pinned at
the box, while the unconstrained residual is bounded but spikes by more than an order of
magnitude inside the cycle.}"""),
    # M2.5 / M3.3 -- the body text quoted a decrease as growth
    (r"""($800$ layers total). The constrained network contracts (residual
$\numv{\NumResConsHead}\to\numv{\NumResConsEnd}$) and its reconstruction error stays within a narrow
band; the unconstrained network's residual grows ($\numv{\NumResUncHead}\to\numv{\NumResUncEnd}$) ---
it drifts --- while the classical iteration sits at its own fixed point
(residual $\approx\numv{\NumResClsEnd}$).""",
     r"""($800$ layers total). Neither unrolled network contracts monotonically. The constrained
residual cycles inside a narrow band ($\numv{\NumResConsHead}$ at the first reported step,
$\numv{\NumResConsEnd}$ at the last) with $\infn{\chi}$ pinned at $\numv{\chimax}$, so its
reconstruction error stays bounded. The unconstrained residual is \emph{not} monotone either,
and quoting its endpoints would understate it: it spikes to $\numv{\NumResUncSpike}$ inside the
cycle --- more than an order of magnitude above the constrained band --- while $|\chi|$ is still
clamped by $\Pi$. The drift is therefore visible in the residual, not in the state; the
classical iteration sits at its own fixed point
(residual $\approx\numv{\NumResClsEnd}$)."""),
    # M2.4 -- ablation reading (c): replace "explodes" with the measured state/error split
    (r"""\emph{(c) Bounding the trunk input is the enabling mechanism.} The third arm (unconstrained,
$\Pi$ off, feature bounding off) is the only one that leaves the floating-point range. With an
unbounded input map, the correction magnitude grows with the state, the $\chi$-box cannot absorb
it, and the rollout explodes --- the behaviour originally reported for unconstrained unrolling.
Feature bounding is therefore not an implementation detail but a load-bearing structural choice,
and it is what makes Theorem~\ref{thm:bounded} non-vacuous.""",
     r"""\emph{(c) What the guard rails actually buy --- a negative result about error as a proxy.} No
arm leaves the floating-point range: all eight are numerically finite over the whole horizon.
What separates them is the \emph{state}, not the error. With $\Pi$ on, $\infn{\chi}$ is pinned
at $\numv{\chimax}$ in every arm; with $\Pi$ off it reaches
$\numv{\NumAblUncPOffFbOnChi}$--$\numv{\NumAblUncPOffFbOffChi}$, i.e.\ it leaves the admissible
box while the reconstruction error stays finite --- and is in fact slightly \emph{lower}
($\numv{\NumAblUncPOffFbOffRre}$ against $\numv{\NumAblUncPOnFbOnRre}$). A finite error is
therefore \emph{not} evidence of the certificate: the certificate is a statement about the
state, and it is the projection, not the observed accuracy, that enforces it. We report this
explicitly because the opposite reading --- ``the unconstrained rollout diverges'' --- is what
the earlier unrolling literature reports and what a reader would expect here, and it is not
what our data show. Feature bounding remains load-bearing for a different reason: it is what
makes the amplitude budget $\bar\epsilon$ attainable by construction
(Remark~\ref{rem:featscale})."""),
    # M2.2 -- say which contrast path the per-layer gain was measured on
    (r"""\caption{Measured spectral norm of each layer's Jacobian at the learned step sizes (power""",
     r"""\caption{Measured spectral norm of each layer's Jacobian at the learned step sizes, on the
\emph{as-implemented} contrast path ($n_{\mathrm{cg}}=\numv{\NumNCg}$); these are contrast-path
quantities, shown for comparability across layers, not as certificates (power"""),
]

CN = [
    # M2.3
    (r"""学习步长处 $L\in[\numv{\NumSigMin},\numv{\NumSigMax}]$（表~\ref{tab:cert}）。""",
     r"""收敛的固定步长网格 $\alpha=\gamma\in[0.05,1]$ 上 $L\in[\numv{\NumSigMin},\numv{\NumSigMax}]$——\emph{该网格上的每一点}都非压缩，而不只是 $\alpha_c$ 以上者（表~\ref{tab:cert}）；在学习所得步长下 $K=8$ 循环复合层为 $\numv{\NumSigCompCons}$。"""),
    # M2.1 / M3.4
    (r"""对复合层，$M_J=\numv{\NumMJCons}$，对应 $\rho=\numv{\NumRhoComp}$、$\sigma_{\max}=\numv{\NumSigCompCons}$，非正规间隙 $\sigma_{\max}/\rho=\numv{\NumGapRatio}$（表~\ref{tab:cert}）。""",
     r"""对 $K=8$ 循环复合层，$M_J=\numv{\NumMJCons}$，对应 $\rho=\numv{\NumRhoComp}$、$\sigma_{\max}=\numv{\NumSigCompCons}$，非正规间隙 $\sigma_{\max}/\rho=\numv{\NumGapRatio}$（表~\ref{tab:cert}）。三者属于\emph{同一个}算子——$K=8$ rollout 的复合 Jacobian——故合并报告；固定 $\alpha=\gamma=\tfrac12$ 的逐层值 $\numv{\NumSigCompLayerCons}$ 是另一个算子，另行列出。"""),
    (r"""复合 $\sigma_{\max}$，约束 / 无约束 & --- & --- & $\numv{\NumSigCompCons}$ / $\numv{\NumSigCompUnc}$ \\""",
     r"""复合 $\sigma_{\max}$（$K=8$ 循环，与 $\rho$、$M_J$ 同源） & --- & --- & $\numv{\NumSigCompCons}$ / $\numv{\NumSigCompUnc}$ \\
\quad\;\; 固定 $\alpha=\gamma=\tfrac12$ 的逐层复合（另一算子） & --- & --- & $\numv{\NumSigCompLayerCons}$ / $\numv{\NumSigCompLayerUnc}$ \\"""),
    # M2.4 / M3.1
    (r"""\caption{消融 I：$800$ 层展开后的重构误差；“发散”标记逸出浮点范围的那一组。}""",
     r"""\caption{消融 I：$800$ 层展开后的重构误差。\emph{八组在整段时程上全部数值有界}——没有一组逸出浮点范围，故图中不出现“发散”标记；区分它们的是状态，而不是误差（表~\ref{tab:abl}）。}"""),
    # M2.5 / M3.3 caption
    (r"""\caption{训练所得 $K=8$ 层的循环深度外推（额外 $100$ 步，即 $800$ 层）：(a) 不动点残差；(b) 重构误差。约束网络收缩；无约束网络漂移但在投影约束下保持有界；经典迭代收敛到自身不动点。}""",
     r"""\caption{训练所得 $K=8$ 层的循环深度外推（额外 $100$ 步，即 $800$ 层）：(a) 不动点残差；(b) 重构误差。约束网络收缩；无约束网络漂移但在投影约束下保持有界；经典迭代收敛到自身不动点。两种展开都\emph{不}单调收缩：约束臂的残差在窄带内循环、$\infn{\chi}$ 被钉在盒上；无约束臂的残差有界，但在循环内部出现超过一个数量级的尖峰。}"""),
    # M2.5 / M3.3 body
    (r"""图~\ref{fig:extrap} 把训练所得 $K=8$ 层循环外推 $100$ 步（共 $800$ 层）。约束网络收缩（残差 $\numv{\NumResConsHead}\to\numv{\NumResConsEnd}$），重构误差保持在窄带内；无约束网络的残差在增长（$\numv{\NumResUncHead}\to\numv{\NumResUncEnd}$）——即漂移——而经典迭代停留在自身不动点（残差 $\approx\numv{\NumResClsEnd}$）。""",
     r"""图~\ref{fig:extrap} 把训练所得 $K=8$ 层循环外推 $100$ 步（共 $800$ 层）。两种展开都\emph{不}单调收缩。约束臂的残差在窄带内循环（首个报告步 $\numv{\NumResConsHead}$，末步 $\numv{\NumResConsEnd}$），且 $\infn{\chi}$ 被钉在 $\numv{\chimax}$，故重构误差有界。无约束臂的残差同样非单调，只报两个端点会低估它：它在循环内部尖峰到 $\numv{\NumResUncSpike}$——比约束带高一个数量级以上——而 $|\chi|$ 仍被 $\Pi$ 钳住。故漂移体现在残差而非状态上；经典迭代停留在自身不动点（残差 $\approx\numv{\NumResClsEnd}$）。"""),
    # M2.4 ablation (c)
    (r"""\emph{(c) 主干输入有界化是使能机制。} 第三组（无约束、$\Pi$ 关、特征有界关）是唯一逸出浮点范围的一组。输入映射无界时，修正幅度随状态增长，$\chi$ 盒无法吸收，轨迹爆炸——这正是早期文献对无约束展开所报告的行为。可见特征有界化并非实现细节，而是承重的结构选择，也是定理~\ref{thm:bounded} 非空洞的前提。""",
     r"""\emph{(c) 护栏究竟买到了什么——一个关于“误差不能作代理”的负结果。} 没有任何一组逸出浮点范围：八组在整段时程上全部数值有界。区分它们的是\emph{状态}，而不是误差。$\Pi$ 开时每一组的 $\infn{\chi}$ 都被钉在 $\numv{\chimax}$；$\Pi$ 关时它达到 $\numv{\NumAblUncPOffFbOnChi}$--$\numv{\NumAblUncPOffFbOffChi}$，即\emph{逸出}可行盒，而重构误差仍然有界——并且实际略\emph{低}（$\numv{\NumAblUncPOffFbOffRre}$ 对 $\numv{\NumAblUncPOnFbOnRre}$）。因此“误差有限”\emph{不能}作为证书成立的证据：证书是关于状态的陈述，而强制它的是投影而不是观测到的精度。我们显式报告这一点，因为相反的读法（“无约束展开会发散”）既是以往展开文献所报告的，也是读者在此处会预期的，但它不是本文数据所显示的。特征有界化仍然承重，理由不同：它是幅度预算 $\bar\epsilon$ 可\emph{按构造}达到的原因（注~\ref{rem:featscale}）。"""),
    # M2.2
    (r"""\caption{在学习所得步长下实测的各层 Jacobian 谱范数（对实数化 Jacobian 做幂迭代，并在线性测试映射上与精确 SVD 对照验证）。}""",
     r"""\caption{在学习所得步长下实测的各层 Jacobian 谱范数，测于\emph{按实现}的对比度路径（$n_{\mathrm{cg}}=\numv{\NumNCg}$）；这些是对比度路径量，为跨层可比而给出，不作为证书（对实数化 Jacobian 做幂迭代，并在线性测试映射上与精确 SVD 对照验证）。}"""),
]

if __name__ == "__main__":
    patch("paper_en/main.tex", EN)
    patch("paper_cn/main.tex", CN)
    print("OK")
