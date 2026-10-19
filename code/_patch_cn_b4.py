# -*- coding: utf-8 -*-
"""Mirror the remaining R1-mandated revisions into paper_cn/main.tex.

Batch 4: abstract (iii) duplicate fix; Setup (sweep driver + reproducibility);
Tab. cert (proved/measured); Table narrative; Sec. noise (fig:alphac + rewrite);
Sec. price paragraph; Discussion (i) + (viii); Conclusion; Tab. trace.
Every replacement is assert-guarded on count==1.
"""
import io

P = r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\paper_cn\main.tex"
s = io.open(P, encoding="utf-8").read()
applied = []


def rep(tag, old, new):
    global s
    n = s.count(old)
    assert n == 1, "[%s] anchor count=%d (expected 1)" % (tag, n)
    s = s.replace(old, new)
    applied.append(tag)


# ---------------------------------------------------------------- 1. abstract (iii) dup
rep("abs_iii",
    r"""(iii) \emph{保真与稳定性}：无限深极限落在物理不动点周围半径 $\bar\epsilon/(1-\beta_0)$ 的先验球内，噪声增益以 $\Lambda/(1-\rho)$ 为界（$\rho$ 为实测谱半径）。(iii) \emph{保真与稳定性，并如实划界}：""",
    r"""(iii) \emph{保真与稳定性，并如实划界}：""")

# ---------------------------------------------------------------- 2. Setup
rep("setup_repro",
    r"""并在表~\ref{tab:cls} 中报告其\emph{最优}配置——因此下文比较的对象是调参后的经典基线，而非任取的一组设置。""",
    r"""并在表~\ref{tab:cls} 中报告其\emph{最优}配置——因此下文比较的对象是调参后的经典基线，而非任取的一组设置。该扫描由独立驱动脚本（\texttt{run\_classical\_sweep.py}）完成，与深度曲线分离，因为经典分支本身按设计不做调参；两者写入同一结果目录，表~\ref{tab:cls} 与本文全部数值宏均由该目录生成，故本文没有任何数字是手工转录的。
\emph{可复现性}：表~\ref{tab:cert} 中每个量均由同一个审计驱动脚本产生，它以有限差分幂迭代配合解析反向模式伴随计算引理~\ref{lem:blocks} 的各分块 Jacobian。仪器做两次自校验，且两次结果均在文中报告：(i) 在随机线性测试映射上把估计值与 \texttt{torch.linalg.matrix\_norm} 给出的精确奇异值比较（相对误差 $\numv{\NumInstrErr}$），这验证了幂迭代在已知答案问题上的正确性；(ii) 在层本身把实测耦合块与引理~\ref{lem:blocks} 的闭式 $\sigma_{\max}(C)=\theta\alpha\mathcal{E}$ 比较（实测/已证之比 $\numv{\NumSigCRatio}$），这验证了伴随与状态排序与理论所预测的结构一致。需强调 $\sigma_{\max}(A)$ \emph{没有}闭式——对比度步经场链 $\chi\to w^{+}\to E$ 依赖 $\chi$——因此 $A$ 作为实测量报告，绝不作为证书使用。推导、本文编号陈述与实现函数之间的对应见表~\ref{tab:trace}。""")

# ---------------------------------------------------------------- 3. Tab. cert
OLD_TAB = r"""\begin{table}[t]
\centering
\caption{层映射的谱分解。所有变体固定 $\alpha=\gamma=0.5$，从而使差异仅归因于修正项（$K=8$，$16$ 个测试状态）。}
\label{tab:cert}
\begin{tabular}{lcc}
\toprule
量 & 认证/界 & 实测 \\
\midrule
场块 $w\!\to\!w^{+}$（定理~\ref{thm:field}） & $\le\numv{\NumGammaW}$ & $\numv{\NumPhysWBlock}$ \\
对比度块 $\chi\!\to\!\chi^{+}$ & --- & $\numv{\NumPhysChiBlock}$ \\
物理核心 $\sigma_{\max}$（无修正） & --- & $\numv{\NumPhysSigma}$ \\
修正 $\operatorname{Lip}(H_k)$，约束 & $\le\numv{\epsbar}$ & $\numv{\NumLipHCons}$ \\
修正 $\operatorname{Lip}(H_k)$，无约束 & --- & $\numv{\NumLipHUnc}$ \\
复合 $\sigma_{\max}$，约束 & --- & $\numv{\NumSigCompCons}$ \\
复合 $\sigma_{\max}$，无约束 & --- & $\numv{\NumSigCompUnc}$ \\
先验漂移半径 $\bar\epsilon/(1-\beta_0)$ & $\numv{\NumFidRadius}$ & 漂移 $\numv{\NumRreConsExt}$ \\
\bottomrule
\end{tabular}
\end{table}
表~\ref{tab:cert} 给出分解结果。三点观察：(1) 场块以 $\numv{\NumPhysWBlock}$ 收缩，低于先验界 $1-\alpha(1-\beta_0)=\numv{\NumGammaW}$——定理~\ref{thm:field} 的证书有效但偏保守。(2) 约束网络的修正 Lipschitz 常数为 $\numv{\NumLipHCons}$，不高于设计值 $\bar\epsilon=\numv{\epsbar}$；无约束为 $\numv{\NumLipHUnc}$，高出一个数量级以上：幅度--斜率参数化完全按设计工作。(3) 然而两种变体的\emph{复合}增益都超过 1（$\numv{\NumSigCompCons}$ 与 $\numv{\NumSigCompUnc}$），确认了命题~\ref{prop:imposs}。双线性耦合 $\chi\odot E$ 使忠实复合算子无论修正控制得多好都非压缩——这正说明我们能提供的保证为何是有界性、保真半径与噪声增益，而非全局压缩率。"""

NEW_TAB = r"""\begin{table*}[t]
\centering
\caption{层映射的谱分解。除注明外所有变体固定 $\alpha=\gamma=1/2$，从而使差异仅归因于修正项（$K=8$，$16$ 个测试状态）。“已证”列是在实测工作点上取值的严格不等式；“实测”列为数值量，不作为任何定理的前提。分块记号沿用引理~\ref{lem:blocks}。}
\label{tab:cert}
\begin{tabular}{llcc}
\toprule
分块 / 量 & 取值 & 已证 & 实测 \\
\midrule
$D$：场块 $\sigma_{\max}$（定理~\ref{thm:field}） & $1-\alpha(1-\beta_0)=\numv{\NumGammaW}$ & $\numv{\NumThmFieldD}$ & $\numv{\NumPhysWBlock}$ \\
$C$：耦合块，闭式（引理~\ref{lem:blocks}） & $\alpha\mathcal{E}=\numv{\NumSigCLower}$ & $\numv{\NumSigCLower}$ & $\numv{\NumSigC}$ \\
$A$：对比度块，$\sigma_{\max}$ & --- & --- & $\numv{\NumSigA}$ \\
$B$：修正侧耦合，$\sigma_{\max}$ & --- & --- & $\numv{\NumSigB}$ \\
最强严格下界 $\max(\sigma_C,\sigma_A,\sigma_B)$ & （由上述三条界） & $\numv{\NumSigLowBound}$ & --- \\
峰值场列范数 $\mathcal{E}$ / $\ell_\infty$ 之比 & $\numv{\NumEColMax}$ / $\numv{\NumEInfRatio}$ & --- & 是 \\
非压缩阈值 $\alpha_c=(\mathcal{E})^{-1}$ & $\numv{\NumAlphaC}$ & --- & 是 \\
物理核心 $\sigma_{\max}(J)$（无修正） & --- & --- & $\numv{\NumPhysSigma}$ \\
Born 对比度变体（强制 $B\equiv0$）$\sigma_{\max}(J)$ & --- & --- & $\numv{\NumBornSigmaFull}$ \\
$\operatorname{Lip}(H_k)$ 约束 / 无约束 & 设计值 $\le\numv{\epsbar}$ & --- & $\numv{\NumLipHCons}$ / $\numv{\NumLipHUnc}$ \\
复合 $\sigma_{\max}$，约束 / 无约束 & --- & --- & $\numv{\NumSigCompCons}$ / $\numv{\NumSigCompUnc}$ \\
复合 $\rho_\chi$（推论~\ref{cor:fidelity}），约束 & 假设 $<1$ & 不成立 & $\numv{\NumRhoChiCons}$ \\
复合 $\rho$ 与瞬态 $M_J$（推论~\ref{cor:noisenormal}） & --- & --- & $\numv{\NumRhoComp}$ / $\numv{\NumMJCons}$ \\
非正规间隙 $\sigma_{\max}/\rho$ & --- & --- & $\numv{\NumGapRatio}$ \\
噪声增益 $\Lambda$：界 \eqref{eq:Lambda} / 实测 & $\ge\numv{\NumLambdaBound}$ & --- & $\numv{\NumLambdaMeas}$ \\
\bottomrule
\end{tabular}
\end{table*}
表~\ref{tab:cert} 给出分解结果，并把“已证”与“实测”两列严格分开。四点观察。(1) 场块以 $\numv{\NumPhysWBlock}$ 收缩，低于先验界 $1-\alpha(1-\beta_0)=\numv{\NumGammaW}$：定理~\ref{thm:field} 的证书有效但偏保守。(2) 障碍就在耦合块上，且它有闭式：$\sigma_{\max}(C)=\alpha\mathcal{E}=\numv{\NumSigCLower}$，其中 $\mathcal{E}=\numv{\NumEColMax}$，阈值 $\alpha_c=\numv{\NumAlphaC}<\tfrac12$——因此在本表所用的固定步长下，非压缩性是\emph{已证}的，而不只是被观测到的。这里也正是两种范数不可混淆之处：$\ell_\infty$ 给出 $\max_r|E(r)|$，它把 $\mathcal{E}$ 低估了一个因子 $\numv{\NumEInfRatio}$；若审计脚本引用 $\alpha\max_r|E|$，此处会得到小于 1 的值并\emph{错误地}得出该界为空。我们的仪器正是针对该闭式校准（实测/已证之比 $\numv{\NumSigCRatio}$，第 2 行），并针对随机线性映射的精确奇异值校准（相对误差 $\numv{\NumInstrErr}$）；两次校验也是我们敢于报告分块数值的原因。(3) 两个数值求值的通道还要大得多（$\sigma_{\max}(A)=\numv{\NumSigA}$，$\sigma_{\max}(B)=\numv{\NumSigB}$），给出最强严格下界 $\numv{\NumSigLowBound}$；这些是严格的\emph{不等式}，其右端由已校准的仪器算出。(4) 两种变体的复合增益都超过 1（$\numv{\NumSigCompCons}$ 与 $\numv{\NumSigCompUnc}$），而修正被控制得很好（$\operatorname{Lip}(H_k)=\numv{\NumLipHCons}$，不超过设计值 $\bar\epsilon=\numv{\epsbar}$，无约束为 $\numv{\NumLipHUnc}$，高出一个数量级以上），因此非压缩性\emph{不能}归因于修正失控。双线性耦合 $\chi\odot E$ 使忠实复合算子无论修正控制得多好都非压缩——这正是我们能提供的保证为有界性与逐层核算、而非全局压缩率的原因。表末五行给出各几何界所依赖的量：实测 $\rho_\chi>1$（故此处漂移半径不可得）、实测 $\rho$ 与瞬态 $M_J$ 及间隙 $\sigma_{\max}/\rho$、以及噪声增益 $\Lambda$ 与构造性界 \eqref{eq:Lambda} 的对比。"""

rep("tab_cert", OLD_TAB, NEW_TAB)

# ---------------------------------------------------------------- 4. Sec. noise
rep("noise_alphac",
    r"""\subsection{噪声稳定性}\label{sec:noise}
\begin{figure*}[t]
\centering
\includegraphics[width=0.86\textwidth]{../figures/fig_noise.pdf}""",
    r"""\subsection{噪声稳定性}\label{sec:noise}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{../figures/fig_alphac.pdf}
\caption{非压缩阈值。物理核心层 Jacobian 的实测 $\sigma_{\max}$ 随步长 $\alpha=\gamma$ 的变化（标记点），对照命题~\ref{prop:imposs} 的严格下界 $\alpha\mathcal{E}$（虚线）与阈值 $\alpha_c=(\mathcal{E})^{-1}$（点线），其中 $\mathcal{E}=\max_r\norm{E_{\cdot,r}}_2$ 为跨入射的逐单元峰值场能量。$\alpha_c$ 左侧该界不具结论性；实测值全程高于 1，但只有 $\alpha_c$ 右侧的区域是\emph{已证}的。}
\label{fig:alphac}
\end{figure}
\begin{figure*}[t]
\centering
\includegraphics[width=0.86\textwidth]{../figures/fig_noise.pdf}""")

rep("noise_para",
    r"""图~\ref{fig:noise} 在 $\SI{30}{\decibel}$ 至 $\SI{5}{\decibel}$ 范围内评估两种网络。二者都单调退化，且 $\SI{20}{\decibel}$ 处敏感度分别为 $\numv{\NumSensCons}$（约束）与 $\numv{\NumSensUnc}$（无约束），十分接近：在本实验中幅度预算\emph{并未}带来可见的噪声优势。它带来的是定理~\ref{thm:noise} 的先验因子 $\Lambda/(1-\rho)$，其中 $\Lambda$ 由构造有界，即一条对未见数据与架构均成立的保证，而非在某一测试集上观测到的改进。我们将其作为否定性经验结果如实报告：约束族的价值在于可认证性与有界性，而非精度或实测鲁棒性。""",
    r"""图~\ref{fig:noise} 在 $\SI{30}{\decibel}$ 至 $\SI{5}{\decibel}$ 范围内评估两种网络。二者都单调退化，且 $\SI{20}{\decibel}$ 处敏感度分别为 $\numv{\NumSensCons}$（约束）与 $\numv{\NumSensUnc}$（无约束），十分接近：在本实验中幅度预算\emph{并未}带来可见的噪声优势。它带来的并不是形如 $\Lambda/(1-\rho)$ 的界——如上所述，该界是条件的，且其前提在此处不成立——而是进入精确核算 \eqref{eq:noiseexact} 的\emph{量} $\Lambda$ 与 $M_J$：它们可对任意网格重算，并可在训练过程中被跟踪。我们把这一点作为明确的否定性经验结果报告：约束族的价值在于可认证性与有界性，而非精度或实测鲁棒性；要为本架构取得噪声保证，或需采用注~\ref{rem:averaged} 的平均算子构造（并承担其精度代价），或需采用带实测 $\Lambda$ 与 $M_J$ 的深度受限部署。""")

# ---------------------------------------------------------------- 5. Sec. price paragraph
rep("price_para",
    r"""图~\ref{fig:depth} 的精度间隙 $\numv{\NumPriceDelta}$，就是为定理~\ref{thm:field}--\ref{thm:noise} 与表~\ref{tab:abl} 付出的代价。直白地说：无约束展开是更好的估计器，约束族是更好的\emph{制成品}——它附带保证。""",
    r"""图~\ref{fig:depth} 的精度间隙 $\numv{\NumPriceDelta}$，就是为定理~\ref{thm:bounded} 的有界性证书、定理~\ref{thm:field} 的场子系统速率，以及表~\ref{tab:abl}--\ref{tab:abl2} 的审计所付出的代价。直白地说：无约束展开是更好的估计器，约束族是更好的\emph{制成品}——它附带一份保修，而这份保修的内容是有界性加可重算常数，并非压缩率（命题~\ref{prop:imposs}）。""")

# ---------------------------------------------------------------- 6. Discussion (i)
rep("disc_i",
    r"""(i) 全局压缩不可能（命题~\ref{prop:imposs}）；因此先验保真半径是以局部压缩假设为条件的，该假设对完整复合算子我们采用数值验证而非证明。""",
    r"""(i) 在 $\alpha>\alpha_c$ 上全局压缩不可能（命题~\ref{prop:imposs}），本文实测阈值为 $\alpha_c=(\theta\mathcal{E})^{-1}=\numv{\NumAlphaC}$；因此由压缩性导出的各几何界——漂移半径 $\bar\epsilon/(1-\rho_\chi)$ 与噪声增益 $\Lambda/(1-\rho)$——都是\emph{条件}结论，其前提被本文实测所否证（$\rho_\chi=\numv{\NumRhoChiCons}>1$，$\sigma_{\max}=\numv{\NumPhysSigma}>1$）。我们改而报告这些实测量；并认为在不给出前提的情况下引用任一半径，都构成对本文理论所证内容的歪曲。""")

# ---------------------------------------------------------------- 7. Discussion (viii)
rep("disc_viii",
    r"""(vii) 学习修正改善条件数与速度，但不增加信息量：噪声敏感度地板由测量算子决定。

\section{结论}""",
    r"""(vii) 学习修正改善条件数与速度，但不增加信息量：噪声敏感度地板由测量算子决定。
(viii) 不可能性命题是按工作点陈述的：$\alpha_c$ 依赖 $\mathcal{E}=\max_r\norm{E_{\cdot,r}}_2$，而后者在测试集上、乃至沿轨迹都会变化。我们报告的是表~\ref{tab:cert} 所用实测工作点处的值，并在图~\ref{fig:alphac} 中做了扫描；要去掉对工作点的依赖，需要对场的集中度给出一个下界，而我们既没有、也不声称拥有这样一个下界。

\section{结论}""")

# ---------------------------------------------------------------- 8. Conclusion
rep("concl",
    r"""本文给出展开式逆散射网络的不动点与收敛性理论，并在此过程中指出该问题长期悬置的结构性原因：忠实的 Born/对比源复合算子对任意阻尼步长都\emph{不是}全局压缩的，且任何与状态无关的可学习修正都无法使其成为压缩。因此分析被重建为逐层形式，证明了确实可得的保证：先验场子系统压缩（$\beta_0=\numv{\betacert}<1$）；非扩张物理先验投影下整条轨迹的认证有界性（显式不变紧集，对任意深度与任意训练参数成立）；无限深外推的先验漂移半径 $\bar\epsilon/(1-\beta_0)$；以及通过实测谱半径的噪声稳定界 $\Lambda/(1-\rho)$。两组 $800$ 层消融把\emph{必要}与\emph{充分}分离开来：投影与有界特征映射是结构性护栏，而幅度预算买到的是显式常数与保真半径。实用结论很短：\emph{对展开式逆散射，请投影对比度、对主干输入做有界化、并逐子系统认证；不要指望全局压缩——也不要因为重训后网络能存活，就推断该防护不必要。}""",
    r"""本文给出展开式逆散射网络的不动点与收敛性理论，并在此过程中指出该问题长期悬置的结构性原因：忠实的 Born/对比源复合算子在网络实际使用的步长范围内\emph{不是}全局压缩的，且任何低于相应余量的修正都无法使其成为压缩。我们把该论断的非正式版本替换为精确版本——层 Jacobian 的耦合块恰为 $\theta\alpha\,\mathrm{diag}(E)$，故 $\sigma_{\max}(J)\ge\theta\alpha\mathcal{E}$，当 $\alpha>\alpha_c=(\theta\mathcal{E})^{-1}$ 时非压缩性可证——同时明确说明在 $\alpha_c$ 以下该分块结构\emph{不}能给出什么：我们报告网格上实测的 $\sigma_{\max}(\alpha)$，而不去外推一个证明。随后分析被重建为逐层形式，证明了确实可得的保证：先验场子系统压缩（$\beta_0=\numv{\betacert}<1$）；非扩张物理先验投影下整条轨迹的认证有界性（显式不变紧集，对任意深度与任意训练参数成立）；精确的有限深度噪声核算，以及数据敏感度 $\Lambda$ 的构造性界；以及两条\emph{显式条件}的几何陈述——其前提被我们实测所否证。两组 $800$ 层消融把\emph{必要}与\emph{充分}分离开来：投影与有界特征映射是结构性护栏，而幅度预算买到的是显式常数。实用结论很短：\emph{对展开式逆散射，请投影对比度、对主干输入做有界化、并逐子系统认证；不要指望全局压缩——也不要因为重训后网络能存活，就推断该防护不必要。}""")

# ---------------------------------------------------------------- 9. Tab. trace
rep("tab_trace",
    r"""\begin{thebibliography}{30}
\bibitem{born_lit}""",
    r"""\begin{table}[t]
\centering
\caption{可追溯对照表：推导、本文编号陈述、以及实现函数。第一列中的“式/节”指随稿推导笔记，其编号与本文不同。}
\label{tab:trace}
\begin{tabular}{p{0.40\linewidth}lp{0.30\linewidth}}
\toprule
推导 / 陈述 & 本文位置 & 实现 \\
\midrule
场步、对比度步 & \eqref{eq:fieldstep}--\eqref{eq:contraststep} & \texttt{UnrolledISP.layer} \\
CG 最小二乘解（算法 1） & \eqref{eq:contraststep} & \texttt{UnrolledISP.\_cg\_ls} \\
带松弛 $\theta$ 的层 & \eqref{eq:layer-code} & \texttt{layer}（$\theta$ 参数） \\
投影 $\Pi$ & \eqref{eq:proj} & \texttt{layer}（clamp） \\
精确分块结构 & 引理~\ref{lem:blocks} & \texttt{run\_bjac.core\_parts} \\
不可能性界 / $\alpha_c$ & 命题~\ref{prop:imposs} & \texttt{run\_bjac.measure} \\
场子系统速率 & 定理~\ref{thm:field} & \texttt{run\_gain\_decomp}（块 A） \\
不变盒 & 定理~\ref{thm:bounded} & \texttt{run\_ablation}（外推） \\
保真半径 / $\rho_\chi$ & 推论~\ref{cor:fidelity} & \texttt{run\_bjac}（L4） \\
噪声核算 / $\Lambda$ & 命题~\ref{prop:noise} & \texttt{run\_bjac}（L3） \\
非正规间隙 & 推论~\ref{cor:noisenormal} & \texttt{exact\_cycle\_rho} / \texttt{run\_bjac}（L5） \\
\bottomrule
\end{tabular}
\end{table}

\begin{thebibliography}{30}
\bibitem{born_lit}""")

io.open(P, "w", encoding="utf-8").write(s)
print("CN batch4 ok", len(applied), applied)
