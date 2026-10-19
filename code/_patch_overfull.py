# -*- coding: utf-8 -*-
"""Fix the overfull hboxes flagged by pdflatex: two display equations that exceed the
column and the monospace column of Table trace."""
import io

ROOT = r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究"
applied = []


def rep(path, tag, old, new):
    s = io.open(path, encoding="utf-8").read()
    assert s.count(old) == 1, "[%s] count=%d" % (tag, s.count(old))
    io.open(path, "w", encoding="utf-8").write(s.replace(old, new))
    applied.append(tag)


EN = ROOT + r"\paper_en\main.tex"
CN = ROOT + r"\paper_cn\main.tex"

# ---------------------------------------------------------------- eq:impossbound
rep(EN, "EN eq:impossbound",
    r"""\begin{equation}\label{eq:impossbound}
\sigma_{\max}(J)\;\ge\;
\underbrace{\theta\alpha\,\mathcal{E}}_{\sigma_{\max}(C),\ \text{closed form}}
\;,\qquad
\sigma_{\max}(J)\;\ge\;\sigma_{\max}(B),\qquad
\sigma_{\max}(J)\;\ge\;\sigma_{\max}(A),
\end{equation}""",
    r"""\begin{equation}\label{eq:impossbound}
\begin{aligned}
\sigma_{\max}(J)&\;\ge\;
\underbrace{\theta\alpha\,\mathcal{E}}_{\sigma_{\max}(C),\ \text{closed form}},\\
\sigma_{\max}(J)&\;\ge\;\sigma_{\max}(B),\qquad
\sigma_{\max}(J)\;\ge\;\sigma_{\max}(A),
\end{aligned}
\end{equation}""")

rep(CN, "CN eq:impossbound",
    r"""\begin{equation}\label{eq:impossbound}
\sigma_{\max}(J)\;\ge\;
\underbrace{\theta\alpha\,\mathcal{E}}_{\sigma_{\max}(C),\ \text{闭式}}
\;,\qquad
\sigma_{\max}(J)\;\ge\;\sigma_{\max}(B),\qquad
\sigma_{\max}(J)\;\ge\;\sigma_{\max}(A),
\end{equation}""",
    r"""\begin{equation}\label{eq:impossbound}
\begin{aligned}
\sigma_{\max}(J)&\;\ge\;
\underbrace{\theta\alpha\,\mathcal{E}}_{\sigma_{\max}(C),\ \text{闭式}},\\
\sigma_{\max}(J)&\;\ge\;\sigma_{\max}(B),\qquad
\sigma_{\max}(J)\;\ge\;\sigma_{\max}(A),
\end{aligned}
\end{equation}""")

# ---------------------------------------------------------------------- eq:box
rep(EN, "EN eq:box",
    r"""\begin{equation}\label{eq:box}
\infn{\chi^{K}}\le\chi_{\max},\qquad
\norm{w^{K}}\;\le\;\gamma_w^{K}\norm{w^{0}}+\frac{\bar\theta\bar\alpha\,\chi_{\max}\norm{E^i}}{1-\gamma_w},
\quad \gamma_w:=1-\bar\theta\bar\alpha\,(1-\beta_0),
\end{equation}""",
    r"""\begin{equation}\label{eq:box}
\begin{aligned}
\infn{\chi^{K}}&\le\chi_{\max},\\
\norm{w^{K}}&\;\le\;\gamma_w^{K}\norm{w^{0}}
+\frac{\bar\theta\bar\alpha\,\chi_{\max}\norm{E^i}}{1-\gamma_w},\\
\gamma_w&:=1-\bar\theta\bar\alpha\,(1-\beta_0),
\end{aligned}
\end{equation}""")

rep(CN, "CN eq:box",
    r"""\begin{equation}\label{eq:box}
\infn{\chi^{K}}\le\chi_{\max},\qquad
\norm{w^{K}}\;\le\;\gamma_w^{K}\norm{w^{0}}+\frac{\bar\theta\bar\alpha\,\chi_{\max}\norm{E^i}}{1-\gamma_w},
\quad \gamma_w:=1-\bar\theta\bar\alpha\,(1-\beta_0),
\end{equation}""",
    r"""\begin{equation}\label{eq:box}
\begin{aligned}
\infn{\chi^{K}}&\le\chi_{\max},\\
\norm{w^{K}}&\;\le\;\gamma_w^{K}\norm{w^{0}}
+\frac{\bar\theta\bar\alpha\,\chi_{\max}\norm{E^i}}{1-\gamma_w},\\
\gamma_w&:=1-\bar\theta\bar\alpha\,(1-\beta_0),
\end{aligned}
\end{equation}""")

# ------------------------------------------------------------------- tab:trace
rep(EN, "EN tab:trace width",
    r"""\begin{tabular}{p{0.44\linewidth}lp{0.30\linewidth}}
\toprule
Derivation / statement & Here & Implementation \\
\midrule
Field step, contrast step & \eqref{eq:fieldstep}--\eqref{eq:contraststep} & \texttt{UnrolledISP.layer} \\
CG least-squares solve (Alg.~1) & \eqref{eq:contraststep} & \texttt{UnrolledISP.\_cg\_ls} \\
Layer with relaxation $\theta$ & \eqref{eq:layer-code} & \texttt{layer} ($\theta$ arg.) \\
Projection $\Pi$ & \eqref{eq:proj} & \texttt{layer} (clamp) \\
Exact block structure & Lem.~\ref{lem:blocks} & \texttt{run\_bjac.core\_parts} \\
Impossibility bound / $\alpha_c$ & Prop.~\ref{prop:imposs} & \texttt{run\_bjac.measure} \\
Field-subsystem rate & Thm.~\ref{thm:field} & \texttt{run\_gain\_decomp} (block A) \\
Invariant box & Thm.~\ref{thm:bounded} & \texttt{run\_ablation} (extrapolation) \\
Fidelity radius / $\rho_\chi$ & Cor.~\ref{cor:fidelity} & \texttt{run\_bjac} (L4) \\
Noise accounting / $\Lambda$ & Prop.~\ref{prop:noise} & \texttt{run\_bjac} (L3) \\
Non-normality gap & Cor.~\ref{cor:noisenormal} & \texttt{exact\_cycle\_rho} / \texttt{run\_bjac} (L5) \\
\bottomrule
\end{tabular}""",
    r"""\footnotesize
\begin{tabular}{p{0.42\linewidth}lp{0.36\linewidth}}
\toprule
Derivation / statement & Here & Implementation \\
\midrule
Field step, contrast step & \eqref{eq:fieldstep}--\eqref{eq:contraststep} & \texttt{layer} \\
CG least-squares solve (Alg.~1) & \eqref{eq:contraststep} & \texttt{\_cg\_ls} \\
Layer with relaxation $\theta$ & \eqref{eq:layer-code} & \texttt{layer} ($\theta$ arg.) \\
Projection $\Pi$ & \eqref{eq:proj} & \texttt{layer} (clamp) \\
Exact block structure & Lem.~\ref{lem:blocks} & \texttt{core\_parts} \\
Impossibility bound / $\alpha_c$ & Prop.~\ref{prop:imposs} & \texttt{measure} \\
Field-subsystem rate & Thm.~\ref{thm:field} & \texttt{gain\_decomp} (A) \\
Invariant box & Thm.~\ref{thm:bounded} & \texttt{ablation} (extrap.) \\
Fidelity radius / $\rho_\chi$ & Cor.~\ref{cor:fidelity} & \texttt{L45} (L4) \\
Noise accounting / $\Lambda$ & Prop.~\ref{prop:noise} & \texttt{bjac} (L3) \\
Non-normality gap & Cor.~\ref{cor:noisenormal} & \texttt{bjac} (L5) \\
\bottomrule
\end{tabular}""")

rep(CN, "CN tab:trace width",
    r"""\begin{tabular}{p{0.40\linewidth}lp{0.30\linewidth}}
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
\end{tabular}""",
    r"""\footnotesize
\begin{tabular}{p{0.40\linewidth}lp{0.34\linewidth}}
\toprule
推导 / 陈述 & 本文位置 & 实现 \\
\midrule
场步、对比度步 & \eqref{eq:fieldstep}--\eqref{eq:contraststep} & \texttt{layer} \\
CG 最小二乘解（算法 1） & \eqref{eq:contraststep} & \texttt{\_cg\_ls} \\
带松弛 $\theta$ 的层 & \eqref{eq:layer-code} & \texttt{layer}（$\theta$） \\
投影 $\Pi$ & \eqref{eq:proj} & \texttt{layer}（clamp） \\
精确分块结构 & 引理~\ref{lem:blocks} & \texttt{core\_parts} \\
不可能性界 / $\alpha_c$ & 命题~\ref{prop:imposs} & \texttt{measure} \\
场子系统速率 & 定理~\ref{thm:field} & \texttt{gain\_decomp}（A） \\
不变盒 & 定理~\ref{thm:bounded} & \texttt{ablation}（外推） \\
保真半径 / $\rho_\chi$ & 推论~\ref{cor:fidelity} & \texttt{L45}（L4） \\
噪声核算 / $\Lambda$ & 命题~\ref{prop:noise} & \texttt{bjac}（L3） \\
非正规间隙 & 推论~\ref{cor:noisenormal} & \texttt{bjac}（L5） \\
\bottomrule
\end{tabular}""")

print("overfull fixes applied: %d" % len(applied))
for a in applied:
    print("  -", a)
