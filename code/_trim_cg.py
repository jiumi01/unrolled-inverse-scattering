import io
p = r'D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\paper_mdpi\main.tex'
with io.open(p,'r',encoding='utf-8') as f:
    lines = f.readlines()

# CG sweep subsection: line 1374 to 1431 (1-indexed) = idx 1373..1430
# Keep \subsection line (1374) and table (1394-1409), replace narrative
new_block = r'''\subsection{Sensitivity to the contrast-solve truncation}
\label{sec:cgsweep}
We sweep $n_{\mathrm{cg}}$ at frozen trained weights and a fixed operating point
(Table~\ref{tab:cgsweep}). The CG-free blocks $C,D$ are bit-identical across a
$\numv{\NumCgRatio}$-fold change in $n_{\mathrm{cg}}$, the closed forms of
Lem.~\ref{lem:blocks} reproduced to every digit. The contrast-path blocks $A,B$ move
by order one, so a certificate on $A$ or $B$ would be a statement about the solver
rather than the physics; one on $C$ or $D$ is not. The deployed network is calibrated
to the residual of its trained truncation, so truncation is part of the operating
point, not an efficiency knob, which is why the analysis is anchored on the CG-free
blocks.

\begin{table}[t]
\centering
\caption{Contrast-solve truncation sweep at a fixed operating point, $n_{\mathrm{cg}}\in\{\numv{\NumCgNMin},\dots,\numv{\NumCgNMax}\}$, double precision. The CG-free blocks $C,D$ are bit-identical across the whole sweep; the contrast-path blocks $A,B$ are not.}
\label{tab:cgsweep}
\small
\begin{tabular}{lccc}
\toprule
quantity & min & max & spread \\
\midrule
$\sigma_{\max}(C)$ (CG-free) & $\numv{\NumCgCMin}$ & $\numv{\NumCgCMax}$ & $\numv{\NumCgCFlatPct}\%$ \\
$\sigma_{\max}(D)$ (CG-free) & $\numv{\NumCgDMin}$ & $\numv{\NumCgDMax}$ & $\numv{\NumCgDFlatPct}\%$ \\
$\sigma_{\max}(A)$ (contrast) & $\numv{\NumCgAMin}$ & $\numv{\NumCgAMax}$ & --- \\
$\sigma_{\max}(B)$ (contrast) & $\numv{\NumCgBMin}$ & $\numv{\NumCgBMax}$ & --- \\
\bottomrule
\end{tabular}
\end{table}
'''

# Find boundaries
start = None; end = None
for i, l in enumerate(lines):
    if l.startswith(r'\subsection{Sensitivity to the contrast-solve truncation}'):
        start = i
    if start is not None and l.startswith(r'\subsection{Depth convergence}'):
        end = i
        break
print(f'replacing lines {start+1}..{end} (1-indexed), removing {end-start-1} lines')
out = lines[:start] + [new_block] + lines[end:]
with io.open(p,'w',encoding='utf-8') as f:
    f.writelines(out)
print('done')
