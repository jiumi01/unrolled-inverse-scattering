import io
with io.open(r'D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\paper_mdpi\main.tex','r',encoding='utf-8') as f:
    lines = f.readlines()

new_block = r'''\emph{Provenance.} All rows are produced by \texttt{code/run\_gain\_decomp.py} at the Born--LS initial state, $\alpha=\gamma=1/2$ fixed, projection on, float64; the $K{=}8$ cyclic rows use a separate driver at learned step sizes. The as-implemented $n_{\mathrm{cg}}$ values differ from the converged-solve values on the contrast-path blocks only; the CG-free blocks $C$ and $D$ are bit-identical across the sweep, which is what makes the certificates CG-free. A single-precision re-run overestimates the contrast-path singular values by order one because the dominant triplet is not resolved; all tabulated values here are double precision.

Four conclusions. (1) The field block contracts at $\numv{\NumPhysWBlock}$, below the a-priori bound $1-\alpha(1-\beta_0)=\numv{\NumGammaW}$: the certificate of Theorem~\ref{thm:field} is valid but conservative. (2) The coupling block is the obstruction and it is available in closed form: $\sigma_{\max}(C)=\alpha\mathcal{E}=\numv{\NumSigCLower}>1$ with threshold $\alpha_c=\numv{\NumAlphaC}<\tfrac12$, so at the operating step size the non-contractivity is \emph{proved}, not merely observed. (3) The contrast-path blocks give $\sigma_{\max}(A)=\numv{\NumSigA}$ and $\sigma_{\max}(B)=\numv{\NumSigB}$; the measured $\rho_\chi=\numv{\NumRhoChiCons}>1$, so the fidelity corollary drift radius is not available in this regime. (4) The composite gain exceeds one in every configuration measured, but the closed-form coupling block $\sigma_{\max}(C)=\alpha\mathcal{E}>1$ already forces $\sigma_{\max}(DT_k)>1$ for \emph{any} correction by Prop.~\ref{prop:imposs}. The bilinear coupling $\chi\odot E$ makes the faithful composite non-contractive regardless of how the correction is controlled, which is why the guarantees we offer are boundedness and layer-wise accounting rather than a per-layer contraction rate.
'''

out = lines[:1610] + [new_block] + lines[1701:]
with io.open(r'D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\paper_mdpi\main.tex','w',encoding='utf-8') as f:
    f.writelines(out)
print('done, removed', 1701-1610, 'lines')
