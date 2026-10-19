# -*- coding: utf-8 -*-
"""Apply the 6 bibliography corrections found by independent >=3-source verification."""
import io

ROOT = r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究"
applied = []


def rep(path, tag, old, new, n=1):
    s = io.open(path, encoding="utf-8").read()
    assert s.count(old) == n, "[%s] anchor count=%d (expected %d)" % (tag, s.count(old), n)
    s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8").write(s)
    applied.append(tag)


EN = ROOT + r"\paper_en\main.tex"
CN = ROOT + r"\paper_cn\main.tex"

# ------------------------------------------------------------------ EN
rep(EN, "EN born_lit",
    r"""T.\ M. Habashy, R.\ W. Groom, and B.\ R. Spies, ``Beyond the Born and Rytov approximations: A
general approach to approximate scattering,'' \emph{J. Geophys. Res.}, vol.\ 98, no.\ B2,
pp.\ 1759--1775, 1993.""",
    r"""T.\ M. Habashy, R.\ W. Groom, and B.\ R. Spies, ``Beyond the Born and Rytov approximations: A
nonlinear approach to electromagnetic scattering,'' \emph{J. Geophys. Res.}, vol.\ 98, no.\ B2,
pp.\ 1759--1775, 1993.""")

rep(EN, "EN som_lit",
    r"""X.\ Chen, ``Subspace-based optimization method for solving inverse-scattering problems,''
\emph{IEEE Trans. Microw. Theory Techn.}, vol.\ 58, no.\ 1, pp.\ 42--49, 2010.""",
    r"""X.\ Chen, ``Subspace-based optimization method for solving inverse-scattering problems,''
\emph{IEEE Trans. Geosci. Remote Sens.}, vol.\ 48, no.\ 1, pp.\ 42--49, 2010.""")

rep(EN, "EN alista",
    r"""J.\ Liu \emph{et al.}, ``ALISTA: Analytic weights are as good as learned weights in LISTA,'' in
\emph{Proc. AISTATS}, 2019.""",
    r"""J.\ Liu, X.\ Chen, Z.\ Wang, and W.\ Yin, ``ALISTA: Analytic weights are as good as learned
weights in LISTA,'' in \emph{Proc. ICLR}, 2019.""")

rep(EN, "EN algomapping",
    r"""``On iterative algorithms with different mapping in each iteration,'' \emph{Algorithms}, vol.\ 19,
no.\ 6, p.\ 470, 2026.""",
    r"""D.\ B. Tay, ``On iterative algorithms with different mapping in each iteration,''
\emph{Algorithms}, vol.\ 19, no.\ 6, Art.\ no.\ 470, 2026.""")

rep(EN, "EN cui_reg",
    r"""Z.\ Cui, Q.\ Zhu, J.\ Cheng, and D.\ Liang, ``Deep unfolding as iterative regularization for
imaging inverse problems,'' \emph{Inverse Problems}, 2024.""",
    r"""Z.-X.\ Cui, Q.\ Zhu, J.\ Cheng, B.\ Zhang, and D.\ Liang, ``Deep unfolding as iterative
regularization for imaging inverse problems,'' \emph{Inverse Problems}, vol.\ 40, no.\ 2,
Art.\ no.\ 025011, 2024.""")

rep(EN, "EN sherry",
    r"""``Designing stable neural networks using convex analysis and ODEs,'' \emph{Physica D}, vol.\ 463,
p.\ 134150, 2024.""",
    r"""``Designing stable neural networks using convex analysis and ODEs,'' \emph{Physica D}, vol.\ 463,
Art.\ no.\ 134159, 2024.""")

# ------------------------------------------------------------------ CN
rep(CN, "CN born_lit",
    r"""\bibitem{born_lit} Habashy T M, Groom R W, Spies B R. Beyond the Born and Rytov approximations: A general approach to approximate scattering. J. Geophys. Res., 98(B2): 1759--1775, 1993.""",
    r"""\bibitem{born_lit} Habashy T M, Groom R W, Spies B R. Beyond the Born and Rytov approximations: A nonlinear approach to electromagnetic scattering. J. Geophys. Res., 98(B2): 1759--1775, 1993.""")

rep(CN, "CN som_lit",
    r"""\bibitem{som_lit} Chen X. Subspace-based optimization method for solving inverse-scattering problems. IEEE Trans. Microw. Theory Techn., 58(1): 42--49, 2010.""",
    r"""\bibitem{som_lit} Chen X. Subspace-based optimization method for solving inverse-scattering problems. IEEE Trans. Geosci. Remote Sens., 48(1): 42--49, 2010.""")

rep(CN, "CN alista",
    r"""\bibitem{alista} Liu J, et al. ALISTA: Analytic weights are as good as learned weights in LISTA. AISTATS, 2019.""",
    r"""\bibitem{alista} Liu J, Chen X, Wang Z, Yin W. ALISTA: Analytic weights are as good as learned weights in LISTA. ICLR, 2019.""")

rep(CN, "CN algomapping",
    r"""\bibitem{algomapping} On iterative algorithms with different mapping in each iteration. Algorithms, 19(6): 470, 2026.""",
    r"""\bibitem{algomapping} Tay D B. On iterative algorithms with different mapping in each iteration. Algorithms, 19(6): 470, 2026.""")

rep(CN, "CN cui_reg",
    r"""\bibitem{cui_reg} Cui Z, Zhu Q, Cheng J, Liang D. Deep unfolding as iterative regularization for imaging inverse problems. Inverse Problems, 2024.""",
    r"""\bibitem{cui_reg} Cui Z-X, Zhu Q, Cheng J, Zhang B, Liang D. Deep unfolding as iterative regularization for imaging inverse problems. Inverse Problems, 40(2): 025011, 2024.""")

rep(CN, "CN sherry",
    r"""\bibitem{sherry_nonexp} Sherry F, Celledoni E, Ehrhardt M J, Murari D, Owren B, Sch\"onlieb C-B. Designing stable neural networks using convex analysis and ODEs. Physica D, 463: 134150, 2024.""",
    r"""\bibitem{sherry_nonexp} Sherry F, Celledoni E, Ehrhardt M J, Murari D, Owren B, Sch\"onlieb C-B. Designing stable neural networks using convex analysis and ODEs. Physica D, 463: 134159, 2024.""")

print("bibliography fixes applied: %d" % len(applied))
for a in applied:
    print("  -", a)
