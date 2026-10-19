# -*- coding: utf-8 -*-
r"""_apply_rho_verdict.py -- rewrite the rho_chi verdict in all three manuscripts.

WHY A SCRIPT AND NOT HAND EDITS
-------------------------------
The 2026-10-19 regeneration moved every measured chi-map quantity from "above one" to
"below one" (block sigma 1.092 -> 0.467, chi-row norm 2.295 -> 0.553, F's spectral radius at
fixed alpha 1.053 -> 0.020).  Three manuscripts assert the OLD direction in the abstract, in
the remark that explains why the hypothesis is an assumption, and in one table-verdict cell.
Rewriting prose against numbers that are still being regenerated is exactly how the
"1.2% gap / different operating point" claim survived for weeks, so this script is
fail-closed:

  1. read the FINAL artefacts (chi_lip.json, L45.json);
  2. ASSERT the inequality directions the new prose depends on;
  3. only then apply the text edits, asserting every anchor exists;
  4. refuse to write anything if (2) or (3) fails.

Run:  python code/_apply_rho_verdict.py [--check-only]
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")


def load(name):
    with io.open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def dig(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def patch(path, pairs):
    p = os.path.join(ROOT, path)
    s = io.open(p, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in s else "\n"
    for old, new in pairs:
        a = old.replace("\n", nl)
        if a not in s:
            raise SystemExit("[anchor missing] %s :: %r" % (path, old[:80]))
        s = s.replace(a, new.replace("\n", nl))
    io.open(p, "w", encoding="utf-8", newline="").write(s)
    print("  patched %s (%d edit%s)" % (path, len(pairs), "" if len(pairs) == 1 else "s"))


EN_REMARK_OLD = r'''Every one of these exceeds one, so the attribution does not depend on
the object or on the norm used: the measured $\chi$-map is expansive in this regime, and in this
regime
a stronger statement holds at the fixed protocol: the $\chi\!\to\!\chi$ block of $F$ has a
\emph{spectral radius} of $\numv{\NumChiRhoFFixUnc}>1$ (unconstrained) at $\alpha=\gamma=\tfrac12$,
against $\numv{\NumChiRhoFFixCons}$ (constrained). A spectral radius above one is norm-independent,
 no choice of norm makes such a map contractive, whereas a Lipschitz constant above one is
merely one norm's verdict; we therefore report the two protocols separately rather than pooling
them, since they support conclusions of different strength.
Corollary~\ref{cor:fidelity} supplies \emph{no} finite a-priori radius, and the certified content of
the theory is boundedness (Thm.~\ref{thm:bounded}) plus the field-subsystem rate
(Thm.~\ref{thm:field}). Reporting the measured $\rho_\chi$ rather than the radius is the honest'''

EN_REMARK_NEW = r'''These numbers do \emph{not} all fall on one side of one, and the side they fall on is what
decides whether the hypothesis holds. In the constrained arm every proxy is below one -- on $\hat G$,
$\numv{\NumRhoChiCons}$ and $\numv{\NumChiLipRowCons}$; on $F$, $\numv{\NumChiFBlockCons}$ and
$\numv{\NumChiLipRowFCons}$ -- so nothing measured there refutes $\rho_\chi<1$. In the unconstrained
arm the row norm of $\hat G$ reaches $\numv{\NumChiLipRowUnc}$, which does refute it for that arm.
Nothing here \emph{verifies} the hypothesis in either arm: the block value is a lower bound and the
row norm is exact only for the layer map, whereas the reduced $\chi$-map derivative is a Schur
complement whose norm neither of them bounds. We therefore do not claim the general failure of the
geometric hypothesis that an earlier draft of this work asserted, and we keep the drift radius
conditional throughout: Cor.~\ref{cor:fidelity} supplies \emph{no} finite a-priori radius here, and
the certified content of the theory is boundedness (Thm.~\ref{thm:bounded}) plus the field-subsystem
rate (Thm.~\ref{thm:field}). Reporting the measured $\rho_\chi$ rather than the radius is the honest'''

MD_REMARK_OLD = r'''Lipschitz constant of the $\chi$-map and the former only a lower bound on it, as Table~\ref{tab:cert} shows; every one of the four pairs exceeds one, so the attribution does not
depend on the object or on the norm used. A stronger, norm-independent statement holds at the
fixed protocol: the \emph{spectral radius} of the $\chi\!\to\!\chi$ block of $F$ is
$\numv{\NumChiRhoFFixUnc}>1$ (unconstrained) at $\alpha=\gamma=\tfrac12$, against
$\numv{\NumChiRhoFFixCons}$ (constrained).'''

MD_REMARK_NEW = r'''Lipschitz constant of the $\chi$-map and the former only a lower bound on it, as Table~\ref{tab:cert} shows. These numbers do \emph{not} all fall on one side of one: in the constrained arm
every pair is below one (on $\hat G$, $\numv{\NumRhoChiCons}$ and $\numv{\NumChiLipRowCons}$; on $F$,
$\numv{\NumChiFBlockCons}$ and $\numv{\NumChiLipRowFCons}$), so nothing measured there refutes
$\rho_\chi<1$, whereas in the unconstrained arm the row norm of $\hat G$ reaches
$\numv{\NumChiLipRowUnc}$ and does refute it. Neither arm is \emph{verified} either: the block value is
a lower bound and the row norm is exact only for the layer map, while the reduced $\chi$-map derivative
is a Schur complement whose norm neither of them bounds. We therefore report the measurement arm by
arm and do not claim a general failure of the geometric hypothesis.'''

ABS_OLD = r'''\emph{conditional}, and we say so plainly: their hypotheses, a per-layer contraction rate
$\rho<1$ in Cor.~\ref{cor:noisegeo}, and $\rho_\chi<1$ in Cor.~\ref{cor:fidelity}, are measured
here to \emph{fail} (per-layer $\sigma_{\max}=\numv{\NumPhysSigma}>1$, $\rho_\chi=\numv{\NumRhoChiCons}>1$).'''

ABS_NEW = r'''\emph{conditional}, and we say so plainly. The per-layer contraction-rate hypothesis,
$\rho<1$ in Cor.~\ref{cor:noisegeo}, is refuted by measurement: the per-layer Lipschitz constant is
$\sigma_{\max}(J)=\numv{\NumPhysSigma}>1$ and no rate below one exists. The second hypothesis,
$\rho_\chi<1$ in Cor.~\ref{cor:fidelity}, is \emph{not decided} by our measurements: in the
constrained arm every $\chi$-map proxy we measure is below one ($\numv{\NumRhoChiCons}$ for the block,
$\numv{\NumChiLipRowCons}$ for the row), so nothing refutes it, while the reduced derivative is a
Schur complement whose norm those proxies do not bound, so nothing verifies it either; in the
unconstrained arm the row norm reaches $\numv{\NumChiLipRowUnc}$ and does refute it.'''

CN_ABS_OLD = r'''其前提被本文自身实测所否证（$\sigma_{\max}=\numv{\NumPhysSigma}>1$ 且实测 $\rho_\chi=\numv{\NumRhoChiCons}>1$）'''

CN_ABS_NEW = r'''其前提要分开看：逐层收缩率 $\rho<1$ 被实测否证（逐层 $\sigma_{\max}(J)=\numv{\NumPhysSigma}>1$），而 $\rho_\chi<1$ 则\emph{未被判定}——约束臂上我们测得的每个 $\chi$ 映射代理量都在 1 以下（分块 $\numv{\NumRhoChiCons}$、整行 $\numv{\NumChiLipRowCons}$），故它未被否证；但缩减导数是 Schur 补，其范数不被这些代理量所界，故也未被证明；无约束臂的整行范数达 $\numv{\NumChiLipRowUnc}$，仅该臂被否证'''

TB_OLD = r'''Composite $\rho_\chi$ as measured = $\sigma_{\max}$ of the $\chi\!\to\!\chi$ block, a \emph{lower} bound on the $\chi$-map Lipschitz constant (Cor.~\ref{cor:fidelity}) & hypothesis $<1$ & fails & $\numv{\NumRhoChiCons}$ / $\numv{\NumRhoChiUnc}$ \\'''

TB_NEW = r'''Composite $\rho_\chi$ as measured = $\sigma_{\max}$ of the $\chi\!\to\!\chi$ block, a \emph{lower} bound on the $\chi$-map Lipschitz constant (Cor.~\ref{cor:fidelity}) & hypothesis $<1$ & not decided & $\numv{\NumRhoChiCons}$ / $\numv{\NumRhoChiUnc}$ \\'''


def main():
    chi = load("chi_lip.json")
    l45 = load("L45.json")
    v = {
        "Ghat_sigma_cons": dig(chi, "chi_lip_constrained.G_hat_learned.sigma_chi_to_chi"),
        "Ghat_row_cons": dig(chi, "chi_lip_constrained.G_hat_learned.lip_chi_row"),
        "Ghat_sigma_unc": dig(chi, "chi_lip_unconstrained.G_hat_learned.sigma_chi_to_chi"),
        "Ghat_row_unc": dig(chi, "chi_lip_unconstrained.G_hat_learned.lip_chi_row"),
        "F_sigma_cons": dig(chi, "chi_lip_constrained.F_learned.sigma_chi_to_chi"),
        "F_row_cons": dig(chi, "chi_lip_constrained.F_learned.lip_chi_row"),
        "F_sigma_unc": dig(chi, "chi_lip_unconstrained.F_learned.sigma_chi_to_chi"),
        "F_row_unc": dig(chi, "chi_lip_unconstrained.F_learned.lip_chi_row"),
        "rho_Ghat_cons": dig(chi, "chi_lip_constrained.G_hat_learned.rho_chi_to_chi"),
        "rho_agrees_cons": dig(chi, "chi_lip_constrained.G_hat_learned.rho_agrees"),
        "comp_sigma_cons": dig(l45, "L4_constrained_composite.sigma_max_chi_block"),
        "comp_sigma_unc": dig(l45, "L4_unconstrained_composite.sigma_max_chi_block"),
    }
    for k in sorted(v):
        print("  %-18s %s" % (k, v[k]))

    missing = [k for k, x in v.items() if x is None]
    if missing:
        raise SystemExit("[FAIL] missing quantities: %s" % missing)
    cons = ("Ghat_sigma_cons", "Ghat_row_cons", "F_sigma_cons", "F_row_cons")
    if not all(v[k] < 1 for k in cons):
        raise SystemExit("[FAIL] constrained arm is not uniformly below one: %s"
                         % {k: v[k] for k in cons})
    if not v["Ghat_row_unc"] > 1:
        raise SystemExit("[FAIL] unconstrained G-hat row norm is not above one (%s); "
                         "re-argue the sentence, do not renumber it" % v["Ghat_row_unc"])
    if abs(v["Ghat_sigma_cons"] - v["comp_sigma_cons"]) / v["comp_sigma_cons"] > 0.05:
        raise SystemExit("[FAIL] chi-block sigma disagrees across drivers: %.4f vs %.4f"
                         % (v["Ghat_sigma_cons"], v["comp_sigma_cons"]))
    print("  [ok] inequality directions hold")

    if "--check-only" in sys.argv:
        print("check-only: nothing written")
        return

    patch("paper_en/main.tex", [(EN_REMARK_OLD, EN_REMARK_NEW), (ABS_OLD, ABS_NEW)])
    patch("paper_mdpi/main.tex", [(MD_REMARK_OLD, MD_REMARK_NEW), (ABS_OLD, ABS_NEW)])
    patch("paper_cn/main.tex", [(CN_ABS_OLD, CN_ABS_NEW)])
    for p in ("paper_en/main.tex", "paper_mdpi/main.tex", "paper_cn/main.tex"):
        s = io.open(os.path.join(ROOT, p), encoding="utf-8").read()
        if TB_OLD in s:
            patch(p, [(TB_OLD, TB_NEW)])
        else:
            print("  [skip] table verdict cell absent in %s" % p)


if __name__ == "__main__":
    main()
