# -*- coding: utf-8 -*-
"""_patch_fix3.py — three follow-ups after the overfull pass.

1. gen_numbers.Fsci emitted "$1.05\\times10^{-7}$" *with* dollar signs, and that
   macro is used inside math mode in both papers ($\\numv{\\NumInstrErr}$), so
   xelatex raised "Missing $ inserted" four times.  Fsci must be mode-agnostic:
   emit \\ensuremath{...}.  Also make an exact zero print as 0 rather than "--",
   which would silently turn a genuine zero into a missing value.

2. CN tab:trace is still 7.67 pt too wide at p{0.40}/p{0.34} -> 0.38/0.32 with a
   tabcolsep of 3 pt.

3. CN eq:noiseexact is 1.44 pt too wide -> drop \\Big to \\big.
   A global \\emergencystretch absorbs the remaining sub-3 pt CJK line breaks
   (paragraph at 314--317) without editing any sentence.
"""
import io, os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GEN = os.path.join(ROOT, "code", "gen_numbers.py")
CN = os.path.join(ROOT, "paper_cn", "main.tex")


def patch(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for old, new, tag in pairs:
        n = s.count(old)
        assert n == 1, "%s [%s]: %d occurrences" % (os.path.basename(path), tag, n)
        s = s.replace(old, new)
        print("  ok  %-14s %s" % (os.path.basename(path), tag))
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)


patch(GEN, [
    (
        '''    """Scientific-notation formatter for quantities whose magnitude is the message
    (e.g. a instrument validation error of 1e-7 must not print as '0.0')."""
    if x is None:
        return "--"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(xf) or xf == 0.0:
        return "--"
    e = math.floor(math.log10(abs(xf)))
    m = xf / (10.0 ** e)
    return "$%.*f\\\\times10^{%d}$" % (nd, m, e)''',
        '''    """Scientific-notation formatter for quantities whose magnitude is the message
    (e.g. an instrument validation error of 1e-7 must not print as '0.0').

    The result is wrapped in \\ensuremath so the same macro can be cited in text
    mode and inside $...$ (both usages occur).  A genuine zero prints as 0 and a
    non-finite value as the explicit placeholder; only the latter is "missing".
    """
    if x is None:
        return "--"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(xf):
        return "--"
    if xf == 0.0:
        return r"\\ensuremath{0}"
    e = math.floor(math.log10(abs(xf)))
    m = xf / (10.0 ** e)
    return r"\\ensuremath{%.*f\\times10^{%d}}" % (nd, m, e)''',
        "Fsci ensuremath",
    ),
])

patch(CN, [
    (
        "\\begin{document}",
        "\\begin{document}\n\\setlength{\\emergencystretch}{2em}",
        "emergencystretch",
    ),
    (
        "\\begin{tabular}{p{0.40\\linewidth}lp{0.34\\linewidth}}",
        "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{p{0.38\\linewidth}lp{0.32\\linewidth}}",
        "tab:trace widths",
    ),
    (
        "\\le\\Big[\\sum_{k=0}^{K-1}\\Lambda_k\\!\\!\\prod_{j=k+1}^{K-1}\\!\\!L_j\\Big]\\norm{\\Delta y}.",
        "\\le\\big[\\sum_{k=0}^{K-1}\\Lambda_k\\!\\!\\prod_{j=k+1}^{K-1}\\!\\!L_j\\big]\\norm{\\Delta y}.",
        "eq:noiseexact big",
    ),
])
print("done")
