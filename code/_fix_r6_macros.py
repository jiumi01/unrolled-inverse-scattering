# -*- coding: utf-8 -*-
"""One-shot fixer for the R6 macro citations (see gen_numbers.py: control words may not
contain digits, and a citation must carry the backslash the pruner looks for)."""
import io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

RENAME = {
    "NumGridGb32": "NumGridGbLo", "NumGridGb64": "NumGridGbMd", "NumGridGb128": "NumGridGbHi",
    "NumGridBeta32": "NumGridBetaLo", "NumGridBeta64": "NumGridBetaMd", "NumGridBeta128": "NumGridBetaHi",
    "NumGridE32": "NumGridELo", "NumGridE64": "NumGridEMd", "NumGridE128": "NumGridEHi",
    "NumGridAlphaC32": "NumGridAlphaCLo", "NumGridAlphaC64": "NumGridAlphaCMd",
    "NumGridAlphaC128": "NumGridAlphaCHi",
    "NumGridGs32": "NumGridGsLo", "NumGridGs64": "NumGridGsMd", "NumGridGs128": "NumGridGsHi",
    "NumGridGbDense32": "NumGridGbDense",
    "NumCgR15E15": "NumCgRImplImpl", "NumCgR15E200": "NumCgRImplConv",
    "NumCgR200E15": "NumCgRConvImpl", "NumCgR200E200": "NumCgRConvConv",
    "NumCsiAtK8": "NumCsiAtKEight",
}
# order matters: longer keys first so "NumGridGb32" is not shadowed by a prefix rule
ORDER = sorted(RENAME, key=len, reverse=True)

for rel in ("paper_en/main.tex", "paper_cn/main.tex", "code/gen_numbers.py"):
    p = os.path.join(ROOT, rel)
    s = io.open(p, encoding="utf-8").read()
    orig = s
    for old, new in RENAME.items():
        s = s.replace(old, new)
    if rel.endswith("main.tex"):
        # insert the missing backslash: \numv{NumXxx} -> \numv{\NumXxx}
        s = re.sub(r"\\numv\{Num", r"\\numv{\\Num", s)
    if s != orig:
        io.open(p, "w", encoding="utf-8").write(s)
        print("patched", rel)
    else:
        print("unchanged", rel)

# report any remaining bare \numv{Num...} (a citation the pruner would still miss)
for rel in ("paper_en/main.tex", "paper_cn/main.tex"):
    s = io.open(os.path.join(ROOT, rel), encoding="utf-8").read()
    bad = re.findall(r"\\numv\{Num[A-Za-z]*\}", s)
    print(rel, "remaining bare:", bad[:5], "count", len(bad))
