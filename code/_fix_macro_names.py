# -*- coding: utf-8 -*-
"""One-off: rename digit-bearing macro citations to their sanitised control-sequence names.

Why: gen_numbers.py sanitises control words (digits are illegal in TeX control words) and
then PRUNES macros the manuscript does not cite, testing the *sanitised* name.  A manuscript
that cites \\NumT1RelOff therefore (a) cites a name that is never emitted and (b) causes the
emitted name \\NumTOneRelOff to be pruned.  Both halves of the same trap.  Fixed here by
citing the sanitised names.

Written as a real file, not a heredoc: a shell heredoc collapses \\N and the file contents
differ from what was intended (known trap in this project).
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SUBS = [
    ("\\NumT1RelOff", "\\NumTOneRelOff"),
    ("\\NumT1RelOn", "\\NumTOneRelOn"),
    ("\\NumF64SigErr", "\\NumFSixFourSigErr"),
]

for rel in ("paper_en/main.tex", "paper_cn/main.tex"):
    p = os.path.join(ROOT, rel)
    s = io.open(p, encoding="utf-8").read()
    n0 = len(s)
    hits = {}
    for a, b in SUBS:
        hits[a] = s.count(a)
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8").write(s)
    print("%-22s %d -> %d chars  %s" % (rel, n0, len(s), hits))
