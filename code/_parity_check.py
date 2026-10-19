# -*- coding: utf-8 -*-
"""Replicate gate G5 (structural parity EN vs CN) standalone."""
import io, os, re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shape(p):
    s = io.open(os.path.join(ROOT, p, "main.tex"), encoding="utf-8").read()
    secs = len(re.findall(r"\\section\{|\\subsection\{", s))
    envs = Counter(re.findall(
        r"\\begin\{(theorem|lemma|proposition|corollary|remark|definition|table\*?|figure\*?|algorithm)\}", s))
    labs = len(re.findall(r"\\label\{(?:sec|eq|thm|lem|prop|cor|rem|tab|fig|alg):", s))
    return secs, dict(sorted(envs.items())), labs


a, b = shape("paper_en"), shape("paper_cn")
print("EN", a)
print("CN", b)
print("MATCH", a == b)
