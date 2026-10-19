# -*- coding: utf-8 -*-
"""_swap_abstract.py -- splice the compressed English abstract into paper_en/main.tex.

Why: R3 n3.4 raised the abstract word-count convention.  Under the "math stripped"
convention the abstract sat at 249 of IEEE TAP's 250 words -- one word of margin -- and
under the more conservative "every formula counts as one word" convention it was 263, i.e.
over the limit.  Rather than argue the convention, the abstract is shortened until BOTH
conventions fit, and gate G10 is tightened to check both.  The Chinese abstract is
mirrored so the two manuscripts stay equivalent.

This script is deliberately dumb and verifiable: it reports the before/after counts from
the same counter the gate imports (code/_abstract_words.py), so the numbers in the cover
letter, in the response letter and in the gate cannot diverge.
"""
import importlib.util, io, os, shutil, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("_aw", os.path.join(ROOT, "code", "_abstract_words.py"))
AW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AW)

LIMIT = 250


def counts(body):
    n, m = AW.strip_latex(body)
    return n, m


def swap(p, new_body, tag):
    src = io.open(p, encoding="utf-8").read()
    a, b = src.find("\\begin{abstract}"), src.find("\\end{abstract}")
    assert a >= 0 and b > a, "abstract environment not found in %s" % p
    old = src[a + len("\\begin{abstract}"):b]
    n0, m0 = counts(old)
    n1, m1 = counts(new_body)
    shutil.copyfile(p, p + ".bak_" + tag)
    src = src[:a + len("\\begin{abstract}")] + new_body + src[b:]
    io.open(p, "w", encoding="utf-8", newline="\n").write(src)
    print("%-18s before A=%d B=%d -> after A=%d B=%d   (limit %d)" % (tag, n0, m0, n1, m1, LIMIT))
    assert n1 <= LIMIT and m1 <= LIMIT, "compressed abstract still exceeds the limit"
    return n1, m1


def main():
    cand = io.open(os.path.join(ROOT, "results", "_abs_cand.txt"), encoding="utf-8").read()
    n, m = swap(os.path.join(ROOT, "paper_en", "main.tex"), cand, "preabs3")
    print("[ok] EN abstract now satisfies both conventions: A=%d, B=%d" % (n, m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
