# -*- coding: utf-8 -*-
"""_dup_sentence.py — find sentences that appear twice inside one manuscript.

Class of defect (found in the round-3 self-check): a patch that inserts a revised
version of a sentence without deleting the old one leaves the *same* claim stated
twice in one paragraph.  Nothing in G1-G10 sees it: the macros resolve, the text
compiles, no number is wrong -- it is purely a text-integrity defect, and the
duplicate is trivially detectable by normalising whitespace and comparing.

First implementation reported four "duplicates" that were LaTeX boilerplate
(`\\end{figure} \\begin{figure}[t] \\centering \\includegraphics[...]{...}`), i.e. a
false positive of the instrument rather than a defect of the manuscript.  Per the
project's audit discipline the fix is to make the instrument discriminate, not to
raise the threshold: structural fragments are excluded before comparison, and a unit
must carry a minimum of prose (CJK characters, or three-letter Latin words) to count
as a sentence at all.

`--selftest` injects a known duplicate and asserts that it is reported, so a clean
run is evidence rather than a rubber stamp.

Usage: python _dup_sentence.py [paper_en/main.tex paper_cn/main.tex]
Exit status 1 if any sentence longer than --minchars recurs.
"""
import argparse
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

STRUCT = re.compile(r"\\(?:begin|end|includegraphics|caption|label|centering|toprule|midrule"
                    r"|bottomrule|footnotesize|small|setlength|tabcolsep|item|end\{|begin\{)")


def is_prose(s):
    """A unit counts as prose if it is not a structural fragment and carries content."""
    if STRUCT.search(s):
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    words = len(re.findall(r"[A-Za-z]{3,}", s))
    return cjk >= 8 or words >= 6


def sentences(src, minchars=40):
    body = src.split(r"\begin{thebibliography}")[0]
    body = re.sub(r"(?<!\\)%.*", " ", body)
    body = re.sub(r"\\(cite|ref|eqref|label)\{[^{}]*\}", " ", body)
    body = re.sub(r"\s+", " ", body)
    out = []
    for p in re.split(r"(?<=[。．\.\!\?])\s*", body):
        s = p.strip()
        if len(s) >= minchars and is_prose(s):
            out.append(s)
    return out


def duplicates(src, minchars=40):
    seen, reps, total = set(), [], 0
    for s in sentences(src, minchars):
        total += 1
        if s in seen:
            reps.append(s)
        else:
            seen.add(s)
    return total, reps


def selftest():
    ok = True
    base = ("This is a sufficiently long sentence about the layer Jacobian block structure "
            "and it appears exactly once. ")
    src = base + ("A different long sentence about the field subsystem contraction that is "
                  "also unique in the document. ")
    t, r = duplicates(src)
    good = (t == 2 and not r)
    print("  selftest clean input    -> %d sentences, %d dup %s"
          % (t, len(r), "PASS" if good else "FAIL"))
    ok &= good

    src2 = src + base          # inject the duplicate
    t2, r2 = duplicates(src2)
    good2 = (len(r2) == 1)
    print("  selftest injected dup   -> %d sentences, %d dup %s"
          % (t2, len(r2), "PASS" if good2 else "FAIL"))
    ok &= good2

    # the instrument must NOT fire on LaTeX boilerplate
    boiler = (r"\end{figure} \begin{figure}[t] \centering "
              r"\includegraphics[width=\linewidth]{../figures/fig_extrap.pdf} ")
    src3 = baseline = (boiler * 3) + base
    t3, r3 = duplicates(src3)
    good3 = (len(r3) == 0)
    print("  selftest boilerplate    -> %d sentences, %d dup %s"
          % (t3, len(r3), "PASS" if good3 else "FAIL (false positive)"))
    ok &= good3

    print("  SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="*",
                    default=[os.path.join(ROOT, "paper_en", "main.tex"),
                             os.path.join(ROOT, "paper_cn", "main.tex")])
    ap.add_argument("--minchars", type=int, default=40)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    bad = 0
    for t in a.tex:
        src = io.open(t, encoding="utf-8").read()
        total, reps = duplicates(src, a.minchars)
        print("%s: %d prose sentence(s) >= %d chars, %d duplicate(s)"
              % (os.path.relpath(t, ROOT), total, a.minchars, len(reps)))
        for r in reps:
            print("   DUP: %s" % r[:160])
        bad += len(reps)
    print("%s" % ("PASS: no duplicated prose sentence" if bad == 0
                  else "FAIL: %d duplicated sentence(s)" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
