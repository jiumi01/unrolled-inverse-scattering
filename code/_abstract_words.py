"""Abstract word-count gate (R3 M3.5 / m3.5 / Q3.4).

IEEE TAP's conventional upper bound on the abstract is ~250 words.  This script
extracts the `abstract` environment from a manuscript, strips LaTeX markup, and
reports two counts:

  * `words_no_math`  --- whitespace tokens that contain at least one letter,
                        after every inline/display formula has been removed.
                        This is the count the reviewers used (they reported 279
                        for the round-2 English abstract).
  * `words_with_math` --- the same, but each formula contributes one token.
                        Reported alongside so the convention is explicit and the
                        number is not a function of a silent preprocessing choice.

Exit status is non-zero when the chosen count exceeds `--limit`, so the script
doubles as a gate.  Run with `--selftest` for the falsifiability probe: a
synthetic abstract of known length must be counted correctly, and a synthetic
over-length abstract must be rejected.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)

# commands whose braced argument is *content*: unwrap, keep the argument
UNWRAP = [
    r"\emph", r"\textit", r"\textbf", r"\mathrm", r"\operatorname", r"\norm",
    r"\numv", r"\SI", r"\text",
]
# commands whose braced argument is *metadata*: drop entirely
DROP = [r"\cite", r"\label", r"\ref", r"\eqref", r"\bibliography", r"\input"]


def strip_latex(text):
    """Return (words_no_math, words_with_math) for one abstract body."""
    # 1. comments
    text = re.sub(r"(?<!\\)%.*", "", text)
    # 2. metadata commands -> empty
    for cmd in DROP:
        text = re.sub(re.escape(cmd) + r"\s*\{[^{}]*\}", " ", text)
    # 3. display math -> one token
    text = re.sub(r"\\\[.*?\\\]", " MATHTOK ", text, flags=re.S)
    # 4. inline math -> one token
    text = re.sub(r"\$[^$]*\$", " MATHTOK ", text)
    # 5. unwrap content commands (repeat: nesting is shallow but real)
    pat = "|".join(re.escape(c) + r"\s*\{" for c in UNWRAP)
    for _ in range(6):
        new = re.sub(r"(" + pat + r")", " ", text)
        if new == text:
            break
        text = new
    # 6. remaining control sequences -> space
    text = re.sub(r"\\[A-Za-z@]+\s*", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("~", " ").replace("---", " ").replace("--", " ")
    # 7. count
    toks = text.split()
    with_math = [t for t in toks if any(ch.isalpha() for ch in t) or t == "MATHTOK"]
    no_math = [t for t in toks if t != "MATHTOK" and any(ch.isalpha() for ch in t)]
    return len(no_math), len(with_math)


def selftest():
    """Falsifiability probe: the counter must be right on known inputs."""
    body = " ".join(["alpha"] * 40)
    n, m = strip_latex(body)
    ok = (n == 40 and m == 40)
    print("  selftest 1 (40 plain words)          -> %d / %d  %s" % (n, m, "PASS" if ok else "FAIL"))

    body2 = " ".join(["alpha"] * 40) + " $x^2+y$ " + " ".join(["beta"] * 10)
    n2, m2 = strip_latex(body2)
    ok2 = (n2 == 50 and m2 == 51)
    print("  selftest 2 (50 words + 1 formula)    -> %d / %d  %s" % (n2, m2, "PASS" if ok2 else "FAIL"))

    body3 = "word " * 300
    n3, _ = strip_latex(body3)
    ok3 = (n3 == 300)
    print("  selftest 3 (300 words, must exceed)  -> %d  %s" % (n3, "PASS" if ok3 else "FAIL"))

    body4 = r"see \cite{foo} and \eqref{eq:bar} here"
    n4, _ = strip_latex(body4)
    ok4 = (n4 == 2)  # "see", "and", "here" -> 3? no: see/and/here = 3
    ok4 = (n4 == 3)
    print("  selftest 4 (drop cites/refs)         -> %d  %s" % (n4, "PASS" if ok4 else "FAIL"))

    good = ok and ok2 and ok3 and ok4
    print("  SELFTEST %s" % ("PASS" if good else "FAIL"))
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="?", default=os.path.join(ROOT, "paper_en", "main.tex"))
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    src = open(a.tex, encoding="utf-8").read()
    ms = ABSTRACT_RE.findall(src)
    if len(ms) != 1:
        print("[fatal] expected exactly 1 abstract environment, found %d" % len(ms))
        sys.exit(2)
    n, m = strip_latex(ms[0])
    print("%s" % a.tex)
    print("  words (formulae removed)      = %d   limit %d" % (n, a.limit))
    print("  words (one token per formula) = %d" % m)
    ok = n <= a.limit
    print("  %s" % ("PASS" if ok else "FAIL: %d words, limit %d" % (n, a.limit)))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
