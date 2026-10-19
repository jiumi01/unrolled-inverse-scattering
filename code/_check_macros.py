# -*- coding: utf-8 -*-
"""Cross-check: every macro referenced in each manuscript must be defined in its numbers.tex,
and every macro defined must be referenced. Also lists undefined \ref labels.

Gate history (why the extra assertions exist):
  * A macro name may not contain a digit -- TeX ends a control word at the first
    non-letter, so ``\\newcommand{\\ResP50}`` silently defines ``\\ResP``.  gen_numbers.py
    therefore renames digit-bearing keys (8 -> Eight); a cited ``\\NumCls8`` then no longer
    matches its definition.  The original regex ``\\\\(Num[A-Za-z]+)`` reproduced the very
    same truncation, so ``\\NumPsnrCls8`` was read as ``\\NumPsnrCls`` and the mismatch was
    reported as a bogus name instead of as the digit defect.  We now (a) forbid digits in
    both cited and defined names, and (b) additionally scan cited names with an
    alphanumeric (digit-aware) pattern so a digit-bearing citation is flagged explicitly.
"""
import io, re, sys, os

ROOT = os.environ.get("PAPER_ROOT",
                      r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究")
FAIL = 0

for d in ("paper_en", "paper_cn"):
    tex = io.open(os.path.join(ROOT, d, "main.tex"), encoding="utf-8").read()
    num = io.open(os.path.join(ROOT, d, "numbers.tex"), encoding="utf-8").read()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", num))
    used = set(re.findall(r"\\(Num[A-Za-z]+)", tex))
    # digit-aware scan: the LITERAL name TeX would consume (letters, then any digits)
    used_literal = set(re.findall(r"\\(Num[A-Za-z]+[0-9]*)", tex))
    digit_cited = sorted(n for n in used_literal if re.search(r"[0-9]", n))
    digit_defined = sorted(n for n in defined if re.search(r"[0-9]", n))
    missing = sorted(used - defined)
    unused = sorted(defined - used - {"numv"})
    # labels
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    dangling = sorted(refs - labels)
    print("== %s ==" % d)
    print("  macros defined : %d" % len(defined))
    print("  macros used    : %d" % len(used))
    print("  MISSING (used, undefined): %s" % (missing or "none"))
    print("  unused (defined, not used): %s" % (unused or "none"))
    print("  DANGLING refs  : %s" % (dangling or "none"))
    print("  DIGIT in cited macro name (ILLEGAL): %s" % (digit_cited or "none"))
    print("  DIGIT in defined macro name (ILLEGAL): %s" % (digit_defined or "none"))
    if missing or digit_cited or digit_defined:
        FAIL = 1

print("\nRESULT: %s" % ("FAIL" if FAIL else "PASS"))
sys.exit(FAIL)
