# -*- coding: utf-8 -*-
"""_check_placeholders.py — gate: no numeric macro may render as a placeholder.

Scans both manuscripts for every macro defined in numbers.tex, resolves each one, and
reports (a) macros whose value is still the missing-data placeholder '--', (b) macros
that are defined but never cited (dead), and (c) macros cited but not defined (which
would be an undefined control sequence, i.e. a compile error).

Exit code 1 if any cited macro renders as '--'.
"""
import io, os, re, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
USE = re.compile(r"\\([A-Za-z]{3,})")


def defines(path):
    d = {}
    for line in io.open(path, encoding="utf-8"):
        m = DEF.match(line.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


def main():
    bad = 0
    for paper in ("paper_en", "paper_cn"):
        nums = defines(os.path.join(ROOT, paper, "numbers.tex"))
        src = io.open(os.path.join(ROOT, paper, "main.tex"), encoding="utf-8").read()
        # drop the bibliography so that author initials are not counted as macros
        body = src.split(r"\begin{thebibliography}")[0]
        used = set(USE.findall(body))
        missing = sorted(m for m in used if m in nums and nums[m].strip() in ("--",))
        dead = sorted(m for m in nums if m not in used)
        unknown = sorted(m for m in used
                         if m.startswith(("Num", "Numu", "efe"))
                         and m not in nums)
        print("== %s ==" % paper)
        print("  cited macros rendering as '--': %d" % len(missing))
        for m in missing:
            print("     - \\%s" % m)
        print("  defined but never cited (dead): %d" % len(dead))
        for m in dead:
            print("     ~ \\%s" % m)
        if unknown:
            print("  cited, not defined (compile error): %s" % unknown)
        bad += len(missing) + len(unknown)
    print("\nplaceholders+unknown =", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
