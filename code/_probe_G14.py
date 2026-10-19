# -*- coding: utf-8 -*-
"""_probe_G14.py -- falsification probe for gate G14 (cover-letter word count).

Discipline: a check that has only ever been seen to PASS is not evidence.  R3 made the same
point about G9 in its round-3 report: the first perturbation there hit docstring text, G9
stayed green, and the perturbation had quietly not taken effect.  So this probe runs G14 on
three inputs and demands the *right* verdict from each:

  1. the shipped letter                              -> must not FAIL
  2. a copy whose footer numbers are corrupted        -> must FAIL
  3. a copy whose footer is deleted                   -> must FAIL

Injections are made into a temporary copy and pointed at through WB_COVER_LETTER, so the
shipped deliverable is never written to; the probe also md5-checks the real letter before and
after to prove that.
"""
import hashlib, io, os, re, subprocess, sys, tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LETTER = os.path.join(ROOT, "docs", "03_投稿信.md")
GATE = os.path.join(ROOT, "code", "_gate_deliverable.py")


def md5(p):
    return hashlib.md5(io.open(p, "rb").read()).hexdigest()


def g14_verdict(letter_path=None):
    env = dict(os.environ)
    if letter_path:
        env["WB_COVER_LETTER"] = letter_path
    p = subprocess.run([sys.executable, GATE], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    lines = [l for l in out.splitlines() if "cover letter" in l or "G14" in l]
    failed = any("[FAIL]" in l for l in lines)
    return failed, lines


def main():
    before = md5(LETTER)
    src = io.open(LETTER, encoding="utf-8").read()
    results, ok = [], True
    tmpdir = tempfile.mkdtemp(prefix="g14probe_")

    f_ok, lines = g14_verdict(None)
    results.append(("shipped letter", "not-FAIL", "FAIL" if f_ok else "not-FAIL", not f_ok))
    for l in lines:
        print("   |", l.strip())

    corrupt = re.sub(r"abstract_words=\d+", "abstract_words=1", src)
    p1 = os.path.join(tmpdir, "corrupt.md")
    io.open(p1, "w", encoding="utf-8", newline="\n").write(corrupt)
    f1, lines1 = g14_verdict(p1)
    results.append(("corrupted footer (A=1)", "FAIL", "FAIL" if f1 else "not-FAIL", f1))
    for l in lines1:
        print("   |", l.strip())

    p2 = os.path.join(tmpdir, "nofooter.md")
    io.open(p2, "w", encoding="utf-8", newline="\n").write(
        re.sub(r"<!--\s*check:.*?-->", "", src, flags=re.S))
    f2, lines2 = g14_verdict(p2)
    results.append(("footer deleted", "FAIL", "FAIL" if f2 else "not-FAIL", f2))
    for l in lines2:
        print("   |", l.strip())

    after = md5(LETTER)
    print("\n=== probe report ===")
    for name, want, got, good in results:
        print("  %-26s want %-9s got %-9s %s" % (name, want, got, "PASS" if good else "**FAIL**"))
        ok = ok and good
    print("  %-26s %s" % ("shipped letter unmodified", "PASS" if before == after
                          else "**FAIL** %s -> %s" % (before, after)))
    ok = ok and before == after
    print("\nPROBE %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
