# -*- coding: utf-8 -*-
"""Falsifiability probe for _check_macros.py (deliverable-gate discipline: an
instrument that has never been shown to fail on a known-bad input is not evidence).

Builds a throwaway copy of both manuscripts with a digit-bearing macro citation
re-injected (the exact defect that motivated the digit assertions) and asserts the
gate reports FAIL.  Also asserts the gate PASSES on the unmodified copies, so a
gate that always fails is caught too.
"""
import io
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
PROBE = os.path.join(os.environ.get("TEMP", "/tmp"), "gateprobe")


def run_gate(root):
    env = dict(os.environ, PAPER_ROOT=root, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, os.path.join(HERE, "_check_macros.py")],
                       capture_output=True, text=True, env=env, encoding="utf-8")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def stage():
    if os.path.isdir(PROBE):
        shutil.rmtree(PROBE)
    for d in ("paper_en", "paper_cn"):
        os.makedirs(os.path.join(PROBE, d))
        for f in ("main.tex", "numbers.tex"):
            shutil.copy(os.path.join(ROOT, d, f), os.path.join(PROBE, d, f))


def main():
    ok = True
    stage()

    # (1) clean copy must PASS -- otherwise the probe proves nothing
    rc, out = run_gate(PROBE)
    print("clean copy      : exit=%d  (expect 0)" % rc)
    ok &= (rc == 0)

    # (2) injection A: digit-bearing CITATION
    p = os.path.join(PROBE, "paper_en", "main.tex")
    s = io.open(p, encoding="utf-8").read()
    good = "\\numv{\\NumPsnrClsEight}"
    assert s.count(good) == 1, "anchor not found (%d)" % s.count(good)
    io.open(p, "w", encoding="utf-8").write(s.replace(good, "\\numv{\\NumPsnrCls8}"))
    rc, out = run_gate(PROBE)
    hit = "DIGIT in cited macro name" in out and "NumPsnrCls8" in out
    print("digit CITATION  : exit=%d  flagged=%s  (expect exit=1, flagged=True)" % (rc, hit))
    ok &= (rc == 1 and hit)

    # (3) injection B: digit-bearing DEFINITION in numbers.tex only
    stage()
    pn = os.path.join(PROBE, "paper_en", "numbers.tex")
    sn = io.open(pn, encoding="utf-8").read()
    assert sn.count("\\NumPsnrClsEight") == 1
    io.open(pn, "w", encoding="utf-8").write(
        sn.replace("\\NumPsnrClsEight", "\\NumPsnrCls8"))
    rc, out = run_gate(PROBE)
    hit = "DIGIT in defined macro name" in out
    print("digit DEFINITION: exit=%d  flagged=%s  (expect exit=1, flagged=True)" % (rc, hit))
    ok &= (rc == 1 and hit)

    shutil.rmtree(PROBE, ignore_errors=True)
    print("\nPROBE: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
