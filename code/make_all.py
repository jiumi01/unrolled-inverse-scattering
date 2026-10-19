# -*- coding: utf-8 -*-
"""make_all.py — one-shot post-processing: figures -> numbers -> compile both manuscripts.

Run with the managed interpreter (needs numpy/matplotlib):
  conda run -n wirenet python -u make_all.py
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
MINE = sys.executable


def step(msg):
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


def run(cmd, **kw):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="ignore")
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join([l for l in out.splitlines() if "CONDA_NO_PLUGINS" not in l
                      and "Alternatively" not in l and "conda install" not in l
                      and "Example:" not in l])
    print(tail[-3000:])
    return r.returncode


def main():
    step("[1/3] figures")
    run([MINE, os.path.join(HERE, "make_figs.py")])

    step("[2/3] numbers.tex")
    run([MINE, os.path.join(HERE, "gen_numbers.py")])

    step("[3/3] compile")
    rc_en = run([MINE, os.path.join(HERE, "build_tex.py"),
                 os.path.join("paper_en", "main.tex"), "pdflatex"])
    rc_cn = run([MINE, os.path.join(HERE, "build_tex.py"),
                 os.path.join("paper_cn", "main.tex"), "xelatex"])
    pdf_en = os.path.join(ROOT, "paper_en", "main.pdf")
    pdf_cn = os.path.join(ROOT, "paper_cn", "main.pdf")
    print("\nEN pdf:", os.path.exists(pdf_en), pdf_en)
    print("CN pdf:", os.path.exists(pdf_cn), pdf_cn)
    print("rc:", rc_en, rc_cn)
    if rc_en or rc_cn:
        sys.exit(2)


if __name__ == "__main__":
    main()
