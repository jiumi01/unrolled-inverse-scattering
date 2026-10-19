# -*- coding: utf-8 -*-
"""build_tex.py <texfile> [pdflatex|xelatex] — compile with MiKTeX missing-package auto-install."""
import subprocess, sys, re, os

BINDIR = r"C:\Users\up\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding='utf-8', errors='ignore')
    return r

def main():
    tex = os.path.abspath(sys.argv[1])
    engine = sys.argv[2] if len(sys.argv) > 2 else "pdflatex"
    d = os.path.dirname(tex)
    base = os.path.splitext(os.path.basename(tex))[0]
    for i in range(25):
        r = run([os.path.join(BINDIR, engine + ".exe"), "-interaction=nonstopmode",
                 os.path.basename(tex)], d)
        logp = os.path.join(d, base + ".log")
        log = open(logp, encoding='utf-8', errors='ignore').read() if os.path.exists(logp) else ""
        missing = sorted(set(re.findall(r"File `([^']+?\.sty)' not found", log)) |
                         set(re.findall(r"! LaTeX Error: File `([^']+?\.sty)' not found", log)))
        cls_missing = sorted(set(re.findall(r"File `([^']+?\.cls)' not found", log)))
        pkgs = [m[:-4] for m in missing + cls_missing]
        if pkgs:
            print("missing packages:", pkgs)
            for p in pkgs:
                run([os.path.join(BINDIR, "miktex.exe"), "packages", "install", p], d)
            continue
        # errors?
        errs = re.findall(r"^! (.*)$", log, re.M)
        print(f"--- pass {i+1}: errors={len(errs)}")
        for e in errs[:8]:
            print("   !", e)
        if errs:
            sys.exit(2)
        break
    # two more passes for cross-references
    for _ in range(2):
        run([os.path.join(BINDIR, engine + ".exe"), "-interaction=nonstopmode",
             os.path.basename(tex)], d)
    pdf = os.path.join(d, base + ".pdf")
    print("PDF:", pdf, os.path.exists(pdf))
    # undefined refs
    log = open(os.path.join(d, base + ".log"), encoding='utf-8', errors='ignore').read()
    undef = re.findall(r"Reference `([^']+)' on page", log)
    if undef:
        print("UNDEFINED REFS:", sorted(set(undef)))

if __name__ == "__main__":
    main()
