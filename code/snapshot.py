# -*- coding: utf-8 -*-
"""snapshot.py — freeze an immutable manuscript snapshot for a review round.

Discipline (Dr Wang's standing requirement): every revision round gets its own
directory; nothing is ever overwritten in place.  This script copies, for both the
English and the Chinese manuscript, the LaTeX source, the generated numbers.tex and
the compiled PDF, plus the result JSONs that produced the numbers, and writes an md5
manifest so a later round can be diffed or rolled back.

Usage:  python snapshot.py round_r1
"""
import hashlib, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

# Frozen artefacts.  The result JSONs, the figures and the code are enumerated dynamically:
# hard-coding the list is exactly how ngc.json and L45.json came to be missing from the
# round-1 snapshot while a dozen macros in numbers.tex sourced their values from them
# (review R2 m2.10).
def _build_src():
    items = [
        ("paper_en/main.tex", "paper_en/main.tex"),
        ("paper_en/numbers.tex", "paper_en/numbers.tex"),
        ("paper_en/main.pdf", "paper_en/main.pdf"),
        ("paper_en/main.log", "paper_en/main.log"),
        ("paper_cn/main.tex", "paper_cn/main.tex"),
        ("paper_cn/numbers.tex", "paper_cn/numbers.tex"),
        ("paper_cn/main.pdf", "paper_cn/main.pdf"),
        ("paper_cn/main.log", "paper_cn/main.log"),
    ]
    res = os.path.join(ROOT, "results")
    if os.path.isdir(res):
        for f in sorted(os.listdir(res)):
            if f.endswith(".json") or f.endswith("_progress.txt"):
                items.append(("results/" + f, "results/" + f))
    figs = os.path.join(ROOT, "figures")
    if os.path.isdir(figs):
        for f in sorted(os.listdir(figs)):
            if f.endswith(".pdf"):
                items.append(("figures/" + f, "figures/" + f))
    return items


SRC = _build_src()



def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2:
        print("usage: python snapshot.py <round_dir_name>"); sys.exit(2)
    rnd = sys.argv[1]
    dst = os.path.join(ROOT, "reviews", rnd, "snapshot")
    os.makedirs(dst, exist_ok=True)
    lines, missing, std = [], [], []
    for src_rel, dst_rel in SRC:
        s = os.path.join(ROOT, src_rel)
        if not os.path.exists(s):
            missing.append(src_rel)
            continue
        d = os.path.join(dst, dst_rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        lines.append("%s  %10d  %s" % (md5(d), os.path.getsize(d), dst_rel))
        std.append("%s  %s" % (md5(d), dst_rel.replace(os.sep, "/")))
    # also freeze the code that produced the numbers (every script, including the audits
    # and the patch scripts, so a round can be re-derived end to end)
    code_files = []
    for f in sorted(os.listdir(HERE)):
        if not f.endswith(".py") or f.endswith(".bak_beforefix"):
            continue
        if f == "snapshot.py":
            continue
        s = os.path.join(HERE, f)
        if not os.path.exists(s):
            missing.append("code/" + f)
            continue
        d = os.path.join(dst, "code", f)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        code_files.append(f)
        lines.append("%s  %10d  %s" % (md5(d), os.path.getsize(d), "code/" + f))
        std.append("%s  %s" % (md5(d), "code/" + f))

    # MANIFEST.md5 is written in the *standard* `md5sum -c` format -- "<md5>  <path>", one
    # space-separated pair per line, LF endings, no size column and no comment lines -- so a
    # reviewer can verify the whole snapshot with a single `md5sum -c MANIFEST.md5` from the
    # snapshot directory.  R3 m3.16: the previous format carried a size column and CRLF, and
    # `md5sum -c` reported "listed files could not be read" even though every hash matched.
    # Human-readable provenance lives in MANIFEST.info instead.
    man = os.path.join(dst, "MANIFEST.md5")
    st = sorted(std)
    with open(man, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(st) + "\n")
    with open(os.path.join(dst, "MANIFEST.info"), "w", encoding="utf-8", newline="\n") as f:
        f.write("frozen  : %s\nround   : %s\nfiles   : %d\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), rnd, len(st)))
        f.write("format  : standard md5sum -c (\"<md5>  <path>\", LF, sorted); size column dropped\n")
        f.write("verify  : cd <snapshot> && md5sum -c MANIFEST.md5\n")
        f.write("note    : main.log is included so gate check G6 can be re-run on this frozen tree\n")
        # R2 m3.10 asked that a reader be able to tell the sealed drivers from the single-use
        # repair/probe scripts.  The files are NOT moved into separate directories -- the frozen
        # manifest records paths, so a move would invalidate every checksum in it -- but the two
        # classes are listed separately here, under the naming convention the manuscript's
        # Data and Code Availability section states: drivers carry no leading underscore,
        # single-use scripts do.
        _drv = sorted(f for f in code_files if not f.startswith("_"))
        _one = sorted(f for f in code_files if f.startswith("_"))
        f.write("\nsealed drivers (%d, no leading underscore):\n  %s\n"
                % (len(_drv), "\n  ".join(_drv)))
        f.write("\nsingle-use repair/probe scripts (%d, leading underscore):\n  %s\n"
                % (len(_one), "\n  ".join(_one)))
        f.write("\nsize  path\n")
        f.write("\n".join(sorted(lines)))
        f.write("\n")
    with open(os.path.join(dst, "MISSING.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(missing) + ("\n" if missing else ""))
    print("snapshot -> %s" % dst)
    print("  frozen files : %d" % len(lines))
    if missing:
        print("  MISSING      : %s" % ", ".join(missing))
    print("  manifest     : %s" % man)


if __name__ == "__main__":
    main()
