# -*- coding: utf-8 -*-
"""临时：按“最近引用页”复核 float 位置（判据 |Δ|<=2）。跑完即删。"""
import re, subprocess, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
# 传参则按 pdf 所在目录取 aux（支持任意树）；不传则默认 cmes_submission。
pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "cmes_submission", "main.pdf")
if not os.path.isabs(pdf):
    pdf = os.path.abspath(pdf)
assert os.path.exists(pdf), "pdf not found: %s" % pdf
aux = open(os.path.join(os.path.dirname(pdf), "main.aux"), "r",
           encoding="utf8", errors="ignore").read()

def grab(s, i):
    """s[i] == '{' ；返回 (内容, 结束后下标)，支持嵌套花括号。"""
    assert s[i] == "{"
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced")

lab = {}
for m in re.finditer(r"\\newlabel\{", aux):
    i = m.end() - 1
    key, i = grab(aux, i)
    while aux[i] in " \n":
        i += 1
    body, _ = grab(aux, i)          # {{num}{page}...}
    j = 0
    while body[j] != "{":
        j += 1
    num, j = grab(body, j)
    while body[j] in " \n":
        j += 1
    pg, _ = grab(body, j)
    pg = re.sub(r"[^0-9]", "", pg) or "0"
    lab[key] = (num.strip(), int(pg))


txt = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True,
                     encoding="utf8", errors="ignore").stdout
pages = txt.split("\f")

def refs(num, kind):
    out = []
    kw = "Fig(?:ure)?" if kind == "Fig" else "Table"
    pat = re.compile(r"\b%s\.?\s*~?\s*%s\b" % (kw, re.escape(num)))
    cap = re.compile(r"^\s*%s\s*%s[:.]" % (kw, re.escape(num)))
    for i, p in enumerate(pages, 1):
        for line in p.splitlines():
            if cap.match(line):
                continue
            if pat.search(line):
                out.append(i)
                break
    return out

rows = []
for k, (num, fpage) in sorted(lab.items(), key=lambda kv: kv[1][1]):
    if "@cref" in k:
        continue
    if k.startswith("fig:"):
        kind = "Fig"
    elif k.startswith("tab:"):
        kind = "Table"
    else:
        continue
    r = refs(num, kind)
    d = min((abs(fpage - x) for x in r), default=None)
    rows.append((kind, num, fpage, r[:4], d))

print("floats: %d" % len(rows))
bad = [r for r in rows if r[4] is None or r[4] > 2]
for r in rows:
    flag = "  " if (r[4] is not None and r[4] <= 2) else "!!"
    print("%s %-5s %-3s p%-3d refs=%-14s d=%s" % (flag, r[0], r[1], r[2], str(r[3]), r[4]))
print("beyond 2 pages: %s" % ("none" if not bad else [ (b[0]+b[1]) for b in bad]))
