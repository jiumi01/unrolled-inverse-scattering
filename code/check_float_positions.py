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
# aux 必须与 pdf 同名（main_cn.pdf -> main_cn.aux）；曾写死 main.aux，
# 对中文稿会读英文 aux，页码全错且 refs 全空（假红）。
auxname = os.path.splitext(os.path.basename(pdf))[0] + ".aux"
auxpath = os.path.join(os.path.dirname(pdf), auxname)
assert os.path.exists(auxpath), "aux not found: %s" % auxpath
aux = open(auxpath, "r", encoding="utf8", errors="ignore").read()

# 标签前缀：中文稿用 图/表（本稿已 renew \figurename/\tablename）。
# 由文件名判定，或第三个参数显式给 cn/en。
lang = sys.argv[2] if len(sys.argv) > 2 else ("cn" if "_cn" in os.path.basename(pdf) else "en")
KW = {"Fig": "图" if lang == "cn" else "Fig(?:ure)?",
      "Table": "表" if lang == "cn" else "Table"}


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
    if "@cref" in key:      # cleveref 的重写条目：{num}{num}...，页码位不可用
        continue
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
npage = len([p for p in pages if p.strip()])
# aux 与 pdf 必须同源：aux 里出现超出 pdf 总页数的页码，说明读错了 aux。
over = [(k, v[1]) for k, v in lab.items() if v[1] > npage]
assert not over, "aux/pdf mismatch: labels beyond page %d -> %s" % (npage, over[:5])

def refs(num, kind):
    out = []
    kw = KW[kind]
    if lang == "cn":
        # CJK 是 \w，前后加 \b 会失效（"图~5与" 里 5\b 不成立），改用数字负前瞻。
        pat = re.compile(r"%s\s*~?\s*%s(?!\d)" % (kw, re.escape(num)))
    else:
        pat = re.compile(r"\b%s\.?\s*~?\s*%s\b" % (kw, re.escape(num)))
    cap = re.compile(r"^\s*%s\s*%s[:.．]" % (kw, re.escape(num)))
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
