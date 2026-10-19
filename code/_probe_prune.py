import re, io, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = ""
for paper in ("paper_en", "paper_cn"):
    fp = os.path.normpath(os.path.join(HERE, "..", paper, "main.tex"))
    if os.path.exists(fp):
        src += io.open(fp, encoding="utf-8").read()

# Replicate the exact prune regex from gen_numbers.py
for k in ["NumFresnelGb", "NumFresnelSigmaLBSynth", "NumFresnelKb", "NumFresnelBeta"]:
    pat = r"\\%s(?![A-Za-z])" % k
    m = re.search(pat, src)
    print(f"{k:26s} regex_match={bool(m)}  group={m.group() if m else ''!r}")

# Also list every occurrence of the bare token in source with the char before it
print("--- occurrences of 'NumFresnelGb' with preceding char ---")
for k in ["NumFresnelGb", "NumFresnelSigmaLBSynth"]:
    for mm in re.finditer(re.escape(k), src):
        pre = src[mm.start()-1] if mm.start()>0 else "<start>"
        print(f"  {k}: pre={pre!r} at pos {mm.start()}  ctx={src[max(0,mm.start()-40):mm.start()+15]!r}")

