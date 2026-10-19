# -*- coding: utf-8 -*-
"""Remove now-dead numeric macros from gen_numbers.py (no longer referenced by either
manuscript after the Tab. cert rework). Guarded: each line must exist exactly once."""
import io, re

P = r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\code\gen_numbers.py"
s = io.open(P, encoding="utf-8").read()

DEAD = ["NumFidRadius", "NumRreConsExt", "NumPhysChiBlock", "NumUncTwo",
        "LipHUnc", "NumBetaConsNoProj",
        "NumRepUncOffFinite", "NumRepUncOffRre", "NumRepUncOnFinite", "NumRepUncOnRre"]

removed = []
for k in DEAD:
    pat = re.compile(r"^[ \t]*\"" + k + r"\":.*\n", re.M)
    m = pat.findall(s)
    if not m:
        print("  skip (not found as macro line): %s" % k)
        continue
    assert len(m) == 1, "%s appears %d times" % (k, len(m))
    s = pat.sub("", s)
    removed.append(k)

# also drop them from ALL_KEYS / indep-dict listings if present
for k in DEAD:
    for pat in (r"^[ \t]*\"" + k + r"\",[ \t]*\n", r"^[ \t]*" + k + r"=[^,\n]*,[ \t]*\n"):
        s = re.sub(pat, "", s, flags=re.M)

io.open(P, "w", encoding="utf-8").write(s)
print("removed %d dead macros: %s" % (len(removed), removed))
