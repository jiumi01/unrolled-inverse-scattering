"""check_refs.py -- cross-check citations against the bibliography.

Checks, in this order:
  1. every \\cite/\\citep/\\citet key resolves to a \\bibitem;
  2. every \\bibitem is cited at least once (no uncited entries);
  3. each entry carries a year, a venue (journal / booktitle / publisher / arXiv id)
     and either a volume+pages pair or a DOI -- reported as WARN, not as an error,
     because a few legacy entries in this manuscript are book chapters;
  4. the printed order in thebibliography matches the order of first citation.

Usage:  python code/check_refs.py [path/to/main.tex]
Exit code 0 when there is no FAIL.
"""
import io
import re
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "cmes_submission/main.tex"
src = io.open(P, encoding="utf8", newline="").read()

body, _, biblio = src.partition(r"\begin{thebibliography}")
if not biblio:
    print("FAIL  no thebibliography environment")
    sys.exit(1)

cite_pat = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
order = []          # keys in order of first citation
for m in cite_pat.finditer(body):
    for k in m.group(1).split(","):
        k = k.strip()
        if k and k not in order:
            order.append(k)

bib_keys = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", biblio)
cin = [k for k in order if k not in bib_keys]
unc = [k for k in bib_keys if k not in order]
dups = [k for k in set(bib_keys) if bib_keys.count(k) > 1]

print("citations (unique keys) : %d" % len(order))
print("bibliography entries    : %d" % len(bib_keys))
print("cited but undefined     : %s" % (cin or "none"))
print("defined but uncited     : %s" % (unc or "none"))
print("duplicate bibitem keys  : %s" % (dups or "none"))

# per-entry field audit
entries = re.split(r"\\bibitem(?:\[[^\]]*\])?\{", biblio)[1:]
warn = []
for e in entries:
    key = e.split("}", 1)[0]
    txt = e.split("}", 1)[1]
    txt = txt.split("\\bibitem")[0]
    has_year = bool(re.search(r"\b(19|20)\d\d\b", txt))
    # This manuscript uses the compact "Authors. Title. JournalAbbrev. YYYY;vol(issue):pp."
    # style, so the venue is what sits between the title and the year field.
    has_venue = bool(re.search(r"\.\s+[A-Za-z][^.;]{1,60}\.\s*(19|20)\d\d\s*;", txt)
                     or re.search(r"In:\s*[^;]{2,80};\s*(19|20)\d\d", txt)
                     or re.search(r"arXiv", txt) or re.search(r"Press", txt))
    has_vol = bool(re.search(r"(19|20)\d\d\s*;[^.]*\d", txt)
                   or re.search(r"\bp\.\s*\d", txt))
    has_doi = "doi" in txt.lower()
    miss = []
    if not has_year:
        miss.append("year")
    if not has_venue:
        miss.append("venue")
    if not (has_vol or has_doi):
        miss.append("volume/pages or DOI")
    if miss:
        warn.append((key, ", ".join(miss)))

print("\nfield audit (WARN only):")
if not warn:
    print("  all %d entries carry year + venue + volume/pages or DOI" % len(bib_keys))
for k, m in warn:
    print("  %-16s missing: %s" % (k, m))

print("\nfirst-citation order vs printed order:")
mis = [(i, k) for i, k in enumerate(order) if k in bib_keys and bib_keys.index(k) != i]
if mis:
    for i, k in mis[:10]:
        print("  pos %2d: cited %-16s printed at %2d" % (i + 1, k, bib_keys.index(k) + 1))
else:
    print("  identical")

fail = bool(cin or dups)
print("\nVERDICT:", "FAIL" if fail else "PASS")
sys.exit(1 if fail else 0)
