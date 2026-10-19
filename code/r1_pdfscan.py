import sys
try:
    from pypdf import PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        print("NO_PDF_LIB", e); sys.exit(0)
r = PdfReader("reviews/round_r3/snapshot/paper_en/main.pdf")
txt = "\n".join((p.extract_text() or "") for p in r.pages)
for kw in ["0.285", "1.053", "0.3192", "1.743", "13.032", "13.682", "14.419",
           "14.594", "2.793", "0.9152", "0.8303", "2.240", "0.3169", "3.155"]:
    print(kw, "->", txt.count(kw))
i = txt.find("1.053")
print("---context---")
print(txt[i - 400:i + 200] if i >= 0 else "1.053 not found")
print("dashdash count:", txt.count("--"))
