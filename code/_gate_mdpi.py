# -*- coding: utf-8 -*-
"""_gate_mdpi.py — single shutter for the MDPI (Mathematics) submission tree.

This is the MDPI-scoped sibling of _gate_deliverable.py, which guards the
paper_en / paper_cn (AIMS / IEEE-TAP) pair.  It reuses the SAME check classes
but is scoped to paper_mdpi alone and drops the cross-manuscript and
IEEE-specific checks:

  dropped  G4  en/cn bibliography-parity   -> replaced by a single-paper
                                                completeness check (G4m)
  dropped  G5  en/cn structural-parity     -> N/A (only one MDPI manuscript)
  dropped  G13 round response-letter        -> N/A (round-specific; the MDPI
                                                snapshot has no response.md yet)
  dropped  G14 IEEE cover-letter word-count -> N/A (MDPI needs its own letter)
  changed  G10 abstract length              -> MDPI "around 200 words" dual
                                                convention (NOT the IEEE 250)

Kept verbatim from the canonical gate: G1 (macro resolution), G2/G3 (macro
hygiene), G6 (build cleanliness), G7 (artefact freshness), G8 (macro-name
legality + cross-ref integrity + no orphan supplementary), G9 (artefact
metadata vs driver), G11 (prose duplication), G12 (ambiguity ledger).

G7 numbers.tex freshness: paper_mdpi/numbers.tex is a *copy* of the canonical
paper_en/numbers.tex.  Two outcomes are both acceptable: (a) the two files are
byte-identical, in which case the MDPI copy inherits paper_en's already-verified
freshness; or (b) they differ, in which case the MDPI copy itself must be newer
than every results/*.json.  Either way the gate refuses a stale numeric source.
"""
import io, os, re, sys
from collections import Counter
import importlib.util

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAPER = "paper_mdpi"
DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
USE = re.compile(r"\\([A-Za-z]{3,})")

fails = []
warns = []


def report(ok, tag, detail=""):
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", tag, detail))
    if not ok:
        fails.append(tag)


def warn(ok, tag, detail=""):
    if ok:
        print("  [PASS] %-46s %s" % (tag, detail))
    else:
        print("  [WARN] %-46s %s" % (tag, detail))
        warns.append(tag)


def defines(path):
    d = {}
    for line in io.open(path, encoding="utf-8"):
        m = DEF.match(line.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


def driver_str_const(path, name):
    import ast
    try:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
    except SyntaxError as e:
        return None, "cannot parse: %s" % e
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    try:
                        return ast.literal_eval(node.value), None
                    except ValueError as e:
                        return None, "not a literal: %s" % e
    return None, "constant %s not found" % name


def main():
    # ---- macro resolution / hygiene (G1, G2, G3) ------------------------
    nums = defines(os.path.join(ROOT, PAPER, "numbers.tex"))
    src = io.open(os.path.join(ROOT, PAPER, "main.tex"), encoding="utf-8").read()
    body = src.split(r"\begin{thebibliography}")[0]
    used = set(USE.findall(body))
    macros_bad = sorted(m for m in used if m in nums and nums[m].strip() == "--")
    # DEAD means "defined in numbers.tex and cited by NO manuscript".  The earlier rule
    # tested the MDPI tree alone, which was self-consistent only while numbers.tex was
    # pruned to MDPI's usage.  gen_numbers prunes to the UNION of the three trees (it must:
    # paper_en and paper_cn are shipped counterparts that cite macros MDPI does not), so the
    # single-tree rule reported every EN/CN-only macro as a failure -- a gate defect, not a
    # manuscript defect.  Per-tree orphan counts are still printed below, as information.
    _all_src = ""
    for _pp in ("paper_en", "paper_cn", "paper_mdpi"):
        _fp = os.path.join(ROOT, _pp, "main.tex")
        if os.path.exists(_fp):
            _all_src += io.open(_fp, encoding="utf-8").read()
    macros_dead = sorted(m for m in nums
                         if not re.search(r"\\%s(?![A-Za-z])" % m, _all_src))
    orphans_here = sorted(m for m in nums
                          if not re.search(r"\\%s(?![A-Za-z])" % m, src))
    macros_undef = sorted(m for m in used
                          if m.startswith("Num") and m not in nums)

    print("G1 numeric macros resolve (no '--')")
    report(not macros_bad, "%s placeholders" % PAPER,
           "%d %s" % (len(macros_bad), macros_bad[:6]))
    print("G2/G3 macro hygiene")
    report(not macros_dead, "%s dead macros (cited by no manuscript)" % PAPER,
           str(macros_dead))
    print("  [info] %d macro(s) cited by the other trees but not by %s: %s"
          % (len(orphans_here), PAPER, orphans_here[:8]))
    report(not macros_undef, "%s undefined macros" % PAPER, str(macros_undef))

    # ---- G4m: single-paper bibliography completeness --------------------
    print("G4m bibliography completeness (single paper)")
    s = src
    bib_body = s.split(r"\begin{thebibliography}")[1]
    labs = re.findall(r"\\bibitem\{([^}]*)\}", bib_body)
    cited = set()
    for m in re.findall(r"\\cite[a-z]*\{([^}]*)\}", s.split(r"\begin{thebibliography}")[0]):
        cited.update(x.strip() for x in m.split(","))
    report(not (set(labs) - cited), "no bib entry defined but uncited",
           str(sorted(set(labs) - cited)))
    report(not (cited - set(labs)), "no citation dangling", str(sorted(cited - set(labs))))

    # ---- G6: build cleanliness ------------------------------------------
    print("G6 build cleanliness")
    lg = os.path.join(ROOT, PAPER, "main.log")
    if not os.path.exists(lg):
        warn(False, "%s log present" % PAPER,
             "absent: run pdflatex twice in paper_mdpi/ and re-run this gate.")
    else:
        s = io.open(lg, encoding="utf-8", errors="ignore").read()
        err = len(re.findall(r"^! ", s, re.M))
        over = len(re.findall(r"Overfull \\hbox", s))
        pages = re.findall(r"Output written on main\.pdf \((\d+) pages", s)
        report(err == 0 and over == 0, "%s errors/overfull" % PAPER,
               "errors=%d overfull=%d pages=%s" % (err, over, pages[0] if pages else "?"))

    # ---- G8: macro-name legality, ref integrity, orphan supplementary ----
    print("\nG8 macro-name legality and float integrity")
    cited_dig = sorted(set(n for n in re.findall(r"\\(Num[A-Za-z]*[0-9][A-Za-z0-9]*)", src)))
    def_dig = sorted(set(re.findall(r"\\newcommand\{\\([A-Za-z]*[0-9][A-Za-z0-9]*)\}",
                                    io.open(os.path.join(ROOT, PAPER, "numbers.tex"),
                                            encoding="utf-8").read())))
    report(not cited_dig, "%s no digit in cited macro" % PAPER, str(cited_dig))
    report(not def_dig, "%s no digit in defined macro" % PAPER, str(def_dig))
    labs = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", src))
    report(not (refs - labs), "%s no dangling ref" % PAPER, str(sorted(refs - labs)))
    cites_alg = sorted(r for r in refs if r.startswith("alg:"))
    if cites_alg:
        ok = "algorithm" in set(re.findall(r"\\begin\{(algorithm)\}", src))
        report(ok, "%s alg: label has a float" % PAPER,
               "" if ok else "cites %s but no algorithm environment exists" % cites_alg)
    sm = re.findall(r"supplementary material|补充材料", src)
    report(not sm, "%s no orphan supplementary ref" % PAPER, "%d occurrence(s)" % len(sm))

    # ---- G15: parenthetical-reference discipline -------------------------
    # Prose must not carry references as parentheticals — "(Thm. 4.4)",
    # "(Table 6)", "(Lem. 4.1), ..." — they read as afterthoughts and were a
    # user-flagged defect class.  Rewrite as "by Thm. X" / "as Table X shows".
    # Allowed parens: math, abbreviation definitions, panel tags, enumerators.
    print("\nG15 parenthetical-reference discipline")
    bad = re.findall(
        r"\((?:Sec|Sect|Thm|Lem|Prop|Cor|Eq|Remark|Table|Tab|Fig|Ablation)s?\.?~?\\(?:ref|eqref)\{[^}]*\}[^)]*\)",
        src)
    # hard-coded numbering without \ref — "(Thm. 4.4)" would silently drift
    bad += re.findall(
        r"\((?:Sec|Sect|Thm|Lem|Prop|Cor|Eq|Remark|Table|Tab|Fig|Ablation)s?\.?\s?\d+(?:\.\d+)?\)",
        src)
    report(not bad, "%s no parenthetical theorem/table/figure refs" % PAPER,
           str([b[:60] for b in bad]))

    # ---- G7: artefact freshness -----------------------------------------
    print("\nG7 artefact freshness (figures and numbers vs their data sources)")
    FIGS = {
        "fig_depth": ["depth_curve.json", "classical_curve.json", "depth_curve_40ep.json"],
        "fig_extrap": ["extrap.json"],
        "fig_noise": ["noise.json"],
        "fig_gain": ["gain.json"],
        "fig_pareto": ["pareto.json"],
        "fig_gradnorm": ["gradnorm.json"],
        "fig_ablation": ["ablation.json"],
        "fig_frozen": ["ablation.json"],
        "fig_images": ["meta.json"],
        "fig_alphac": ["bjac.json", "ngc.json"],
        "fig_alphac_converged": ["ngc.json"],
    }
    derived = {}
    mf = os.path.join(ROOT, "code", "make_figs.py")
    if os.path.exists(mf):
        txt = io.open(mf, encoding="utf-8").read()
        secs = re.split(r"^\s*#\s*-{4,}\s*(fig_\w+)[^\n]*$", txt, flags=re.M)
        for i in range(1, len(secs) - 1, 2):
            got = sorted(set(re.findall(r'load\(\s*"([^"]+\.json)"\s*\)', secs[i + 1])))
            if got:
                derived[secs[i]] = got
    for fig in sorted(set(FIGS) | set(derived)):
        dec, der = set(FIGS.get(fig, [])), set(derived.get(fig, []))
        if der and dec != der:
            warns.append("G7 %s: declared sources %s != sources read by make_figs.py %s"
                         % (fig, sorted(dec - der) or "-", sorted(der - dec) or "-"))
    figdir = os.path.join(ROOT, "figures")
    for fig in sorted(set(FIGS) | set(derived)):
        srcs = sorted(set(FIGS.get(fig, [])) | set(derived.get(fig, [])))
        if not srcs:
            continue
        fp = os.path.join(figdir, fig + ".pdf")
        if not os.path.exists(fp):
            continue
        newest, who = 0.0, None
        for s_ in srcs:
            sp = os.path.join(ROOT, "results", s_)
            if os.path.exists(sp) and os.path.getmtime(sp) > newest:
                newest, who = os.path.getmtime(sp), s_
        if who is None:
            report(False, "%s data source present" % fig, "none of %s exists" % srcs)
            continue
        ok = os.path.getmtime(fp) >= newest
        report(ok, "%s newer than %s" % (fig, who),
               "" if ok else "figure is STALE (older than its data source)")

    # numbers.tex: copy of paper_en's; byte-identical => inherit, else mtime-check.
    mdpi_num = os.path.join(ROOT, PAPER, "numbers.tex")
    en_num = os.path.join(ROOT, "paper_en", "numbers.tex")
    if os.path.exists(mdpi_num) and os.path.exists(en_num):
        a = io.open(mdpi_num, encoding="utf-8").read()
        b = io.open(en_num, encoding="utf-8").read()
        if a == b:
            report(True, "%s/numbers.tex == paper_en/numbers.tex" % PAPER,
                   "byte-identical copy of the canonical source")
        else:
            allres = [os.path.join(ROOT, "results", f) for f in os.listdir(
                os.path.join(ROOT, "results"))
                if f.endswith(".json") and f != "ambiguity.json"]
            newest_res = max((os.path.getmtime(x) for x in allres), default=0.0)
            ok = os.path.getmtime(mdpi_num) >= newest_res
            report(ok, "%s/numbers.tex newer than results/*.json" % PAPER,
                   "" if ok else "MDPI numbers.tex DIVERGES from paper_en AND is older "
                                 "than a results file: re-sync the copy.")
    else:
        report(False, "%s/numbers.tex present" % PAPER, "missing")

    # ---- G9: artefact metadata vs driver script -------------------------
    print("\nG9 artefact metadata vs its driver script")
    note, err = driver_str_const(os.path.join(ROOT, "code", "run_seeds.py"), "SEED_NOTE")
    art = os.path.join(ROOT, "results", "seeds.json")
    if err is not None:
        warn(False, "run_seeds.SEED_NOTE readable", err)
    elif not os.path.exists(art):
        warn(False, "seeds.json present", "missing")
    else:
        got = io.open(art, encoding="utf-8") and __import__("json").load(
            io.open(art, encoding="utf-8")).get("config", {}).get("note")
        if got == note:
            report(True, "seeds.json note == run_seeds.SEED_NOTE", "%d chars" % len(note))
        else:
            report(False, "seeds.json note == run_seeds.SEED_NOTE",
                   "STALE: artifact note differs from the driver (artifact says %r...)"
                   % (got or "")[:48])
        import ast, json
        drv = os.path.join(ROOT, "code", "run_seeds.py")
        tree = ast.parse(io.open(drv, encoding="utf-8").read())
        decl = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in ("K", "EPOCHS", "SEEDS", "VARIANTS"):
                        try:
                            decl[t.id] = ast.literal_eval(node.value)
                        except ValueError:
                            decl[t.id] = None
        cfg = json.load(io.open(art, encoding="utf-8")).get("config", {})
        vnames = None
        vsrc = io.open(drv, encoding="utf-8").read()
        mvar = re.search(r"^VARIANTS\s*=\s*\{(.*?)^\}", vsrc, re.S | re.M)
        if mvar:
            vnames = re.findall(r'^\s*"([A-Za-z0-9_]+)"\s*:', mvar.group(1), re.M)
        exp = {
            "K": decl.get("K"),
            "epochs": decl.get("EPOCHS"),
            "seeds": [s for _, s in decl["SEEDS"]] if decl.get("SEEDS") else None,
            "variants": vnames,
        }
        for key in ("K", "epochs", "seeds", "variants"):
            if exp[key] is None:
                warn(False, "run_seeds.%s declarable" % key, "could not read from driver")
                continue
            got_v = cfg.get(key)
            report(got_v == exp[key], "seeds.json config.%s == driver" % key,
                   "artifact=%r driver=%r" % (got_v, exp[key]))
        keys = set(json.load(io.open(art, encoding="utf-8")).get("runs", {}))
        want = set("r%d_%s" % (r, v) for r, _ in (decl.get("SEEDS") or [])
                   for v in (vnames or []))
        if want:
            report(want <= keys, "seeds.json holds every declared run",
                   "missing %s" % sorted(want - keys) if not want <= keys
                   else "%d runs" % len(want))

    # ---- G10: abstract length (MDPI ~200 words, dual convention) --------
    print("\nG10 abstract length (MDPI 'around 200 words')")
    spec = importlib.util.spec_from_file_location(
        "_abs_words", os.path.join(ROOT, "code", "_abstract_words.py"))
    AW = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AW)
    # MDPI author instructions: "a concise and factual abstract ... around 200 words".
    # (Confirm against the current Mathematics template; 150-250 is also seen.)  We hold
    # the tighter 200 so the abstract is safe under either stated bound.
    ABS_LIMIT = 200

    def extract_mdpi_abstract(s):
        """MDPI puts the abstract in \abstract{...}, a command, not an environment."""
        i = s.find(r"\abstract{")
        if i < 0:
            return None
        i += len(r"\abstract{")
        depth, out, j = 1, [], i
        while j < len(s) and depth > 0:
            c = s[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(c)
            j += 1
        return "".join(out)

    abs_mdpi = extract_mdpi_abstract(src)
    if abs_mdpi is None:
        report(False, "%s abstract present" % PAPER, "no \\abstract{...} found")
    else:
        n, m = AW.strip_latex(abs_mdpi)
        report(n <= ABS_LIMIT and m <= ABS_LIMIT, "%s abstract words" % PAPER,
               "A=%d, B=%d (limit %d = %d words + %d formula tokens)"
               % (n, m, ABS_LIMIT, n, m - n))

    # ---- G11: prose duplication -----------------------------------------
    print("\nG11 prose duplication inside the manuscript")
    spec2 = importlib.util.spec_from_file_location(
        "_dup_sent", os.path.join(ROOT, "code", "_dup_sentence.py"))
    DS = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(DS)
    total, reps = DS.duplicates(src)
    report(not reps, "%s duplicated prose sentence" % PAPER,
           "%d of %d sentences" % (len(reps), total)
           + ("" if not reps else " | first: %r" % reps[0][:70]))

    # ---- G12: ambiguity ledger ------------------------------------------
    print("\nG12 ambiguous dotted-path lookups")
    amb_p = os.path.join(ROOT, "results", "ambiguity.json")
    if not os.path.exists(amb_p):
        report(False, "results/ambiguity.json present",
               "missing: re-run code/gen_numbers.py (it writes the ledger)")
    else:
        led = __import__("json").load(io.open(amb_p, encoding="utf-8"))
        nn = led.get("n_ambiguous", -1)
        report(nn == 0, "no ambiguous lookup in the last generation run",
               "0 recorded" if nn == 0 else "%d recorded" % nn)
        report(len(led.get("lookups", [])) == nn, "ambiguity ledger internally consistent",
               "%d entries vs n_ambiguous=%s" % (len(led.get("lookups", [])), nn))
        report(led.get("complete") is True, "generation run completed (not SIGPIPE-truncated)",
               "" if led.get("complete") is True else "results/ambiguity.json has no complete flag")

    print("\n%s (%d failing check(s), %d warning(s))"
          % ("GATE PASS" if not fails else "GATE FAIL", len(fails), len(warns)))
    if warns:
        for w in warns:
            print("  WARN: %s" % w)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
