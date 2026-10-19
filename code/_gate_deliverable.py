# -*- coding: utf-8 -*-
"""_gate_deliverable.py — one shutter for the pair of manuscripts.

Run before freezing any review snapshot.  Nine independent checks; the script prints a
PASS/FAIL line per check and exits non-zero if any fails.

  G1  every macro cited by either manuscript resolves to a real number (no '--')
  G2  no macro is defined in numbers.tex but cited nowhere (dead macro; R1 m1.6d)
  G3  no macro is cited but undefined (would be a compile error)
  G4  the two bibliographies are identical in labels and order, fully cited, no dangling
  G5  the two manuscripts have the same section/environment/label structure
  G6  both PDF logs report 0 errors, 0 undefined control sequences, 0 overfull boxes
  G7  every figure is newer than its data source, and numbers.tex is newer than the numeric
      results inputs (the generator's own ambiguity.json ledger is excluded by construction)
  G8  macro-name digit legality, cross-reference integrity, float integrity
  G9  artifact metadata agrees with the driver script that produced it (R3 m3.12)
  G10 the English abstract fits the IEEE TAP ~250-word convention under BOTH counting
      conventions -- math stripped, and every formula counted as one word (R3 m3.5/n3.4)
  G11 no prose sentence occurs twice inside one manuscript (round-3 self-check)
  G12 no ambiguous dotted-path lookup in the last generation run (R3 n3.2)
  G13 every macro / label / file path named in the response letter exists (R1 t1/t2)
  G14 the cover letter's abstract-word-count statement matches the counter (R3 n3.4)

G6 reads <paper>/main.log, so both manuscripts must have been compiled after the last
edit; a stale log is reported as such rather than silently passing.  A *missing* log (a
frozen snapshot need not carry one) is a WARN, not a FAIL -- R3 m3.13: a gate that cannot be
re-run on the frozen artifact it guards is not a gate.
"""
import io, os, re, sys
from collections import Counter

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAPERS = ("paper_en", "paper_cn")
DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
USE = re.compile(r"\\([A-Za-z]{3,})")

fails = []
warns = []


def report(ok, tag, detail=""):
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", tag, detail))
    if not ok:
        fails.append(tag)


def warn(ok, tag, detail=""):
    """Soft check: a missing prerequisite is a WARN, a present-but-bad one is a FAIL.

    R3 m3.13/Q3.3: G6 read <paper>/main.log, which is not part of a frozen snapshot, so the
    gate necessarily failed on the frozen tree while the authors reported "all PASS".  A gate
    that cannot be run on the frozen artifact is not a gate, so the log is now snapshotted
    (snapshot.py) *and* its absence is downgraded to a WARN in a tree that has no log at all.
    """
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
    """Read a module-level string constant out of a driver script *without importing it*.

    R3 m3.12 identified this blind spot explicitly: G7 compares figures with their data files
    and numbers.tex with results/*.json, but nothing compared results/*.json with the driver
    script that produced it, so a stale note inside the artifact survived.  The driver imports
    torch, so the gate must not import it; ast + literal_eval needs neither torch nor a GPU.
    """
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
    macros_cited, macros_bad, macros_dead, macros_undef = {}, {}, {}, {}
    # A macro is DEAD when no manuscript cites it.  Deciding "dead" per manuscript is wrong
    # for a single-source setup: gen_numbers prunes to the UNION of the trees, so a macro that
    # only paper_mdpi cites (the far-field / geometry / contrast-grid tables live there) is
    # reported dead by paper_en and paper_cn alike -- 52 false failures in this gate alone, and
    # the mirror image of the same defect in _gate_mdpi.  The union MUST include paper_mdpi:
    # omitting it reproduces the very defect being fixed.  Per-tree orphans are kept as info.
    _union_src = ""
    for paper in tuple(dict.fromkeys(tuple(PAPERS) + ("paper_mdpi",))):
        _fp = os.path.join(ROOT, paper, "main.tex")
        if os.path.exists(_fp):
            _union_src += io.open(_fp, encoding="utf-8").read()
    macros_orphans = {}
    for paper in PAPERS:
        nums = defines(os.path.join(ROOT, paper, "numbers.tex"))
        src = io.open(os.path.join(ROOT, paper, "main.tex"), encoding="utf-8").read()
        body = src.split(r"\begin{thebibliography}")[0]
        used = set(USE.findall(body))
        macros_cited[paper] = used
        macros_bad[paper] = sorted(m for m in used if m in nums and nums[m].strip() == "--")
        macros_dead[paper] = sorted(m for m in nums
                                    if not re.search(r"\\%s(?![A-Za-z])" % m, _union_src))
        macros_orphans[paper] = sorted(m for m in nums
                                       if not re.search(r"\\%s(?![A-Za-z])" % m, src))
        macros_undef[paper] = sorted(m for m in used
                                     if m.startswith("Num") and m not in nums)

    print("G1 numeric macros resolve")
    for p in PAPERS:
        report(not macros_bad[p], "%s placeholders" % p,
               "%d %s" % (len(macros_bad[p]), macros_bad[p][:6]))
    print("G2/G3 macro hygiene")
    for p in PAPERS:
        report(not macros_dead[p], "%s dead macros (cited by no manuscript)" % p,
               str(macros_dead[p]))
        print("  [info] %d macro(s) cited only by other trees: %s"
              % (len(macros_orphans[p]), macros_orphans[p][:6]))
        report(not macros_undef[p], "%s undefined macros" % p, str(macros_undef[p]))

    print("G4 bibliography")
    def bibitems(p):
        s = io.open(os.path.join(ROOT, p, "main.tex"), encoding="utf-8").read()
        body = s.split(r"\begin{thebibliography}")[1]
        labs = re.findall(r"\\bibitem\{([^}]*)\}", body)
        cited = set()
        for m in re.findall(r"\\cite[a-z]*\{([^}]*)\}",
                            s.split(r"\begin{thebibliography}")[0]):
            cited.update(x.strip() for x in m.split(","))
        return labs, cited
    labs_en, cited_en = bibitems("paper_en")
    labs_cn, cited_cn = bibitems("paper_cn")
    report(labs_en == labs_cn, "labels and order identical",
           "%d items" % len(labs_en))
    report(not (set(labs_en) - cited_en), "no entry defined but uncited",
           str(sorted(set(labs_en) - cited_en)))
    report(not (cited_en - set(labs_en)), "no citation dangling", str(sorted(cited_en - set(labs_en))))

    print("G5 structural parity")
    def shape(p):
        s = io.open(os.path.join(ROOT, p, "main.tex"), encoding="utf-8").read()
        secs = len(re.findall(r"\\section\{|\\subsection\{", s))
        envs = Counter(re.findall(
            r"\\begin\{(theorem|lemma|proposition|corollary|remark|definition|table\*?|figure\*?|algorithm)\}", s))
        labs = len(re.findall(r"\\label\{(?:sec|eq|thm|lem|prop|cor|rem|tab|fig|alg):", s))
        return secs, dict(sorted(envs.items())), labs
    a, b = shape("paper_en"), shape("paper_cn")
    report(a == b, "sections/environments/labels", "EN %s | CN %s" % (a, b))

    print("G6 build cleanliness")
    for p in PAPERS:
        lg = os.path.join(ROOT, p, "main.log")
        if not os.path.exists(lg):
            # No log at all: either the tree has never been compiled, or it is a frozen
            # snapshot that did not carry the log.  That is a WARN, not a FAIL -- see warn().
            warn(False, "%s log present" % p,
                 "absent (no log in this tree: page count / overfull not verifiable here). "
                 "In the authoring tree run pdflatex twice and re-run this gate.")
            continue
        s = io.open(lg, encoding="utf-8", errors="ignore").read()
        err = len(re.findall(r"^! ", s, re.M))
        over = len(re.findall(r"Overfull \\hbox", s))
        pages = re.findall(r"Output written on main\.pdf \((\d+) pages", s)
        report(err == 0 and over == 0, "%s errors/overfull" % p,
               "errors=%d overfull=%d pages=%s" % (err, over, pages[0] if pages else "?"))

    print("\nG8 macro-name legality and float integrity")
    # A LaTeX control word ends at the first non-letter, so \NumCls8 defines/uses
    # \NumCls.  gen_numbers.py renames such keys; a manuscript that then cites the
    # original name gets a silently truncated macro.  Note that this gate's own
    # USE regex (\\([A-Za-z]{3,})) truncates in exactly the same way, so the check
    # must be digit-aware to be able to see the defect at all.
    for p in PAPERS:
        src = io.open(os.path.join(ROOT, p, "main.tex"), encoding="utf-8").read()
        num = io.open(os.path.join(ROOT, p, "numbers.tex"), encoding="utf-8").read()
        cited_dig = sorted(set(n for n in re.findall(r"\\(Num[A-Za-z]*[0-9][A-Za-z0-9]*)", src)))
        def_dig = sorted(set(re.findall(r"\\newcommand\{\\([A-Za-z]*[0-9][A-Za-z0-9]*)\}", num)))
        report(not cited_dig, "%s no digit in cited macro" % p, str(cited_dig))
        report(not def_dig, "%s no digit in defined macro" % p, str(def_dig))
        # cross-reference integrity: every \ref/\eqref target must exist, and no float may
        # be referenced by a hand-written number (that is how "Alg. 1, supplementary
        # material" survived with no algorithm environment in the document).  The check is
        # language-agnostic: it triggers on a \ref to an alg: label, not on the word
        # "Algorithm", which the Chinese manuscript does not use.
        labs = set(re.findall(r"\\label\{([^}]+)\}", src))
        refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", src))
        report(not (refs - labs), "%s no dangling ref" % p, str(sorted(refs - labs)))
        cites_alg = sorted(r for r in refs if r.startswith("alg:"))
        if cites_alg:
            ok = "algorithm" in set(re.findall(r"\\begin\{(algorithm)\}", src))
            report(ok, "%s alg: label has a float" % p,
                   "" if ok else "cites %s but no algorithm environment exists" % cites_alg)
        # a reference to supplementary material is dangling unless the paper has one
        sm = re.findall(r"supplementary material|补充材料", src)
        report(not sm, "%s no orphan supplementary ref" % p,
               "%d occurrence(s)" % len(sm))

    print("\nG7 artefact freshness (figures and numbers vs their data sources)")
    # R3/M3.2 was exactly this failure: the text was updated but the figure was not, so one
    # point of fig:alphac disagreed with the same quantity in Table cert by 5.2x.  Neither G1
    # (macros resolve) nor G6 (compiles cleanly) can see that, because the figure is a binary
    # blob and the macro is self-consistent -- the figure is simply older than its inputs.
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
    # Class-level fix (round-3 self-check): the hand-maintained FIGS map can silently rot --
    # add a load("new.json") to a figure section and nothing complains that the new input is no
    # longer freshness-checked, which is exactly the defect class R3 m3.2 caught once by hand.
    # So the sources are ALSO derived from make_figs.py itself; freshness is checked against the
    # union, and any divergence between the derived and the declared list is reported.
    derived = {}
    mf = os.path.join(ROOT, "code", "make_figs.py")
    if os.path.exists(mf):
        txt = io.open(mf, encoding="utf-8").read()
        # The marker line may carry a parenthetical suffix, e.g.
        #   "# ------------- fig_ablation (training-time arms) -------------".
        # The first version of this regex required the dashes immediately after the name, so two
        # sections were not split and their bodies were attributed to the PREVIOUS section --
        # which produced two spurious "declared != derived" warnings.  Instruments first.
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
            continue                      # an optional figure need not exist
        newest, who = 0.0, None
        for s in srcs:
            sp = os.path.join(ROOT, "results", s)
            if os.path.exists(sp) and os.path.getmtime(sp) > newest:
                newest, who = os.path.getmtime(sp), s
        if who is None:
            report(False, "%s data source present" % fig, "none of %s exists" % srcs)
            continue
        ok = os.path.getmtime(fp) >= newest
        report(ok, "%s newer than %s" % (fig, who),
               "" if ok else "figure is STALE (older than its data source)")

    # R3 n3.2 / format self-check: ambiguity.json is written *by* gen_numbers.py as its own
    # completion ledger, so it is always younger than numbers.tex by construction.  Requiring
    # numbers.tex to be newer than it would make G7 fail on every clean run.  It is excluded
    # from the freshness comparison; the data freshness that matters is numbers.tex vs the
    # numeric inputs (gain_decomp.json, seeds.json, ...), which ambiguity.json is not.
    allres = [os.path.join(ROOT, "results", f) for f in os.listdir(os.path.join(ROOT, "results"))
              if f.endswith(".json") and f != "ambiguity.json"]
    newest_res = max(os.path.getmtime(x) for x in allres) if allres else 0.0
    for p in PAPERS:
        np_ = os.path.join(ROOT, p, "numbers.tex")
        if os.path.exists(np_):
            ok = os.path.getmtime(np_) >= newest_res
            report(ok, "%s/numbers.tex newer than results/*.json" % p,
                   "" if ok else "numbers.tex is OLDER than a results file: re-run gen_numbers.py")

    print("\nG9 artefact metadata vs its driver script")
    # R3 m3.12: results/seeds.json:config.note asserted "r=0 ... must reproduce depth_curve.json",
    # which the same file's own numbers (0.40180 vs 0.40673) falsify.  Nothing in G1-G8 could
    # see it.  The note is now a single constant in run_seeds.py; the artifact must carry it
    # verbatim, and the two may only change together.
    note, err = driver_str_const(os.path.join(ROOT, "code", "run_seeds.py"), "SEED_NOTE")
    art = os.path.join(ROOT, "results", "seeds.json")
    if err is not None:
        warn(False, "run_seeds.SEED_NOTE readable", err)
    elif not os.path.exists(art):
        warn(False, "seeds.json present", "missing")
    else:
        import json
        got = json.load(io.open(art, encoding="utf-8")).get("config", {}).get("note")
        if got == note:
            report(True, "seeds.json note == run_seeds.SEED_NOTE", "%d chars" % len(note))
        else:
            report(False, "seeds.json note == run_seeds.SEED_NOTE",
                   "STALE: artifact note differs from the driver; run "
                   "code/_restamp_seeds_note.py (artifact says %r...)"
                   % (got or "")[:48])

        # R3 n3.2 (G9 extended): compare the run *set*, not only the note string.  A stale
        # note is caught above, but editing the driver to add a seed / variant / depth while
        # the artifact still holds the old one was invisible.  The driver's declarations are
        # read with ast (no import: the driver needs torch), and the artifact's config must
        # agree with all of K, epochs, the seed list and the variant list.
        import ast
        drv = os.path.join(ROOT, "code", "run_seeds.py")
        try:
            tree = ast.parse(io.open(drv, encoding="utf-8").read())
        except SyntaxError as e:
            report(False, "run_seeds.py parses", str(e))
            tree = None
        if tree is not None:
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
            # VARIANTS is a dict of dict(...) calls, so ast.literal_eval cannot read it;
            # read its keys textually instead.  A missing declaration is a WARN (the guard
            # must still work on a tree where the driver was refactored).
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
            # and every declared (repeat, variant) pair must actually be present
            keys = set(json.load(io.open(art, encoding="utf-8")).get("runs", {}))
            want = set("r%d_%s" % (r, v) for r, _ in (decl.get("SEEDS") or [])
                       for v in (vnames or []))
            if want:
                report(want <= keys, "seeds.json holds every declared run",
                       "missing %s" % sorted(want - keys) if not want <= keys
                       else "%d runs" % len(want))

    print("\nG12 ambiguous dotted-path lookups")
    # Class of defect: the resolver in jsonpath.py tolerates periods inside keys by trying
    # the longest join first.  That same rule over-matches when a flat key and a nested
    # path both resolve: {"alpha.beta": 99, "alpha": {"beta": 1}}.  dig_path then returns a
    # value *silently*, and a wrong number in the PDF is worse than a visible "--".
    # gen_numbers.py records every ambiguous lookup it performs into results/ambiguity.json;
    # here we fail if the last generation run contained one.  Probe P5 in
    # code/_probe_jsonpath.py establishes that the detector can fire (P5a/P5b) and that it
    # stays silent on unambiguous input (P5c).
    amb_p = os.path.join(ROOT, "results", "ambiguity.json")
    if not os.path.exists(amb_p):
        report(False, "results/ambiguity.json present",
               "missing: re-run code/gen_numbers.py (it writes the ledger)")
    else:
        led = json.load(io.open(amb_p, encoding="utf-8"))
        n = led.get("n_ambiguous", -1)
        report(n == 0, "no ambiguous lookup in the last generation run",
               "0 recorded" if n == 0 else "%d recorded: %s"
               % (n, [a.get("path") for a in led.get("lookups", [])][:3]))
        # structural self-check: the ledger must be consistent with itself
        report(len(led.get("lookups", [])) == n, "ambiguity ledger internally consistent",
               "%d entries vs n_ambiguous=%s" % (len(led.get("lookups", [])), n))
        # R4 self-check, class: "a run that was killed half-way is indistinguishable from a
        # clean one".  gen_numbers.py is often inspected as `... | head`; when stdout is
        # truncated the child dies of SIGPIPE after writing numbers.tex but before writing this
        # ledger, so the only visible symptom is a freshness failure elsewhere.  The ledger's
        # `complete` flag makes that state detectable, and the writer now emits all artefacts
        # before its large stdout dump.  Asserting the flag here means a truncated generation
        # run fails loudly at its own check instead of silently ageing numbers.tex.
        report(led.get("complete") is True, "generation run completed (not SIGPIPE-truncated)",
               "" if led.get("complete") is True else
               "results/ambiguity.json has no complete flag: gen_numbers.py was killed before "
               "finishing (do not pipe it into head/tail); re-run it to completion")

    # ---- G13: response letter vs the tree it describes (R1 t1/t2, R3 m3.2) ---------
    print("\nG13 response letter references resolve inside the tree")
    # Class of defect R1 named twice ("声明已改、快照未体现"): the response letter claims a
    # fix, the tree does not contain it, and every other gate is blind because each checks
    # internal consistency of the tree alone.  The checkable proxy is that *every concrete
    # identifier the letter cites must exist in the tree*: a macro must be defined, a label
    # must be referenced somewhere, a file path must exist.  It cannot prove a prose claim
    # true, but it does catch "the letter names a macro/label/file that does not exist",
    # which is exactly how the two R1 items looked from the outside.
    import glob as _glob
    rounds = sorted(_glob.glob(os.path.join(ROOT, "reviews", "round_r*", "response.md")))
    if not rounds:
        warn(False, "a response letter exists", "no reviews/round_r*/response.md")
    else:
        rp = rounds[-1]
        rtxt = io.open(rp, encoding="utf-8").read()
        nums_src = io.open(os.path.join(ROOT, "paper_en", "numbers.tex"),
                           encoding="utf-8").read()
        # The letter describes the DELIVERED tree.  Round 9's letter is written against
        # paper_mdpi (its §6.3/§6.4/§6.6/§6.8 and tab:contrast are MDPI numbers and labels), so
        # resolving references against paper_en+paper_cn alone reported four false failures of
        # the same class as the dead-macro defect: the rule was scoped to two trees while the
        # object it describes now has three.  Resolve against all shipped trees.
        _letter_trees = tuple(dict.fromkeys(tuple(PAPERS) + ("paper_mdpi",)))
        main_src = "".join(io.open(os.path.join(ROOT, p, "main.tex"), encoding="utf-8").read()
                           for p in _letter_trees if os.path.exists(
                               os.path.join(ROOT, p, "main.tex")))
        # (a) macros named in the letter must be defined.
        #     False positive to avoid (found on the first run): a letter that *retracts* an
        #     identifier legitimately names one that no longer exists ("NumDfCgBlockImpl has
        #     been removed").  A guard that cannot tell a retraction from a stale claim would
        #     force authors to stop documenting what they deleted, which is worse than the
        #     defect it guards.  So any line mentioning a removal is exempt, and the exemption
        #     is reported so that it is visible rather than silent.
        REMOVAL = ("remov", "delet", "retract", "删除", "移除", "不再输出", "已删")
        lines = rtxt.splitlines()
        mac_toks = sorted(set(re.findall(r"\\?(Num[A-Z][A-Za-z]{2,})", rtxt)))
        exempt = set()
        for ln in lines:
            if any(k in ln.lower() for k in REMOVAL):
                exempt.update(re.findall(r"\\?(Num[A-Z][A-Za-z]{2,})", ln))
        bad_mac = [m for m in mac_toks
                   if m not in exempt and ("\\newcommand{\\%s}" % m) not in nums_src]
        report(not bad_mac, "%s macros named in letter are defined" % os.path.basename(
            os.path.dirname(rp)), "%d named (%d exempt as retracted), %d undefined%s"
            % (len(mac_toks), len(exempt), len(bad_mac),
               "" if not bad_mac else " -> %s" % bad_mac[:4]))
        # (b) LaTeX labels named in the letter must exist in a manuscript
        lab_toks = sorted(set(re.findall(
            r"\b((?:sec|tab|fig|eq|lem|thm|prop|cor|rem|alg):[A-Za-z][A-Za-z0-9]*)", rtxt)))
        bad_lab = [l for l in lab_toks if ("label{%s}" % l) not in main_src]
        report(not bad_lab, "labels named in letter exist", "%d named, %d missing%s"
               % (len(lab_toks), len(bad_lab), "" if not bad_lab else " -> %s" % bad_lab[:4]))
        # (c) artefact / driver paths named in the letter must exist
        paths = sorted(set(re.findall(r"\b((?:results|code|paper_en|paper_cn|reviews)/[A-Za-z0-9_.\-/]+"
                                      r"(?:\.json|\.py|\.tex|\.md|\.pdf|\.txt))", rtxt)))
        bad_p = [p for p in paths
                 if not os.path.exists(os.path.join(ROOT, p.replace("/", os.sep)))]
        report(not bad_p, "paths named in letter exist", "%d named, %d missing%s"
               % (len(paths), len(bad_p), "" if not bad_p else " -> %s" % bad_p[:4]))
        # (d) INTERNAL cross-references inside the letter.  Found for real: the letter said
        #     "see §4.1" (the depth-schedule control) while §4.1 did not exist -- the letter had
        #     gone 4.0 -> 4.2.  (a)-(c) are blind to this because they only look outward into the
        #     tree.  Any "§N" or "§N.M" the letter cites must match a heading of the letter.
        heads = set(re.findall(r"^#{2,4}\s+(\d+(?:\.\d+)*)", rtxt, flags=re.M))
        # ...and a "§N.M" may also point INTO the manuscript (round 9 does exactly that:
        # "§6.4 provenance", "§6.3 tab:fresnel").  Those numbers are not literal text in the
        # sources, so derive them from the \section / \subsection order of every shipped tree
        # and accept a reference that lands on a real section path there.
        def _section_paths(tex):
            paths, sec, sub = set(), 0, 0
            for m in re.finditer(r"\\(section|subsection)\*?\{", tex):
                if m.group(1) == "section":
                    sec += 1
                    sub = 0
                    paths.add(str(sec))
                else:
                    sub += 1
                    paths.add("%d.%d" % (sec, sub))
            return paths
        man_paths = set()
        for _p in _letter_trees:
            _fp = os.path.join(ROOT, _p, "main.tex")
            if os.path.exists(_fp):
                man_paths |= _section_paths(io.open(_fp, encoding="utf-8").read())
        heads |= man_paths
        refs = set(re.findall(r"§\s*(\d+(?:\.\d+)*)", rtxt))
        bad_ref = sorted(r for r in refs if r not in heads)
        report(not bad_ref, "section cross-references inside letter resolve",
               "%d cited, headings %s, missing%s"
               % (len(refs), sorted(heads), "" if not bad_ref else " -> %s" % bad_ref[:4]))
        print("  [nb]   source letter: %s" % os.path.relpath(rp, ROOT).replace(os.sep, "/"))

    print("\nG10 abstract length")
    # R3 m3.5/Q3.4: the English abstract must fit IEEE TAP's ~250-word convention.  The count
    # is not re-implemented here -- the gate imports the same counter the authors use, so the
    # number in the response letter and the number in this check cannot diverge.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_abs_words", os.path.join(ROOT, "code", "_abstract_words.py"))
    AW = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AW)
    ABS_LIMIT = 250
    for p in PAPERS:
        src = io.open(os.path.join(ROOT, p, "main.tex"), encoding="utf-8").read()
        ms = AW.ABSTRACT_RE.findall(src)
        if len(ms) != 1:
            report(False, "%s one abstract environment" % p, "found %d" % len(ms))
            continue
        n, m = AW.strip_latex(ms[0])
        if p == "paper_en":
            # Tightened after R3 n3.4: the abstract must fit under BOTH word-count
            # conventions, not just the one we happen to prefer.  "math stripped" is
            # convention A; "every formula counts as one word" is the more conservative
            # convention B, which is what a journal's own counter is closer to.
            report(n <= ABS_LIMIT and m <= ABS_LIMIT, "%s abstract words" % p,
                   "A=%d, B=%d (limit %d = %d words + %d formula tokens)"
                   % (n, m, ABS_LIMIT, n, m - n))
        else:
            print("  [nb]   %-46s A=%d B=%d (IEEE limit not applied to the Chinese abstract)"
                  % ("%s abstract words" % p, n, m))

    print("\nG11 prose duplication inside one manuscript")
    # Class of defect: a patch inserts a revised sentence without deleting the old one, so the
    # same claim is stated twice in one paragraph.  Compiles fine, all macros resolve, no number
    # is wrong -- G1-G10 are blind to it.  The detector excludes LaTeX structural fragments
    # (its own first version reported float boilerplate as a "duplicate"); see _dup_sentence.py,
    # which also carries a selftest that injects a known duplicate AND checks that boilerplate
    # does not fire.
    spec2 = importlib.util.spec_from_file_location(
        "_dup_sent", os.path.join(ROOT, "code", "_dup_sentence.py"))
    DS = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(DS)
    for p in PAPERS:
        src = io.open(os.path.join(ROOT, p, "main.tex"), encoding="utf-8").read()
        total, reps = DS.duplicates(src)
        report(not reps, "%s duplicated prose sentence" % p,
               "%d of %d sentences" % (len(reps), total)
               + ("" if not reps else " | first: %r" % reps[0][:70]))

    print("\nG14 cover-letter word-count statement vs the counter")
    # R3 n3.4: the cover letter states the abstract word-count convention.  A stated
    # convention is a *counting assertion* of exactly the kind that is true only on the day
    # it is written: shorten the abstract and the letter silently becomes false.  So the
    # letter carries a machine-readable footer
    #     <!-- check: abstract_words=A formulas_as_words=B limit=L -->
    # and this check recomputes A and B with the same counter G10 uses and requires an exact
    # match.  A missing footer is a FAIL, not a skip: the letter is part of the submission.
    # WB_COVER_LETTER exists only so that _probe_G14.py can point this check at a known-bad
    # copy; the default is always the shipped letter.
    CL = os.environ.get("WB_COVER_LETTER") or os.path.join(ROOT, "docs", "03_投稿信.md")
    if not os.path.exists(CL):
        report(False, "cover letter exists", "docs/03_投稿信.md not found")
    else:
        s = io.open(CL, encoding="utf-8").read()
        mm = re.search(r"<!--\s*check:\s*abstract_words=(\d+)\s+formulas_as_words=(\d+)"
                       r"\s+limit=(\d+)\s*-->", s)
        if not mm:
            report(False, "cover letter carries a check footer", "no <!-- check: ... --> line")
        else:
            ca, cb, cl = (int(x) for x in mm.groups())
            src = io.open(os.path.join(ROOT, "paper_en", "main.tex"), encoding="utf-8").read()
            ms = AW.ABSTRACT_RE.findall(src)
            n, m = AW.strip_latex(ms[0]) if len(ms) == 1 else (None, None)
            report(n is not None and (ca, cb, cl) == (n, m, ABS_LIMIT),
                   "cover letter counts == counter",
                   "letter (A=%d, B=%d, limit=%d) vs counter (A=%s, B=%s, limit=%d)"
                   % (ca, cb, cl, n, m, ABS_LIMIT))
            stated = re.search(r"one word", s) and re.search(r"[Mm]ath stripped", s)
            report(bool(stated), "cover letter states both conventions",
                   "mentions 'math stripped' and 'one word'"
                   if stated else "one of the two conventions is not described")

    print("\n%s (%d failing check(s), %d warning(s))"
          % ("GATE PASS" if not fails else "GATE FAIL", len(fails), len(warns)))
    if warns:
        for w in warns:
            print("  WARN: %s" % w)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
