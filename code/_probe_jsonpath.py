# -*- coding: utf-8 -*-
"""_probe_jsonpath.py — falsifiability probe for jsonpath.dig_path.

Dr Wang's discipline: an instrument must first be shown to FAIL on a known-bad input
before its "all clear" means anything.  This probe checks three properties, and the
second and third matter as much as the first:

  P1  a key that contains a period resolves through the dotted path
      (this is the defect that gate G1 caught: chi_lip.json["F_fixed_ag_0.5"])
  P2  a plain nested key still resolves (no regression)
  P3  a genuinely absent path still returns None
      (over-permissiveness would be just as bad: it would silently substitute a
       *wrong* number for a missing one, which is worse than printing "--")

It also runs the naive split-based lookup on the P1 input to demonstrate that the
probe input really does break the old implementation -- otherwise the probe would be
passing for the wrong reason.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from jsonpath import dig_path, dig_candidates  # noqa: E402


def naive(d, path):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def main():
    doc = {
        "outer": {
            "F_fixed_ag_0.5": {"rho_chi_to_chi": 1.0531},
            "plain": {"x": 7},
        }
    }
    ok = True

    # P0 -- the input must genuinely break the old lookup, else P1 proves nothing
    old = naive(doc, "outer.F_fixed_ag_0.5.rho_chi_to_chi")
    good0 = (old is None)
    print("  P0 naive split fails on dot-bearing key   -> %s  %s"
          % (old, "PASS" if good0 else "FAIL (probe input does not discriminate)"))
    ok &= good0

    # P1 -- the new resolver must find it
    v = dig_path(doc, "outer.F_fixed_ag_0.5.rho_chi_to_chi")
    good1 = (v == 1.0531)
    print("  P1 dig_path resolves dot-bearing key      -> %s  %s"
          % (v, "PASS" if good1 else "FAIL"))
    ok &= good1

    # P2 -- plain nested key unaffected
    v = dig_path(doc, "outer.plain.x")
    good2 = (v == 7)
    print("  P2 dig_path resolves plain nested key     -> %s  %s"
          % (v, "PASS" if good2 else "FAIL"))
    ok &= good2

    # P3 -- absent path still None (no silent wrong-number substitution)
    v = dig_path(doc, "outer.F_fixed_ag_0.5.does_not_exist")
    v2 = dig_path(doc, "nope.at.all")
    good3 = (v is None and v2 is None)
    print("  P3 dig_path returns None when absent      -> %s / %s  %s"
          % (v, v2, "PASS" if good3 else "FAIL"))
    ok &= good3

    # P4 -- against the real archive: chi_lip.json must expose the fixed-step entry
    import json
    cl = os.path.join(os.path.dirname(HERE), "results", "chi_lip.json")
    if os.path.exists(cl):
        doc2 = json.load(open(cl, encoding="utf-8"))
        v = dig_path(doc2, "chi_lip_constrained.F_fixed_ag_0.5.rho_chi_to_chi")
        vu = dig_path(doc2, "chi_lip_unconstrained.F_fixed_ag_0.5.rho_chi_to_chi")
        good4 = isinstance(v, float) and isinstance(vu, float)
        print("  P4 real chi_lip.json fixed-step rho        -> cons=%s unc=%s  %s"
              % (v, vu, "PASS" if good4 else "FAIL"))
        ok &= good4
    else:
        print("  P4 skipped (results/chi_lip.json absent)")
        ok = False

    # P5 -- the over-match case (round-3 R3 n3.2).  The rule that fixes the dot-bearing
    # key also over-matches: when a flat key and a nested path both resolve, dig_path
    # silently returns the flat one.  The precedence is documented, but silence is not
    # acceptable -- the ambiguity has to be *detectable*, so that (a) the probe can show
    # the two readings really do differ, and (b) gen_numbers can record every ambiguous
    # lookup it performs and the gate can refuse a run that contained one.
    amb = {"alpha.beta": 99, "alpha": {"beta": 1, "gamma": 1}}
    v_flat = dig_path(amb, "alpha.beta")
    cands = dig_candidates(amb, "alpha.beta")
    vals = sorted(str(c[1]) for c in cands)
    good5a = (v_flat == 99 and len(cands) == 2 and vals == ["1", "99"])
    print("  P5a ambiguous path: two readings differ  -> dig_path=%s key_chains=%s  %s"
          % (v_flat, [c[0] for c in cands], "PASS" if good5a else "FAIL"))
    ok &= good5a

    # P5b -- the naive split would have returned the nested 1, so the two implementations
    # genuinely disagree on this input: the probe input is discriminating.
    good5b = (naive(amb, "alpha.beta") == 1 and v_flat == 99)
    print("  P5b precedence is longest-key-first      -> naive=%s dig_path=%s  %s"
          % (naive(amb, "alpha.beta"), v_flat, "PASS" if good5b else "FAIL"))
    ok &= good5b

    # P5c -- unambiguous paths must report exactly one candidate (no false alarm: a guard
    # that always cries ambiguity is as useless as one that never does).
    n_plain = len(dig_candidates(doc, "outer.plain.x"))
    n_dot = len(dig_candidates(doc, "outer.F_fixed_ag_0.5.rho_chi_to_chi"))
    n_absent = len(dig_candidates(doc, "nope.at.all"))
    good5c = (n_plain == 1 and n_dot == 1 and n_absent == 0)
    print("  P5c unambiguous/nested/absent candidate counts -> %d/%d/%d  %s"
          % (n_plain, n_dot, n_absent, "PASS" if good5c else "FAIL"))
    ok &= good5c

    print("  PROBE %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
