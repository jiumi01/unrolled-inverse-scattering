# -*- coding: utf-8 -*-
"""_regen_gain_decomp.py -- regenerate results/gain_decomp.json to pick up the
PROVENANCE block added to run_gain_decomp.py after round-3 review (R1 t3/Q3.2),
WITHOUT letting the regeneration silently move any load-bearing number.

Why this guard exists
---------------------
The artifact in results/ (11:15) predates the driver patch (12:44): the driver now
writes out["provenance"] naming the exact operating point (Born-LS initial state,
proj on, alpha=gamma=0.5 forced, float64, batch 16, Lanczos tol 1e-5, k=argmax),
but the shipped JSON does not carry it.  Re-running the driver is the only honest
way to get that block -- provenance must be produced by the run, not typed in
afterwards.  But re-running also re-solves the Lanczos problems, and those are the
numbers Table~\\ref{tab:cert} quotes.  So the run is bracketed:

  1. copy results/gain_decomp.json -> results/gain_decomp_preprov.json (immutable
     backup, with an archive note),
  2. run the driver as a subprocess,
  3. compare EVERY numeric leaf of the new artifact against the backup with a
     relative tolerance of 1e-9 (and every list element / every key),
  4. require out["provenance"] to be present and to name the state, proj, step
     sizes, layer index, checkpoint and driver command,
  5. on any violation, restore the backup and exit non-zero.

A failure here is information, not an obstacle: it would mean the archiving
precision of the two runs differs, which is exactly the class of defect the
round-3 review (R1 t3) is about.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
ART = os.path.join(RES, "gain_decomp.json")
BAK = os.path.join(RES, "gain_decomp_preprov.json")
TOL = 1e-9
REQUIRED_PROV = ("state", "proj", "step_sizes", "layer_index", "checkpoint",
                 "driver_command")


def leaves(o, path=""):
    """Yield (path, value) for every scalar/list-element leaf."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, "%s.%s" % (path, k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, "%s[%d]" % (path, i))
    else:
        yield path, o


def main():
    if not os.path.exists(ART):
        print("[FAIL] %s missing" % ART)
        return 2
    old = json.load(open(ART))
    n_old = sum(1 for _ in leaves(old))

    note = {
        "role": "immutable pre-provenance backup of results/gain_decomp.json",
        "archived_because": ("the shipped artifact predates the PROVENANCE patch to "
                             "code/run_gain_decomp.py (round-3 review R1 t3/Q3.2)"),
        "artifact_mtime_before": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ART))),
        "numbers": old,
    }
    if not os.path.exists(BAK):
        json.dump(note, open(BAK, "w"), indent=1)
        print("[backup] wrote %s" % os.path.basename(BAK))
    else:
        print("[backup] %s already exists, left untouched (immutable)"
              % os.path.basename(BAK))

    def restore(reason):
        """Put the backup's numbers back as the shipped artifact."""
        b = json.load(open(BAK))["numbers"]
        json.dump(b, open(ART, "w"), indent=1)
        print("[ROLLBACK] %s -- restored the pre-provenance artifact" % reason)

    print("[run] python code/run_gain_decomp.py ... (this re-solves A/B/C)")
    t0 = time.time()
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "2"
    p = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_gain_decomp.py")],
                       env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    tail = "\n".join((p.stdout or "").strip().splitlines()[-8:])
    print(tail)
    if p.returncode != 0:
        print("[FAIL] driver exited %d" % p.returncode)
        print((p.stderr or "")[-1500:])
        restore("driver failed")
        return 4
    print("[run] done in %.0fs" % (time.time() - t0))

    new = json.load(open(ART))
    old_nums = dict(leaves(old))
    new_nums = dict(leaves(new))
    missing = sorted(set(old_nums) - set(new_nums))
    extra = sorted(set(new_nums) - set(old_nums))
    drift, worst = [], (0.0, "")
    for k in sorted(set(old_nums) & set(new_nums)):
        a, b = old_nums[k], new_nums[k]
        if isinstance(a, bool) or isinstance(b, bool):
            continue
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        d = abs(b - a) / max(abs(a), 1e-30)
        if d > worst[0]:
            worst = (d, k)
        if d > TOL:
            drift.append((k, a, b, d))

    prov = new.get("provenance") or {}
    prov_missing = [k for k in REQUIRED_PROV if not prov.get(k)]

    print("\n=== guard report ===")
    print("leaves: old %d / new %d ; missing %d ; extra %d"
          % (n_old, len(new_nums), len(missing), len(extra)))
    print("worst relative drift: %.3e at %s" % (worst[0], worst[1]))
    ok = True
    if missing:
        print("[FAIL] keys lost: %s" % missing[:8]); ok = False
    if extra:
        print("[note] new keys: %s" % extra[:8])
    if drift:
        print("[FAIL] %d numeric leaf/leaves moved beyond %.0e:" % (len(drift), TOL))
        for k, a, b, d in drift[:10]:
            print("   %-46s %r -> %r  (rel %.3e)" % (k, a, b, d))
        ok = False
    if prov_missing:
        print("[FAIL] provenance incomplete, missing %s" % prov_missing); ok = False
    else:
        print("[PASS] provenance present: %s" % ", ".join(REQUIRED_PROV))

    out = {"numbers_identical": not drift and not missing,
           "n_leaves_old": n_old, "n_leaves_new": len(new_nums),
           "worst_rel_drift": worst[0], "worst_rel_drift_key": worst[1],
           "tolerance": TOL, "keys_missing": missing, "keys_added": extra,
           "provenance_fields": sorted(prov.keys()),
           "guard_passed": bool(ok)}
    json.dump(out, open(os.path.join(RES, "gaindecomp_regen_guard.json"), "w"), indent=1)

    if not ok:
        restore("guard violations above")
        return 3
    print("\n[OK] artifact regenerated with provenance; numbers unchanged "
          "(worst rel drift %.2e)" % worst[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
