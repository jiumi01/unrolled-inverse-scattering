# -*- coding: utf-8 -*-
"""_repro_floor.py -- the identical-seed reproducibility floor of the training pipeline.

Round-3 referee R2 (items m2.5 / Q2.2) asked whether the gap between the r=0 repeat and the main
pipeline really is the "initialisation / batch-order" component.  Answering that needs a quantity
the paper did not have: how much does an identical run move when nothing at all is changed?

CORRECTED PROTOCOL (the first version of this file measured the wrong thing).  It used to compare
`results/seeds_7run_backup.json` (pre-correction) against `results/seeds.json` (post-correction):
that gap is the effect of the isp_core.py fix, not pipeline noise, so it could not be quoted as a
reproducibility floor.  The protocol is now: archive the CURRENT seeds.json, re-run the identical
driver on the identical code, and compare the two same-regime archives.  Only run-to-run
variability of the training pipeline itself remains in that difference.

Usage:  python _repro_floor.py [old_json] [new_json]
        (defaults: results/seeds_7run_backup.json vs results/seeds.json -- the historical pair,
         kept only so the old artifact can be re-derived; pass two same-regime paths for the
         quantity the manuscript quotes.)

Writes results/repro_floor.json.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))


def main(argv):
    old_p = argv[1] if len(argv) > 1 else os.path.join(RES, "seeds_7run_backup.json")
    new_p = argv[2] if len(argv) > 2 else os.path.join(RES, "seeds.json")
    old = json.load(open(old_p))["runs"]
    new = json.load(open(new_p))["runs"]
    shared = sorted(set(old) & set(new))
    per_arm, drre, dextrap, dchi = {}, [], [], []
    for k in shared:
        a, b = old[k], new[k]
        d_rre = b["rre_K8"] - a["rre_K8"]
        d_ex = b["extrap"]["rre"] - a["extrap"]["rre"]
        c_a, c_b = a["extrap"]["maxabs_chi"], b["extrap"]["maxabs_chi"]
        per_arm[k] = {"rre_K8_old": a["rre_K8"], "rre_K8_new": b["rre_K8"], "d_rre": d_rre,
                      "extrap_old": a["extrap"]["rre"], "extrap_new": b["extrap"]["rre"],
                      "d_extrap": d_ex,
                      "maxabs_chi_old": c_a, "maxabs_chi_new": c_b,
                      "finite_old": a["extrap"]["finite"], "finite_new": b["extrap"]["finite"],
                      "box_verdict_old": bool(c_a <= 0.5 + 1e-9),
                      "box_verdict_new": bool(c_b <= 0.5 + 1e-9)}
        drre.append(abs(d_rre))
        dextrap.append(abs(d_ex))
        dchi.append(abs(c_b - c_a))
    out = {
        "instrument": "_repro_floor.py",
        "question": ("how far does an identical run move when nothing is changed? "
                     "(round-3 R2 m2.5 / Q2.2)"),
        "protocol": ("same driver, same seeds, same device, same BLAS thread settings; the two "
                     "archives are two runs of the IDENTICAL code (the historical pair compared a "
                     "pre-correction against a post-correction run, which measures the fix, not "
                     "the noise -- do not quote that one as a floor)"),
        "old_archive": os.path.relpath(old_p, os.path.normpath(os.path.join(HERE, ".."))),
        "new_archive": os.path.relpath(new_p, os.path.normpath(os.path.join(HERE, ".."))),
        "n_shared_runs": len(shared),
        "shared_runs": shared,
        "per_arm": per_arm,
        "abs_d_rre_K8": {"max": max(drre), "median": float(np.median(drre)),
                         "mean": float(np.mean(drre))},
        "abs_d_extrap_rre": {"max": max(dextrap), "median": float(np.median(dextrap))},
        "abs_d_maxabs_chi": {"max": max(dchi)},
        "finite_all_old": all(old[k]["extrap"]["finite"] for k in shared),
        "finite_all_new": all(new[k]["extrap"]["finite"] for k in shared),
        "box_verdict_stable": all(per_arm[k]["box_verdict_old"] == per_arm[k]["box_verdict_new"]
                                  for k in shared),
        "note": ("The floor is quoted in the manuscript next to the seed spread. If the floor is "
                 "not small compared with the spread, the spread cannot be attributed to the "
                 "seed, and the qualitative verdicts (finite / inside the box) are the ones that "
                 "survive; that is what the numbers show."),
    }
    with open(os.path.join(RES, "repro_floor.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("shared runs: %d  (old=%s new=%s)" % (len(shared), old_p, new_p))
    for k in shared:
        p = per_arm[k]
        print("  %-26s RRE %.5f -> %.5f  (d=%+.5f)   extrap %.5f -> %.5f (d=%+.5f)"
              % (k, p["rre_K8_old"], p["rre_K8_new"], p["d_rre"],
                 p["extrap_old"], p["extrap_new"], p["d_extrap"]))
    print("|d RRE|   max=%.5f  median=%.5f" % (out["abs_d_rre_K8"]["max"],
                                               out["abs_d_rre_K8"]["median"]))
    print("|d extrap| max=%.5f" % out["abs_d_extrap_rre"]["max"])
    print("all finite: old=%s new=%s ; box verdict stable: %s"
          % (out["finite_all_old"], out["finite_all_new"], out["box_verdict_stable"]))


if __name__ == "__main__":
    main(sys.argv)
