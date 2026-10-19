# -*- coding: utf-8 -*-
"""_restamp_seeds_note.py — re-stamp results/seeds.json:config.note from run_seeds.SEED_NOTE.

R3 m3.12: the archived seeds.json carried the *old* note ("r=0 ... must reproduce
depth_curve.json"), which the artifact's own numbers falsify (r0_constrained.rre_K8 = 0.40180
against depth_curve.json:constrained_K8 = 0.40673; the training-data seed is shared but the
initialisation and batch order are not).  The driver was already corrected; only the archived
copy was stale.

The note is *metadata*, not a measurement, so re-stamping it does not require re-running the
seven trainings.  Everything else in the file is preserved byte-for-byte in value, and the
script refuses to touch the measured fields.  It also records where the text came from, so a
later reader can tell that this file was re-stamped rather than re-run.

Verification: run code/_gate_deliverable.py afterwards -- G9 asserts the artifact note equals
run_seeds.SEED_NOTE, so the two can never drift apart again.
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from _gate_deliverable import driver_str_const  # noqa: E402

ART = os.path.join(ROOT, "results", "seeds.json")
DRV = os.path.join(HERE, "run_seeds.py")

# Fields this script must never alter: they are measurements.
MEASURED = ("runs",)


def main():
    note, err = driver_str_const(DRV, "SEED_NOTE")
    if err is not None:
        print("[fatal] cannot read SEED_NOTE from run_seeds.py: %s" % err)
        return 2
    if not os.path.exists(ART):
        print("[fatal] %s missing" % ART)
        return 2

    d = json.load(io.open(ART, encoding="utf-8"))
    before = json.dumps({k: v for k, v in d.items() if k in MEASURED}, sort_keys=True)
    old = d.get("config", {}).get("note")

    if old == note:
        print("[noop] seeds.json:config.note already matches run_seeds.SEED_NOTE")
        return 0

    d.setdefault("config", {})["note"] = note
    d["config"]["note_source"] = ("run_seeds.py::SEED_NOTE (re-stamped by "
                                  "code/_restamp_seeds_note.py; the measured fields are "
                                  "untouched, this is metadata only)")

    after = json.dumps({k: v for k, v in d.items() if k in MEASURED}, sort_keys=True)
    if before != after:
        print("[fatal] refusing to write: a measured field changed")
        return 2

    with io.open(ART, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    print("[stamp] seeds.json:config.note updated")
    print("        old: %r..." % (old or "")[:70])
    print("        new: %r..." % note[:70])
    return 0


if __name__ == "__main__":
    sys.exit(main())
