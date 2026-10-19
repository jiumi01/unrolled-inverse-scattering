# -*- coding: utf-8 -*-
"""_regen_all.py -- orchestrator for the post-fix regeneration of every artifact.

Context.  isp_core.py was corrected on 2026-10-18 08:35 (forward_exact's least-squares
column scaling), and a float32 dtype regression was found in run_bjac.py on 10-19.
Every manuscript number is single-sourced from results/*.json through
code/gen_numbers.py, so the artifacts below are re-generated on the corrected code
BEFORE any number is re-quoted.  Scope was closed programmatically: every
load("...json") in gen_numbers.py was checked against the fix timestamp and against
its own input dependencies (a fresh mtime is not sufficient -- cg_sweep.json was
rewritten on 10-18 but still consumed the PRE-fix cg_cg200 checkpoint).

Two chains, each strictly serial, running concurrently (they share no artifact):

  CPU chain (float64 audits; CPU-pinned so they never contend with GPU training and
  so every evaluation of the same operator sits on the same arithmetic -- GPU vs CPU
  differ by ~3% through the ill-conditioned contrast solve):
    noisefloor -> imgq(+selftest) -> cost_probe -> ngc -> bjac -> L45 -> chi_lip
      -> gain_decomp -> gain_decomp_pair(selfcheck+clean) -> t1_probe
      -> gaindecomp_prov -> prov_state -> gaindecomp_prov (2nd pass: the two prov
         drivers read each other's artifact, so the pair is run twice)
      -> gain_refine

  GPU chain (training-class jobs, float32, serial):
    classical_sweep -> [backup seeds.json] run_seeds -> repro_floor(new protocol)
      -> _retrain_depth -> cg_sweep_verify -> _retrain_cg200 -> run_cg_sweep
      -> _cgloss_probe -> imgq_protocol

Idempotent: a finished job leaves results/_regen/done/<job>.done and is skipped on
the next invocation, so an interrupted run resumes where it stopped.  --force
re-runs everything; --chain cpu|gpu runs one chain only.

Usage:
  python code/_regen_all.py                 # both chains, resume
  python code/_regen_all.py --force         # ignore markers
  python code/_regen_all.py --chain cpu     # one chain

Per-job logs: results/_regen/logs/<job>.log ; heartbeat: results/_regen/heartbeat.log
and results/_regen/status.json (rewritten every 30 s).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
REGEN = os.path.join(RES, "_regen")
LOGS = os.path.join(REGEN, "logs")
PY = "C:/Users/up/miniconda3/envs/wirenet/python.exe"
SEEDS_BACKUP = os.path.join(REGEN, "seeds_prefloor_backup.json")


def script(name):
    return os.path.join(HERE, name)


def backup_seeds():
    """Archive the current (post-fix) seeds.json BEFORE the identical re-run, so
    _repro_floor.py compares two same-regime runs instead of the historical
    pre-fix/post-fix pair (which measured the fix, not the pipeline noise)."""
    src = os.path.join(RES, "seeds.json")
    if os.path.exists(SEEDS_BACKUP):
        return "backup already present"
    shutil.copy2(src, SEEDS_BACKUP)
    return "archived %s -> %s" % (src, SEEDS_BACKUP)


# --------------------------------------------------------------------------- #
# job table
# --------------------------------------------------------------------------- #
def J(name, chain, steps, pre=None, truncate=()):
    return dict(name=name, chain=chain, steps=steps, pre=pre,
                truncate=list(truncate))


JOBS = [
    # ---------------- CPU chain ----------------
    J("noisefloor", "cpu", [[PY, script("_noisefloor_probe.py")]]),
    J("imgq", "cpu", [[PY, script("_img_quality.py"), "--selftest"],
                      [PY, script("_img_quality.py")]]),
    J("cost_probe", "cpu", [[PY, script("_cost_probe.py")]]),
    J("ngc", "cpu", [[PY, script("_ngc_audit.py")]],
      truncate=["ngc_progress.txt"]),
    J("bjac", "cpu", [[PY, script("run_bjac.py")]],
      truncate=["bjac_progress.txt"]),
    J("L45", "cpu", [[PY, script("_run_L45.py")]]),
    J("chi_lip", "cpu", [[PY, script("_chi_lip_audit.py")]]),
    J("gain_decomp", "cpu", [[PY, script("run_gain_decomp.py")]]),
    J("gain_decomp_pair", "cpu", [[PY, script("run_gain_decomp_pair.py"), "selfcheck"],
                                  [PY, script("run_gain_decomp_pair.py"), "clean"]]),
    J("t1_probe", "cpu", [[PY, script("_probe_T1_blockLL.py")]]),
    J("gaindecomp_prov_p1", "cpu", [[PY, script("_gaindecomp_prov.py"), "cpu"]]),
    J("prov_state", "cpu", [[PY, script("_prov_state_probe.py"), "cpu"]]),
    J("gaindecomp_prov_p2", "cpu", [[PY, script("_gaindecomp_prov.py"), "cpu"]]),
    J("gain_refine", "cpu", [[PY, script("run_gain_refine.py")]]),
    # ---------------- GPU chain ----------------
    J("classical_sweep", "gpu", [[PY, script("run_classical_sweep.py")]]),
    J("seeds_rerun", "gpu", [[PY, script("run_seeds.py")]], pre=backup_seeds),
    J("repro_floor", "gpu", [[PY, script("_repro_floor.py"), SEEDS_BACKUP,
                              os.path.join(RES, "seeds.json")]]),
    J("retrain_depth", "gpu", [[PY, script("_retrain_depth.py")]]),
    J("cg_sweep_verify", "gpu", [[PY, script("_cg_sweep_verify.py")]]),
    J("retrain_cg200", "gpu", [[PY, script("_retrain_cg200.py")]]),
    J("cg_sweep", "gpu", [[PY, script("run_cg_sweep.py")]]),
    J("cgloss_probe", "gpu", [[PY, script("_cgloss_probe.py")]]),
    J("imgq_protocol", "gpu", [[PY, script("_imgq_protocol_check.py")]]),
]

STATE = {}          # job name -> dict(state, started, finished, rc, log)
LOCK = threading.Lock()
CHAIN_STOP = {"cpu": threading.Event(), "gpu": threading.Event()}


def logline(msg):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(REGEN, "heartbeat.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status():
    with LOCK:
        snap = {k: dict(v) for k, v in STATE.items()}
    doc = {"updated": time.strftime("%m-%d %H:%M:%S"),
           "jobs": {k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in snap.items()}}
    tmp = os.path.join(REGEN, "status.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, os.path.join(REGEN, "status.json"))


def heartbeat():
    while True:
        with LOCK:
            running = [(k, STATE[k]["started"]) for k, v in STATE.items()
                       if v["state"] == "running"]
        for k, st in running:
            p = os.path.join(LOGS, k + ".log")
            sz = os.path.getsize(p) if os.path.exists(p) else 0
            el = time.time() - st
            logline("alive  %-20s elapsed=%6.0fs  log=%dB" % (k, el, sz))
        write_status()
        if (not running and CHAIN_STOP["cpu"].is_set()
                and CHAIN_STOP["gpu"].is_set()):
            break
        time.sleep(30)
    write_status()


def run_job(job):
    name = job["name"]
    logp = os.path.join(LOGS, name + ".log")
    with LOCK:
        STATE[name] = dict(state="running", started=time.time(), finished=None,
                           rc=None, log=logp)
    logline("START %-20s (%s)" % (name, job["chain"]))
    t0 = time.time()
    try:
        for fn in job["truncate"]:
            p = os.path.join(RES, fn)
            if os.path.exists(p):
                open(p, "w").close()          # fresh progress log per regeneration
        if job["pre"] is not None:
            msg = job["pre"]()
            logline("  pre  %-20s %s" % (name, msg))
        with open(logp, "a", encoding="utf-8") as lf:
            lf.write("\n===== %s start %s =====\n" % (name, time.strftime("%m-%d %H:%M:%S")))
            for cmd in job["steps"]:
                lf.write("\n$ %s\n" % " ".join(cmd))
                lf.flush()
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
                if rc != 0:
                    raise RuntimeError("step failed rc=%d: %s" % (rc, " ".join(cmd)))
        with LOCK:
            STATE[name].update(state="done", finished=time.time(), rc=0)
        with open(os.path.join(REGEN, "done", name + ".done"), "w", encoding="utf-8") as f:
            json.dump({"job": name, "wallclock_s": time.time() - t0,
                       "finished": time.strftime("%m-%d %H:%M:%S")}, f, indent=1)
        logline("DONE  %-20s %.0fs" % (name, time.time() - t0))
        return True
    except Exception as e:                       # noqa: BLE001 - report and stop the chain
        with LOCK:
            STATE[name].update(state="failed", finished=time.time(), rc=1)
        with open(os.path.join(REGEN, "fail", name + ".fail"), "w", encoding="utf-8") as f:
            f.write("%s\n%s\n" % (e, time.strftime("%m-%d %H:%M:%S")))
        logline("FAIL  %-20s %.0fs  %s" % (name, time.time() - t0, e))
        return False


def run_chain(chain, force):
    try:
        for job in JOBS:
            if job["chain"] != chain:
                continue
            if CHAIN_STOP[chain].is_set():
                logline("chain %s: halted after a failed job" % chain)
                return
            marker = os.path.join(REGEN, "done", job["name"] + ".done")
            if os.path.exists(marker) and not force:
                logline("SKIP  %-20s (marker present)" % job["name"])
                with LOCK:
                    STATE[job["name"]] = dict(state="skipped", started=None,
                                              finished=None, rc=0,
                                              log=os.path.join(LOGS, job["name"] + ".log"))
                continue
            ok = run_job(job)
            if not ok:
                # Halt only THIS chain.  The two chains share no artifact (verified:
                # no GPU-chain job reads a CPU-chain output and vice versa), so the
                # sibling chain's work is not wasted by a failure here.
                CHAIN_STOP[chain].set()
                return
    finally:
        CHAIN_STOP[chain].set()      # normal completion also releases the heartbeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ignore done markers and re-run everything")
    ap.add_argument("--chain", choices=["cpu", "gpu"], default=None,
                    help="run only one chain")
    args = ap.parse_args()

    for d in (REGEN, LOGS, os.path.join(REGEN, "done"), os.path.join(REGEN, "fail")):
        os.makedirs(d, exist_ok=True)

    if args.force:
        for job in JOBS:
            for d in ("done", "fail"):
                p = os.path.join(REGEN, d, job["name"] + "." + d)
                if os.path.exists(p):
                    os.remove(p)
        logline("force: markers cleared")

    logline("orchestrator start (python=%s, chains=%s)" %
            (PY, args.chain or "cpu+gpu"))

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    chains = [args.chain] if args.chain else ["cpu", "gpu"]
    workers = []
    for ch in chains:
        t = threading.Thread(target=run_chain, args=(ch, args.force), name="chain-" + ch)
        t.start()
        workers.append(t)
    for t in workers:
        t.join()
    CHAIN_STOP["cpu"].set()
    CHAIN_STOP["gpu"].set()
    hb.join(timeout=5)

    with LOCK:
        states = {k: v["state"] for k, v in STATE.items()}
    n_done = sum(1 for s in states.values() if s in ("done", "skipped"))
    n_fail = sum(1 for s in states.values() if s == "failed")
    logline("orchestrator end: %d ok, %d failed, %d not run" %
            (n_done, n_fail, len(JOBS) - n_done - n_fail))
    write_status()
    return 0 if n_fail == 0 and n_done == len(JOBS) else 1


if __name__ == "__main__":
    sys.exit(main())
