# -*- coding: utf-8 -*-
"""run_fresnel_baselines.py -- reference reconstructions for the measured-data figure.

The measured-data study (Sec. 6.11) reported only the recovered contrast next to the
published phantom, so a reader had no way to tell what the unrolled physics kernel
bought over the standard alternatives on the same data.  This script produces two:

  born    -- one linearised (Born) contrast solve at the incident field, i.e. the
             first iterate of the same kernel with neither iteration nor step search;
  fixed   -- the same kernel run for 30 layers at the canonical alpha=gamma=0.5,
             i.e. no per-target step search (the archived arm picks the pair with the
             smallest label-free residual out of three candidates).

Both use the identical geometry, calibration constant, projection box and CG solve as
run_fresnel_recon_v2.py (imported, not re-implemented), so the comparison isolates the
algorithmic difference.  Nothing here is trained; the correction network is not involved,
because the trained weights collapse on this geometry (Sec. 6.11).

Writes results/fresnel_baselines.json and results/fresnel_baselines.npz.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_fresnel_recon_v2 as R

RES = R.RES
FREQ = 3.0
TARGETS = R.TARGETS


def rre(chi, truth):
    d = np.linalg.norm(chi - truth)
    return float(d / np.linalg.norm(truth))


def main():
    t0 = time.time()
    rows, imgs = [], {}
    # one load per phantom: the instrument constant is needed both for its own case and
    # as a calibration donor for the other two
    cinfo = dict((t, R.instr_const(t, FREQ)) for t in TARGETS)
    for t in TARGETS:
        _, d, fi, g = cinfo[t]
        # per-target calibration constant: mean of the OTHER two phantoms' constants,
        # exactly as in run_fresnel_recon_v2.main (target-model-assisted, not blind)
        a = complex(np.mean([cinfo[o][0] for o in TARGETS if o != t]))
        Es = d["escat"][fi] / a
        Y = R.true_geom(t, FREQ).ravel()

        # (0) adjoint / DSM: a single CG step from zero, i.e. the matched filter that
        #     the measured-data literature uses for qualitative imaging
        chi_dsm = np.clip(R.cg_contrast(g, g.Einc, Es, n_cg=1), 0.0, R.CHI_MAX)
        # (1) Born: the same regularised LS solve run to the paper's CG budget, with E
        #     still held at the incident field
        chi_born = np.clip(R.cg_contrast(g, g.Einc, Es), 0.0, R.CHI_MAX)
        # (2) the same kernel as the paper, 30 layers, canonical alpha=gamma=0.5, no search
        chi_fix, w_fix = R.unrolled_run(g, Es, K_max=30, alpha=0.5, gamma=0.5, proj=True)

        for arm, chi, mw in [("dsm", chi_dsm, None),
                             ("born", chi_born, None),
                             ("fixed", chi_fix, float(np.abs(w_fix).max()))]:
            rows.append(dict(target=t, freq=FREQ, arm=arm, rre=rre(chi, Y),
                             mse=float(np.mean((chi - Y) ** 2)),
                             r_net=R.residual(g, chi, Es),
                             max_chi=float(np.abs(chi).max()), max_w=mw))
            imgs["%s_%s" % (t, arm)] = chi.reshape(R.N, R.N).astype(np.float32)
        print("%-13s dsm RRE=%.3f  born RRE=%.3f  fixed RRE=%.3f" %
              (t, rows[-3]["rre"], rows[-2]["rre"], rows[-1]["rre"]), flush=True)

    # the archived arm, same numbers Table 12 quotes, so the four panels share one scale
    for t in TARGETS:
        pair = np.load(os.path.join(RES, "fresnel_recon_v2_recons.npz"))["%s_3" % t]
        imgs["%s_ours" % t] = pair[0].astype(np.float32)
        Y = pair[1].ravel()
        chi = pair[0].ravel()
        rows.append(dict(target=t, freq=FREQ, arm="ours", rre=rre(chi, Y),
                         mse=float(np.mean((chi - Y) ** 2)),
                         r_net=None, max_chi=float(np.abs(chi).max()), max_w=None))

    with open(os.path.join(RES, "fresnel_baselines.json"), "w") as f:
        json.dump(rows, f, indent=1)
    np.savez_compressed(os.path.join(RES, "fresnel_baselines.npz"), **imgs)
    print("[baselines done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
