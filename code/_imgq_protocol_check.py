"""Protocol reconciliation for the image-quality metrics (review R2 m2.8).

`_img_quality.py` reports a relative-L2 error of ~0.348 (constrained, K=8) while
the manuscript quotes ~0.407 from depth_curve.json.  Those two numbers are not
allowed to sit in the same paper unexplained, so this script decides WHY they
differ instead of guessing:

  H1 (subset)   -- depth_curve uses eval_model(chi_te_d) i.e. the whole test set;
                   images.npz is written from chi_te_d[0:4], i.e. 4 examples.
  H2 (formula)  -- the relative-L2 definitions differ (norm, denominator, or
                   averaging order).

H1 is tested by evaluating the SAME checkpoints on chi_te_d[0:4] and on the full
test set with the SAME rre() function, and comparing each against the value
recovered from images.npz.  If the 4-example value reproduces images.npz, H1 is
confirmed and the two numbers are simply two protocols -- which is what the
manuscript must then say explicitly.  If it does not, H2 is live and the metric
is not trustworthy for inclusion.
"""
import os
import sys
import io
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_experiments import (init_state, rre, eval_model, load_data, UnrolledISP,  # noqa: E402
                           NX, NY, N_INC)

RES = os.path.join(os.path.dirname(HERE), "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # load_data() is the exact call path that produced depth_curve.json and
    # images.npz; reusing it guarantees the same test set and the same split.
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)
    # Gb, Gs, Ei are numpy arrays at this point (load_data torchifies only the data);
    # UnrolledISP handles the conversion internally, exactly as in run_experiments.
    print("test set: chi_te %s  (device %s)" % (tuple(chi_te_d.shape), DEV))

    npz = np.load(os.path.join(RES, "images.npz"))
    nx = ny = 32

    def stored_rel_l2(row, n_ex):
        gt = npz["gt"].reshape(-1, nx, ny).astype(np.float64)
        x = npz[row].reshape(-1, nx, ny).astype(np.float64)
        return float(np.mean([np.linalg.norm(x[i] - gt[i]) / max(np.linalg.norm(gt[i]), 1e-12)
                              for i in range(n_ex)]))

    print("\n%-24s %10s %10s %10s %10s" % ("arm", "full-set", "first-4", "npz(4)", "match?"))
    rec = {"n_test_full": int(chi_te_d.shape[0]), "n_displayed": 4,
           "note": ("full-set = eval_model over chi_te_d (the protocol behind "
                    "depth_curve.json and every RRE quoted in the manuscript); "
                    "first-4 = chi_te_d[0:4], the subset stored in images.npz and "
                    "shown in Fig. images"), "arms": {}}
    all_ok = True
    for mode, K in [("classical", 8), ("constrained", 8), ("unconstrained", 8)]:
        m = UnrolledISP(K, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV)
        if mode != "classical":
            m.load_state_dict(torch.load(os.path.join(RES, "ckpt_%s_K%d.pt" % (mode, K)),
                                         map_location=DEV))
        full = eval_model(m, chi_te_d, y_te_d)
        with torch.no_grad():
            n4 = 4
            c0, w0 = init_state(m, y_te_d[:n4], n4)
            c4, _ = m(c0, w0, y_te_d[:n4])
            first4 = rre(c4, chi_te_d[:n4])
        stored = stored_rel_l2({"classical": "classical8", "constrained": "constrained",
                                "unconstrained": "unconstrained"}[mode], n4)
        ok = abs(first4 - stored) < 5e-3
        all_ok &= ok
        rec["arms"]["%s_K%d" % (mode, K)] = {
            "rre_full_test": round(float(full), 6),
            "rre_first_4": round(float(first4), 6),
            "rre_recovered_from_npz": round(float(stored), 6),
            "subset_hypothesis_confirmed": bool(ok),
        }
        print("%-24s %10.4f %10.4f %10.4f %10s"
              % ("%s_K%d" % (mode, K), full, first4, stored, "YES" if ok else "NO"))
        del m

    rec["H1_subset_confirmed"] = bool(all_ok)
    with io.open(os.path.join(RES, "imgq_protocol.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    print("\nwrote results/imgq_protocol.json   H1(subset)=%s" % all_ok)

    print("\nIf 'match?' is YES everywhere, the manuscript must state BOTH protocols;")
    print("the PSNR/SSIM table is over the 4 displayed examples and must be labelled so.")


if __name__ == "__main__":
    main()
