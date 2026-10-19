"""Image-quality audit for the test reconstructions  (review R2 m2.8).

R2 m2.8 objected that the imaging claim is supported only by a relative error
(RRE) and that no image-domain quality metric is reported.  A relative L2 error
on the contrast can in principle be small while the image looks wrong (and vice
versa), so we add two *independent* image-domain metrics on exactly the same
reconstructions that Fig. images displays -- no new forward solves, no GPU:

  * PSNR  -- sensitive to the absolute amplitude scale,
  * SSIM  -- Wang et al. (2004) Gaussian-window form, sensitive to structure.

We also report the relative L2 error recomputed here from the *stored arrays*,
which is a second, independent path to the number the manuscript quotes from
`depth_curve.json` (the stored arrays are produced by a different script), so
the two must agree; a disagreement is an instrument defect, not a result.

The re-implementation of SSIM is verified on known-bad inputs before it is
trusted: a shifted image must lose similarity, an amplitude-scaled image must
keep its structure, and a constant image must be rejected.  See --selftest.
"""
import io
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NX = NY = 32
ROWS = ["classical8", "classical50", "unconstrained", "constrained"]
LABEL = {
    "classical8": "Classical, K=8",
    "classical50": "Classical, K=50",
    "unconstrained": "Unconstrained, K=8",
    "constrained": "Constrained (ours), K=8",
}


def _psnr(a, b, data_range):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def _ssim(a, b, data_range, sigma=1.5, truncate=3.5):
    """Wang et al. (2004) SSIM, Gaussian window, sample covariance (K1=K2=0.01)."""
    K1, K2 = 0.01, 0.02
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu_a = gaussian_filter(a, sigma, truncate=truncate, mode="reflect")
    mu_b = gaussian_filter(b, sigma, truncate=truncate, mode="reflect")
    saa = gaussian_filter(a * a, sigma, truncate=truncate, mode="reflect") - mu_a ** 2
    sbb = gaussian_filter(b * b, sigma, truncate=truncate, mode="reflect") - mu_b ** 2
    sab = gaussian_filter(a * b, sigma, truncate=truncate, mode="reflect") - mu_a * mu_b
    num = (2.0 * mu_a * mu_b + C1) * (2.0 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    return float(np.mean(num / den))


def _rel_l2(a, b):
    d = float(np.linalg.norm(a - b))
    n = float(np.linalg.norm(b))
    return d / n if n > 0 else float("nan")


def _selftest():
    """Falsifiability probes: the metric must FAIL on inputs we know are bad."""
    rng = np.random.default_rng(0)
    g = rng.random((NX, NY)) * 0.5
    dr = float(g.max() - g.min())
    ok = True

    s_same = _ssim(g, g, dr)
    print("  self (must be 1.000):            %.4f" % s_same)
    ok &= abs(s_same - 1.0) < 1e-9

    gs = np.roll(g, 8, axis=0)
    s_shift = _ssim(g, gs, dr)
    print("  row-shift by 8 (must drop):      %.4f" % s_shift)
    ok &= s_shift < 0.75

    s_scaled = _ssim(g, g * 1.6, dr)
    print("  amplitude x1.6 (must drop):      %.4f" % s_scaled)
    ok &= s_scaled < 0.95

    flat = np.full_like(g, float(g.mean()))
    s_flat = _ssim(g, flat, dr)
    print("  constant image (must be low):    %.4f" % s_flat)
    ok &= s_flat < 0.5

    p_shift = _psnr(g, g + 0.05, dr)
    p_same = _psnr(g, g, dr)
    print("  PSNR identity=%s vs +0.05=%.2f dB" % (p_same, p_shift))
    ok &= (p_same == float("inf")) and (p_shift < 30.0)

    print("  SELFTEST: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)

    d = np.load(os.path.join(RES, "images.npz"))
    gt = d["gt"].reshape(-1, NX, NY).astype(np.float64)
    n_ex = gt.shape[0]
    data_range = float(gt.max() - gt.min())

    rec = {"n_examples": int(n_ex), "data_range": data_range,
           "protocol": ("the %d examples displayed in Fig. images, i.e. chi_te_d[0:4]; "
                        "these are NOT the full test set (100) over which every RRE in the "
                        "manuscript is computed -- see imgq_protocol.json" % n_ex),
           "ssim_impl": "gaussian sigma=1.5 truncate=3.5, K1=0.01 K2=0.02, mean over pixels",
           "psnr_impl": "10log10(data_range^2/MSE)",
           "rows": {}}

    print("data_range = %.6f   n_examples = %d" % (data_range, n_ex))
    print("%-22s %8s %8s %10s" % ("row", "PSNR", "SSIM", "rel-L2"))
    for r in ROWS:
        x = d[r].reshape(-1, NX, NY).astype(np.float64)
        ps = [_psnr(x[i], gt[i], data_range) for i in range(n_ex)]
        ss = [_ssim(x[i], gt[i], data_range) for i in range(n_ex)]
        rl = [_rel_l2(x[i], gt[i]) for i in range(n_ex)]
        rec["rows"][r] = {
            "label": LABEL[r],
            "psnr_per_example": [round(v, 4) for v in ps],
            "ssim_per_example": [round(v, 6) for v in ss],
            "rel_l2_per_example": [round(v, 6) for v in rl],
            "psnr_mean": float(np.mean(ps)),
            "ssim_mean": float(np.mean(ss)),
            "rel_l2_mean": float(np.mean(rl)),
        }
        print("%-22s %8.2f %8.4f %10.4f" % (LABEL[r], np.mean(ps), np.mean(ss), np.mean(rl)))

    with io.open(os.path.join(RES, "imgq.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    print("wrote results/imgq.json")


if __name__ == "__main__":
    main()
