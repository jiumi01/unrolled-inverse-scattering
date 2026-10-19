"""Effective noise floor of the "noiseless" evaluation set (R2 m3.9).

The depth and extrapolation studies are run on a synthesis set generated with
snr_db = 300, which the paper calls "effectively noiseless".  That phrase is
only defensible if the *stored* representation can actually hold a 300 dB SNR:
the generator computes in float64 and then casts the field to complex64, so the
delivered data cannot be quieter than the complex64 representation error.  This
probe measures that floor instead of inferring it.

Protocol: regenerate the production test set (100 samples, seed 2, snr_db=300,
the exact call in run_experiments.load_data) in float64, cast a copy to
complex64 exactly as the generator does, and report the relative round-trip
error.  No GPU, no training, no checkpoint.

Writes results/noisefloor.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isp_core as ic          # noqa: E402
import run_experiments as RE   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    Gb, Gs, pts, rx = ic.build_matrices(RE.NX, RE.NY, dx=0.02)
    dirs = ic.make_dirs(RE.N_INC)
    kb = 2 * np.pi
    Ei = ic.incident_fields(pts, dirs, kb)

    rng = np.random.default_rng(2)
    y64 = []
    chi64 = []
    for _ in range(100):
        chi = ic.random_contrast(RE.NX, RE.NY, rng)
        _, _, y = ic.forward_exact(Gb, Gs, Ei, chi)
        y64.append(ic.add_noise(y, 300.0, rng))
        chi64.append(chi)
    y64 = np.stack(y64)          # complex128, post-noise, pre-cast
    chi128 = np.stack(chi64)     # float64

    y_c64 = y64.astype(np.complex64).astype(np.complex128)
    d_y = np.linalg.norm(y_c64 - y64) / np.linalg.norm(y64)
    chi_f32 = chi128.astype(np.float32).astype(np.float64)
    # relative error against the per-sample contrast norm (what an RRE would see)
    d_chi = (np.linalg.norm(chi_f32 - chi128, axis=1)
             / (np.linalg.norm(chi128, axis=1) + 1e-30)).mean()

    out = {
        "protocol": {
            "n_samples": 100,
            "seed": 2,
            "snr_db_requested": 300.0,
            "note": "synthesis in float64 (as the generator computes it), then the "
                    "generator's own complex64/float32 cast applied to a copy; the "
                    "relative round-trip error is the quietest signal the stored data "
                    "can represent",
        },
        "field_roundtrip_rel": float(d_y),
        "field_floor_db": float(20 * np.log10(d_y)),
        "contrast_roundtrip_rel_mean": float(d_chi),
        "contrast_floor_db": float(20 * np.log10(d_chi)),
        "nominal_snr_db": 300.0,
        "effective_snr_db": float(-20 * np.log10(d_y)),
        "verdict_nominal_exceeds_storage": bool(300.0 > -20 * np.log10(d_y)),
    }
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "noisefloor.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("\nNOISE-FLOOR PROBE DONE")


if __name__ == "__main__":
    main()
