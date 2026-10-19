# -*- coding: utf-8 -*-
"""_probe_T1_blockLL.py -- falsification probe for the layer-Jacobian decomposition.

Round-3 referee R1 (item T1) claims the sentence

    "The full layer Jacobian is DT_k = J + DH_k ... three lower bounds ... hence
     a fortiori on DT_k"

cannot hold: the projection Pi is applied to the COMPOSITE argument
chi + theta(chi_t - chi) + H_k, whereas J is the Jacobian of the projected CORE
chi + theta(chi_t - chi).  Once H_k pushes a cell across a clamp bound the two
projection indicators differ, so DT_k - J - DH_k != 0.

The probe is deliberately falsifiable -- every load-bearing claim is paired with a
control that must go the other way, so it cannot pass for the wrong reason:

  P0  NEGATIVE CONTROL.  If the identity happened to hold here (e.g. no cell is clamped
      at this operating point) then P1 would prove nothing.  P0 asserts that
      DT_k = J + DH_k FAILS at this operating point first.
  P1  POSITIVE CONTROL.  With the projection switched OFF the same code path must
      satisfy the identity to machine precision.  Otherwise P0 could be an artefact of a
      broken Jacobian rather than of the projection.
  P2  THE EXACT BLOCK.  The lower-left block of DT_k (w_out <- chi) is exactly
      C = theta*alpha*diag(E), independent of H_k and of Pi, because the w-channel update
      is neither projected nor corrected.  Checked by comparing DT_k u for a
      chi-supported u against C u.
  P3  THE NEW BOUND.  sigma_max(DT_k) >= sigma_max(C) = theta*alpha*Ecal on the same
      state with the projection active -- the unconditional statement that replaces Weyl.
  P4  THE WEYL STATUS from the shipped archive alone: the physics-core sigma_max minus
      the pure Lip(H_k) is compared with the composite sigma_max (the three values are
      read from results/gain_decomp.json at run time, never hand-typed).  On the CORRECTED
      float64 operator the bound is MET, so the paper must not read a numerical
      contradiction off these three numbers; the remark's argument is structural.  The
      pre-fix archive (sigma_max(J)=14.42, composite=13.03) did violate it, and that is
      what the earlier text reported -- the flip itself is the evidence that the operator,
      not the theorem, had moved.

Run:  conda run -n wirenet python code/_probe_T1_blockLL.py
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(True)
from isp_core import UnrolledISP
from spec_tools import JacOp, sigma_max_L, as_float64_model
from run_experiments import load_data, init_state, NX, NY, N_INC
from run_gain_decomp import make_flatten, make_H, flat_state

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
# CPU-only by design: same reason as the other float64 audit drivers (run_gain_decomp,
# run_bjac, run_prec_audit) -- never contend with the GPU training chain, and every
# double-precision evaluation of this operator must sit on the same arithmetic (the GPU
# and CPU values of sigma_max differ by ~3% through the ill-conditioned contrast solve).
DEV = "cpu"
ALPHA = GAMMA = 0.5
BATCH = 16
TOL = 1e-5

FAIL = []


def check(ok, tag, detail):
    print("  %-62s %s  %s" % (tag, "PASS" if ok else "FAIL", detail), flush=True)
    if not ok:
        FAIL.append(tag)


def resid_identity(m, zr, y_b, k, n, block=False):
    """min over 4 random tangents of ||DTv - Jv - DHv|| / ||DTv||.

    `block=True` also returns the two block-wise residuals.  They matter: the w-channel
    update contains neither H_k nor Pi, so the w block must vanish identically in BOTH
    arms, and only the chi block may move.  A residual outside the chi block would mean
    the instrument, not the mathematics, is being measured.

    TRAP (measured, not hypothetical).  The residual must be formed AFTER reshaping to
    (B, R).  A first version of this probe concatenated the flat tangents instead,
    `cat([H, zeros_like(DT[...])])`; the lengths agree but the per-sample block offsets do
    not, so H lands on sample 0's chi and part of its w block rather than on the chi blocks
    of all B samples.  That version reported a spurious 7.1e-3 for the projection-OFF
    control (the true value is 1e-17), i.e. it manufactured exactly the non-additivity the
    probe exists to detect.  ``_diag_probe_bug.py`` keeps the side-by-side comparison.
    """
    B = zr.shape[0]
    f_full = make_flatten(m, k, ALPHA, GAMMA, True, y_b, "full")
    f_core = make_flatten(m, k, ALPHA, GAMMA, False, y_b, "full")
    f_H = make_H(m, k, y_b)
    of, oc, oh = JacOp(f_full, zr), JacOp(f_core, zr), JacOp(f_H, zr)
    assert oh.Nout == B * n, "make_H output is not (B, n)"
    assert of.Nout == B * (n + 2 * m.n_inc * n), "full map output shape changed"
    # The tangents come from numpy, not from torch.randn: the two backends do not share an
    # RNG stream, so a torch-drawn tangent makes this artifact depend on the device (measured:
    # 7.08e-2 on the host against 7.15e-2 on the GPU for the same state and code path).
    rng = np.random.default_rng(0)
    out = None
    brest, wchg = None, None
    for _ in range(4):
        v = torch.as_tensor(rng.standard_normal(of.Nin), dtype=zr.dtype, device=DEV)
        v = v / v.norm()
        DTv = of.jvp(v).reshape(B, -1)
        Jv = oc.jvp(v).reshape(B, -1)
        Hv = oh.jvp(v).reshape(B, n)
        res_chi = (DTv[:, :n] - Jv[:, :n] - Hv).norm().item()
        res_w = (DTv[:, n:] - Jv[:, n:]).norm().item()
        rel = ((DTv[:, :n] - Jv[:, :n] - Hv).norm() ** 2 + res_w ** 2).sqrt().item() \
            / (DTv.norm().item() + 1e-30)
        out = rel if out is None else min(out, rel)
        brest = res_chi if brest is None else min(brest, res_chi)
        wchg = res_w if wchg is None else min(wchg, res_w)
    return (out, brest, wchg) if block else out


def main():
    t0 = time.time()
    Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
    chi_te_d = chi_te.to(DEV).to(torch.float64)
    y_te_d = y_te.to(DEV).to(torch.complex128)
    assert y_te_d.is_complex(), "incident field lost its imaginary part"

    ck = os.path.join(RES, "ckpt_constrained_K8.pt")
    m = as_float64_model(
        UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained",
                    eps_bar=0.03, proj=True).to(DEV), Gb, Gs, Ei)
    m.load_state_dict(torch.load(ck, map_location=DEV))
    m.eval()
    with torch.no_grad():
        m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])

    chi_b, y_b = chi_te_d[:BATCH], y_te_d[:BATCH]
    chi0, w0 = init_state(m, y_b, BATCH)
    chi0 = chi0.to(torch.float64)
    zr = flat_state(chi0, w0)
    n, I = m.nx * m.ny, m.n_inc
    print("=== op point: init_state, BATCH=%d, alpha=gamma=%.2f, k=%d, ckpt_constrained_K8 "
          "=== (dev=%s)" % (BATCH, ALPHA, 0, DEV), flush=True)

    # ---- P0 / P1: the identity with the projection ON, then OFF ----------------
    rel_on, chi_on, w_on = resid_identity(m, zr, y_b, 0, n, block=True)
    check(rel_on > 1e-4,
          "P0 DT_k = J + DH_k FAILS (proj ON)   [negative control]",
          "min rel = %.3e  (chi-block %.3e, w-block %.3e)" % (rel_on, chi_on, w_on))

    m.proj = False
    rel_off, chi_off, w_off = resid_identity(m, zr, y_b, 0, n, block=True)
    m.proj = True
    check(w_on < 1e-14 and w_off < 1e-14,
          "P0b the w block is identical in both arms (H_k and Pi are absent there)",
          "w-block residuals ON=%.3e OFF=%.3e" % (w_on, w_off))
    check(rel_off < 1e-12,
          "P1 same identity HOLDS (proj OFF)    [positive control]",
          "min rel = %.3e  (chi-block %.3e, w-block %.3e)" % (rel_off, chi_off, w_off))
    check(rel_on / max(rel_off, 1e-300) > 1e6,
          "P1b the violation is switched by the projection alone",
          "ON/OFF ratio = %.3e" % (rel_on / max(rel_off, 1e-300)))

    # ---- P2: the lower-left block is exactly C = theta*alpha*diag(E) -----------
    rng2 = np.random.default_rng(1)      # seeded, like P0/P1: an unseeded torch.randn
    with torch.no_grad():                # would make the artifact depend on ambient RNG
        w_t = (1 - ALPHA) * w0 + ALPHA * chi0.unsqueeze(1) * (
            m.Ei.unsqueeze(0) + m._Gb_apply(w0))
        E = m.Ei.unsqueeze(0) + m._Gb_apply(w0)         # field at z (Lem. blocks)
        v = torch.as_tensor(rng2.standard_normal((BATCH, n)),
                            dtype=chi0.dtype, device=DEV)
        u = torch.zeros(BATCH, zr.shape[1], dtype=chi0.dtype, device=DEV)
        u[:, :n] = v
        f_full = make_flatten(m, 0, ALPHA, GAMMA, True, y_b, "full")
        dtv = JacOp(f_full, zr).jvp(u.reshape(-1)).reshape(BATCH, -1)[:, n:]
        cv = (ALPHA * v.unsqueeze(1) * E)               # (B, I, n) complex
        cv_r = torch.cat([cv.real.reshape(BATCH, -1), cv.imag.reshape(BATCH, -1)], dim=1)
        err = (dtv - cv_r).norm().item() / (cv_r.norm().item() + 1e-30)
    check(err < 1e-9,
          "P2 lower-left block of DT_k == C = alpha*diag(E) exactly",
          "relative error = %.3e" % err)

    # ---- P3: sigma_max(DT_k) >= sigma_max(C) on the same state -----------------
    s_DT, nvDT, _ = sigma_max_L(JacOp(f_full, zr), tol=TOL)
    Ecal = float(E.reshape(BATCH, I, n).norm(dim=1).max().item())
    sig_C = ALPHA * Ecal
    check(s_DT >= sig_C * (1 - 1e-6),
          "P3 sigma_max(DT_k) >= alpha*Ecal  (proj ON)",
          "sigma_max(DT_k)=%.4f  alpha*Ecal=%.4f  (matvecs=%d)" % (s_DT, sig_C, nvDT))
    check(sig_C > 1.0,
          "P3b the bound exceeds 1 at alpha=0.5  (1/(alpha*Ecal)=%.4f)" % (1.0 / Ecal),
          "alpha*Ecal = %.4f" % sig_C)

    # ---- P4: Weyl bound MET on the corrected operator --------------------------
    # The assertion direction was FLIPPED on 2026-10-19 together with the operator fix.
    # Pre-fix archives gave sigma_max(J)=14.42 and composite=13.03 with Lip(H)=0.32, i.e.
    # 13.03 < 14.42-0.32, and the manuscript read that as a numerical certificate that the
    # additive splitting fails.  After the float64 correction the same three fields satisfy
    # the bound, so no numerical contradiction can be claimed; the structural argument of
    # Eq. (dTvsJ) is unaffected (it does not use magnitudes at all).  Asserting the corrected
    # direction keeps the probe honest: it now checks the claim the text actually makes.
    with open(os.path.join(RES, "gain_decomp.json")) as f:
        gd = json.load(f)
    sj = gd["A_physics_core"]["sigma_max_full"]
    sdt = gd["C_constrained"]["sigma_max_composite_max"]
    ell = gd["B_constrained"]["Lip_H_pure_max"]
    check(sdt >= sj - ell,
          "P4 Weyl lower bound is MET on the corrected operator",
          "%.4f - %.4f = %.4f <= %.4f" % (sj, ell, sj - ell, sdt))

    # ---- artefact: the manuscript macros are read from here, never hand-typed --
    art = {
        "instrument": "_probe_T1_blockLL.py",
        "probe": "falsification probe for DT_k = J + DH_k (round-3 referee R1, item T1)",
        "operating_point": ("init_state(model, y_test[:16]) of the K=8 constrained "
                            "checkpoint, alpha=gamma=0.5, theta=1, layer k=0, float64"),
        "P0_rel_DT_minus_J_minus_DH_projON": rel_on,
        "P0_chi_block_projON": chi_on,
        "P0_w_block_projON": w_on,
        "P1_rel_DT_minus_J_minus_DH_projOFF": rel_off,
        "P1_chi_block_projOFF": chi_off,
        "P1_w_block_projOFF": w_off,
        "P1b_on_off_ratio": rel_on / max(rel_off, 1e-300),
        "P2_lower_left_block_rel_error": err,
        "P3_sigma_max_DT_k0_projON": s_DT,
        "P3_alpha_times_Ecal": sig_C,
        "P3_alpha_c_measured": 1.0 / Ecal,
        "P4_sigma_max_J_archive": sj,
        "P4_Lip_H_archive": ell,
        "P4_weyl_rhs": sj - ell,
        "P4_sigma_max_composite_archive": sdt,
        "P4_weyl_met": bool(sdt >= sj - ell),
        "P4_weyl_violated": bool(sdt < sj - ell),
        "all_assertions_passed": not FAIL,
    }
    with open(os.path.join(RES, "t1_probe.json"), "w") as f:
        json.dump(art, f, indent=1)

    print("\nPROBE %s   (%.0fs)" % ("PASS" if not FAIL else "FAIL " + str(FAIL),
                                    time.time() - t0), flush=True)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
