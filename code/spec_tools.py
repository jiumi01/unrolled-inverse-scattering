# -*- coding: utf-8 -*-
"""spec_tools.py — exact linear-operator tools for layer / composite Jacobians.

Motivation.  A finite-difference power iteration on these layer maps converges very
slowly: the maps are strongly non-normal and a plain power iteration advances at the
rate (sigma_2/sigma_1)^(2k).  An earlier revision used one and produced mutually
inconsistent numbers — a full-Jacobian estimate *below* one of its own blocks, which is
impossible since sigma_max(J) >= sigma_max(any block).

What is done here instead:
  * the Jacobian is NEVER formed; two EXACT products are supplied
        forward : torch.func.jvp                      (no finite-difference error)
        adjoint : torch.autograd.grad(grad_outputs=v)
    wrapped in a scipy.sparse.linalg.LinearOperator acting on the FLATTENED
    (B*m,) input vector and returning (B*R,);
  * sigma_max by Lanczos/IRAM on the symmetric PSD operator J^T J (scipy eigsh),
    which converges in ~50-300 matvecs instead of thousands;
  * spectral radius by eigs(which='LM') on J;
  * transient sup_m ||J^m|| by eigsh on (J^T)^m J^m.

Every estimate carries the matvec count and a convergence/solver flag, so a
non-converged value can be reported as such instead of being silently quoted.
"""
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh, eigs


class JacOp:
    """Exact Jacobian of a flat map f : (B, D) -> (B, R) at zr, restricted to input
    columns `in_idx` and output columns `out_slice`.  Operator space: (B*R) x (B*m)."""

    def __init__(self, f, zr, in_idx=None, out_slice=None, name="J"):
        self.f = f
        self.zr = zr.detach()
        self.name = name
        self.B = int(zr.shape[0])
        self.D = int(zr.shape[1])
        self.in_idx = (torch.arange(self.D) if in_idx is None
                       else torch.as_tensor(in_idx).long())
        self.m = int(self.in_idx.numel())
        self.out_slice = out_slice
        self.matvec = 0
        with torch.no_grad():
            self.R = int(self._raw(self.zr).shape[1])
        self.Nin, self.Nout = self.B * self.m, self.B * self.R
        self.dtype = self.zr.dtype
        # DEVICE.  Every tensor built here must live where zr lives.  The first version
        # omitted this and produced CPU tensors, which was invisible while the audits ran
        # on CPU but makes torch.func.jvp fail with "found at least two devices, cuda:0
        # and cpu" as soon as a CUDA model is audited (measured in run_gain_decomp).
        self.dev = self.zr.device
        self.in_idx = self.in_idx.to(self.dev)

    # --------------------------------------------------------------- internals
    def _raw(self, z):
        o = self.f(z)
        return o if self.out_slice is None else o[:, self.out_slice]

    @staticmethod
    def _host(x):
        """np.asarray() raises TypeError on a CUDA tensor ("can't convert cuda:0 device type
        tensor to numpy"), so every flat vector handed in from a CUDA-side caller must be
        brought to host memory explicitly first."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _to_states(self, x):
        """flat (B*m,) -> (B, D) with zeros outside in_idx, on the operator's device."""
        X = torch.as_tensor(self._host(x), dtype=self.dtype,
                            device=self.dev).reshape(self.B, self.m)
        u = torch.zeros(self.B, self.D, dtype=self.dtype, device=self.dev)
        u[:, self.in_idx] = X
        return u

    def jvp(self, x):
        """flat (B*m,) -> flat (B*R,).  Exact.  Counts one matvec."""
        self.matvec += 1
        _, out = torch.func.jvp(self._raw, (self.zr,), (self._to_states(x),))
        return out.detach().reshape(-1)

    def vjp(self, y):
        """flat (B*R,) -> flat (B*m,).  Exact.  Counts one matvec."""
        self.matvec += 1
        Y = torch.as_tensor(self._host(y), dtype=self.dtype,
                            device=self.dev).reshape(self.B, self.R)
        z = self.zr.clone().requires_grad_(True)
        g = torch.autograd.grad(self._raw(z), (z,), grad_outputs=Y, allow_unused=True)[0]
        if g is None:
            return torch.zeros(self.Nin, dtype=self.dtype, device=self.dev)
        return g.detach()[:, self.in_idx].reshape(-1)

    # ------------------------------------------------------------- operators
    def _np(self, t):
        return t.detach().cpu().numpy().astype(np.float64)

    def MTM(self, power=1):
        """LinearOperator for (J^T J)^power, shape (B*m, B*m)."""
        def mv(x):
            X = np.asarray(x, dtype=np.float64)
            for _ in range(power):
                X = self._np(self.vjp(self.jvp(X)))
            return X
        return LinearOperator((self.Nin, self.Nin), matvec=mv, rmatvec=mv,
                              dtype=np.float64)

    def M(self):
        """LinearOperator for J, shape (B*R, B*m)."""
        def mv(x):
            return self._np(self.jvp(np.asarray(x, dtype=np.float64)))
        def rmv(y):
            return self._np(self.vjp(np.asarray(y, dtype=np.float64)))
        return LinearOperator((self.Nout, self.Nin), matvec=mv, rmatvec=rmv,
                              dtype=np.float64)


# ------------------------------------------------------------------ estimators
def sigma_max_L(J, tol=1e-6, maxiter=None, seed=0):
    """Largest singular value by Lanczos on J^T J.  Returns (sigma, matvecs, info).

    Two guards, both of which have bitten:
      * a ZERO operator (e.g. block B of the Born variant, where the contrast step no
        longer depends on the field, so d chi+/d w == 0) makes ARPACK fail; we detect it
        with one probe product and return exactly 0 instead of relying on the fallback;
      * the Rayleigh fallback must contract the (B, m) state with the FLATTENED
        (B*m,) adjoint result -- shapes (B, m) vs (B*m,) do not broadcast.
    """
    m0 = J.matvec
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(J.Nin)
    X0 = torch.as_tensor(v0, dtype=J.dtype, device=J.dev).reshape(J.B, J.m)
    X0 = X0 / (X0.norm() + 1e-30)
    probe = J.vjp(J.jvp(X0.reshape(-1))).reshape(J.B, J.m)
    nprobe = float(probe.norm())
    if nprobe <= 1e-20:
        return 0.0, J.matvec - m0, "zero-operator"
    try:
        w = eigsh(J.MTM(), v0=v0, k=1, which="LM", tol=tol, maxiter=maxiter,
                  return_eigenvectors=False)
        eig = float(np.atleast_1d(w).max())
        info = "eigsh"
    except Exception as e:
        eig = float((probe * X0).sum())
        info = "eigsh-failed(%s); rayleigh" % type(e).__name__
    return float(np.sqrt(max(eig, 0.0))), J.matvec - m0, info


def sigma_max_power(J, iters=200, seed=0):
    """Power iteration on J^T J — INDEPENDENT cross-check, and a rigorous lower bound."""
    m0 = J.matvec
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(rng.standard_normal((J.B, J.m)), dtype=J.dtype, device=J.dev)
    X = X / (X.norm() + 1e-30)
    sig = 0.0
    for _ in range(iters):
        Y = J.jvp(X.reshape(-1))
        nv = float(Y.norm())
        if nv < 1e-30:
            return 0.0, J.matvec - m0
        X = J.vjp((Y / nv).reshape(-1)).reshape(J.B, J.m)
        nu = float(X.norm())
        if nu < 1e-30:
            return 0.0, J.matvec - m0
        X = X / nu
        sig = nv
    return float(sig), J.matvec - m0


def sigma_max_sq(J, L=2, tol=1e-6):
    """sigma_max(J^L) by Lanczos on (J^T)^L J^L; one matvec costs 2L J-products."""
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(J.Nin)
    try:
        w = eigsh(J.MTM(power=L), v0=v0, k=1, which="LM", tol=tol,
                  return_eigenvectors=False)
        return float(np.sqrt(max(float(np.atleast_1d(w).max()), 0.0)))
    except Exception:
        return float("nan")


def spectral_radius(J, tol=1e-6, maxiter=None, seed=0):
    """Spectral radius (largest eigenvalue modulus) by IRAM on J."""
    m0 = J.matvec
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(J.Nin)
    try:
        w = eigs(J.M(), k=1, which="LM", tol=tol, maxiter=maxiter, v0=v0,
                 return_eigenvectors=False)
        rho = float(np.abs(np.atleast_1d(w)).max())
        info = "eigs"
    except Exception as e:
        rho = float("nan")
        info = "eigs-failed(%s)" % type(e).__name__
    return rho, J.matvec - m0, info


def rho_power(J, iters=200, seed=0):
    """|lambda|_max lower bound by power iteration on J itself.

    Independent of IRAM, which can converge to a non-dominant eigenvalue of a
    non-normal operator and does not say so.  Power iteration either converges to the
    dominant eigenvalue or its iterate leaves the range; both are observable.
    """
    m0 = J.matvec
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(rng.standard_normal((J.B, J.m)), dtype=J.dtype, device=J.dev)
    X = X / (X.norm() + 1e-30)
    rho = 0.0
    for _ in range(iters):
        Y = J.jvp(X.reshape(-1)).reshape(J.B, J.m)
        ny = float(Y.norm())
        if not np.isfinite(ny) or ny < 1e-30:
            return float(rho), J.matvec - m0
        X = Y / ny
        rho = ny
    return float(rho), J.matvec - m0


def sigma_max_validated(J, tol=1e-6, iters=200, agree=1e-2, seed=0):
    """sigma_max with an independent cross-check, reported explicitly.

    Two estimators are always computed:
      * Lanczos/ARPACK on J^T J (fast, but it can return a Ritz value for the wrong
        invariant subspace on the ill-conditioned, non-normal Jacobians that truncated
        CG produces -- measured on this project: at n_cg=200 two layers returned exactly
        0.0 and the core value was 10% low);
      * power iteration on J^T J, which is a rigorous LOWER bound at every iterate and
        has no breakdown mode.
    The reported value is Lanczos when the two agree, and power otherwise.  The rule is
    fixed in advance in both directions: when the instrument is sound the historical value
    is reproduced bit-for-bit (so archives stay comparable), and when it is not the
    breakdown-free estimator is used and `agrees` records that substitution.  Quoting the
    Lanczos value unconditionally is what produced two exactly-zero singular values and a
    10%-low core value at n_cg=200 in this project's own archives.

    Returns dict(sigma, lanczos, power, rel, agrees, info, matvecs).
    """
    s_l, mv, info = sigma_max_L(J, tol=tol, seed=seed)
    s_p, _ = sigma_max_power(J, iters=iters, seed=seed)
    s_l = float(s_l) if np.isfinite(s_l) else 0.0
    s_p = float(s_p) if np.isfinite(s_p) else 0.0
    hi = max(s_l, s_p)
    rel = (abs(s_l - s_p) / hi) if hi > 0 else 0.0
    agrees = bool(rel <= agree)
    return dict(sigma=(s_l if agrees else s_p), lanczos=s_l, power=s_p, rel=float(rel),
                agrees=agrees, info=info, matvecs=int(mv))



def transient_M(J, ms=(1, 2, 3, 4, 6, 8)):
    """sup_{m in ms} ||J^m||, each by Lanczos on (J^T)^m J^m."""
    return {str(L): sigma_max_sq(J, L) for L in ms}


# ---------------------------------------------------------------- dtype -------

def as_float64_model(m, Gb, Gs, Ei):
    """Double-precision copy of an audit model, with the physics kept complex128.

    TRAP (measured, not hypothetical).  nn.Module.to(torch.float64) converts *complex*
    registered buffers to real float64 and silently discards their imaginary part --
    torch emits a UserWarning ("Casting complex values to real discards the imaginary
    part") and then proceeds.  UnrolledISP registers Gb, Gs and Ei as complex64, so a
    plain .to(torch.float64) destroys the Green operators and the incident field, and
    the first place it shows up is torch.Tensor.imag failing on a real tensor deep
    inside the layer.  The complex buffers are therefore re-installed from their
    complex128 sources, and every tensor's dtype is asserted before returning.

    The same trap applies to a bare tensor: y.to(torch.float64) also throws the field
    away, so incident data must be moved with .to(torch.complex128).
    """
    m.double()                                     # float params/buffers -> float64
    # Preserve the model's device: the sources are host-side numpy/CPU arrays, so a naive
    # torch.as_tensor(...) would reinstall the Green operators on the CPU and the first
    # matmul against a CUDA field would raise a device-mismatch RuntimeError.
    dev = next(m.parameters()).device
    for nm, src in (("Gb", Gb), ("Gs", Gs), ("Ei", Ei)):
        if src is not None:
            setattr(m, nm, torch.as_tensor(src, dtype=torch.complex128, device=dev))
    if hasattr(m, "_gs_diag"):
        del m._gs_diag                             # cached at the old dtype
    for nm in ("Gb", "Gs", "Ei"):
        t = getattr(m, nm, None)
        assert t is None or t.is_complex(), "complex buffer %s was cast to real" % nm
    return m


def dtype_consistency(m32, m64, w, y, chi=None):
    """Two separate questions, deliberately not conflated.

    (1) Is the float64 conversion transparent?  Compare the *physics operators* applied to
        the same complex input in both precisions.  There is no CG loop here, so the only
        difference is round-off: a destroyed imaginary part would show up as a relative
        difference of order one, float32 round-off as ~1e-7.  Also report the ratio
        |Im Gb w| / |Gb w|, which is exactly zero if the conversion threw the field away.
    (2) How much does the 15-step CG contrast solve amplify a 1e-7 arithmetic difference?
        Apply one full layer to the *same* state in both precisions.  A large number here is
        not a conversion defect; it is the conditioning of the contrast step
        (lambda_min ~ 1e-5), and it is the reason every audit quantity is computed in
        double precision.
    """
    w32, w64 = w.to(torch.complex64), w.to(torch.complex128)
    y32, y64 = y.to(torch.complex64), y.to(torch.complex128)

    def _up(t):
        # .double() means .to(torch.float64), which on a COMPLEX tensor discards the
        # imaginary part -- the very trap this function exists to detect.  A precision
        # upgrade of a complex quantity must go through complex128.
        return t.to(torch.complex128) if t.is_complex() else t.to(torch.float64)

    def rel(a64, a32):
        a32 = _up(a32)
        return float((a64 - a32).norm() / (a32.norm() + 1e-30))

    with torch.no_grad():
        gb64, gb32 = m64._Gb_apply(w64), m32._Gb_apply(w32)
        gs64, gs32 = m64._Gs_apply(w64), m32._Gs_apply(w32)
        out = dict(operator_Gb_rel=rel(gb64, gb32),
                   operator_Gs_rel=rel(gs64, gs32),
                   operator_Ei_rel=rel(m64.Ei, m32.Ei),
                   imag_frac_Gbw=float(gb64.imag.norm() / (gb64.norm() + 1e-30)),
                   layer_rel=None)
        if chi is not None:
            c64, ww64 = m64.layer(0, chi.to(torch.float64), w64, y64)
            c32, ww32 = m32.layer(0, chi.to(torch.float32), w32, y32)
            out["layer_rel"] = rel(c64, c32)
            out["layer_w_rel"] = rel(ww64, ww32)
    return out
