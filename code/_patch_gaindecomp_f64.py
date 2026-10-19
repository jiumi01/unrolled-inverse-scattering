# -*- coding: utf-8 -*-
"""Patch run_gain_decomp.py to run entirely in double precision (review R2 m2.11).

Why.  Table cert row 11 (Lip(H_k)) and row 13 (per-layer composite at fixed alpha=gamma=1/2)
are sourced from gain_decomp.json, and that driver built its models with a plain
`.to(DEV)` -- i.e. float32 -- while the surrounding text claims the Table's spectral
quantities are double precision.  R2 showed the attribution of the 1.869-vs-2.793 gap to
"the as-implemented n_cg" is wrong: gain_decomp's 1.869 matches bjac.json's
float32_comparison to the digit, whereas the same operating point in float64 at the
as-implemented n_cg gives 14.594.  So the gap is dominated by precision, not by n_cg.

Fix.  Build every audit model through spec_tools.as_float64_model (which reinstalls the
complex buffers as complex128 -- a plain .double() would silently discard the imaginary
part of Gb/Gs/Ei), move the incident data with .to(torch.complex128) -- NOT float64, which
has the same trap -- and record the dtype in the output JSON so the table's dtype claim is
checkable rather than asserted.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "run_gain_decomp.py")
s = io.open(P, encoding="utf-8").read()
n = s


def sub(old, new, why):
    global n
    assert n.count(old) == 1, "anchor not unique (%d): %s" % (n.count(old), why)
    n = n.replace(old, new)
    print("  patched: %s" % why)


# 1) import the conversion helper
sub("from spec_tools import JacOp, sigma_max_L, sigma_max_power",
    "from spec_tools import JacOp, sigma_max_L, sigma_max_power, as_float64_model",
    "import as_float64_model")

# 2) data moved to double / complex128
sub("    chi_te_d, y_te_d = chi_te.to(DEV), y_te.to(DEV)\n",
    "    # complex data must go to complex128; .to(torch.float64) would discard the field\n"
    "    chi_te_d = chi_te.to(DEV).to(torch.float64)\n"
    "    y_te_d = y_te.to(DEV).to(torch.complex128)\n"
    "    DT = torch.float64\n"
    "    assert y_te_d.is_complex(), 'incident field lost its imaginary part'\n",
    "data -> float64/complex128")

# 3) record the dtype in the artifact
sub('"alpha_fixed": ALPHA, "gamma_fixed": GAMMA, "batch": BATCH}',
    '"alpha_fixed": ALPHA, "gamma_fixed": GAMMA, "batch": BATCH,\n'
    '           "dtype": "float64 (complex128 fields)",\n'
    '           "note": ("R2 m2.11: this driver previously ran in float32, which is why its "\n'
    '                    "sigma_max_full disagreed with the float64 audit; the gap to the "\n'
    '                    "converged-solve value was dominated by precision, not by n_cg")}',
    "record dtype in JSON")

# 4) base audit model in double precision
sub('    m = UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV)\n',
    '    m = as_float64_model(\n'
    '        UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),\n'
    '        Gb, Gs, Ei)\n',
    "base model -> float64")

# 5) the K=8 audit models in double precision
sub('        mk = UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV)\n',
    '        mk = as_float64_model(\n'
    '            UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode=mode, eps_bar=0.03).to(DEV),\n'
    '            Gb, Gs, Ei)\n',
    "K=8 models -> float64")

# 6) the state built by init_state must be double as well
sub("    chi0, w0 = init_state(m, y_b, BATCH)\n    zr = flat_state(chi0, w0)\n"
    "    idx_chi = torch.arange(0, n)\n",
    "    chi0, w0 = init_state(m, y_b, BATCH)\n"
    "    chi0 = chi0.to(DT)          # w0 stays complex128\n"
    "    assert w0.is_complex(), 'float64 conversion lost the complex field state'\n"
    "    zr = flat_state(chi0, w0)\n"
    "    idx_chi = torch.arange(0, n)\n",
    "initial state -> float64")

# 7) dtype guard before writing
sub('    out["L0_layer_equivalence"] = selfcheck_layer(m, chi_b, w0, y_b)',
    '    assert next(m.parameters()).dtype == torch.float64, "audit model is not float64"\n'
    '    assert m.Gb.is_complex() and m.Gb.dtype == torch.complex128, "Gb is not complex128"\n'
    '    out["L0_layer_equivalence"] = selfcheck_layer(m, chi_b, w0, y_b)',
    "dtype assertions")

import ast  # noqa: E402
ast.parse(n)
io.open(P, "w", encoding="utf-8").write(n)
print("run_gain_decomp.py patched (ast.parse OK)")
