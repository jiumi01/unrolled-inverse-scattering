# -*- coding: utf-8 -*-
"""诊断3：同一 zr 下，我的复合层实现 vs 冻结 run_gain_decomp 实现 的 sigma_max 直接对比。"""
import os, sys, torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
torch.set_grad_enabled(True)
import run_gain_decomp as R
from isp_core import UnrolledISP
from spec_tools import as_float64_model
from run_experiments import load_data, init_state, NX, NY, N_INC

DEV = "cpu"
Gb, Gs, pts, dirs, kb, Ei, chi_tr, y_tr, chi_te, y_te = load_data()
chi_te_d = chi_te.to(DEV).to(torch.float64); y_te_d = y_te.to(DEV).to(torch.complex128)
m = as_float64_model(UnrolledISP(1, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", proj=True).to(DEV),
                     Gb, Gs, Ei)
with torch.no_grad():
    m._cg_ls(m.Ei.unsqueeze(0).expand(1, -1, -1), y_te_d[:1])
chi0, w0 = init_state(m, y_te_d[:16], 16); chi0 = chi0.to(torch.float64)
zr = R.flat_state(chi0, w0)
print("feat_scale in model:", m.feat_scale.flatten()[:8].tolist(), "...",
      float(m.feat_scale.max()), float(m.feat_scale.min()), flush=True)

mk = as_float64_model(UnrolledISP(8, Gb, Gs, Ei, NX, NY, N_INC, mode="constrained", eps_bar=0.03).to(DEV),
                      Gb, Gs, Ei)
mk.load_state_dict(torch.load("results/ckpt_constrained_K8.pt", map_location=DEV))
mk.eval()
print("mk.eps_bar=", mk.eps_bar, " mk.chi_max=", mk.chi_max, " mk.featbound=", mk.featbound,
      " feat_scale max/min=", float(mk.feat_scale.max()), float(mk.feat_scale.min()), flush=True)
print("model feat_scale vs mk feat_scale equal:",
      bool(torch.equal(mk.feat_scale, m.feat_scale)), flush=True)

k = 3
# frozen
lvc, lpc, _, _ = R.s_of(R.make_H(mk, k, y_te_d[:16]), zr, name="H")
cvc, cpc, _, _ = R.s_of(R.make_flatten(mk, k, 0.5, 0.5, True, y_te_d[:16], "full"), zr, name="comp")
print("FROZEN  Lip(H)=%.6f  composite=%.6f" % (lvc, cvc), flush=True)

# 我的实现：重建 corr 字典
sd = torch.load("results/ckpt_constrained_K8.pt", map_location=DEV)
corr = {}
for c in ("c1", "c2", "c3"):
    corr[c + "w"] = sd["corrs.%d.%s.conv.weight" % (k, c)].to(torch.float64)
    corr[c + "b"] = sd["corrs.%d.%s.conv.bias" % (k, c)].to(torch.float64)
print("corr weights equal to mk:", bool(torch.equal(corr["c1w"], mk.corrs[k].c1.conv.weight)),
      flush=True)

# 用我的模块函数，但复用它内部的 Gb/Gs/Ei/zr：这里用 monkeypatch 方式不便，
# 改为直接调用 r1_indep_corr 里的闭包生成器需要其 main 上下文；简化：重算 feature 对比
with torch.no_grad():
    ch = zr[:, :1024]
    wr = zr[:, 1024:1024 + 8 * 1024].reshape(16, 8, 1024)
    wi = zr[:, 1024 + 8 * 1024:].reshape(16, 8, 1024)
    w = wr + 1j * wi
    w_t = 0.5 * w + 0.5 * ch.unsqueeze(1) * (m.Ei.unsqueeze(0) + m._Gb_apply(w))
    E = m.Ei.unsqueeze(0) + m._Gb_apply(w_t)
    resid = m._Gs_apply(w_t) - y_te_d[:16]
    resid_dom = torch.matmul(m.Gs.conj().T, resid.reshape(-1, resid.shape[-1]).T).T.reshape(16, 8, -1)
    feat = torch.cat([resid_dom.real.reshape(16, -1), resid_dom.imag.reshape(16, -1),
                      E.real.reshape(16, -1), E.imag.reshape(16, -1),
                      w_t.real.reshape(16, -1), w_t.imag.reshape(16, -1), ch], dim=1)
    feat = feat.reshape(16, -1, 32, 32)
    feat_s = torch.tanh(feat / mk.feat_scale)
    dchi = mk.corrs[k](feat_s).flatten(1)
    chi_t = 0.5 * ch + 0.5 * m._cg_ls(E, y_te_d[:16])
    chi_out = chi_t + dchi
    chi_cl = chi_out.clamp(0.0, 0.5)
    nclamp_hi = int((chi_out > 0.5).sum()); nclamp_lo = int((chi_out < 0.0).sum())
    ntot = chi_out.numel()
    print("clamp stats composite: >0.5 =", nclamp_hi, " <0 =", nclamp_lo, " of", ntot, flush=True)
    # 核心(无修正)的 clamp 统计
    c2 = chi_t.clamp(0.0, 0.5)
    print("clamp stats core: >0.5 =", int((chi_t > 0.5).sum()), " <0 =", int((chi_t < 0).sum()), flush=True)
    # dchi 幅度
    print("|dchi| max=%.4e  mean=%.4e ; |chi_t| max=%.4f ; #chi_t in (0,0.5)=%d/%d"
          % (float(dchi.abs().max()), float(dchi.abs().mean()), float(chi_t.abs().max()),
             int(((chi_t > 0) & (chi_t < 0.5)).sum()), ntot), flush=True)
    print("feat max|.| =", float(feat_s.abs().max()), " (tanh saturation -> small deriv)",
          flush=True)
