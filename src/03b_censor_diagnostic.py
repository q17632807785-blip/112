"""03b_censor_diagnostic.py — 只针对噪声模型的诊断（预注册第 3.4 节 R1 理由中的数字，在当前划分下重算）。
(a) 开发集删失点中“该眼该点每次都删失”的比例；(b) 噪声模型自检模拟（OLS 模板，下限 0 dB）的删失比例；
(c) 只把“每次都删失”的点模板设为 −10 dB 的诊断模拟删失比例。不运行任何参赛方法，不生成 sim_train/sim_test。"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
src = open(os.path.join(HERE, "03_noise_model.py"), encoding="utf-8").read()
src = src[:src.index("model = {")].replace("__file__", repr(os.path.join(HERE, "03_noise_model.py")))
ns = {}; exec(src, ns)
np, Cn, groups, F, L, ter, T = ns["np"], ns["Cn"], ns["groups"], ns["F"], ns["L"], ns["ter"], ns["T"]
always = np.zeros_like(Cn)
for eid, idx in groups.items(): always[idx] = Cn[idx].all(0)[None, :]
frac_abs = float((Cn & always).sum() / Cn.sum())
def simulate(tmpl, seed, frac_always_below0=False):
    rng = np.random.default_rng(seed)
    sd_off_v = np.array(ns["off_sd_deconv"])[ter]
    s = tmpl + sd_off_v[:, None] * rng.standard_normal((len(tmpl), 1)) + (rng.standard_normal((len(tmpl), 59)) @ L.T) * ns["sd_point"](np.clip(tmpl, 0, None))
    if frac_always_below0: return float((s[always] < 0).mean())   # “每次都删失”的点在模拟中落到 0 以下的比例
    return float((s < 0).mean())
base = np.clip(F, 0, 35); diag = np.where(always, -10.0, base)
out = dict(dev_censor_frac=float(Cn.mean()), frac_censored_points_always_censored=frac_abs,
           sim_censor_frac_ols_floor0=[simulate(base, ns["cfg"]["seed"] + i) for i in range(5)],
           ols_floor0_always_censored_cells_sim_below0=[simulate(base, ns["cfg"]["seed"] + i, True) for i in range(5)],
           sim_censor_frac_always_to_minus10=[simulate(diag, ns["cfg"]["seed"] + i) for i in range(5)])
json.dump(out, open(os.path.join(ROOT, "results", "censor_diagnostic.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
