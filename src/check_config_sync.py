"""check_config_sync.py — 预注册第 5 部分 yaml 块与 config.yaml 逐字节一致性检查，外加关键字段与正文的交叉核对。
用法：python src/check_config_sync.py [preregistration.md] [config.yaml]；任何不一致都会 assert 失败（退出码非 0）。"""
import os, sys, re, datetime as dt
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PRE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "preregistration.md")
CFG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "config.yaml")
raw = open(PRE, "rb").read()
nl = b"\r\n" if b"\r\n" in raw else b"\n"
sec5 = raw.index("## 5. 预先定死的超参数".encode())
start = raw.index(b"```yaml" + nl, sec5) + len(b"```yaml" + nl)
end = raw.index(nl + b"```", start) + len(nl)          # 块内容含最后一行的换行
block = raw[start:end]; cfgb = open(CFG, "rb").read()
assert raw.count(b"```yaml") == 1, "预注册中应只有一个 yaml 块"
assert block == cfgb, f"config.yaml 与第 5 部分 yaml 块不一致（{len(block)} vs {len(cfgb)} 字节）"
EOL = 'CRLF' if nl == b'\r\n' else 'LF'
print(f'[OK] 逐字节一致：{len(cfgb)} 字节，行尾 {EOL}')
c = yaml.safe_load(cfgb.decode("utf-8"))
text = raw.decode("utf-8").replace("\r\n", "\n")
sec = lambda a, b: text[text.index(a):text.index(b)]
s12, s13, s34, s35 = sec("### 1.2", "### 1.3"), sec("### 1.3", "### 1.4"), sec("### 3.4", "### 3.5"), sec("### 3.5", "### 3.6")

checks = []
def chk(name, cond, detail=""):
    checks.append((name, bool(cond), detail)); print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")

# 1) 检验顺序
chk("evaluation.test_order == [H1,H3,H4,H2]", c["evaluation"]["test_order"] == ["H1", "H3", "H4", "H2"], str(c["evaluation"]["test_order"]))
m = re.search(r"\*\*(H\d)\s*→\s*(H\d)\s*→\s*(H\d)\s*→\s*(H\d)\*\*", s12)
chk("1.2 正文顺序与 yaml 一致", m and list(m.groups()) == c["evaluation"]["test_order"], m.group(0) if m else "未找到")
# 2) 删失目标
tcf = c["noise_model"]["target_censor_frac"]; pct = f"{tcf * 100:.2f}%"
chk("noise_model.target_censor_frac == 0.0415（重建开发集实测）", tcf == 0.0415, str(tcf))
chk(f"3.5 正文“开发集实测 {pct}”", f"开发集实测 {pct}" in s35)
chk(f"3.4 R1 理由“真实开发集 {pct}”", f"真实开发集 {pct}" in s34)
chk("全文不再出现旧删失目标 3.99% / 0.0399", "3.99%" not in text and "0.0399" not in text)
dd = os.path.join(ROOT, "data", "noise_model.json")
if os.path.exists(dd):
    import json; nmj = json.load(open(dd, encoding="utf-8"))
    chk("target_censor_frac == round(noise_model.json 开发集删失比例, 4)", round(nmj["censoring"]["dev_censor_frac"], 4) == tcf)
chk("sim_calibration_tolerance.censor_frac_abs 0.01 ↔ 3.5“±1 个百分点”", c["sim_calibration_tolerance"]["censor_frac_abs"] == 0.01 and "±1 个百分点" in s35)
# 3) 真值尺度
chk("simulator.truth_scale_primary == observed_floor_0db", c["simulator"]["truth_scale_primary"] == "observed_floor_0db")
chk("simulator.truth_scale_secondary == latent", c["simulator"]["truth_scale_secondary"] == "latent")
chk("1.3 主分析 = max(f(t),0) 的 OLS；描述性 = 不截断 f(t)", "主分析（观察尺度）" in s13 and r"\max(f(t), 0)" in s13 and "描述性分析（潜在尺度）" in s13 and "不截断" in s13)
chk("2.6 校准真值用观察尺度", "用观察尺度" in sec("### 2.6", "## 3."))
# 4) 解封日期与周次
w1 = dt.date.fromisoformat(c["week1_start"]); un = dt.date.fromisoformat(c["sim_test_seal"]["unseal_not_before_date"])
chk("week1_start 是周一", w1.weekday() == 0, str(w1))
chk("unseal_not_before_date == 2027-01-04 == 第 12 周第一天", un == dt.date(2027, 1, 4) == w1 + dt.timedelta(weeks=11), str(un))
chk("3.4 正文“2027-01-04（第 12 周第一天）”", "2027-01-04（第 12 周第一天）" in s34)
chk("sim_test_seal.sealed == true 且 sha256 为空", c["sim_test_seal"]["sealed"] is True and c["sim_test_seal"]["sha256"] is None)
for wk, a, b in [(3, "2026-11-02", "11-08"), (5, "2026-11-16", "11-22"), (11, "2026-12-28", "2027-01-03")]:
    d0 = w1 + dt.timedelta(weeks=wk - 1)
    chk(f"第 {wk} 周日期 {a} 起", d0 == dt.date.fromisoformat(a) and (d0 + dt.timedelta(days=6)).isoformat().endswith(b) and a in text)
# 5) 其他数值交叉核对
chk("seed / seed_sim_train 20261019；seed_sim_test 20261119", c["seed"] == c["simulator"]["seed_sim_train"] == 20261019 and c["simulator"]["seed_sim_test"] == 20261119 and "种子 20261119" in s34)
chk("model_seeds 20261019–20261023", c["model_seeds"] == list(range(20261019, 20261024)) and "20261019–20261023" in text)
chk("data.expected 与第 0 节 263/144/1115", (c["data"]["expected"]["n_eyes"], c["data"]["expected"]["n_subjects"], c["data"]["expected"]["n_visits"]) == (263, 144, 1115) and "263 眼 / 144 人 / 1115 次" in text)
chk("neighbor_resid_corr 0.5067/−0.0353 ↔ 第 6.3 条 0.507/−0.035", round(c["locations"]["neighbor_resid_corr"], 3) == 0.507 and round(c["locations"]["other_resid_corr"], 3) == -0.035 and "**0.507**" in text and "−0.035" in text)
chk("n_bootstrap 2000 ↔ 正文", c["evaluation"]["n_bootstrap"] == 2000 and "2000 次" in text)
chk("PoPLR n_perm 1000 ↔ 2.4", c["methods"]["PoPLR"]["n_perm"] == 1000 and "1000 次" in text)
chk("real_detection_n_shuffles 200 ↔ 3.8", c["evaluation"]["real_detection_n_shuffles"] == 200 and "打乱 200 次" in text)
chk("B4 fallback 24 小时 ↔ 2.4", c["methods"]["B4"]["fallback_if_est_hours_over"] == 24 and "24 小时" in text)
chk("template_baseline_clip_db [-10,35] ↔ 3.4", c["simulator"]["template_baseline_clip_db"] == [-10, 35] and "−10–35 dB" in s34)
chk("revisions 中的 R1/R2 ↔ 第 7 部分", all(r.split()[0] in [x.split()[0] for x in c["revisions"]] for r in ["R1", "R2"]) and "| R1 |" in text and "| R2 |" in text)

n_fail = sum(not ok for _, ok, _ in checks)
print(f"\n{len(checks)} 项交叉核对，失败 {n_fail} 项")
assert n_fail == 0

# 英文版：yaml 块应与 config.yaml 内容相同（忽略行尾差异）
EN = os.path.join(os.path.dirname(PRE), "preregistration_EN.md")
if os.path.exists(EN):
    et = open(EN, encoding="utf-8").read().replace("\r\n", "\n")
    assert et.count("```yaml\n") == 1
    s0 = et.index("```yaml\n") + 8; e0 = et.index("\n```", s0) + 1
    assert et[s0:e0] == cfgb.decode("utf-8").replace("\r\n", "\n"), "preregistration_EN.md 的 yaml 块与 config.yaml 不一致"
    print("[OK] preregistration_EN.md 的 yaml 块与 config.yaml 内容一致（忽略行尾）")
