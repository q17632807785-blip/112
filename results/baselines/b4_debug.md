# B4 调试记录（第 5 周检查点失败 → 修复 → 通过）

日期：2026-10-21（Asia/Shanghai）  
对照：`sim_train` k=4 子样本（每情景 150 条）与全量检查点。  
约束：未打开 `sim_test`；未推送；未改 sealed 模拟器；`spatial_impl` 仍为 R8 锁定的 `spaMM_matern_within_hemifield`。

## 假设与检验

| ID | 假设 | 结果 |
|---|---|---|
| H1 | 层次 BLUP 把斜率收缩到训练群体 `beta1≈−0.23`，损害异质情景 RMSE | **成立**：旧 B4 eqRMSE≈1.50；去掉层次、仅「空间平滑+簇 OLS」→ **1.34 &lt; B1 1.36** |
| H2 | 无空间的纯层次 BLUP 更差 | **成立**：eqRMSE≈1.52 |
| H3 | `spat_var/(spat_var+nugget)≈0.95`，向**访视总均值**做 kriging，把半视野拉成一团，破坏旁中心/弓形结构 | **成立**：应改为相对**训练位点均值**的残差 Matérn |
| H4 | `var_cl_int=1.0` 硬编码、且 BLUP 的 R 用的是平滑前残差方差 → 过度收缩 | **成立**：平滑后 R 应重估；`var_cl_int` 应从数据估 |
| H5 | 半视野划分 / 坐标单位错误 | **未支持**：块划分与 `sim_core` 的 `y≥0` 一致；距离为度 |
| H6 | 静默回退 B1 / provisional 路径 | **未支持**：`spatial_impl` 正确，非 provisional |
| H7 | 模拟器空间相关与 Matérn 根本不匹配 | 空间+OLS 已优于 B1 → **优先修实现，不改模拟器** |

## 根因

1. **主因**：簇级两重 BLUP 使用过大的残差方差（及硬编码 `var_cl_int`），把眼/簇斜率过度拉向群体均值，RMSE 系统性差于 B1。
2. **次因**：空间步把观测拉向**访视均值**而非位点先验，损伤局灶情景。

半视野分割、坐标单位、训练访视、泄漏、静默回退均未证实。

## 修复（仍在 R8 `spatial_impl` 字符串内）

文件：`src/methods/lmm.py`（SHA-256 `69b7293bb6393cb19a534b3c7d53b6bf99310d3c2afb758fd22e32c603aff477`）

1. **残差 Matérn**：在每个半视野内，对相对训练位点均值的残差做 Matérn(ν=3/2) 平滑，再加回位点均值（不向访视总均值 kriging）。
2. **VC**：在平滑后的簇轨迹上 MoM 估计 `var_cl_int`、`sigma2`、`spat_var`/`nugget`（不再硬编码 `var_cl_int=1`）。
3. **预测**：平滑后簇 OLS，再对偏离群体 `beta1` 做 EB 收缩；当平滑后 SE 小时权重→1，行为接近 spaMM 弱先验下的 BLUP。

未改锁定空间定义字符串 → **不需要**发布改 spatial 的 R9。

## 修复后检查点（同协议重跑）

脚本：`src/18_b4_vs_b1_checkpoint.py`  
记录：`results/baselines/b4_vs_b1_checkpoint.json`（SHA-256 `a93624977e71083c0129ba4486f24339b603ad7faccdc488155624a53badc2ba`）

| 统计量 | 失败（旧） | 修复后 |
|---|---:|---:|
| RMSE_B4 | 1.476 | **0.926** |
| RMSE_B1 | 1.379 | 1.379 |
| RMSE_B4 − RMSE_B1 | +0.097 | **−0.453** |
| 95% CI | [+0.019, +0.177] | **[−0.493, −0.405]** |
| 判定 | FAIL | **PASS** |

子样本快速对照（每情景 150）：固定 B4 eqRMSE≈0.911 vs B1≈1.361。

拟合后 VC 摘要：`spat/(spat+nugget)≈0.85`，`var_cl_int≈17.6`，`sigma2≈2.05`，`beta1≈−0.22`。

## 未做

- 未改 / 未重生成模拟器；未打开 `sim_test`
- 未推送；未发布预注册修订
- 因实现错误已解释失败，**未**起草「模拟器相对 Matérn 误设」修订提案
