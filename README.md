# GRAPE Direction A — preregistration (through revision R9)

- `preregistration.md` is authoritative (Chinese); `preregistration_EN.md` is a faithful English translation.
- `config.yaml` is byte-identical to the YAML block in Part 5; verify with `python src/check_config_sync.py`.
- `data/g1_locations.csv`: G1 location map. Raw GRAPE data and per-subject derived files are NOT included.
- `.gitattributes` (`* -text`) prevents line-ending conversion; `preregistration.md` and `config.yaml` use CRLF.
- **R8 (2026-10-21)**: locks B4 spatial term to `spaMM_matern_within_hemifield` after formal timing (early vs week-3 window); see section 7.3.
- **R9 (2026-10-21)**: records week-5 B4 vs B1 checkpoint **pass** (RMSE_B4 = 0.926384, RMSE_B1 = 1.379045, 95% CI entirely < 0); discloses early run on 2026-10-21; updates `src/methods/lmm.py` SHA-256 to `69b7293bb6393cb19a534b3c7d53b6bf99310d3c2afb758fd22e32c603aff477` after residual-Matérn / EB-BLUP fix **without** changing the locked `spatial_impl` string (section 7.4). Includes `src/18_b4_vs_b1_checkpoint.py` and `results/baselines/b4_vs_b1_checkpoint.json` + `b4_debug.md`.
- Simulator (R5) and sealed `sim_test` hash (R7) unchanged; `sim_test.npz` is not in this repository.
- Method code under `src/methods/` and evaluation scripts are included for reproducibility of baselines; no `.npz` simulations, no `.pt` weights, no per-person parquet.

中文摘要：修订 R9 记录第 5 周检查点通过（提前于 2026-10-21 运行），并更新 B4 实现文件哈希；R8 锁定的 `spaMM_matern_within_hemifield` 字符串未改。不含原始数据、模拟 npz、模型权重。
