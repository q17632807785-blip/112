# GRAPE Direction A — preregistration (through revision R8)

- `preregistration.md` is authoritative (Chinese); `preregistration_EN.md` is a faithful English translation.
- `config.yaml` is byte-identical to the YAML block in Part 5; verify with `python src/check_config_sync.py`.
- `data/g1_locations.csv`: G1 location map. Raw GRAPE data and per-subject derived files are NOT included.
- `.gitattributes` (`* -text`) prevents line-ending conversion; `preregistration.md` and `config.yaml` use CRLF.
- **R8 (2026-10-21)**: locks B4 spatial term to `spaMM_matern_within_hemifield` after formal timing on 20 development eyes (seed 20261021). Extrapolation to 185 development eyes ≈ 0.079 h ≤ 24 h. Timing was completed **early** relative to the announced week-3 window 2026-11-02–11-08, at the investigator's request (disclosed in section 7.3). Implementation: `src/methods/lmm.py`, timing script `src/17b_b4_timing_formal.py`.
- Simulator (R5) and sealed `sim_test` hash (R7) unchanged; `sim_test.npz` is not in this repository.
- Method code under `src/methods/` and evaluation scripts (`10_`–`17b_`) are included for reproducibility of baselines; no `.npz` simulations, no `.pt` weights, no per-person parquet.

中文摘要：修订 R8 锁定 B4 空间项为同半视野 spaMM Matérn；计时提前于 2026-10-21 完成并披露。不含原始数据、模拟 npz、模型权重。
