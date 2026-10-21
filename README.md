# GRAPE Direction A — preregistration (through revision R10)

- `preregistration.md` is authoritative (Chinese); `preregistration_EN.md` is a faithful English translation.
- `config.yaml` is byte-identical to the YAML block in Part 5; verify with `python src/check_config_sync.py`.
- `data/g1_locations.csv`: G1 location map. Raw GRAPE data and per-subject derived files are NOT included.
- `.gitattributes` (`* -text`) prevents line-ending conversion; `preregistration.md` and `config.yaml` use CRLF.
- **R8**: locks B4 spatial term to `spaMM_matern_within_hemifield` (section 7.3).
- **R9**: week-5 B4 vs B1 checkpoint **pass** on `sim_train`; early run disclosed; `lmm.py` hash `69b7293…` (section 7.4).
- **R10 (2026-10-21)**: fixes B4 **point-level SE** (no cluster-trajectory broadcast onto points); updates `src/methods/lmm.py` SHA-256 to `0bc51e64a750445755ed099b9ac567f0a60344876b5f9eae03171ef74e8a2817`; cluster RMSE / R9 checkpoint numbers **unchanged**; `spatial_impl` string **unchanged** (section 7.5).
- Optional local summary (no private paths): `results/最终结果_解封前.md`.
- Simulator (R5) and sealed `sim_test` hash (R7) unchanged; `sim_test.npz` is not in this repository.
- No `.npz` simulations, no `.pt` weights, no per-person parquet.

中文摘要：修订 R10 仅修复 B4 逐点 SE 并更新实现哈希；R8/R9 锁定的空间项字符串与第 5 周簇级检查点数字均未改。不含原始数据、模拟 npz、模型权重。
