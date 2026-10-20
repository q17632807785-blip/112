#!/usr/bin/env bash
# 发布前数据步骤（预注册 3.1–3.3、第 6 部分）。不运行任何参赛方法，不生成 sim_train/sim_test。
set -euo pipefail
cd "$(dirname "$0")"; PY=../.venv/bin/python
$PY src/check_config_sync.py          | tee logs/check_config_sync.log
$PY src/00_build_long.py              | tee logs/00_build_long.log
Rscript R/00_export_g1_coords.R data/visualFields_g1_coord.csv
$PY src/01_g1_locations.py            | tee logs/01_g1_locations.log
$PY src/01_validate_locations.py      | tee logs/01_validate_locations.log
$PY src/02_split.py                   | tee logs/02_split.log
$PY src/03_noise_model.py             | tee logs/03_noise_model.log
$PY src/03b_censor_diagnostic.py      | tee logs/03b_censor_diagnostic.log
sha256sum data/split.json data/g1_locations.csv data/noise_model.json config.yaml | tee logs/sha256.txt
