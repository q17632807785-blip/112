# GRAPE Direction A — preregistration / 预注册（方向 A：Octopus G1 视野去噪方法基准）
- `preregistration.md` is authoritative (Chinese); `preregistration_EN.md` is a faithful English translation. 中文版为准，英文版为忠实翻译。
- `config.yaml` is byte-identical to the YAML block in Part 5; verify with `python src/check_config_sync.py`. `config.yaml` 与第 5 部分逐字节一致，可用该脚本核对。
- `data/g1_locations.csv`: G1 location map (61 GRAPE indices, coordinates, hemifield, 10 clusters). G1 点位与 10 簇表。
- Raw GRAPE data and per-subject derived files are NOT included (obtain GRAPE from Huang et al., Sci Data 2023;10:520). 不含原始数据与逐人派生数据。
- Rebuild pre-publication data steps: place `VF and clinical information.xlsx` one level above this folder, extract GRAPE Fig. 2 to `figures/GRAPE_fig2_source.png`, then run `./run_prepublication.sh` (Python venv at `../.venv`; R with visualFields). 重建步骤同左。
- SHA-256 of the non-public files (`split.json`, `noise_model.json`) are listed in Part 6. 未公开文件的哈希见第 6 部分。
- `.gitattributes` (`* -text`) prevents line-ending conversion; `preregistration.md` and `config.yaml` use CRLF. 行尾为 CRLF，请勿转换。
