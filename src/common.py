"""方向 A 公用函数：读配置、路径、日志。"""
from __future__ import annotations
import datetime, json, logging, os, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]          # directionA/


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_config(updates: dict[str, dict]) -> None:
    """按“段.键”就地改写 config.yaml 中的标量行（保留注释）。"""
    p = ROOT / "config.yaml"
    lines = p.read_text(encoding="utf-8").splitlines()
    for section, kv in updates.items():
        in_sec = False
        sec_path = section.split(".")
        depth_found = 0
        for i, ln in enumerate(lines):
            stripped = ln.lstrip()
            indent = len(ln) - len(stripped)
            if depth_found < len(sec_path):
                if stripped.startswith(sec_path[depth_found] + ":") and indent == 2 * depth_found:
                    depth_found += 1
                continue
            if indent < 2 * len(sec_path) and stripped and not stripped.startswith("#"):
                break
            for k, v in kv.items():
                if stripped.startswith(k + ":") and indent == 2 * len(sec_path):
                    comment = ""
                    if "#" in ln.split(":", 1)[1]:
                        comment = "  #" + ln.split("#", 1)[1]
                    lines[i] = " " * indent + f"{k}: {yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}" + comment
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_logger(name: str) -> logging.Logger:
    (ROOT / "logs").mkdir(exist_ok=True)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(ROOT / "logs" / f"{name}.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        lg.addHandler(fh)
        lg.addHandler(sh)
    lg.info("start %s (python %s)", name, sys.version.split()[0])
    return lg


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def load_arrays(eye_subset=None):
    """读长表 -> 按检查排列的 (n_visits, 59) 数组及检查表。eye_subset：只保留这些 eye_id。"""
    import numpy as np, pandas as pd
    d = ROOT / "data"
    vis = pd.read_parquet(d / "visits.parquet")
    lg = pd.read_parquet(d / "vf_long.parquet")
    lg = lg.sort_values(["eye_id", "visit", "loc"])
    Y = lg["y"].to_numpy().reshape(-1, 59)
    C = lg["censored"].to_numpy().reshape(-1, 59)
    vis = vis.sort_values(["eye_id", "visit"]).reset_index(drop=True)
    if eye_subset is not None:
        m = vis["eye_id"].isin(set(eye_subset)).to_numpy()
        vis, Y, C = vis[m].reset_index(drop=True), Y[m], C[m]
    return vis, Y, C


def pointwise_ols_residuals(vis, Y):
    """每只眼、每个点对时间做 OLS。返回 fitted, resid_corrected(×sqrt(n/(n-2))), slope, intercept, se 等。"""
    import numpy as np
    fitted = np.empty_like(Y, dtype=float)
    resid = np.empty_like(Y, dtype=float)
    per_eye = {}
    for eye, idx in vis.groupby("eye_id").indices.items():
        t = vis["t"].to_numpy()[idx]
        y = Y[idx].astype(float)
        n = len(t)
        X = np.column_stack([np.ones(n), t])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        f = X @ beta
        r = y - f
        fitted[idx] = f
        resid[idx] = r * np.sqrt(n / (n - 2))
        sxx = ((t - t.mean()) ** 2).sum()
        rss = (r ** 2).sum(0)
        per_eye[eye] = dict(n=n, slope=beta[1], intercept=beta[0], sxx=sxx, rss=rss)
    return fitted, resid, per_eye
