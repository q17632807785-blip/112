"""sim_test 封存守卫：任何评估/训练路径必须经此模块加载；默认拒绝读内容。"""
from __future__ import annotations

import hashlib
import os
from datetime import date

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sim_test_hash_only() -> str:
    """只核对 config 中的 SHA-256，不打开文件内容。"""
    cfg = _cfg()
    seal = cfg["sim_test_seal"]
    path = os.path.join(ROOT, "data", "sim_test.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    got = sha256_file(path)
    expect = seal["sha256"]
    if got != expect:
        raise RuntimeError(f"sim_test.npz SHA-256 与预注册不符：got={got} expect={expect}")
    return got


def load_sim_test(*, UNSEAL: bool = False):
    """读取 sim_test.npz。UNSEAL 必须显式为 True，且不得早于解封日。默认 False。"""
    cfg = _cfg()
    seal = cfg["sim_test_seal"]
    if not UNSEAL:
        raise PermissionError(
            "拒绝加载 sim_test.npz：UNSEAL=False（预注册封存；评估脚本不得读内容，只可核对哈希）。"
        )
    if seal.get("sealed", True):
        unseal = date.fromisoformat(seal["unseal_not_before_date"])
        if date.today() < unseal:
            raise PermissionError(
                f"拒绝加载 sim_test.npz：今日早于解封日 {seal['unseal_not_before_date']}。"
            )
    verify_sim_test_hash_only()
    import numpy as np
    return np.load(os.path.join(ROOT, "data", "sim_test.npz"), allow_pickle=True)


def load_sim_train():
    """允许读取 sim_train（开发集模拟，未封存）。"""
    import numpy as np
    path = os.path.join(ROOT, "data", "sim_train.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)
