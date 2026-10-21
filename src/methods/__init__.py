"""对照与参赛方法（预注册第 2 节）。"""
from .base import Estimate, Series, GeometryInfo, cluster_slopes_from_denoised, global_slope_from_denoised
from .simple import B0, B1, B2, B5
from .eb import EB, EBGK
from .lmm import B3, B4
from .poplr import PoPLR

METHODS = {
    "B0": B0,
    "B1": B1,
    "B2": B2,
    "B3": B3,
    "B4": B4,
    "B5": B5,
    "EB": EB,
    "EB_GK": EBGK,
    "PoPLR": PoPLR,
}

__all__ = [
    "Estimate", "Series", "GeometryInfo", "METHODS",
    "B0", "B1", "B2", "B3", "B4", "B5", "EB", "EBGK", "PoPLR",
]
