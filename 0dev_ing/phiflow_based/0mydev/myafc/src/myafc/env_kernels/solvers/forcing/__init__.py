"""
Forcing 场生成模块

提供不同类型的外力场生成器，用于 CFD 仿真中的体力项。
"""
from .constant_forcing import ConstantForcing
from .kolmogorov_forcing import KolmogorovForcing

__all__ = ['ConstantForcing', 'KolmogorovForcing']
