"""
myafc.engines - 通用引擎模块

包含:
- validators: AFC 通用验证引擎
"""
from .validators import AFCValidator, ValidatorConfig

__all__ = ['AFCValidator', 'ValidatorConfig']
