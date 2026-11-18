# -*- coding: utf-8 -*-
"""
myafc - 主动流动控制 (Active Flow Control) 工具包

包含：
- afc_supervision: 基于监督学习的 AFC 训练框架
- afc_drl_sb3: 基于 Stable-Baselines3 的深度强化学习 AFC 框架
- env_kernels: 流场仿真环境核心模块
- utils: 通用工具函数
"""

# ============================================================
# 解决 OpenMP 运行时库冲突问题（Windows 平台常见）
# 必须在任何其他 import 之前设置
# 原因：PyTorch、NumPy、MKL 等可能各自链接了 libiomp5md.dll
# ============================================================
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

__version__ = '0.1.0'
__author__ = 'myafc contributors'
