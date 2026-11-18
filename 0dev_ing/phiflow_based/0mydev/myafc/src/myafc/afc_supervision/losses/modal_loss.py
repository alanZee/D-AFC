"""
可微分 DMD/POD 模态分解 Loss

实现可微分的模态分解用于训练:
1. DMD (Dynamic Mode Decomposition) - 动态模态分解
2. POD (Proper Orthogonal Decomposition) - 本征正交分解

核心思想: 分解流场为模态，最小化非稳态模态的能量。

Note:
    底层 POD/DMD 实现位于 myafc.utils.differentiable_ops
    这些实现使用 PyTorch 原生操作，完全可微分。
"""
from __future__ import annotations
from typing import Optional, Tuple, List
import torch

from myafc.utils.differentiable_ops import (
    differentiable_pod,
    differentiable_dmd,
)


def pod_mode_loss(
    snapshots: torch.Tensor,
    n_modes: int = 5,
    exclude_first_k: int = 1,
    mode_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """POD 模态 Loss
    
    最小化非稳态模态的能量（奇异值）。
    
    Args:
        snapshots: 快照矩阵 (n_snapshots, H, W) 或 (n_snapshots, n_features)
        n_modes: 考虑的模态数量
        exclude_first_k: 排除前 k 个主模态（通常是稳态/平均场）
        mode_weights: 模态权重，None = 等权重
    
    Returns:
        loss: 模态能量加权和
    """
    _, _, singular_values = differentiable_pod(snapshots, n_modes)
    
    # 排除前几个主模态
    if exclude_first_k < n_modes:
        target_svals = singular_values[exclude_first_k:]
    else:
        target_svals = singular_values
    
    # 归一化
    total_energy = singular_values.sum() + 1e-10
    normalized_svals = target_svals / total_energy
    
    # 加权
    if mode_weights is not None:
        weights = torch.tensor(mode_weights[:len(normalized_svals)], device=snapshots.device)
        weights = weights / weights.sum()
        loss = (normalized_svals * weights).sum()
    else:
        loss = normalized_svals.mean()
    
    return loss


def dmd_mode_loss(
    snapshots: torch.Tensor,
    n_modes: int = 5,
    dt: float = 1.0,
    stability_threshold: float = 0.99,
) -> torch.Tensor:
    """DMD 模态 Loss
    
    最小化非稳态模态的幅值。
    稳态模态定义为 |λ| ≈ 1 的模态。
    
    Args:
        snapshots: 快照矩阵 (n_snapshots, H, W) 或 (n_snapshots, n_features)
        n_modes: 考虑的模态数量
        dt: 时间步长
        stability_threshold: 稳态判定阈值 (|λ| > threshold 视为稳态)
    
    Returns:
        loss: 非稳态模态幅值和
    """
    _, eigenvalues, amplitudes = differentiable_dmd(snapshots, n_modes, dt)
    
    # 识别非稳态模态 (|λ| 远离 1)
    abs_eigenvalues = torch.abs(eigenvalues)
    
    # 非稳态指标：与 1 的距离
    instability_indicator = torch.abs(abs_eigenvalues - 1.0)
    
    # 加权：不稳定模态给予更高权重
    weighted_amplitudes = amplitudes * instability_indicator
    
    loss = weighted_amplitudes.mean()
    return loss


def steady_state_residual_loss(
    snapshots: torch.Tensor,
    n_modes: int = 5,
) -> torch.Tensor:
    """稳态残差 Loss
    
    计算原流场减去稳态模态（第一个 POD 模态）的残差。
    
    最小化此 loss 等于鼓励流场趋近稳态。
    
    Args:
        snapshots: 快照矩阵 (n_snapshots, H, W) 或 (n_snapshots, n_features)
        n_modes: POD 模态数量（仅用第一个作为稳态）
    
    Returns:
        loss: 残差均方根
    """
    orig_shape = snapshots.shape
    if snapshots.dim() > 2:
        flat_snapshots = snapshots.view(orig_shape[0], -1)
    else:
        flat_snapshots = snapshots
    
    # POD 分解
    modes, coeffs, _ = differentiable_pod(flat_snapshots, n_modes)
    
    # 仅用第一个模态重构（稳态近似）
    steady_state = coeffs[:, 0:1] @ modes[0:1]
    
    # 残差
    residual = flat_snapshots - steady_state
    
    # RMS loss
    loss = torch.sqrt((residual ** 2).mean())
    return loss
