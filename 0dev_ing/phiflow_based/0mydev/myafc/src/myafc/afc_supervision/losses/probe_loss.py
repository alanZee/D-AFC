"""
探针 Loss 函数

可微分的探针速度损失计算
"""
import torch
from torch import Tensor
from typing import Optional


def probe_mse_loss(
    current_velocity: Tensor,
    target_velocity: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """探针速度 MSE Loss
    
    L = weight × mean((v_current - v_target)²)
    
    Args:
        current_velocity: 当前探针速度 (batch, 2) 或 (2,)
        target_velocity: 目标探针速度 (batch, 2) 或 (2,)
        weight: 权重系数
    
    Returns:
        标量 loss 张量
    """
    diff = current_velocity - target_velocity
    return weight * (diff ** 2).mean()


def probe_mae_loss(
    current_velocity: Tensor,
    target_velocity: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """探针速度 MAE Loss
    
    L = weight × mean(|v_current - v_target|)
    
    Args:
        current_velocity: 当前探针速度 (batch, 2) 或 (2,)
        target_velocity: 目标探针速度 (batch, 2) 或 (2,)
        weight: 权重系数
    
    Returns:
        标量 loss 张量
    """
    diff = current_velocity - target_velocity
    return weight * diff.abs().mean()


def probe_component_loss(
    current_velocity: Tensor,
    target_velocity: Tensor,
    u_weight: float = 1.0,
    v_weight: float = 1.0,
    loss_type: str = 'mse',
) -> Tensor:
    """分量加权的探针速度 Loss
    
    可分别对 u, v 分量设置不同权重
    
    Args:
        current_velocity: 当前探针速度 (batch, 2) 或 (2,)
        target_velocity: 目标探针速度 (batch, 2) 或 (2,)
        u_weight: u 分量权重
        v_weight: v 分量权重
        loss_type: 'mse' 或 'mae'
    
    Returns:
        标量 loss 张量
    """
    diff = current_velocity - target_velocity
    
    if loss_type == 'mse':
        error = diff ** 2
    else:  # mae
        error = diff.abs()
    
    # 分量加权
    if diff.dim() == 1:
        return u_weight * error[0] + v_weight * error[1]
    else:
        return (u_weight * error[..., 0] + v_weight * error[..., 1]).mean()
