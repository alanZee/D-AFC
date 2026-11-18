"""
正则化 Loss 函数

包含:
- amplitude_l2_loss: 控制幅值 L2 正则化
- temporal_smoothness_loss: 时序平滑正则化
- entropy_loss: 熵正则化 (鼓励探索)
- kl_divergence_loss: KL 散度正则化 (稳定更新)
"""
import math
import torch
from torch import Tensor
from typing import Optional


def amplitude_l2_loss(
    amplitude: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """控制幅值 L2 正则化
    
    L = weight × mean(amplitude²)
    
    抑制过大的控制信号幅值
    
    Args:
        amplitude: 控制信号幅值 (batch,) 或标量
        weight: 权重
    
    Returns:
        标量 loss 张量
    """
    if amplitude.dim() == 0:
        return weight * amplitude ** 2
    return weight * (amplitude ** 2).mean()


def temporal_smoothness_loss(
    current_amplitude: Tensor,
    prev_amplitude: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """时序平滑正则化
    
    L = weight × mean((a_t - a_{t-1})²)
    
    惩罚控制信号的剧烈变化
    
    Args:
        current_amplitude: 当前幅值
        prev_amplitude: 上一步幅值
        weight: 权重
    
    Returns:
        标量 loss 张量
    """
    diff = current_amplitude - prev_amplitude
    if diff.dim() == 0:
        return weight * diff ** 2
    return weight * (diff ** 2).mean()


def entropy_loss(
    log_std: Tensor,
    weight: float = 0.01,
) -> Tensor:
    """熵正则化 (高斯分布)
    
    H(π) = Σ 0.5 × log(2πe × σ²) = Σ (0.5 × log(2πe) + log_std)
    
    返回负熵作为 loss (最小化 loss = 最大化熵)
    
    Args:
        log_std: 策略网络的 log 标准差
        weight: 权重 (正值鼓励探索)
    
    Returns:
        标量 loss 张量 (负熵)
    """
    # H = 0.5 * (1 + log(2π) + 2*log_std)
    entropy = 0.5 * (1 + math.log(2 * math.pi)) + log_std
    if entropy.dim() == 0:
        return -weight * entropy
    return -weight * entropy.mean()


def kl_divergence_loss(
    mean_new: Tensor,
    log_std_new: Tensor,
    mean_old: Tensor,
    log_std_old: Tensor,
    weight: float = 0.01,
) -> Tensor:
    """KL 散度正则化 (高斯分布)
    
    KL(π_new || π_old) = Σ [log(σ_old/σ_new) + (σ_new² + (μ_new-μ_old)²)/(2σ_old²) - 0.5]
    
    限制新旧策略分布的差异，避免更新过大
    
    Args:
        mean_new: 新策略均值
        log_std_new: 新策略 log 标准差
        mean_old: 旧策略均值
        log_std_old: 旧策略 log 标准差
        weight: 权重
    
    Returns:
        标量 loss 张量
    """
    std_new = log_std_new.exp()
    std_old = log_std_old.exp()
    var_old = std_old ** 2
    
    # KL 公式
    kl = log_std_old - log_std_new + (std_new ** 2 + (mean_new - mean_old) ** 2) / (2 * var_old) - 0.5
    
    if kl.dim() == 0:
        return weight * kl
    return weight * kl.mean()


def policy_stability_loss(
    current_amplitude: Tensor,
    ema_amplitude: Tensor,
    weight: float = 0.01,
) -> Tensor:
    """策略稳定性正则化
    
    L = weight × mean((a_t - ā_EMA)²)
    
    惩罚当前输出与移动平均的偏差，防止策略过冲
    
    Args:
        current_amplitude: 当前幅值
        ema_amplitude: EMA 移动平均幅值
        weight: 权重
    
    Returns:
        标量 loss 张量
    """
    diff = current_amplitude - ema_amplitude
    if diff.dim() == 0:
        return weight * diff ** 2
    return weight * (diff ** 2).mean()
