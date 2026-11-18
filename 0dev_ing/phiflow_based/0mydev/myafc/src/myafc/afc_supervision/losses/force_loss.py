"""
力系数 Loss 函数

可微分的阻力/升力系数损失计算

包含:
1. 基础力系数 Loss (MSE 式)
2. 阻力减小 Loss
3. 升力抑制 Loss  
4. DRLinFluids 风格的 Reward-based Loss
5. 涡脱抑制 Loss

参考: DRLinFluids - 基于深度强化学习的流体控制
"""
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# ============================================================
# 配置类
# ============================================================
@dataclass
class ForceLossConfig:
    """力系数 Loss 配置
    
    DRLinFluids 风格的奖励函数设计 (论文原始公式):
        reward = cd_baseline - Cd - alpha_lift × |Cl|
    
    其中 cd_baseline 为无控制时的参考阻力系数 (如 Re=100 方柱约为 1.549)
    参考: DRLinFluids 技术文档
    
    Attributes:
        enabled: 是否启用
        
        # === 基础 MSE Loss ===
        cd_target: 目标阻力系数
        cl_target: 目标升力系数
        cd_weight: Cd MSE 权重
        cl_weight: Cl MSE 权重
        
        # === DRLinFluids Reward Loss ===
        drl_enabled: 启用 DRLinFluids 风格 loss
        cd_baseline: 无控制基线 Cd (如 Re=100 方柱为 1.549)
        alpha_lift: 升力惩罚系数 (论文默认 0.1)
        
        # === 涡脱抑制 ===
        vortex_shedding_enabled: 启用涡脱抑制 loss
        vortex_cl_history_len: Cl 历史长度 (用于检测周期性)
    """
    enabled: bool = False
    
    # 基础 MSE
    cd_target: float = 0.0
    cl_target: float = 0.0
    cd_weight: float = 1.0
    cl_weight: float = 1.0
    
    # DRLinFluids Reward (论文原始公式)
    drl_enabled: bool = False
    cd_baseline: float = 1.549  # Re=100 方柱无控制基线 Cd
    alpha_lift: float = 0.1     # 升力惩罚系数 (论文默认 0.1)
    
    # 涡脱抑制
    vortex_shedding_enabled: bool = False
    vortex_cl_history_len: int = 50


# ============================================================
# 基础 Loss
# ============================================================
def force_coeffs_loss(
    cd: Tensor,
    cl: Tensor,
    cd_target: float = 0.0,
    cl_target: float = 0.0,
    cd_weight: float = 1.0,
    cl_weight: float = 1.0,
) -> Tensor:
    """力系数 MSE Loss
    
    L = cd_weight × (Cd - Cd_target)² + cl_weight × (Cl - Cl_target)²
    
    Args:
        cd: 当前阻力系数 (标量或 batch)
        cl: 当前升力系数 (标量或 batch)
        cd_target: 目标阻力系数
        cl_target: 目标升力系数
        cd_weight: Cd 项权重
        cl_weight: Cl 项权重
    
    Returns:
        标量 loss 张量
    """
    cd_loss = cd_weight * (cd - cd_target) ** 2
    cl_loss = cl_weight * (cl - cl_target) ** 2
    
    if cd_loss.dim() > 0:
        return (cd_loss + cl_loss).mean()
    return cd_loss + cl_loss


def drag_reduction_loss(
    cd: Tensor,
    cd_baseline: float,
    weight: float = 1.0,
) -> Tensor:
    """阻力减小 Loss
    
    鼓励 Cd 低于基线值
    L = weight × max(0, Cd - Cd_baseline)²
    
    Args:
        cd: 当前阻力系数
        cd_baseline: 基线阻力系数
        weight: 权重
    
    Returns:
        标量 loss 张量
    """
    excess = torch.relu(cd - cd_baseline)
    return weight * (excess ** 2).mean() if excess.dim() > 0 else weight * excess ** 2


def lift_suppression_loss(
    cl: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """升力抑制 Loss
    
    抑制升力波动 (使 Cl 趋近于 0)
    L = weight × Cl²
    
    Args:
        cl: 当前升力系数
        weight: 权重
    
    Returns:
        标量 loss 张量
    """
    return weight * (cl ** 2).mean() if cl.dim() > 0 else weight * cl ** 2


# ============================================================
# DRLinFluids 风格 Reward-based Loss
# ============================================================
def drl_reward_loss(
    cd: Tensor,
    cl: Tensor,
    cd_baseline: float = 1.549,
    alpha_lift: float = 0.1,
) -> Tensor:
    """DRLinFluids 风格 Reward-based Loss (论文原始公式)
    
    将 reward 转换为 loss: L = -reward (最大化 reward = 最小化 loss)
    
    论文公式:
        reward = cd_baseline - Cd - alpha_lift × |Cl|
    
    其中:
        - cd_baseline: 无控制基线阻力系数 (如 Re=100 方柱为 1.549)
        - alpha_lift: 升力惩罚系数 (论文默认 0.1)
    
    参考: DRLinFluids 技术文档 - 方柱案例
    
    Args:
        cd: 当前阻力系数
        cl: 当前升力系数
        cd_baseline: 无控制基线 Cd (默认 1.549)
        alpha_lift: 升力惩罚系数 (默认 0.1)
    
    Returns:
        标量 loss (负 reward)
    """
    # DRLinFluids 论文原始公式: r = cd_baseline - Cd - alpha_lift × |Cl|
    reward = cd_baseline - cd - alpha_lift * torch.abs(cl)
    
    # 取平均 (如果是 batch)
    if reward.dim() > 0:
        reward = reward.mean()
    
    # 转为 loss (负 reward)
    loss = -reward
    return loss


def compute_control_power(
    control_signal: Tensor,
    power_type: str = 'l2',
) -> Tensor:
    """计算控制功率
    
    Args:
        control_signal: 控制信号 (n_jets,) 或 (batch, n_jets)
        power_type: 'l2' | 'l1' | 'max'
    
    Returns:
        功率标量
    """
    if power_type == 'l2':
        return (control_signal ** 2).sum(dim=-1).sqrt().mean() if control_signal.dim() > 1 else (control_signal ** 2).sum().sqrt()
    elif power_type == 'l1':
        return control_signal.abs().sum(dim=-1).mean() if control_signal.dim() > 1 else control_signal.abs().sum()
    elif power_type == 'max':
        return control_signal.abs().max()
    else:
        return (control_signal ** 2).sum(dim=-1).sqrt().mean() if control_signal.dim() > 1 else (control_signal ** 2).sum().sqrt()


# ============================================================
# 涡脱抑制 Loss
# ============================================================
def vortex_shedding_loss(
    cl_history: Tensor,
    target_frequency: float = 0.0,
    weight: float = 1.0,
) -> Tensor:
    """涡脱抑制 Loss
    
    基于 Cl 历史的 FFT 分析，抑制涡脱频率成分。
    
    Args:
        cl_history: Cl 时间序列 (n_steps,)
        target_frequency: 目标频率 (0 = 抑制所有波动)
        weight: 权重
    
    Returns:
        涡脱抑制 loss
    """
    if cl_history.numel() < 4:
        return torch.tensor(0.0, device=cl_history.device)
    
    # 去均值
    cl_centered = cl_history - cl_history.mean()
    
    # FFT
    fft_result = torch.fft.rfft(cl_centered)
    magnitudes = torch.abs(fft_result)
    
    # 排除 DC 分量
    magnitudes = magnitudes[1:]
    
    if target_frequency == 0.0:
        # 抑制所有非零频率
        loss = magnitudes.sum()
    else:
        # 抑制特定频率附近
        n = cl_history.numel()
        freqs = torch.arange(1, len(magnitudes) + 1, device=cl_history.device) / n
        # 高斯加权
        weights = torch.exp(-((freqs - target_frequency) ** 2) / 0.01)
        loss = (magnitudes * weights).sum()
    
    return weight * loss


def cl_oscillation_loss(
    cl_history: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """Cl 振荡幅值 Loss
    
    简单的升力波动 loss，基于历史 Cl 的标准差。
    
    Args:
        cl_history: Cl 时间序列 (n_steps,)
        weight: 权重
    
    Returns:
        振荡幅值 loss
    """
    if cl_history.numel() < 2:
        return torch.tensor(0.0, device=cl_history.device)
    
    std = cl_history.std()
    return weight * std


# ============================================================
# 综合 Loss
# ============================================================
def combined_force_loss(
    cd: Tensor,
    cl: Tensor,
    control_signal: Optional[Tensor] = None,
    cl_history: Optional[Tensor] = None,
    config: Optional[ForceLossConfig] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """综合力系数 Loss
    
    根据配置组合多种 loss。
    
    Args:
        cd: 当前阻力系数
        cl: 当前升力系数
        control_signal: 控制信号 (用于功率计算)
        cl_history: Cl 历史 (用于涡脱抑制)
        config: ForceLossConfig 配置
    
    Returns:
        total_loss: 总 loss
        loss_dict: 各项 loss 分解
    """
    if config is None:
        config = ForceLossConfig()
    
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=cd.device)
    
    # 基础 MSE Loss
    if config.cd_weight > 0 or config.cl_weight > 0:
        mse_loss = force_coeffs_loss(
            cd, cl,
            config.cd_target, config.cl_target,
            config.cd_weight, config.cl_weight
        )
        loss_dict['mse'] = mse_loss
        total_loss = total_loss + mse_loss
    
    # DRLinFluids Reward Loss (论文原始公式)
    if config.drl_enabled:
        drl_loss = drl_reward_loss(
            cd, cl,
            config.cd_baseline,
            config.alpha_lift,
        )
        loss_dict['drl'] = drl_loss
        total_loss = total_loss + drl_loss
    
    # 涡脱抑制 Loss
    if config.vortex_shedding_enabled and cl_history is not None:
        vortex_loss = vortex_shedding_loss(cl_history)
        loss_dict['vortex'] = vortex_loss
        total_loss = total_loss + vortex_loss
    
    return total_loss, loss_dict
