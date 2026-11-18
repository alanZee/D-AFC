"""
可微分数学操作模块

封装用于 AFC 训练的可微分操作:
1. FFT 模态分析 - 傅里叶变换
2. POD 分解 - 本征正交分解  
3. DMD 分解 - 动态模态分解

所有实现使用 PyTorch 原生操作，完全可微分。

使用示例:
    from myafc.utils.differentiable_ops import differentiable_fft, differentiable_pod, differentiable_dmd
    
    # FFT
    magnitudes, phases = differentiable_fft(field)
    
    # POD
    modes, coeffs, svals = differentiable_pod(snapshots, n_modes=5)
    
    # DMD
    modes, eigenvalues, amplitudes = differentiable_dmd(snapshots, n_modes=5)
"""
from __future__ import annotations
from typing import Tuple, Optional
import torch


# ============================================================
# FFT 可微分操作
# ============================================================
def differentiable_fft(
    field: torch.Tensor,
    return_complex: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """可微分 2D FFT
    
    Args:
        field: 输入场 (H, W) 或 (batch, H, W)
        return_complex: 是否返回复数结果
    
    Returns:
        magnitudes: FFT 幅值 (可微分)
        phases: FFT 相位 (仅 return_complex=False 时)，None if return_complex=True
    """
    if field.dim() == 2:
        field = field.unsqueeze(0)
    
    # 2D FFT (PyTorch 原生，可微分)
    fft_result = torch.fft.fft2(field)  # fft2, rfft2
    
    if return_complex:
        return fft_result, None
    
    magnitudes = torch.abs(fft_result)
    phases = torch.angle(fft_result)
    
    return magnitudes, phases


def fft_get_top_modes(
    magnitudes: torch.Tensor,
    n_modes: int = 5,
    exclude_dc: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取 FFT 幅值最大的 n 个模态
    
    Args:
        magnitudes: FFT 幅值 (batch, H, W) 或 (H, W)
        n_modes: 选取的模态数
        exclude_dc: 是否排除 DC 分量
    
    Returns:
        top_values: top-n 幅值
        top_indices: top-n 索引
    """
    if magnitudes.dim() == 2:
        magnitudes = magnitudes.unsqueeze(0)
    
    batch_size = magnitudes.shape[0]
    flat = magnitudes.view(batch_size, -1)
    
    if exclude_dc:
        flat = flat.clone()
        flat[:, 0] = 0
    
    k = min(n_modes, flat.shape[1])
    top_values, top_indices = torch.topk(flat, k=k, dim=1)
    
    return top_values, top_indices


# ============================================================
# POD 可微分操作
# ============================================================
def differentiable_pod(
    snapshots: torch.Tensor,
    n_modes: int = 5,
    center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """可微分 POD (本征正交分解)
    
    使用 SVD 实现，完全可微分。
    X = U @ diag(S) @ Vh
    
    Args:
        snapshots: 快照矩阵 (n_snapshots, ...) 
                   支持任意形状，自动展平空间维度
        n_modes: 保留的模态数量
        center: 是否中心化 (减去时间均值)
    
    Returns:
        modes: 空间模态 (n_modes, n_features)
        coeffs: 时间系数 (n_snapshots, n_modes)
        singular_values: 奇异值 (n_modes,)
    
    数学关系:
        snapshots ≈ coeffs @ modes + mean (if centered)
        modes[i] 是第 i 个空间模态 (特征向量)
        coeffs[:, i] 是第 i 个模态的时间系数
        singular_values[i]² / sum(singular_values²) 是能量占比
    """
    # 展平空间维度
    orig_shape = snapshots.shape
    if snapshots.dim() > 2:
        snapshots = snapshots.view(orig_shape[0], -1)
    
    n_snapshots, n_features = snapshots.shape
    n_modes = min(n_modes, n_snapshots, n_features)
    
    # 中心化
    if center:
        mean = snapshots.mean(dim=0, keepdim=True)
        centered = snapshots - mean
    else:
        centered = snapshots
    
    # SVD (PyTorch 原生，可微分)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    
    # 截断到 n_modes
    modes = Vh[:n_modes]                    # (n_modes, n_features)
    coeffs = U[:, :n_modes] * S[:n_modes]   # (n_snapshots, n_modes)
    singular_values = S[:n_modes]
    
    return modes, coeffs, singular_values


def pod_reconstruct(
    modes: torch.Tensor,
    coeffs: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """使用 POD 模态重构快照
    
    Args:
        modes: 空间模态 (n_modes, n_features)
        coeffs: 时间系数 (n_snapshots, n_modes)
        mean: 时间均值 (n_features,)
    
    Returns:
        reconstructed: 重构快照 (n_snapshots, n_features)
    """
    reconstructed = coeffs @ modes
    if mean is not None:
        reconstructed = reconstructed + mean
    return reconstructed


# ============================================================
# DMD 可微分操作
# ============================================================
def differentiable_dmd(
    snapshots: torch.Tensor,
    n_modes: int = 5,
    dt: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """可微分 DMD (动态模态分解)
    
    使用 exact DMD 算法，基于 SVD 实现，完全可微分。
    
    DMD 将时序数据分解为:
        X(t) = Σ φ_i × exp(ω_i × t) × b_i
    其中 φ_i 是空间模态，ω_i 是复频率，b_i 是幅值
    
    Args:
        snapshots: 快照矩阵 (n_snapshots, n_features) 或 (n_snapshots, H, W)
        n_modes: 保留的模态数量
        dt: 时间步长 (用于计算频率)
    
    Returns:
        modes: DMD 模态 (n_modes, n_features)，取绝对值后的实数
        eigenvalues: 复特征值 (n_modes,)
        amplitudes: 模态幅值 (n_modes,)
    
    特征值解读:
        |λ| < 1: 衰减模态
        |λ| = 1: 稳态/周期模态
        |λ| > 1: 增长模态
        arg(λ) / dt: 振荡频率
    """
    # 展平空间维度
    orig_shape = snapshots.shape
    if snapshots.dim() > 2:
        snapshots = snapshots.view(orig_shape[0], -1)
    
    n_snapshots, n_features = snapshots.shape
    
    # 构造时间延迟矩阵
    X = snapshots[:-1].T  # (n_features, n_snapshots-1)
    Y = snapshots[1:].T   # (n_features, n_snapshots-1)
    
    # SVD 降维
    n_modes = min(n_modes, n_snapshots - 1, n_features)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    U = U[:, :n_modes]
    S = S[:n_modes]
    V = Vh[:n_modes].T
    
    # 构造低维算子 A_tilde = U^H @ Y @ V @ S^{-1}
    S_inv = 1.0 / (S + 1e-10)
    A_tilde = (U.T @ Y @ V) * S_inv.unsqueeze(0)
    
    # 特征分解 (复数)
    A_tilde_complex = A_tilde.to(torch.complex64)
    eigenvalues, W = torch.linalg.eig(A_tilde_complex)
    
    # 恢复 DMD 模态: Φ = Y @ V @ S^{-1} @ W
    Y_V_Sinv = (Y @ V * S_inv).to(torch.complex64)
    modes_complex = Y_V_Sinv @ W  # (n_features, n_modes)
    
    # 计算幅值
    amplitudes = torch.abs(eigenvalues)
    
    # 转为实数 (取模)
    modes = torch.abs(modes_complex.T)  # (n_modes, n_features)
    
    return modes, eigenvalues, amplitudes


def dmd_get_frequencies(
    eigenvalues: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    """从 DMD 特征值计算频率
    
    ω = Im(log(λ)) / dt
    
    Args:
        eigenvalues: 复特征值
        dt: 时间步长
    
    Returns:
        frequencies: 频率 (Hz)
    """
    # log(λ) = log|λ| + i·arg(λ)
    log_eig = torch.log(eigenvalues + 1e-10)
    frequencies = log_eig.imag / dt / (2 * torch.pi)
    return frequencies.real


def dmd_get_growth_rates(eigenvalues: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """从 DMD 特征值计算增长率
    
    σ = Re(log(λ)) / dt
    σ > 0: 增长; σ < 0: 衰减; σ ≈ 0: 稳态
    
    Args:
        eigenvalues: 复特征值
        dt: 时间步长
    
    Returns:
        growth_rates: 增长率
    """
    log_eig = torch.log(eigenvalues + 1e-10)
    return log_eig.real / dt


# ============================================================
# 涡量计算
# ============================================================
def compute_vorticity(
    velocity: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
) -> torch.Tensor:
    """计算涡量场 (可微分)
    
    ω = ∂v/∂x - ∂u/∂y
    
    Args:
        velocity: 速度场 (2, H, W) - [u, v] 分量
        dx, dy: 网格间距
    
    Returns:
        vorticity: 涡量场 (H-2, W-2) - 中心差分边界缩小
    """
    u, v = velocity[0], velocity[1]
    
    # 中心差分
    dvdx = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    dudy = (u[2:, :] - u[:-2, :]) / (2 * dy)
    
    # 裁剪到相同尺寸
    H, W = min(dvdx.shape[0], dudy.shape[0]), min(dvdx.shape[1], dudy.shape[1])
    vorticity = dvdx[:H, :W] - dudy[:H, :W]
    
    return vorticity


def velocity_magnitude(velocity: torch.Tensor) -> torch.Tensor:
    """计算速度模值 (可微分)
    
    Args:
        velocity: (2, H, W) 或 (H, W)
    
    Returns:
        magnitude: (H, W)
    """
    if velocity.dim() == 2:
        return velocity.abs()
    return torch.sqrt((velocity ** 2).sum(dim=0) + 1e-8)
