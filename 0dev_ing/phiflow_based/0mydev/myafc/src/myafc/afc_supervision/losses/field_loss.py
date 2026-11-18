"""
流场数据 Loss 函数

基于指定区域流场的 loss 计算:
1. 流场 RMS Loss - 直接对流场取均方根
2. FFT 模态 Loss - 傅里叶变换后最小化非零模态幅值  
3. 波动稀疏 Loss - 自动定位波动最大的点并最小化
4. 多场加权 Loss - 对多种场类型(速度/涡量/压力)加权求和

所有 loss 完全可微分。

区域指定规则:
    region: [x_start, y_start, x_end, y_end] (相对坐标 [0,1])
    region=None: 使用整个流场
    示例: region=[0.3, 0.2, 0.8, 0.8] 表示 x∈[0.3,0.8], y∈[0.2,0.8]
"""
from __future__ import annotations
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass, field as dataclass_field
import torch


# ============================================================
# 配置类
# ============================================================
@dataclass
class RegionConfig:
    """区域配置 (相对坐标 [0,1])
    
    Attributes:
        x_start: x 起始位置
        y_start: y 起始位置  
        x_end: x 结束位置
        y_end: y 结束位置
    """
    x_start: float = 0.5
    y_start: float = 0.0
    x_end: float = 1.0
    y_end: float = 1.0
    
    def to_list(self) -> List[float]:
        return [self.x_start, self.y_start, self.x_end, self.y_end]


@dataclass
class FieldLossConfig:
    """流场 Loss 配置
    
    支持的场类型:
        - 'velocity': 速度场 (2, H, W) 或 (H, W) → 取速度模值
        - 'vorticity': 涡量场 (H, W) → 由速度场计算
        - 'pressure': 压力场 (H, W)
    
    Attributes:
        enabled: 是否启用
        region: 区域配置 [x_start, y_start, x_end, y_end]
        
        # === 场类型权重 (>0 启用) ===
        velocity_weight: 速度场权重
        vorticity_weight: 涡量场权重  
        pressure_weight: 压力场权重
        
        # === RMS Loss ===
        rms_enabled: 是否启用 RMS loss
        
        # === FFT 模态 Loss ===
        fft_enabled: 是否启用 FFT loss
        fft_n_modes: 考虑的模态数量
        fft_mode_weights: 各模态权重 (None=等权重)
        fft_exclude_dc: 是否排除 DC 分量
        
        # === 波动稀疏 Loss ===
        fluctuation_enabled: 是否启用波动稀疏 loss
        fluctuation_top_k: 选取 top-k 波动点
        fluctuation_aggregation: 聚合方式 'mean'/'sum'/'max'
    """
    enabled: bool = False
    region: Optional[List[float]] = None
    
    # 场类型权重
    velocity_weight: float = 1.0
    vorticity_weight: float = 0.0
    pressure_weight: float = 0.0
    
    # RMS
    rms_enabled: bool = True
    
    # FFT
    fft_enabled: bool = False
    fft_n_modes: int = 5
    fft_mode_weights: Optional[List[float]] = None
    fft_exclude_dc: bool = True
    
    # 波动稀疏
    fluctuation_enabled: bool = False
    fluctuation_top_k: int = 100
    fluctuation_aggregation: str = 'mean'


# ============================================================
# 工具函数
# ============================================================
def extract_region(
    field: torch.Tensor,
    region: Optional[Union[List[float], RegionConfig]] = None,
) -> torch.Tensor:
    """提取流场指定区域
    
    Args:
        field: 流场 (H, W) 或 (C, H, W) 或 (batch, C, H, W)
        region: [x_start, y_start, x_end, y_end] 或 RegionConfig
    
    Returns:
        提取的区域张量
    """
    if region is None:
        return field
    
    if isinstance(region, RegionConfig):
        x_start, y_start = region.x_start, region.y_start
        x_end, y_end = region.x_end, region.y_end
    else:
        x_start, y_start, x_end, y_end = region
    
    # 获取空间维度
    if field.dim() == 2:
        H, W = field.shape
        x0, x1 = int(x_start * W), int(x_end * W)
        y0, y1 = int(y_start * H), int(y_end * H)
        return field[y0:y1, x0:x1]
    elif field.dim() == 3:
        C, H, W = field.shape
        x0, x1 = int(x_start * W), int(x_end * W)
        y0, y1 = int(y_start * H), int(y_end * H)
        return field[:, y0:y1, x0:x1]
    else:  # 4D: (batch, C, H, W)
        B, C, H, W = field.shape
        x0, x1 = int(x_start * W), int(x_end * W)
        y0, y1 = int(y_start * H), int(y_end * H)
        return field[:, :, y0:y1, x0:x1]


def compute_vorticity(velocity: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """计算涡量场 (可微分)
    
    ω = ∂v/∂x - ∂u/∂y
    
    Args:
        velocity: 速度场 (2, H, W) - [u, v] 分量
        dx, dy: 网格间距
    
    Returns:
        vorticity: 涡量场 (H, W)
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
        模值 (H, W)
    """
    if velocity.dim() == 2:
        return velocity.abs()
    return torch.sqrt((velocity ** 2).sum(dim=0) + 1e-8)


# ============================================================
# 基础 Loss 函数
# ============================================================
def field_rms_loss(
    field: torch.Tensor,
    region: Optional[Union[List[float], RegionConfig]] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """流场均方根 Loss
    
    RMS = sqrt(mean(field²))
    
    Args:
        field: 流场 (H, W) 或 (C, H, W)
        region: 区域配置
        mask: 区域掩码 (可选)
    
    Returns:
        RMS 损失
    """
    field = extract_region(field, region)
    
    if mask is not None:
        field = field * mask
        n_points = mask.sum() + 1e-8
    else:
        n_points = field.numel()
    
    rms = torch.sqrt(torch.sum(field ** 2) / n_points)
    return rms


def fft_mode_loss(
    field: torch.Tensor,
    n_modes: int = 5,
    mode_weights: Optional[List[float]] = None,
    exclude_dc: bool = True,
    region: Optional[Union[List[float], RegionConfig]] = None,
) -> torch.Tensor:
    """FFT 模态 Loss
    
    最小化非零模态(或指定模态)的幅值。
    
    Args:
        field: 流场 (H, W) 或 (batch, H, W)
        n_modes: 考虑的模态数量 (按幅值排序)
        mode_weights: 模态权重 (None=等权重)
        exclude_dc: 排除 DC 分量
        region: 区域配置
    
    Returns:
        FFT 模态损失
    """
    field = extract_region(field, region)
    
    if field.dim() == 2:
        field = field.unsqueeze(0)
    
    # 2D FFT (可微分)
    fft_result = torch.fft.fft2(field)
    magnitudes = torch.abs(fft_result)
    
    batch_size = magnitudes.shape[0]
    flat_mags = magnitudes.view(batch_size, -1)
    
    if exclude_dc:
        flat_mags = flat_mags.clone()
        flat_mags[:, 0] = 0
    
    # 取 top-k 模态
    k = min(n_modes, flat_mags.shape[1])
    top_values, _ = torch.topk(flat_mags, k=k, dim=1)
    
    # 权重
    if mode_weights is None:
        weights = torch.ones(k, device=field.device) / k
    else:
        weights = torch.tensor(mode_weights[:k], device=field.device)
        weights = weights / (weights.sum() + 1e-8)
    
    loss = (top_values * weights).sum(dim=1).mean()
    return loss


def fluctuation_sparse_loss(
    field: torch.Tensor,
    top_k: int = 100,
    aggregation: str = 'mean',
    region: Optional[Union[List[float], RegionConfig]] = None,
) -> torch.Tensor:
    """波动稀疏 Loss
    
    自动定位流场中波动最大的点，用它们作为 loss。
    适用于抑制局部高波动区域。
    
    Args:
        field: 流场 (H, W) 或 (batch, H, W)
        top_k: 选取波动最大的 k 个点
        aggregation: 聚合方式 'mean'/'sum'/'max'
        region: 区域配置
    
    Returns:
        稀疏波动损失
    """
    field = extract_region(field, region)
    
    if field.dim() == 2:
        field = field.unsqueeze(0)
    
    # 计算波动（偏离均值的程度）
    mean_val = field.mean()
    fluctuation = torch.abs(field - mean_val)
    
    flat = fluctuation.view(-1)
    k = min(top_k, flat.numel())
    top_values, _ = torch.topk(flat, k=k)
    
    if aggregation == 'mean':
        return top_values.mean()
    elif aggregation == 'sum':
        return top_values.sum()
    elif aggregation == 'max':
        return top_values.max()
    else:
        return top_values.mean()


# ============================================================
# 多场加权 Loss
# ============================================================
def multi_field_loss(
    velocity: Optional[torch.Tensor] = None,
    pressure: Optional[torch.Tensor] = None,
    vorticity: Optional[torch.Tensor] = None,
    region: Optional[Union[List[float], RegionConfig]] = None,
    velocity_weight: float = 1.0,
    vorticity_weight: float = 0.0,
    pressure_weight: float = 0.0,
    loss_type: str = 'rms',
    **kwargs,
) -> torch.Tensor:
    """多场类型加权 Loss
    
    对多种场类型分别计算 loss 后加权求和。
    
    Args:
        velocity: 速度场 (2, H, W)
        pressure: 压力场 (H, W)
        vorticity: 涡量场 (H, W)，如果为 None 且 vorticity_weight > 0 则从 velocity 计算
        region: 区域配置
        velocity_weight: 速度场权重
        vorticity_weight: 涡量场权重
        pressure_weight: 压力场权重
        loss_type: 'rms' / 'fft' / 'fluctuation'
        **kwargs: 传递给具体 loss 函数的参数
    
    Returns:
        加权总损失
    """
    total_loss = torch.tensor(0.0, device=velocity.device if velocity is not None 
                              else (pressure.device if pressure is not None else 'cpu'))
    total_weight = 0.0
    
    # 选择 loss 函数
    if loss_type == 'rms':
        loss_fn = lambda f: field_rms_loss(f, region=region)
    elif loss_type == 'fft':
        loss_fn = lambda f: fft_mode_loss(f, region=region, **kwargs)
    elif loss_type == 'fluctuation':
        loss_fn = lambda f: fluctuation_sparse_loss(f, region=region, **kwargs)
    else:
        loss_fn = lambda f: field_rms_loss(f, region=region)
    
    # 速度场
    if velocity is not None and velocity_weight > 0:
        vel_mag = velocity_magnitude(velocity)
        total_loss = total_loss + velocity_weight * loss_fn(vel_mag)
        total_weight += velocity_weight
    
    # 涡量场
    if vorticity_weight > 0:
        if vorticity is None and velocity is not None:
            vorticity = compute_vorticity(velocity)
        if vorticity is not None:
            total_loss = total_loss + vorticity_weight * loss_fn(vorticity)
            total_weight += vorticity_weight
    
    # 压力场
    if pressure is not None and pressure_weight > 0:
        total_loss = total_loss + pressure_weight * loss_fn(pressure)
        total_weight += pressure_weight
    
    # 归一化
    if total_weight > 0:
        total_loss = total_loss / total_weight
    
    return total_loss


# ============================================================
# 便捷函数
# ============================================================
def get_default_downstream_region(
    obstacle_x_ratio: float = 0.3,
    downstream_extent: float = 0.5,
) -> RegionConfig:
    """获取默认下游区域配置"""
    return RegionConfig(
        x_start=obstacle_x_ratio,
        y_start=0.0,
        x_end=min(obstacle_x_ratio + downstream_extent, 1.0),
        y_end=1.0,
    )


def region_velocity_loss(
    velocity_field: torch.Tensor,
    region: Optional[Union[List[float], RegionConfig]] = None,
    obstacle_x_ratio: float = 0.3,
    downstream_extent: float = 0.5,
) -> torch.Tensor:
    """区域速度 Loss (向后兼容)
    
    建议使用 multi_field_loss 代替。
    """
    if region is None:
        region = get_default_downstream_region(obstacle_x_ratio, downstream_extent)
    
    return multi_field_loss(
        velocity=velocity_field,
        region=region,
        velocity_weight=1.0,
        loss_type='rms',
    )


# 向后兼容别名
downstream_velocity_loss = region_velocity_loss


# ============================================================
# 向后兼容: 私有函数别名
# ============================================================
_extract_region = extract_region
