"""
Kolmogorov Forcing 场生成器

提供 Kolmogorov 流经典的正弦外力场。
参考: PhiFlow Higher_order_Kolmogorov.ipynb
"""
from __future__ import annotations
from typing import Optional
import math as pymath

from phi.torch.flow import StaggeredGrid, CenteredGrid, Box, vec, math


class KolmogorovForcing:
    """
    Kolmogorov 外力场生成器
    
    经典 Kolmogorov 流使用正弦外力:
        F_x = A * sin(k * y)
        F_y = 0
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        wavenumber: int = 4,
        noise_scale: float = 0.0,
        bounds: Box = None,
        x: int = 64,
        y: int = 64
    ):
        """
        Args:
            amplitude: 力的振幅 A
            wavenumber: 波数 k（正弦周期数）
            noise_scale: 噪声扰动幅度（暂不实现）
            bounds: 物理域边界
            x, y: 分辨率
        """
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.noise_scale = noise_scale
        self.bounds = bounds or Box(x=1, y=1)
        self.resolution = {'x': x, 'y': y}
    
    def get_field(self, template: StaggeredGrid = None) -> StaggeredGrid:
        """
        获取力场
        
        Args:
            template: 模板速度场，用于匹配形状
        """
        L_y = self.bounds.size['y']
        k = 2 * pymath.pi * self.wavenumber / L_y
        A = self.amplitude
        
        # 先使用 CenteredGrid 创建，然后采样到 StaggeredGrid
        forcing_centered = CenteredGrid(
            lambda x, y: vec(x=A * math.sin(k * y), y=0),
            boundary=0,
            bounds=self.bounds,
            **self.resolution
        )
        
        if template is not None:
            # 采样到模板的位置
            return forcing_centered @ template
        
        # 无模板时返回 CenteredGrid（需要调用者自行采样）
        return forcing_centered
