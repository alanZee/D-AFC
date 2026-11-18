"""
定常 Forcing 场生成器

提供空间上均匀或非均匀的定常外力场。
"""
from __future__ import annotations
from typing import List, Optional

from phi.torch.flow import StaggeredGrid, Box, vec


class ConstantForcing:
    """
    定常外力场生成器
    
    支持均匀力场。
    """
    
    def __init__(
        self,
        force: Optional[List[float]] = None,
        bounds: Box = None,
        x: int = 64,
        y: int = 64
    ):
        """
        Args:
            force: 均匀力 [Fx, Fy]
            bounds: 物理域边界
            x, y: 分辨率
        """
        self.force = force if force is not None else [0.0, 0.0]
        self.bounds = bounds or Box(x=1, y=1)
        self.resolution = {'x': x, 'y': y}
    
    def get_field(self, template: StaggeredGrid = None) -> StaggeredGrid:
        """
        获取力场
        
        Args:
            template: 模板速度场，用于匹配形状
        """
        force_vec = vec(x=self.force[0], y=self.force[1])
        
        if template is not None:
            # 直接在模板数据上加常数，保持形状完全一致
            return template * 0 + force_vec
        
        # 无模板时创建新场
        return StaggeredGrid(
            force_vec, boundary=0, bounds=self.bounds, **self.resolution
        )
