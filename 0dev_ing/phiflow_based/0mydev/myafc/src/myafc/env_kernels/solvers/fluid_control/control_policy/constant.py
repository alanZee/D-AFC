"""
定常输出策略

输出固定的速度大小和角度（均为 torch.Tensor）。
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

import torch

from .base import BasePolicy


class ConstantPolicy(BasePolicy):
    """
    定常控制策略
    
    输出固定的速度幅值和方向（均为 torch.Tensor），所有喷嘴使用相同控制信号。
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        angle_deg: float = 0.0,
        num_nozzles: int = 1,
        device: str = 'cuda',
    ):
        """
        Args:
            amplitude: 速度大小（正=吹出，负=吸入）
            angle_deg: 速度方向（度），0°=x+, 90°=y+
            num_nozzles: 喷嘴数量（由 JetObstacle 自动设置）
            device: 计算设备 'cpu' / 'cuda'
        """
        super().__init__(num_nozzles, device)
        self.amplitude = amplitude
        self.angle_deg = angle_deg
    
    def __call__(self, state: Optional[Dict[str, Any]] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """返回所有喷嘴的控制信号（相同，均为 torch.Tensor）"""
        amp = torch.tensor(self.amplitude, dtype=torch.float32, device=self.device)
        angle = torch.tensor(self.angle_deg, dtype=torch.float32, device=self.device)
        return [(amp, angle)] * self.num_nozzles
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'ConstantPolicy':
        return cls(
            amplitude=cfg.get('amplitude', 1.0),
            angle_deg=cfg.get('angle_deg', 0.0),
            num_nozzles=cfg.get('num_nozzles', 1),
            device=cfg.get('device', 'cpu'),
        )
