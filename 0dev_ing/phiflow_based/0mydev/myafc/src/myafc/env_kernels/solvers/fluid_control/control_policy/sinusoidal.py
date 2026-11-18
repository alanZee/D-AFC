"""
正弦输出策略

输出随时间变化的正弦波控制信号（均为 torch.Tensor）。

角度约定（物理绝对坐标系）:
    - 0°   = x轴正方向 (→)
    - 90°  = y轴正方向 (↑)
    - 180° = x轴负方向 (←)
    - 270° = y轴负方向 (↓)
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional

import torch

from .base import BasePolicy


class SinusoidalPolicy(BasePolicy):
    """
    正弦控制策略
    
    输出幅值随时间正弦变化的控制信号（均为 torch.Tensor）:
        amplitude(t) = A * sin(2π * freq * t + phase) + offset
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase_deg: float = 0.0,
        angle_deg: float = 0.0,
        offset: float = 0.0,
        num_nozzles: int = 1,
        use_state_time: bool = True,
        angle_deg_fixed: Optional[float] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            amplitude: 振幅 A
            frequency: 频率 f (Hz)
            phase_deg: 初始相位（度）
            angle_deg: 速度方向（度，物理坐标），0°=x+, 90°=y+
            offset: 幅值偏移量
            num_nozzles: 喷嘴数量（由 JetObstacle 自动设置）
            use_state_time: 是否使用 state['time']（否则用内部时间）
            angle_deg_fixed: 固定角度（度），设置后覆盖 angle_deg
            device: 计算设备 'cpu' / 'cuda'
        """
        super().__init__(num_nozzles, device)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_rad = math.radians(phase_deg)
        self.angle_deg = angle_deg
        self.offset = offset
        self.use_state_time = use_state_time
        self.angle_deg_fixed = angle_deg_fixed
    
    def __call__(self, state: Optional[Dict[str, Any]] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """计算当前时刻的正弦控制信号（均为 torch.Tensor）"""
        t = state.get('time', self._time) if state and self.use_state_time else self._time
        amp_val = self.amplitude * math.sin(2 * math.pi * self.frequency * t + self.phase_rad) + self.offset
        angle_val = self.angle_deg_fixed if self.angle_deg_fixed is not None else self.angle_deg
        
        amp = torch.tensor(amp_val, dtype=torch.float32, device=self.device)
        angle = torch.tensor(angle_val, dtype=torch.float32, device=self.device)
        return [(amp, angle)] * self.num_nozzles
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'SinusoidalPolicy':
        return cls(
            amplitude=cfg.get('amplitude', 1.0),
            frequency=cfg.get('frequency', 1.0),
            phase_deg=cfg.get('phase_deg', 0.0),
            angle_deg=cfg.get('angle_deg', 0.0),
            offset=cfg.get('offset', 0.0),
            num_nozzles=cfg.get('num_nozzles', 1),
            use_state_time=cfg.get('use_state_time', True),
            angle_deg_fixed=cfg.get('angle_deg_fixed'),
            device=cfg.get('device', 'cpu'),
        )
