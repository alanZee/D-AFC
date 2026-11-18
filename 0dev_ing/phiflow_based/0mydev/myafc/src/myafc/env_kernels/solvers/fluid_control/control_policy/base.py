"""
控制策略基类

统一 API:
    - __call__(state) -> List[Tuple[torch.Tensor, torch.Tensor]]
    - reset() -> None
    - update_time(dt) -> None
    - set_num_nozzles(n) -> None
    - from_config(cfg) -> BasePolicy

注意：
1. 所有 Policy 输出的 amp 和 angle 都是 torch.Tensor，以支持可微分训练。
2. device 参数由 solver (runtime.device) 统一控制，配置文件中的 device 参数会被忽略。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

import torch


class BasePolicy(ABC):
    """
    控制策略基类
    
    所有策略需实现:
    - __call__(state): 返回每个喷嘴的 (amplitude, angle_deg) 列表
                       两者均为 torch.Tensor
    
    参数说明:
    - num_nozzles: 喷嘴数量，由 JetObstacle 通过 set_num_nozzles() 自动设置
    - device: 计算设备 'cpu' / 'cuda'，由 solver (runtime.device) 统一控制
    """
    
    def __init__(self, num_nozzles: int = 1, device: str = 'cuda'):
        self.num_nozzles = num_nozzles
        self.device = device
        self._time = 0.0
    
    @abstractmethod
    def __call__(self, state: Optional[Dict[str, Any]] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """计算控制信号，返回 [(amplitude, angle_deg), ...] 均为 torch.Tensor"""
        pass
    
    def reset(self):
        """重置策略状态"""
        self._time = 0.0
    
    def update_time(self, dt: float):
        """更新内部时间"""
        self._time += dt
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'BasePolicy':
        """
        从配置字典创建策略
        
        参数:
            num_nozzles: 喷嘴数量（默认1，由 JetObstacle 自动设置）
            device: 计算设备（由 solver 统一设置，配置文件中的值会被忽略）
        """
        return cls(
            num_nozzles=cfg.get('num_nozzles', 1),
            device=cfg.get('device', 'cpu'),
        )
