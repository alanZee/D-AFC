"""
自定义 Loss 加载接口

支持用户从配置文件指定自定义 loss 函数脚本路径和函数名。

使用方法:
    在配置文件中:
    ```yaml
    loss:
      custom_loss_path: "path/to/custom_loss.py"
      custom_loss_func: "my_custom_loss"
    ```
    
    自定义函数签名:
    ```python
    def my_custom_loss(
        state: Dict[str, Any],
        amplitude: torch.Tensor,
        target: Dict[str, Any],
        **kwargs
    ) -> torch.Tensor:
        ...
    ```
"""
from __future__ import annotations
from typing import Callable, Dict, Any, Optional
from pathlib import Path
import importlib.util
import torch


def load_custom_loss(
    script_path: str,
    func_name: str,
) -> Callable:
    """动态加载自定义 loss 函数
    
    Args:
        script_path: Python 脚本路径
        func_name: 函数名
    
    Returns:
        loss_func: 加载的 loss 函数
        
    Raises:
        FileNotFoundError: 脚本不存在
        AttributeError: 函数不存在
    """
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"自定义 loss 脚本不存在: {script_path}")
    
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("custom_loss_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 获取函数
    if not hasattr(module, func_name):
        raise AttributeError(f"自定义 loss 脚本中不存在函数: {func_name}")
    
    return getattr(module, func_name)


# ============================================================
# 自定义 Loss 示例
# ============================================================

def example_custom_loss(
    state: Dict[str, Any],
    amplitude: torch.Tensor,
    target: Optional[Dict[str, Any]] = None,
    velocity_weight: float = 1.0,
    pressure_weight: float = 0.1,
) -> torch.Tensor:
    """自定义 Loss 示例
    
    目标: 最小化探针采样的速度绝对值和压力绝对值的加权和。
    
    Args:
        state: 环境状态字典，包含 'probes' 等
        amplitude: 当前控制幅值
        target: 目标值字典 (可选)
        velocity_weight: 速度权重
        pressure_weight: 压力权重
    
    Returns:
        loss: 自定义 loss 值
        
    Example:
        state = {
            'probes': {
                'target': {'u': 0.5, 'v': 0.1, 'p': 0.01}
            }
        }
        loss = example_custom_loss(state, amplitude)
    """
    loss = torch.tensor(0.0, dtype=torch.float32)
    
    if 'probes' not in state:
        return loss
    
    probes = state['probes']
    
    for probe_name, probe_data in probes.items():
        # 速度绝对值
        if 'u' in probe_data:
            u = probe_data['u']
            u_tensor = torch.tensor(u) if not isinstance(u, torch.Tensor) else u
            loss = loss + velocity_weight * torch.abs(u_tensor)
        
        if 'v' in probe_data:
            v = probe_data['v']
            v_tensor = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            loss = loss + velocity_weight * torch.abs(v_tensor)
        
        # 压力绝对值
        if 'p' in probe_data:
            p = probe_data['p']
            p_tensor = torch.tensor(p) if not isinstance(p, torch.Tensor) else p
            loss = loss + pressure_weight * torch.abs(p_tensor)
    
    return loss
