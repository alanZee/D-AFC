"""
runner_factory.py - SolverRunner 工厂与配置工具

提供统一的 SolverRunner 创建接口，供以下模块使用:
- JetTrainer (训练/验证阶段)
- AFCValidator (JetTrainer/DRL 模型验证)
- DRLTrainer (训练可视化)

主要功能:
1. create_runner: 创建 SolverRunner 实例
2. build_runner_config: 从验证配置构建 Runner 配置
3. ensure_target_probe: 确保探针配置包含目标探针

日志级别优先级规则:
- 若上层显式传入 verbose，则使用该值（例如 trainer 配置覆盖 solver 配置）
- 若未显式传入，则继承 base_runner_config.verbose（通常来自 env/solver 配置）
- 若两者都缺失，则回退为 INFO
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
import yaml

from myafc.utils import get_logger

if TYPE_CHECKING:
    from myafc.env_kernels.solver_wrappers import SolverRunner
    from myafc.env_kernels.solvers import KarmanTorchSolver

LOGGER = get_logger()


# ============================================================
# Runner 配置构建
# ============================================================
def build_validation_runner_config(
    output_dir: Union[str, Path],
    visuals_config: Dict[str, Any],
    base_runner_config: Optional[Dict[str, Any]] = None,
    target_probe_position: Optional[List[float]] = None,
    verbose: Optional[str] = None,
) -> Dict[str, Any]:
    """构建验证用 Runner 配置
    
    Args:
        output_dir: 输出目录
        visuals_config: 可视化配置，应包含:
            - enabled: bool
            - interval: int
            - quantities: List[str]
            - save_endpoints: bool
            - save_fields: bool
            - smooth: bool (可选)
            - video_fps: int (可选)
            - obstacle_style: str (可选)
        base_runner_config: 基础 runner 配置（如 env_config 中的 runner 节）
        target_probe_position: 目标探针相对位置 [x, y] (0~1)
        verbose: 日志级别。为 None 时沿用 base_runner_config.verbose，若仍缺失则使用 INFO
    
    Returns:
        Runner 配置字典
    """
    base_cfg = base_runner_config.copy() if base_runner_config else {}
    vis = visuals_config
    resolved_verbose = verbose if verbose is not None else base_cfg.get('verbose', 'INFO')
    
    runner_cfg = {
        'output': {
            'enabled': True,
            'root': str(output_dir),
            'subdir': '',
            'skip_log_setup': True,
            'skip_config_snapshot': True,
        },
        'visuals': {
            'enabled': vis.get('enabled', True),
            'interval': vis.get('interval', 50),
            'quantities': vis.get('quantities', ['velocity_magnitude', 'pressure']),
            'save_endpoints': vis.get('save_endpoints', True),
            'folder': 'frames',
            'smooth': vis.get('smooth', True),
            'video_fps': vis.get('video_fps', 10),
            'obstacle_style': vis.get('obstacle_style', 'both'),
        },
        'fields': {
            'save': {
                'enabled': vis.get('save_fields', False),
                'interval': vis.get('interval', 50),
                'filename': 'fields.h5',
                'quantities': ['velocity', 'pressure'],
            },
            'cache': {'enabled': False},
        },
        'force_coeffs': {
            'enabled': True,
            'interval': 1,
            'filename': 'force_coeffs.csv',
        },
        'probes': base_cfg.get('probes', {'enabled': False}),
        'verbose': resolved_verbose,
    }
    
    # 确保目标探针配置
    if target_probe_position:
        ensure_target_probe(runner_cfg, target_probe_position)
    
    return runner_cfg


def build_training_runner_config(
    output_dir: Union[str, Path],
    visuals_config: Dict[str, Any],
    base_runner_config: Optional[Dict[str, Any]] = None,
    verbose: Optional[str] = None,
) -> Dict[str, Any]:
    """构建训练可视化用 Runner 配置
    
    与验证配置的区别:
    - 不记录力系数和探针（由 Trainer 自己记录）
    - 仅用于可视化和流场保存
    
    Args:
        output_dir: 输出目录
        visuals_config: 可视化配置
        base_runner_config: 基础 runner 配置（可选，用于继承 solver/env 的 verbose）
        verbose: 日志级别。为 None 时沿用 base_runner_config.verbose，若仍缺失则使用 INFO
    
    Returns:
        Runner 配置字典
    """
    vis = visuals_config
    base_cfg = base_runner_config.copy() if base_runner_config else {}
    resolved_verbose = verbose if verbose is not None else base_cfg.get('verbose', 'INFO')
    
    runner_cfg = {
        'output': {
            'enabled': True,
            'root': str(output_dir),
            'subdir': '',
            'skip_log_setup': True,
            'skip_config_snapshot': True,
        },
        'visuals': {
            'enabled': True,
            'interval': vis.get('interval', 50),
            'quantities': vis.get('quantities', ['velocity_magnitude', 'pressure']),
            'save_endpoints': vis.get('save_endpoints', True),
            'folder': 'frames',
            'smooth': vis.get('smooth', True),
            'video_fps': vis.get('video_fps', 10),
            'obstacle_style': vis.get('obstacle_style', 'both'),
        },
        'fields': {
            'save': {
                'enabled': True,
                'interval': vis.get('interval', 50),
                'filename': 'fields.h5',
                'quantities': ['velocity', 'pressure'],
            },
            'cache': {'enabled': False},
        },
        # 训练时不通过 runner 记录力系数和探针
        'force_coeffs': {'enabled': False},
        'probes': {'enabled': False},
        'verbose': resolved_verbose,
    }
    
    return runner_cfg


def ensure_target_probe(
    runner_cfg: Dict[str, Any],
    target_position: List[float],
    probe_name: str = 'target',
) -> None:
    """确保 Runner 配置中包含目标探针
    
    若 probes.locations 中不存在名为 probe_name 的探针，则添加。
    
    Args:
        runner_cfg: Runner 配置（会被原地修改）
        target_position: 目标位置 [x_fraction, y_fraction]
        probe_name: 探针名称
    """
    probes_cfg = runner_cfg.get('probes', {})
    
    if not probes_cfg.get('enabled', False):
        return
    
    locations = probes_cfg.get('locations', [])
    
    # 检查是否已存在
    has_target = any(loc.get('name') == probe_name for loc in locations)
    if has_target:
        return
    
    # 添加目标探针
    locations.append({
        'name': probe_name,
        'fraction': [float(target_position[0]), float(target_position[1])]
    })
    probes_cfg['locations'] = locations
    runner_cfg['probes'] = probes_cfg


# ============================================================
# Runner 创建
# ============================================================
def create_runner(
    solver: 'KarmanTorchSolver',
    runner_cfg: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    name: str = 'runner',
) -> 'SolverRunner':
    """创建 SolverRunner 实例
    
    统一的 Runner 创建入口。
    
    Args:
        solver: KarmanTorchSolver 实例
        runner_cfg: Runner 配置字典
        output_dir: 输出目录（可选，会自动创建）
        name: Runner 名称（用于日志）
    
    Returns:
        SolverRunner 实例
    """
    from myafc.env_kernels.solver_wrappers import SolverRunner
    
    # 创建输出目录
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建 Runner 实例
    runner = SolverRunner(solver, {'runner': runner_cfg})
    
    LOGGER.debug(f"[RunnerFactory] {name} runner 已创建")
    return runner


# ============================================================
# 配置加载工具
# ============================================================
def load_env_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """加载环境配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"环境配置不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_target_probe_position_from_config(
    config: Dict[str, Any],
    default: List[float] = None,
) -> List[float]:
    """从配置中获取目标探针位置
    
    支持两种配置格式:
    1. JetTrainer: loss.target_probe_position
    2. DRL: env.probes.target_position
    
    Args:
        config: 配置字典
        default: 默认值
    
    Returns:
        [x_fraction, y_fraction]
    """
    if default is None:
        default = [0.5, 0.5]
    
    # JetTrainer 格式
    loss_cfg = config.get('loss', {})
    if 'target_probe_position' in loss_cfg:
        return loss_cfg['target_probe_position']
    
    # DRL 格式
    env_cfg = config.get('env', {})
    probes_cfg = env_cfg.get('probes', {})
    if 'target_position' in probes_cfg:
        return probes_cfg['target_position']
    
    return default
