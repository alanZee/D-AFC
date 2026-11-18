"""
path_utils - 路径和目录管理工具

为 JetTrainer、AFCValidator 和 DRLTrainer 提供统一的路径管理功能：
1. 时间戳目录生成
2. 配置快照保存
3. 相对路径解析

使用示例:
    from myafc.utils import create_timestamped_dir, save_config_snapshot
    
    # 创建带时间戳的输出目录
    output_dir = create_timestamped_dir('./runs', prefix='exp')
    # → runs/exp_YYYYMMDD_HHMMSS/
    
    # 保存配置快照
    save_config_snapshot(config_dict, output_dir / 'configs')
"""
from __future__ import annotations
from pathlib import Path
from typing import Union, Dict, Optional
from datetime import datetime
import shutil
import yaml

from myafc.utils import get_logger

LOGGER = get_logger()


def create_timestamped_dir(
    base_dir: Union[str, Path],
    prefix: str = 'exp',
    timestamp_format: str = '%Y%m%d_%H%M%S',
    create: bool = True,
) -> Path:
    """创建带时间戳的目录
    
    Args:
        base_dir: 基础目录路径
        prefix: 目录名前缀
        timestamp_format: 时间戳格式 (默认 YYYYMMDD_HHMMSS)
        create: 是否立即创建目录
    
    Returns:
        带时间戳的目录路径
    
    Example:
        create_timestamped_dir('./runs', 'train')
        → Path('runs/train_YYYYMMDD_HHMMSS')
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime(timestamp_format)
    output_dir = base_dir / f'{prefix}_{timestamp}'
    
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.debug(f"[path_utils] 创建目录: {output_dir}")
    
    return output_dir


def save_config_snapshot(
    config: Union[Dict, str, Path],
    output_dir: Union[str, Path],
    filename: str = 'config.yaml',
    copy_original: bool = False,
) -> Path:
    """保存配置快照到指定目录
    
    Args:
        config: 配置来源
            - Dict: 配置字典，保存为 YAML
            - str/Path: 原始配置文件路径，复制或重新保存
        output_dir: 输出目录
        filename: 输出文件名
        copy_original: 如果 config 是路径，是否直接复制文件 (True) 还是重新解析保存 (False)
    
    Returns:
        保存的配置文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    if isinstance(config, dict):
        # 保存字典为 YAML
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    elif isinstance(config, (str, Path)):
        config_path = Path(config)
        if copy_original and config_path.exists():
            # 直接复制原文件
            shutil.copy2(config_path, output_path)
        else:
            # 加载后重新保存 (确保格式一致)
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    else:
        raise TypeError(f"不支持的 config 类型: {type(config)}")
    
    LOGGER.debug(f"[path_utils] 配置保存至: {output_path}")
    return output_path


def resolve_relative_path(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    must_exist: bool = False,
) -> Path:
    """解析相对路径为绝对路径
    
    如果 path 是相对路径，则相对于 base_dir 解析；
    如果已是绝对路径，则直接返回。
    
    Args:
        path: 待解析的路径
        base_dir: 基础目录 (用于解析相对路径)
        must_exist: 是否要求路径存在
    
    Returns:
        解析后的绝对路径
    
    Raises:
        FileNotFoundError: 如果 must_exist=True 且路径不存在
    """
    path = Path(path)
    
    if path.is_absolute():
        resolved = path
    else:
        resolved = (Path(base_dir) / path).resolve()
    
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"路径不存在: {resolved}")
    
    return resolved


def ensure_dir(path: Union[str, Path], parents: bool = True) -> Path:
    """确保目录存在，不存在则创建
    
    Args:
        path: 目录路径
        parents: 是否创建父目录
    
    Returns:
        目录路径
    """
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=True)
    return path


def get_output_subdirs(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """获取标准输出子目录结构
    
    创建标准化的输出目录结构，用于训练器/验证器输出。
    
    Args:
        output_dir: 根输出目录
    
    Returns:
        子目录路径字典
    
    Example:
        subdirs = get_output_subdirs('./runs/exp_001')
        # subdirs = {
        #     'configs': Path('runs/exp_001/configs'),
        #     'checkpoints': Path('runs/exp_001/checkpoints'),
        #     'curves': Path('runs/exp_001/curves'),
        #     'frames': Path('runs/exp_001/frames'),
        #     'fields': Path('runs/exp_001/fields'),
        # }
    """
    output_dir = Path(output_dir)
    
    subdirs = {
        'configs': output_dir / 'configs',
        'checkpoints': output_dir / 'checkpoints',
        'curves': output_dir / 'curves',
        'frames': output_dir / 'frames',
        'fields': output_dir / 'fields',
    }
    
    # 创建所有子目录
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs
