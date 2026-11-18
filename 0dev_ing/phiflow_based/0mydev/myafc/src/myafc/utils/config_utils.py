"""
config_utils - 配置解析与处理工具

为 JetTrainer、AFCValidator 和 DRLTrainer 提供统一的配置处理功能：
1. YAML 加载与科学计数法转换
2. 配置合并 (base + override)
3. 路径解析

使用示例:
    from myafc.utils import load_yaml_config, merge_configs
    
    # 加载配置 (自动转换科学计数法字符串)
    cfg = load_yaml_config('config.yaml')
    
    # 合并配置
    merged = merge_configs(base_cfg, override_cfg)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Union, Optional
import copy
import yaml

from myafc.utils import get_logger

LOGGER = get_logger()


def convert_scientific_notation(cfg_dict: Dict) -> Dict:
    """递归转换配置中的科学计数法字符串为 float
    
    将形如 '1.0e-3' 的字符串转换为 float 类型。
    
    Args:
        cfg_dict: 配置字典
    
    Returns:
        转换后的配置字典 (原地修改)
    
    Example:
        cfg = {'lr': '1.0e-3', 'nested': {'eps': '1e-8'}}
        convert_scientific_notation(cfg)
        # → {'lr': 0.001, 'nested': {'eps': 1e-08}}
    """
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            convert_scientific_notation(value)
        elif isinstance(value, str):
            # 尝试解析科学计数法
            try:
                if 'e' in value.lower() and (value.replace('e', '').replace('E', '').replace('-', '').replace('+', '').replace('.', '').isdigit()):
                    cfg_dict[key] = float(value)
            except (ValueError, AttributeError):
                pass
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    convert_scientific_notation(item)
                elif isinstance(item, str):
                    try:
                        if 'e' in item.lower() and (item.replace('e', '').replace('E', '').replace('-', '').replace('+', '').replace('.', '').isdigit()):
                            value[i] = float(item)
                    except (ValueError, AttributeError):
                        pass
    return cfg_dict


def load_yaml_config(
    path: Union[str, Path],
    convert_scientific: bool = True,
) -> Dict:
    """加载 YAML 配置文件
    
    Args:
        path: 配置文件路径
        convert_scientific: 是否转换科学计数法字符串
    
    Returns:
        配置字典
    
    Raises:
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML 解析错误
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    if cfg is None:
        cfg = {}
    
    if convert_scientific:
        convert_scientific_notation(cfg)
    
    return cfg


def merge_configs(
    base: Dict,
    override: Dict,
    deep: bool = True,
) -> Dict:
    """合并两个配置字典
    
    override 中的值会覆盖 base 中的同名值。
    如果 deep=True，则递归合并嵌套字典。
    
    Args:
        base: 基础配置
        override: 覆盖配置
        deep: 是否深度合并嵌套字典
    
    Returns:
        合并后的新配置字典
    
    Example:
        base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        override = {'b': {'c': 10}}
        merge_configs(base, override)
        # → {'a': 1, 'b': {'c': 10, 'd': 3}}
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and deep and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = merge_configs(result[key], value, deep=True)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def resolve_config_path(
    path: Optional[str],
    base_dir: Union[str, Path],
    key_name: str = 'path',
) -> Optional[Path]:
    """解析配置中的路径引用
    
    将相对路径转换为绝对路径。
    
    Args:
        path: 路径字符串 (可为 None)
        base_dir: 基础目录
        key_name: 键名 (用于日志)
    
    Returns:
        解析后的 Path 对象，如果 path 为 None 则返回 None
    """
    if path is None:
        return None
    
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return (Path(base_dir) / path).resolve()


def get_nested(cfg: Dict, key_path: str, default: Any = None) -> Any:
    """获取嵌套配置值
    
    使用点分隔的路径获取嵌套配置。
    
    Args:
        cfg: 配置字典
        key_path: 点分隔的键路径，如 'training.learning_rate'
        default: 默认值
    
    Returns:
        配置值，不存在时返回 default
    
    Example:
        cfg = {'training': {'lr': 0.01}}
        get_nested(cfg, 'training.lr')  # → 0.01
        get_nested(cfg, 'training.epochs', 100)  # → 100
    """
    keys = key_path.split('.')
    value = cfg
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_nested(cfg: Dict, key_path: str, value: Any) -> Dict:
    """设置嵌套配置值
    
    使用点分隔的路径设置嵌套配置。
    如果中间路径不存在，会自动创建空字典。
    
    Args:
        cfg: 配置字典 (会被修改)
        key_path: 点分隔的键路径
        value: 要设置的值
    
    Returns:
        修改后的配置字典
    
    Example:
        cfg = {}
        set_nested(cfg, 'training.lr', 0.01)
        # → {'training': {'lr': 0.01}}
    """
    keys = key_path.split('.')
    current = cfg
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return cfg


def flatten_config(cfg: Dict, separator: str = '.', prefix: str = '') -> Dict[str, Any]:
    """将嵌套配置展平为单层字典
    
    Args:
        cfg: 嵌套配置字典
        separator: 键分隔符
        prefix: 键前缀
    
    Returns:
        展平后的字典
    
    Example:
        cfg = {'a': {'b': 1, 'c': 2}, 'd': 3}
        flatten_config(cfg)
        # → {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    result = {}
    
    for key, value in cfg.items():
        full_key = f'{prefix}{separator}{key}' if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_config(value, separator, full_key))
        else:
            result[full_key] = value
    
    return result
