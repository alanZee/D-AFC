"""
AFCValidator - AFC 通用验证引擎

支持加载并验证以下两种训练器的模型:
1. JetTrainer: .pt 检查点文件 (含 AFCAgent 网络权重)
2. DRLTrainer: .zip 模型文件 (Stable-Baselines3 格式)

验证流程 (对齐 JetTrainer._run_single_validation):
1. Warmup 阶段: warmup_steps 步 (填充历史 buffer, 不计入统计)
2. 正式验证: validation_steps 步 (记录 Cd/Cl/Amplitude 等)

关键对齐点 (与 JetTrainer 一致):
- control_interval: 控制信号更新间隔
- action_duration: 控制信号持续步数
- warmup_apply_jets: warmup 阶段是否施加射流
- state_history: envstep_state 多步历史状态
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import os
from pathlib import Path
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from datetime import datetime
import yaml
import numpy as np
import torch

from myafc.utils import get_logger, get_target_probe_position_from_config

LOGGER = get_logger()


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class ValidationVisualsConfig:
    """验证可视化配置 (对齐 SolverRunner.visuals)"""
    enabled: bool = True              # 是否启用可视化
    interval: int = 50                # 帧保存间隔 (env-steps)
    quantities: List[str] = dataclass_field(
        default_factory=lambda: [
            'velocity_magnitude', 'velocity_x', 'velocity_y', 
            'pressure', 'vorticity', 'streamlines'
        ]
    )                                 # 与 JetTrainer 默认一致
    save_endpoints: bool = True       # 保存初始/最终帧
    save_fields: bool = False         # 保存原始流场 (raw/fields.npz)
    generate_video: bool = True       # 生成 mp4 视频
    video_fps: int = 10               # 视频帧率
    obstacle_style: str = 'both'      # 障碍物风格: none/outline/fill/both
    smooth: bool = True               # 平滑渲染


@dataclass
class ValidatorConfig:
    """AFCValidator 配置
    
    控制周期参数:
        control_interval: 控制信号更新间隔 (env-steps)
        action_duration: 控制信号持续步数 (env-steps)
            - apply_jets = (step % control_interval < action_duration)
        warmup_apply_jets: warmup 阶段是否施加射流边界条件
    """
    trainer_config_path: str = ''     # Trainer 配置文件路径
    model_path: str = ''              # 模型路径 (.pt/.zip)
    model_type: Literal['auto', 'jet', 'drl'] = 'auto'  # 模型类型
    warmup_steps: int = 100           # 预热步数
    validation_steps: int = 500       # 验证步数
    deterministic: bool = True        # 确定性策略
    use_ema: bool = False             # 使用 EMA 权重 (JetTrainer)
    output_dir: Optional[str] = None  # 输出目录
    device: str = 'auto'              # 计算设备
    
    # 配置覆盖 (JetTrainer / DRL 两种模式通用)
    env_config_path: Optional[str] = None  # 环境配置路径 (覆盖 trainer_config 中的 env_config_path)
    input_features: Optional[Dict] = None  # 输入特征配置 (覆盖 trainer_config 中的 input_features)
    
    # 控制周期
    control_interval: int = 26        # 控制更新间隔
    action_duration: int = 26         # 控制持续步数
    warmup_apply_jets: bool = True    # warmup 是否施加射流
    warmup_update_controls: bool = True  # warmup 是否更新控制
    
    visuals: ValidationVisualsConfig = dataclass_field(default_factory=ValidationVisualsConfig)


def _parse_validator_config(cfg_dict: Dict) -> ValidatorConfig:
    """从字典解析 ValidatorConfig"""
    cfg = ValidatorConfig()
    keys = ['trainer_config_path', 'model_path', 'model_type', 'warmup_steps',
            'validation_steps', 'deterministic', 'use_ema', 'output_dir', 
            'device', 'control_interval', 'action_duration', 
            'warmup_apply_jets', 'warmup_update_controls',
            'env_config_path', 'input_features']
    for key in keys:
        if key in cfg_dict:
            setattr(cfg, key, cfg_dict[key])
    
    if 'visuals' in cfg_dict:
        vis = cfg_dict['visuals']
        default_qtys = [
            'velocity_magnitude', 'velocity_x', 'velocity_y',
            'pressure', 'vorticity', 'streamlines'
        ]
        cfg.visuals = ValidationVisualsConfig(
            enabled=vis.get('enabled', True),
            interval=vis.get('interval', 50),
            quantities=vis.get('quantities', default_qtys),
            save_endpoints=vis.get('save_endpoints', True),
            save_fields=vis.get('save_fields', False),
            generate_video=vis.get('generate_video', True),
            video_fps=vis.get('video_fps', 10),
            obstacle_style=vis.get('obstacle_style', 'both'),
            smooth=vis.get('smooth', True),
        )
    return cfg


# ============================================================
# Trainer 配置解析器
# ============================================================
class TrainerConfigParser:
    """解析 JetTrainer/DRLTrainer 配置文件"""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Trainer 配置不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.raw_dict = yaml.safe_load(f)
        
        # 检测 Trainer 类型
        self.trainer_type = 'jet' if 'agent' in self.raw_dict else 'drl'
    
    def get_env_config_path(self) -> Path:
        """获取环境配置绝对路径
        
        优先级:
        1. 同目录下的 solver_config.yaml ( JetTrainer 训练时保存的完整配置，推荐)
        2. trainer_config 配置中的 env_config_path (相对于 trainer_config 目录解析)
        
        不存在时直接报错，不进行复杂的回退搜索。
        """
        # 优先级 1: 同目录下的 solver_config.yaml（JetTrainer 训练时保存）
        solver_config_path = self.config_path.parent / 'solver_config.yaml'
        if solver_config_path.exists():
            LOGGER.info(f"[TrainerConfigParser] 使用训练保存的 solver_config: {solver_config_path}")
            return solver_config_path
        
        # 优先级 2: 解析 env_config_path
        env_path_str = self.raw_dict.get('env_config_path', '')
        if not env_path_str:
            raise FileNotFoundError(
                f"[TrainerConfigParser] 配置错误:\n"
                f"  - trainer_config: {self.config_path}\n"
                f"  - 同目录下无 solver_config.yaml\n"
                f"  - 配置中未指定 env_config_path\n"
                f"请确保训练时正确保存了 solver_config.yaml，或在 trainer_config 中指定 env_config_path。"
            )
        
        env_path = Path(env_path_str)
        
        # 绝对路径
        if env_path.is_absolute():
            if env_path.exists():
                return env_path
            raise FileNotFoundError(
                f"[TrainerConfigParser] env_config_path 不存在:\n"
                f"  路径: {env_path}\n"
                f"提示: 此路径可能是训练机器的绝对路径，跨机器时请确保 solver_config.yaml 存在。"
            )
        
        # 相对路径 (相对于 trainer_config 目录)
        resolved = (self.config_path.parent / env_path).resolve()
        if resolved.exists():
            return resolved
        
        raise FileNotFoundError(
            f"[TrainerConfigParser] env_config_path 解析失败:\n"
            f"  原始值: {env_path_str}\n"
            f"  解析为: {resolved}\n"
            f"  trainer_config 目录: {self.config_path.parent}\n"
            f"提示: 请检查路径是否正确，或确保同目录下有 solver_config.yaml。"
        )
    
    def get_agent_config(self) -> Optional[Dict]:
        return self.raw_dict.get('agent')
    
    def get_input_features_config(self) -> Optional[Dict]:
        """获取输入特征配置 (统一接口，支持 JetTrainer 和 DRL 两种格式)
        
        配置来源优先级:
        1. input_features (JetTrainer 格式)
        2. env.input_features (DRL 格式)
        """
        # JetTrainer 格式
        if 'input_features' in self.raw_dict:
            return self.raw_dict['input_features']
        # DRL 格式
        env_cfg = self.raw_dict.get('env', {})
        if 'input_features' in env_cfg:
            return env_cfg['input_features']
        return None
    
    def get_training_config(self) -> Dict:
        return self.raw_dict.get('training', {})
    
    def get_loss_config(self) -> Optional[Dict]:
        return self.raw_dict.get('loss')
    
    def get_algorithm_config(self) -> Optional[Dict]:
        return self.raw_dict.get('algorithm')
    
    def get_env_config(self) -> Dict:
        """获取环境配置 (DRL 模式: env 节点)"""
        return self.raw_dict.get('env', {})
    
    def get_target_probe_position(self) -> list:
        """获取目标探针位置 (委托给 utils.get_target_probe_position_from_config)
        
        支持 JetTrainer 和 DRL 两种配置格式:
        1. loss.target_probe_position (JetTrainer 格式)
        2. env.probes.target_position (DRL 格式)
        3. 默认值 [0.5, 0.5]
        
        Returns:
            [x_fraction, y_fraction] 相对坐标 (0~1)
        """
        return get_target_probe_position_from_config(self.raw_dict)


# ============================================================
# AFCValidator 主类
# ============================================================
class AFCValidator:
    """AFC 通用验证引擎
    
    验证流程 (对齐 JetTrainer._run_single_validation):
    1. Warmup: warmup_steps 步 (填充 history, apply_jets 由配置控制)
    2. Validation: validation_steps 步 (记录 Cd/Cl/Amplitude/Loss)
    
    控制逻辑 (对齐 JetTrainer._env_step):
    - should_update = (step % control_interval == 0)
    - should_apply_jets = (step % control_interval < action_duration)
    """
    
    def __init__(
        self,
        config: Optional[Union[str, Path, Dict, ValidatorConfig]] = None,
        trainer_config: Optional[Union[str, Path]] = None,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """初始化
        
        Args:
            config: Validator 配置 (YAML路径/字典/ValidatorConfig)
            trainer_config: Trainer 配置文件路径
            model_path: 模型路径 (.pt/.zip)
            **kwargs: 覆盖配置项
        """
        self.cfg = self._parse_config(config, trainer_config, model_path, kwargs)
        
        # 设备
        if self.cfg.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.cfg.device)
        
        # 解析 Trainer 配置
        self.trainer_parser = TrainerConfigParser(self.cfg.trainer_config_path)
        
        # 从 trainer 配置同步控制参数
        self._sync_control_params_from_trainer()
        
        # 检测模型类型
        if self.cfg.model_type == 'auto':
            suffix = Path(self.cfg.model_path).suffix.lower()
            self.model_type = 'jet' if suffix == '.pt' else 'drl'
        else:
            self.model_type = self.cfg.model_type
        
        # 设置输出目录
        self._setup_output_dir()
        
        # 延迟初始化
        self.solver = None
        self.agent = None
        self.target_point = None
        
        # 状态历史 (对齐 JetTrainer._state_history_buffer)
        self._state_history_buffer: List[Dict] = []
        
        # 力系数缓存 (对齐 JetTrainer._cached_force_coeffs)
        self._cached_force_coeffs = None
        
        # 当前幅值
        self._current_amplitude = 0.0
        
        # 数据缓存
        self._val_losses: List[float] = []
        self._val_cd: List[float] = []
        self._val_cl: List[float] = []
        self._val_amplitudes: List[float] = []
        self._val_velocities: List[Tuple[float, float]] = []
        self._val_pressures: List[float] = []
        
        LOGGER.info(f"[Validator] 初始化 | Model: {self.model_type.upper()} | Device: {self.device}")
    
    def _sync_control_params_from_trainer(self):
        """从 Trainer 配置同步控制参数"""
        training_cfg = self.trainer_parser.get_training_config()
        
        # 同步控制周期参数
        if 'control_interval' in training_cfg or 'control_interval_raw' in training_cfg:
            raw = training_cfg.get('control_interval_raw', training_cfg.get('control_interval', 26))
            action_dur = training_cfg.get('action_duration', 26)
            if isinstance(raw, str) and raw.lower() in ('action_duration', 'equal2action_duration'):
                self.cfg.control_interval = action_dur
            else:
                self.cfg.control_interval = int(raw)
        
        if 'action_duration' in training_cfg:
            self.cfg.action_duration = training_cfg['action_duration']
        
        if 'warmup_apply_jets' in training_cfg:
            self.cfg.warmup_apply_jets = training_cfg['warmup_apply_jets']
        
        if 'warmup_update_controls' in training_cfg:
            self.cfg.warmup_update_controls = training_cfg['warmup_update_controls']
        
        LOGGER.debug(f"[Validator] 控制参数: interval={self.cfg.control_interval}, "
                     f"duration={self.cfg.action_duration}, warmup_jets={self.cfg.warmup_apply_jets}")
    
    def _get_num_probes(self) -> int:
        """获取探针数量 (从 env_config 读取)"""
        env_config_path = self.trainer_parser.get_env_config_path()
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_cfg = yaml.safe_load(f)
        locations = env_cfg.get('runner', {}).get('probes', {}).get('locations', [])
        return len(locations)
    
    def _parse_config(self, config, trainer_config, model_path, kwargs) -> ValidatorConfig:
        """解析配置
        
        trainer_config_path 优先级:
        1. 配置中显式指定的 trainer_config_path
        2. 若为 null/空，根据 model_path 推断: model_path/../configs/trainer_config.yaml
        """
        config_base_dir = None  # 用于解析相对路径
        
        if config is not None:
            if isinstance(config, ValidatorConfig):
                cfg = config
            elif isinstance(config, (str, Path)):
                config_path = Path(config)
                config_base_dir = config_path.parent  # 相对路径的基准目录
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = _parse_validator_config(yaml.safe_load(f))
            else:
                cfg = _parse_validator_config(config)
        else:
            if trainer_config is None or model_path is None:
                raise ValueError("需提供 config 或 (trainer_config + model_path)")
            cfg = ValidatorConfig(
                trainer_config_path=str(trainer_config),
                model_path=str(model_path),
            )
        
        # 解析相对路径为绝对路径
        if config_base_dir is not None:
            if cfg.trainer_config_path and not Path(cfg.trainer_config_path).is_absolute():
                cfg.trainer_config_path = str((config_base_dir / cfg.trainer_config_path).resolve())
            if cfg.model_path and not Path(cfg.model_path).is_absolute():
                cfg.model_path = str((config_base_dir / cfg.model_path).resolve())
        
        # 处理 trainer_config_path: 若为空则根据 model_path 推断
        cfg.trainer_config_path = self._resolve_trainer_config_path(
            cfg.trainer_config_path, cfg.model_path
        )
        
        # 应用覆盖参数
        for key in ['warmup_steps', 'validation_steps', 'output_dir', 'device', 
                    'deterministic', 'use_ema', 'control_interval', 'action_duration',
                    'warmup_apply_jets', 'warmup_update_controls']:
            if key in kwargs and kwargs[key] is not None:
                setattr(cfg, key, kwargs[key])
        
        return cfg
    
    def _resolve_trainer_config_path(self, trainer_config_path: str, model_path: str) -> str:
        """解析 trainer_config_path
        
        优先级:
        1. 显式指定的 trainer_config_path (非空且存在)
        2. 若为空/None，根据 model_path 自动推断:
           - JetTrainer (.pt): checkpoints/ → configs/trainer_config.yaml
           - DRLTrainer (.zip): 同目录或父目录的 config.yaml
        """
        # 优先级 1: 显式指定
        if trainer_config_path:
            path = Path(trainer_config_path)
            if path.exists():
                return str(path)
            raise FileNotFoundError(
                f"[Validator] trainer_config_path 不存在:\n"
                f"  路径: {path}\n"
                f"提示: 请检查路径是否正确，或设为 null 以自动从 model_path 推断。"
            )
        
        # 优先级 2: 根据 model_path 推断
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"[Validator] model_path 不存在: {model_path}")
        
        suffix = model_path.suffix.lower()
        
        if suffix == '.pt':
            # JetTrainer: exp/checkpoints/best.pt → exp/configs/trainer_config.yaml
            exp_dir = model_path.parent.parent
            inferred_path = exp_dir / 'configs' / 'trainer_config.yaml'
        else:
            # DRLTrainer (.zip): 
            # 情况1: exp/final_model.zip → exp/config.yaml
            # 情况2: exp/checkpoints/xxx.zip → exp/config.yaml
            if model_path.parent.name == 'checkpoints':
                exp_dir = model_path.parent.parent
            else:
                exp_dir = model_path.parent
            inferred_path = exp_dir / 'config.yaml'
        
        if inferred_path.exists():
            LOGGER.info(f"[Validator] trainer_config_path 未指定，自动推断: {inferred_path}")
            return str(inferred_path)
        
        raise FileNotFoundError(
            f"[Validator] 无法推断 trainer_config_path:\n"
            f"  model_path: {model_path}\n"
            f"  尝试路径: {inferred_path}\n"
            f"提示: 请显式指定 trainer_config_path。"
        )
    
    def _setup_output_dir(self):
        """设置输出目录
        
        自动推断逻辑:
        - JetTrainer: exp/checkpoints/best.pt → exp/validation_xxx
        - DRLTrainer: exp/final_model.zip → exp/validation_xxx
                      exp/checkpoints/model.zip → exp/validation_xxx
        """
        if self.cfg.output_dir:
            self.output_dir = Path(self.cfg.output_dir)
        else:
            model_path = Path(self.cfg.model_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 判断模型目录结构
            # 如果 parent 是 checkpoints，则 exp_dir = parent.parent
            # 否则 exp_dir = parent (DRLTrainer 的 final_model.zip 直接在 exp 下)
            if model_path.parent.name == 'checkpoints':
                exp_dir = model_path.parent.parent
            else:
                exp_dir = model_path.parent
            
            self.output_dir = exp_dir / f'validation_{timestamp}'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / 'frames'
        self.raw_dir = self.output_dir / 'raw'
        LOGGER.info(f"[Validator] 输出目录: {self.output_dir}")
    
    def _setup_solver(self):
        """初始化求解器
        
        环境配置路径优先级:
        1. Validator 配置中显式指定的 env_config_path (适用于 JetTrainer 和 DRL 两种模式)
        2. trainer_parser.get_env_config_path() (从 trainer_config 解析)
        """
        from myafc.env_kernels.solvers import KarmanTorchSolver
        
        # 优先使用 Validator 配置中的 env_config_path
        if self.cfg.env_config_path:
            env_config_path = Path(self.cfg.env_config_path)
            if not env_config_path.is_absolute():
                # 相对于 trainer_config 目录解析
                env_config_path = (Path(self.cfg.trainer_config_path).parent / env_config_path).resolve()
            if not env_config_path.exists():
                raise FileNotFoundError(f"[Validator] env_config_path 不存在: {env_config_path}")
            LOGGER.info(f"[Validator] 使用 Validator 配置中的 env_config_path: {env_config_path}")
        else:
            # 使用 trainer_parser 解析
            env_config_path = self.trainer_parser.get_env_config_path()
        
        self.solver = KarmanTorchSolver(str(env_config_path))
        
        # 设置确定性模式
        if self.cfg.deterministic and self.solver.jet_obstacle:
            self.solver.jet_obstacle.deterministic = True
        
        LOGGER.info(f"[Validator] 求解器初始化 | 环境: {env_config_path.name}")
        
        # 设置探针点
        self._setup_probe_points()
        
        # 创建 SolverRunner (用于流场保存与可视化)
        self._setup_runner()
    
    def _setup_runner(self):
        """创建 SolverRunner 实例 (使用 runner_factory 工具)
        
        Runner 负责:
        - 流场 h5 保存 (raw/fields.h5)
        - 力系数记录 (raw/force_coeffs.csv)
        - 探针记录 (raw/probes.csv)
        - 可视化帧保存 (frames/)
        - 视频生成
        """
        from myafc.utils import (
            build_validation_runner_config,
            create_runner,
            load_env_config,
        )
        
        # 加载基础 runner 配置
        # 优先使用 Validator 配置中的 env_config_path
        if self.cfg.env_config_path:
            env_config_path = Path(self.cfg.env_config_path)
            if not env_config_path.is_absolute():
                env_config_path = (Path(self.cfg.trainer_config_path).parent / env_config_path).resolve()
        else:
            env_config_path = self.trainer_parser.get_env_config_path()
        env_cfg = load_env_config(env_config_path)
        base_runner_cfg = env_cfg.get('runner', {})
        training_cfg = self.trainer_parser.get_training_config()
        trainer_verbose = training_cfg.get('verbose', None)
        
        # 构建 visuals 配置字典
        vis_cfg = self.cfg.visuals
        visuals_dict = {
            'enabled': vis_cfg.enabled,
            'interval': vis_cfg.interval,
            'quantities': vis_cfg.quantities,
            'save_endpoints': vis_cfg.save_endpoints,
            'save_fields': vis_cfg.save_fields,
            'smooth': vis_cfg.smooth,
            'video_fps': vis_cfg.video_fps,
            'obstacle_style': vis_cfg.obstacle_style,
        }
        
        # 使用工厂方法构建配置
        runner_cfg = build_validation_runner_config(
            output_dir=self.output_dir,
            visuals_config=visuals_dict,
            base_runner_config=base_runner_cfg,
            target_probe_position=self.trainer_parser.get_target_probe_position(),
            # 全局日志级别优先级：trainer.training.verbose 覆盖 solver/env runner.verbose
            verbose=trainer_verbose,
        )
        
        # 创建 runner
        self.runner = create_runner(
            solver=self.solver,
            runner_cfg=runner_cfg,
            output_dir=self.output_dir,
            name='Validation',
        )
        LOGGER.debug(f"[Validator] Runner 已创建 | 输出目录: {self.output_dir}")
    
    def _setup_probe_points(self):
        """加载 env_config 中的所有探针位置 (对齐 JetTrainer._setup_probe_points)"""
        from phi.torch.flow import Point, vec
        from typing import Dict
        
        self._probe_points: Dict[str, Point] = {}
        
        input_cfg = self.trainer_parser.get_input_features_config() or {}
        main_input_type = input_cfg.get('main_input_type', 'probes')
        
        # 如果 main_input_type 为 'target_probe'，则无需加载额外探针
        if main_input_type == 'target_probe':
            LOGGER.debug(f"[Validator] main_input_type='target_probe', 跳过额外探针加载")
            return
        
        env_config_path = self.trainer_parser.get_env_config_path()
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_cfg = yaml.safe_load(f)
        
        runner_cfg = env_cfg.get('runner', {})
        probes_cfg = runner_cfg.get('probes', {})
        locations = probes_cfg.get('locations', [])
        
        if not locations:
            LOGGER.debug("[Validator] env_config 中未定义探针位置")
            return
        
        Lx, Ly = self.solver.cfg.domain.size
        
        for loc in locations:
            name = loc.get('name', f'probe_{len(self._probe_points)}')
            
            # 解析坐标 (支持 fraction 和 units 两种格式)
            if 'fraction' in loc:
                frac = loc['fraction']
                x, y = frac[0] * Lx, frac[1] * Ly
            elif 'units' in loc:
                x, y = loc['units']
            else:
                LOGGER.warning(f"[Validator] 探针 '{name}' 缺少坐标配置, 跳过")
                continue
            
            self._probe_points[name] = Point(vec(x=x, y=y))
        
        LOGGER.info(f"[Validator] 已加载 {len(self._probe_points)} 个探针点")

    def _setup_agent(self):
        """初始化 Agent (JetTrainer 模式)
        
        input_features 配置优先级:
        1. Validator 配置中显式指定的 input_features (用于测试不同特征组合)
        2. trainer_parser.get_input_features_config() (从 trainer_config 解析)
        """
        from myafc.afc_supervision.agents import AFCAgent
        from myafc.afc_supervision.agents.afc_agent import AFCAgentConfig
        
        agent_cfg = self.trainer_parser.get_agent_config() or {}
        
        # 优先使用 Validator 配置中的 input_features
        if self.cfg.input_features:
            input_cfg = self.cfg.input_features
            LOGGER.info("[Validator] 使用 Validator 配置中的 input_features")
        else:
            input_cfg = self.trainer_parser.get_input_features_config() or agent_cfg
        
        n_jets = len(self.solver.cfg.jets.nozzles) if self.solver.cfg.jets.enabled else 1
        
        # 构建 AFCAgentConfig 
        cfg = AFCAgentConfig(
            hidden_dims=agent_cfg.get('hidden_dims', [512, 512]),
            amplitude_max=agent_cfg.get('amplitude_max', 4.0),
            angle_deg_fixed=agent_cfg.get('angle_deg_fixed', 90.0),
            shared_output=agent_cfg.get('shared_output', True),
            normalize_input=agent_cfg.get('normalize_input', True),
            action_scale=agent_cfg.get('action_scale', 'softsign'),
            relative_action=agent_cfg.get('relative_action', False),
            deterministic=self.cfg.deterministic,
            main_input_type=input_cfg.get('main_input_type', 'probes'),
            envstep_state=input_cfg.get('envstep_state', 1),
            include_target_velocity=input_cfg.get('include_target_velocity', False),
            include_prev_amplitude_num=input_cfg.get('include_prev_amplitude_num', 1),
            include_force_coeffs=input_cfg.get('include_force_coeffs', False),
            probe_fields=input_cfg.get('probe_fields', ['u', 'v', 'p']),
        )
        
        self.agent = AFCAgent(config=cfg, num_nozzles=n_jets, device=str(self.device))
        
        # 加载权重 (需要先初始化网络)
        # 创建 dummy state 触发 lazy init
        self._init_agent_network_from_checkpoint()
        
        self.agent.eval()
        LOGGER.info(f"[Validator] Agent 加载完成 | hidden_dims={cfg.hidden_dims}")
    
    def _init_agent_network_from_checkpoint(self):
        """初始化 Agent 网络并加载权重"""
        ckpt = torch.load(self.cfg.model_path, map_location=self.device, weights_only=False)
        
        if self.cfg.use_ema and ckpt.get('ema'):
            state_dict = ckpt['ema'].get('ema_model_state_dict', ckpt.get('model'))
            LOGGER.info("[Validator] 使用 EMA 权重")
        else:
            state_dict = ckpt.get('model', ckpt.get('network_state_dict'))
        
        if state_dict is None:
            raise ValueError(f"检查点中未找到模型权重: {self.cfg.model_path}")
        
        # 需要先初始化网络 (lazy init)
        # 创建 dummy state 触发网络初始化
        dummy_state = self._create_dummy_state()
        _ = self.agent.compute_action(dummy_state, deterministic=True)
        
        # 加载权重 (使用 strict=False 以兼容转换后的模型，可能缺少归一化参数)
        missing_keys, unexpected_keys = self.agent.network.load_state_dict(state_dict, strict=False)
        if missing_keys:
            LOGGER.warning(f"[Validator] 模型缺少以下参数 (将使用默认初始化): {missing_keys}")
        if unexpected_keys:
            LOGGER.warning(f"[Validator] 模型包含未使用的参数: {unexpected_keys}")
        LOGGER.debug(f"[Validator] 模型权重加载成功")
    
    def _create_dummy_state(self) -> Dict:
        """创建用于初始化网络的 dummy state
        
        与 _build_state() 对齐，探针字段严格按 probe_fields 构建:
        1. target 探针 — probe_fields 中的字段
        2. 所有 env_config 探针 — probe_fields 中的字段
        3. force_coeffs (Cd, Cl) — 如果启用
        """
        state = {
            'target_velocity': [0.0, 0.0],
            'prev_amplitudes': [0.0],
            'probes': {},
            'force_coeffs': {'Cd': 0.0, 'Cl': 0.0},
        }
        
        # 获取 probe_fields 配置
        input_cfg = self.trainer_parser.get_input_features_config() or {}
        probe_fields = input_cfg.get('probe_fields', ['u', 'v', 'p'])
        
        # 创建 dummy 探针值字典
        def make_probe_dict():
            return {f: 0.0 for f in probe_fields}
        
        # 添加 target 探针 (JetTrainer 会首先添加 target)
        state['probes']['target'] = make_probe_dict()
        
        # 如果使用探针，需要添加探针数据
        env_config_path = self.trainer_parser.get_env_config_path()
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_cfg = yaml.safe_load(f)
        
        locations = env_cfg.get('runner', {}).get('probes', {}).get('locations', [])
        
        for loc in locations:
            name = loc.get('name', f'probe_{len(state["probes"])}')
            state['probes'][name] = make_probe_dict()
        
        return state
    
    def _compute_input_dim(self, input_cfg: Dict) -> int:
        """计算输入维度 (对齐 AFCAgent._build_input)
        
        两个独立维度:
        - 探针选择: main_input_type='target_probe'(仅 target) / 'probes'(全部)
        - 字段选择: probe_fields=['u','v','p'] 的任意子集
        """
        solver_cfg = self.solver.cfg
        
        # 探针: 从 env_config 的 runner.probes.locations 获取
        n_probes = self._get_num_probes()
        probe_fields = input_cfg.get('probe_fields', ['u', 'v', 'p'])
        
        # main_input_type='probes' 包含 target 探针 + 所有其他探针
        # target 探针按 probe_fields 采样 (与 _build_state 对齐)
        n_total_probes = n_probes + 1  # +1 for target
        n_probe_features = n_total_probes * len(probe_fields)
        
        # 其他特征
        n_jets = len(solver_cfg.jets.nozzles) if solver_cfg.jets.enabled else 1
        include_target = input_cfg.get('include_target_velocity', False)  # 默认 False
        prev_amp_num = input_cfg.get('include_prev_amplitude_num', 1)
        include_force = input_cfg.get('include_force_coeffs', False)
        envstep_state = input_cfg.get('envstep_state', 1)
        
        # 单步特征维度
        dim_single = n_probe_features
        if include_force:
            dim_single += 2  # Cd, Cl
        
        # 总维度 = 单步维度 * envstep_state + 历史幅值
        total_dim = dim_single * envstep_state
        
        # prev_amplitude 不参与 envstep_state 堆叠
        total_dim += n_jets * prev_amp_num
        
        # target_velocity 也不参与堆叠
        if include_target:
            total_dim += 2
        
        return total_dim
    
    def run(self) -> Dict[str, Any]:
        """执行验证"""
        if self.model_type == 'jet':
            return self._run_jet_validation()
        else:
            return self._run_drl_validation()
    
    # ============================================================
    # JetTrainer 验证 (使用 Runner)
    # ============================================================
    
    def _run_jet_validation(self) -> Dict[str, Any]:
        """JetTrainer 模式验证 (使用 Runner)
        
        使用 SolverRunner 处理:
        - 流场 h5 保存 (raw/fields.h5)
        - 力系数记录 (raw/force_coeffs.csv)
        - 探针记录 (raw/probes.csv)
        - 可视化帧保存 (frames/)
        - 视频生成
        """
        from tqdm import tqdm
        from phi.torch.flow import Point, vec
        
        # 初始化
        self._setup_solver()
        self._setup_agent()
        
        cfg = self.cfg
        warmup_steps = cfg.warmup_steps
        val_steps = cfg.validation_steps
        total_steps = warmup_steps + val_steps
        vis_cfg = cfg.visuals
        
        LOGGER.info(f"[Validator] 开始验证 | warmup: {warmup_steps} | 验证: {val_steps}")
        LOGGER.info(f"[Validator] 控制: interval={cfg.control_interval}, duration={cfg.action_duration}")
        
        # 获取 runner (已在 _setup_solver 中创建)
        runner = self.runner
        runner.reset_caches(reset_solver=False)  # 保留 solver 状态
        
        # 重置求解器 (对齐 JetTrainer._reset_env)
        if self.solver.cfg.initial.resume_from:
            self.solver.reset(use_h5=True)
        else:
            self.solver.reset(use_h5=False)
            LOGGER.warning("[Validator] 无 resume_from 配置，使用 inflow 初始化流场")
        
        self._state_history_buffer.clear()
        self._cached_force_coeffs = None
        self._current_amplitude = 0.0
        self.agent.reset()
        
        # 设置目标探针点 (对齐 JetTrainer)
        target_pos = self.trainer_parser.get_target_probe_position()
        loss_cfg = self.trainer_parser.get_loss_config() or {}
        target_u, target_v = loss_cfg.get('target_velocity', [0.0, 0.0])
        Lx, Ly = self.solver.cfg.domain.size
        self.target_point = Point(vec(x=target_pos[0] * Lx, y=target_pos[1] * Ly))
        
        # 保存初始帧 (使用 runner)
        if runner.cfg.visuals.enabled and runner.cfg.visuals.save_endpoints:
            runner._save_visual(0)
        
        # ===== Warmup Phase (对齐 JetTrainer) =====
        LOGGER.info(f"[Validator] Warmup 阶段...")
        for step in range(warmup_steps):
            is_update, apply_jets, state = self._env_step(step, is_warmup=True)
            self._do_solver_step(apply_jets, state)
            self.agent.update_history(state)
            
            # 使用 runner 记录 warmup 数据 (保持时序完整)
            step_num = step + 1
            if runner.cfg.force_coeffs.enabled:
                runner._record_force_coeffs()
            if runner.cfg.probes.enabled:
                runner._record_probes()
        
        # warmup 后清空缓存 (对齐 JetTrainer._detach_graph)
        self._cached_force_coeffs = None
        
        # ===== Validation Phase (对齐 JetTrainer) =====
        steps_in_accumulation = 0
        pbar = tqdm(range(val_steps), desc='Validation')
        
        with torch.no_grad():
            for local_step in pbar:
                env_step = warmup_steps + local_step
                
                # 1. _env_step (对齐 JetTrainer)
                is_update, apply_jets, state = self._env_step(
                    steps_in_accumulation, is_warmup=False
                )
                
                # 2. 更新控制信号
                if is_update:
                    amplitude, _ = self.agent.compute_action(state)
                    self._current_amplitude = amplitude
                    if self.solver.jet_obstacle:
                        angle = self.trainer_parser.get_agent_config().get('angle_deg_fixed', 90.0)
                        self.solver.jet_obstacle.set_controls_external(
                            amplitudes=amplitude,
                            angles=angle,
                        )
                
                # 3. _do_solver_step (对齐 JetTrainer)
                self._do_solver_step(apply_jets, state)
                steps_in_accumulation += 1
                self.agent.update_history(state)
                
                # 4. 采样目标速度 (对齐 _sample_target_velocity)
                u, v = self._sample_target_velocity()
                self._val_velocities.append((float(u), float(v)))
                self._val_pressures.append(self._sample_target_pressure())
                
                # 5. 计算 loss (对齐 JetTrainer._compute_primary_loss)
                # 根据训练配置选择 loss 类型
                loss = self._compute_primary_loss(u, v, target_u, target_v)
                self._val_losses.append(float(loss))
                
                # 6. 记录幅值
                amp = self._get_current_amplitude(apply_jets)
                self._val_amplitudes.append(amp)
                
                # 7. 记录力系数 (对齐 _get_force_coeffs)
                cd, cl = self._get_force_coeffs()
                self._val_cd.append(cd)
                self._val_cl.append(cl)
                
                pbar.set_postfix_str(f"Cd={cd:.4f} |Cl|={abs(cl):.4f}")
                
                # 8. 使用 runner 记录数据
                step_num = env_step + 1
                if runner.cfg.force_coeffs.enabled:
                    runner._record_force_coeffs()
                if runner.cfg.probes.enabled:
                    runner._record_probes()
                if runner.cfg.visuals.enabled and step_num % runner.cfg.visuals.interval == 0:
                    runner._save_visual(step_num)
                if runner.cfg.fields.save.enabled and step_num % runner.cfg.fields.save.interval == 0:
                    runner._save_fields_h5()
        
        # 保存最终帧 (使用 runner)
        if runner.cfg.visuals.enabled and runner.cfg.visuals.save_endpoints:
            runner._save_visual(total_steps, suffix='_final')
        
        # 使用 runner 保存 CSV、图表、视频
        if runner.cfg.output.enabled:
            runner._save_csvs()
            if runner.force_coeffs:
                runner._plot_force_coeffs()
            runner._generate_videos()
        
        # 关闭 h5 文件
        if runner._h5_file is not None:
            runner._h5_file.close()
            runner._h5_file = None
        
        # 保存 validation_curves (独立于 runner)
        results = self._save_results()
        
        return results
    
    # ============================================================
    # 核心方法 (对齐 JetTrainer)
    # ============================================================
    
    def _env_step(
        self, 
        steps_in_accumulation: int,
        is_warmup: bool = False,
    ) -> Tuple[bool, bool, Dict]:
        """环境步准备 (对齐 JetTrainer._env_step)
        
        Returns:
            (should_update_control, should_apply_jets, state)
        """
        cfg = self.cfg
        
        # 控制逻辑 (对齐 JetTrainer)
        if is_warmup:
            should_update = cfg.warmup_update_controls
            should_apply_jets = cfg.warmup_apply_jets
        else:
            should_update = (steps_in_accumulation % cfg.control_interval == 0)
            should_apply_jets = (steps_in_accumulation % cfg.control_interval < cfg.action_duration)
        
        # 构建状态 (对齐 JetTrainer._build_state)
        state = self._build_state()
        
        # 注入历史 (对齐 envstep_state)
        input_cfg = self.trainer_parser.get_input_features_config() or {}
        if input_cfg.get('envstep_state', 1) > 1:
            state['_state_history'] = list(self._state_history_buffer)
        
        return should_update, should_apply_jets, state
    
    def _do_solver_step(self, apply_jets: bool, state: Dict = None):
        """执行求解步 (对齐 JetTrainer._do_solver_step)"""
        import copy
        self.solver.step(update_controls=False, apply_jets=apply_jets)
        
        # 清空力系数缓存
        self._cached_force_coeffs = None
        
        # 更新历史状态缓冲区
        input_cfg = self.trainer_parser.get_input_features_config() or {}
        envstep_state = input_cfg.get('envstep_state', 1)
        if envstep_state > 1 and state is not None:
            state_copy = {k: v for k, v in state.items() if not k.startswith('_')}
            self._state_history_buffer.append(copy.deepcopy(state_copy))
            max_history_len = envstep_state - 1
            if len(self._state_history_buffer) > max_history_len:
                self._state_history_buffer = self._state_history_buffer[-max_history_len:]
    
    def _build_state(self) -> Dict:
        """构建 Agent 输入状态 (对齐 JetTrainer._build_state)
        
        根据 input_features 配置构建状态字典，仅采样 probe_fields 中请求的字段:
        - main_input_type: 'target_probe'(仅目标探针) / 'probes'(全部探针)
        - probe_fields: ['u','v','p'] 的任意子集 — 与探针选择独立组合
        - include_force_coeffs: 是否包含力系数
        
        使用共享的 sample_probe_fields() 确保与所有 Trainer 采样一致。
        """
        from myafc.utils import sample_probe_fields, build_probe_batch, sample_probe_fields_batch
        
        input_cfg = self.trainer_parser.get_input_features_config() or {}
        probe_fields = input_cfg.get('probe_fields', ['u', 'v', 'p'])
        
        state = {
            'time': self.solver.current_time,
            'step': self.solver.current_step,
        }
        
        # target 探针 (始终采样, 验证阶段约定无噪声)
        state['probes'] = {'target': sample_probe_fields(
            self.solver.velocity, self.solver.pressure, self.target_point,
            probe_fields,
        )}
        
        # env 探针 (main_input_type='probes' 时)
        main_input_type = input_cfg.get('main_input_type', 'probes')
        if main_input_type == 'probes' and self._probe_points:
            probe_signature = tuple(self._probe_points.keys())
            if getattr(self, '_probe_batch_signature', None) != probe_signature:
                names, batch_geom, _coords = build_probe_batch(self._probe_points)
                self._probe_batch_names = names
                self._probe_batch_geom = batch_geom
                self._probe_batch_signature = probe_signature

            batch_data = sample_probe_fields_batch(
                self.solver.velocity,
                self.solver.pressure,
                self._probe_batch_names,
                self._probe_batch_geom,
                probe_fields,
            )
            state['probes'].update(batch_data)
        
        # 力系数
        if input_cfg.get('include_force_coeffs', False):
            cd, cl = self._get_force_coeffs()
            state['force_coeffs'] = {'Cd': cd, 'Cl': cl}
        
        return state
    
    def _compute_primary_loss(
        self, u: float, v: float, target_u: float, target_v: float
    ) -> float:
        """计算 primary loss (对齐 JetTrainer._compute_primary_loss)
        
        根据训练配置动态选择 loss 类型:
        - probe_mse_weight > 0: 使用 probe MSE loss
        - force_drl_weight > 0: 使用 DRL reward-based loss
        - 否则: 回退到简单的探针 MSE
        
        Returns:
            loss (float)
        """
        from myafc.afc_supervision.losses import (
            probe_mse_loss, drl_reward_loss
        )
        
        loss_cfg = self.trainer_parser.get_loss_config() or {}
        loss = 0.0
        
        # 探针 MSE loss
        probe_mse_w = loss_cfg.get('probe_mse_weight', 0.0)
        if probe_mse_w > 0:
            current = torch.tensor([u, v], device=self.device)
            target = torch.tensor([target_u, target_v], device=self.device)
            loss += probe_mse_w * float(probe_mse_loss(current, target).item())
        
        # DRL reward-based loss
        force_drl_w = loss_cfg.get('force_drl_weight', 0.0)
        if force_drl_w > 0:
            cd, cl = self._get_force_coeffs()
            cd_t = torch.tensor(cd, device=self.device)
            cl_t = torch.tensor(cl, device=self.device)
            drl_loss = drl_reward_loss(
                cd_t, cl_t,
                cd_baseline=loss_cfg.get('cd_baseline', 0.0),
                alpha_lift=loss_cfg.get('alpha_lift', 0.1),
            )
            loss += force_drl_w * float(drl_loss.item())
        
        # 如果没有配置任何 loss，回退到简单 MSE
        if probe_mse_w == 0 and force_drl_w == 0:
            loss = (u - target_u)**2 + (v - target_v)**2
        
        return loss
    
    def _sample_target_velocity(self) -> Tuple[float, float]:
        """采样目标速度 (对齐 JetTrainer._sample_target_velocity)"""
        from phi.torch.flow import field
        vel_at_target = field.sample(self.solver.velocity, self.target_point)
        u = float(vel_at_target.vector['x'].native().squeeze().cpu())
        v = float(vel_at_target.vector['y'].native().squeeze().cpu())
        return u, v
    
    def _sample_target_pressure(self) -> float:
        """采样目标探针位置的压力 (对齐 JetTrainer._sample_target_pressure)"""
        from phi.torch.flow import field
        p_at_target = field.sample(self.solver.pressure, self.target_point)
        p = float(p_at_target.native().squeeze().cpu())
        return p
    
    def _get_force_coeffs(self) -> Tuple[float, float]:
        """获取力系数 (对齐 JetTrainer._get_force_coeffs 带缓存)"""
        if self._cached_force_coeffs is None:
            if hasattr(self.solver, 'compute_force_coeffs'):
                result = self.solver.compute_force_coeffs()
                if isinstance(result, tuple) and len(result) >= 2:
                    cd, cl = result[0], result[1]
                    cd = float(cd.cpu()) if hasattr(cd, 'cpu') else float(cd)
                    cl = float(cl.cpu()) if hasattr(cl, 'cpu') else float(cl)
                    self._cached_force_coeffs = (cd, cl)
                    return cd, cl
            self._cached_force_coeffs = (0.0, 0.0)
        return self._cached_force_coeffs
    
    def _get_current_amplitude(self, apply_jets: bool) -> float:
        """获取当前控制幅值 - 使用实际施加值 (对齐 JetTrainer._get_current_amplitude)
        
        Args:
            apply_jets: 当前步是否施加射流。若为 False，返回 0（因为实际未施加控制）
        
        Returns:
            当前实际施加的控制幅值（经过裁剪、缩放、时间平滑处理后）
        """
        if not apply_jets:
            # 射流未施加时，实际作用于流场的幅值为 0
            return 0.0
        
        # 优先使用实际施加值（经过 apply_controls 处理后的值）
        if self.solver.jet_obstacle and self.solver.jet_obstacle.current_controls:
            return float(self.solver.jet_obstacle.current_controls[0][0].detach().cpu())
        
        # fallback: 使用网络原始输出
        amp = self._current_amplitude
        if hasattr(amp, 'item'):
            return float(amp.item())
        elif hasattr(amp, 'cpu'):
            return float(amp.cpu())
        return float(amp)
    
    # ============================================================
    # DRLTrainer 验证 (使用 Runner)
    # ============================================================
    
    def _run_drl_validation(self) -> Dict[str, Any]:
        """DRLTrainer 模式验证 (CFD step 级别数据记录)
        
        步数语义 (与 JetTrainer 对齐):
        =============================
        - warmup_steps: CFD steps (env-steps)
        - validation_steps: CFD steps (env-steps)
        - 配置中的步数直接就是 CFD 步数，无需乘以 control_interval
        
        DRL 特有概念:
        - control_interval: 1 DRL step = control_interval 个 CFD steps
        - 每 control_interval 个 CFD step 获取一次新动作
        (DRLTrainer 中 action_duration 始终等于 control_interval)

        
        数据记录粒度:
        - validation_curves: CFD step 粒度 (每步记录 Cd/Cl/amplitude)
        - reward: 每 control_interval 步记录一次 (DRL step 粒度)
        
        输出结构:
        - raw/fields.h5: 流场数据 (CFD step 粒度)
        - raw/force_coeffs.csv: 力系数 (CFD step 粒度)
        - validation_curves.*: 曲线数据 (CFD step 粒度, reward 除外)
        - frames/: 可视化图片 (CFD step 粒度)
        - *.mp4: 视频
        
        输出结构:
        - raw/fields.h5: 流场数据 (CFD step 粒度)
        - raw/force_coeffs.csv: 力系数 (CFD step 粒度)
        - validation_curves.*: 曲线数据 (CFD step 粒度)
        - frames/: 可视化图片 (CFD step 粒度)
        - *.mp4: 视频
        """
        import yaml
        from tqdm import tqdm
        from stable_baselines3 import SAC, PPO, TD3
        from myafc.afc_drl_sb3.envs import PhiFlowEnv
        from myafc.env_kernels.solver_wrappers import SolverRunner
        
        # ===== 1. 初始化环境 (仅用于模型加载和动作处理) =====
        env_cfg_dict = {}
        if self.cfg.trainer_config_path:
            with open(self.cfg.trainer_config_path, 'r', encoding='utf-8') as f:
                trainer_cfg = yaml.safe_load(f)
            
            # 优先使用 validator config 中的 env_config_path
            if self.cfg.env_config_path:
                env_cfg_path = Path(self.cfg.env_config_path)
                if not env_cfg_path.is_absolute():
                    # 相对于 validator config 文件目录解析
                    env_cfg_path = (Path(self.cfg.trainer_config_path).parent / env_cfg_path).resolve()
            else:
                # 使用 trainer_config 中的 env_config_path
                env_cfg_path = trainer_cfg.get('env_config_path', '')
                if env_cfg_path and not Path(env_cfg_path).is_absolute():
                    env_cfg_path = (Path(self.cfg.trainer_config_path).parent / env_cfg_path).resolve()
                else:
                    env_cfg_path = Path(env_cfg_path) if env_cfg_path else None
            
            env_cfg_dict['env_config_path'] = str(env_cfg_path) if env_cfg_path else ''
            
            if 'env' in trainer_cfg:
                env_cfg_dict['env'] = trainer_cfg['env'].copy()
            
            input_features = self.cfg.input_features
            if not input_features:
                input_features = trainer_cfg.get('input_features')
            if input_features:
                env_cfg_dict.setdefault('env', {})['input_features'] = input_features
        
        # 确保 env_config_path 存在
        if env_cfg_dict.get('env_config_path') and not Path(env_cfg_dict['env_config_path']).exists():
            LOGGER.warning(f"[Validator] env_config_path 不存在: {env_cfg_dict['env_config_path']}")
            # 尝试 fallback 到同目录下的 solver_config.yaml
            fallback_path = Path(self.cfg.trainer_config_path).parent / 'solver_config.yaml'
            if fallback_path.exists():
                LOGGER.info(f"[Validator] 使用 fallback: {fallback_path}")
                env_cfg_dict['env_config_path'] = str(fallback_path)
        
        # ⭐ 关键: 将 Validator 的 control_interval 和 warmup 配置传递给 PhiFlowEnv
        # 这样验证时使用的控制周期与训练配置一致，且 warmup 由 PhiFlowEnv 统一处理
        env_cfg_dict.setdefault('env', {})['control_interval'] = self.cfg.control_interval
        env_cfg_dict['env']['warmup_steps'] = self.cfg.warmup_steps  # 传递 warmup 配置
        env_cfg_dict['env']['warmup_apply_jets'] = self.cfg.warmup_apply_jets
        LOGGER.debug(f"[Validator] 设置 PhiFlowEnv control_interval={self.cfg.control_interval}, "
                     f"warmup_steps={self.cfg.warmup_steps}")
        
        self.env = PhiFlowEnv(config=env_cfg_dict) if env_cfg_dict else PhiFlowEnv(config=self.cfg.trainer_config_path)
        
        # 使用统一的 control_interval (来自 Validator 配置)
        control_interval = self.cfg.control_interval
        LOGGER.info(f"[Validator] DRL 环境初始化 | control_interval={control_interval}")
        
        self.solver = self.env.solver
        
        # ===== 2. 计算步数 =====
        # 关键设计: warmup 由 PhiFlowEnv.reset() 统一处理
        # validation_steps 是纯验证阶段的 CFD 步数
        val_cfd_steps = self.cfg.validation_steps
        
        LOGGER.info(f"[Validator] 验证步数 | warmup: {self.cfg.warmup_steps} CFD (由 env.reset 处理) | "
                    f"validation: {val_cfd_steps} CFD | control_interval: {control_interval}")
        
        # ===== 3. 创建 Runner =====
        self._setup_runner()
        runner = self.runner
        runner.reset_caches(reset_solver=False)
        
        # ===== 4. 加载 SB3 模型 =====
        algo_cfg = self.trainer_parser.get_algorithm_config() or {}
        algo_type = algo_cfg.get('type', 'sac').lower()
        algo_class = {'sac': SAC, 'ppo': PPO, 'td3': TD3}[algo_type]
        self.model = algo_class.load(self.cfg.model_path, env=self.env)
        
        # ===== 5. 重置环境 (warmup 在 reset 内部执行) =====
        use_h5 = bool(self.solver.cfg.initial.resume_from)
        obs, reset_info = self.env.reset(options={'use_h5': use_h5})
        LOGGER.info(f"[Validator] 环境 reset 完成 | warmup_steps={reset_info.get('warmup_steps', 0)}")
        
        # 设置目标探针
        from phi.torch.flow import Point, vec
        target_pos = self.trainer_parser.get_target_probe_position()
        Lx, Ly = self.solver.cfg.domain.size
        self.target_point = Point(vec(x=target_pos[0] * Lx, y=target_pos[1] * Ly))
        
        if runner.cfg.visuals.enabled and runner.cfg.visuals.save_endpoints:
            runner._save_visual(0)
        
        # ===== 6. 验证循环 (仅验证阶段, warmup 已在 reset 中完成) =====
        # 关键设计（与 JetTrainer/PhiFlowEnv 对齐）：
        # - smoothing_alpha: 由 JetObstacle.apply_controls() 在每个 CFD step 执行
        # - 所有时间步参数的语义都是 CFD step 级别
        rewards = []               # DRL step 级别的 reward 列表
        drl_trajectory = 0         # 已执行的 DRL step 数
        reward_accumulator = 0.0   # 用于累积当前 DRL step 内的 reward
        
        pbar = tqdm(range(val_cfd_steps), desc='Validation')
        
        for cfd_step in pbar:
            # 计算当前 CFD step 在 DRL step 中的位置
            cfd_step_in_drl = cfd_step % control_interval
            
            # --- 每个 DRL step 开始时获取新动作并设置原始信号 ---
            if cfd_step_in_drl == 0:
                action, _ = self.model.predict(obs, deterministic=self.cfg.deterministic)
                action_value = float(action.flatten()[0]) if hasattr(action, 'flatten') else float(action)
                
                # 注意：不在此处做平滑，直接设置原始动作
                # 平滑由 JetObstacle.apply_controls() 在每个 CFD step 内部处理
                if self.solver.jet_obstacle:
                    self.solver.jet_obstacle.set_controls_external(
                        amplitudes=action_value, angles=90.0
                    )
                
                drl_trajectory += 1
            
            # --- 每个 CFD step 都调用 apply_controls（内部执行平滑） ---
            if self.solver.jet_obstacle:
                self.solver.jet_obstacle.apply_controls()
            
            # --- 执行 1 个 CFD step ---
            self.solver.step(update_controls=False, apply_jets=True)
            
            # 关键: solver.step() 后必须清空力系数缓存，否则 _get_force_coeffs() 返回旧值
            self._cached_force_coeffs = None
            
            # --- 获取平滑后的实际幅值 (用于记录) ---
            current_amplitude = 0.0
            if self.solver.jet_obstacle and self.solver.jet_obstacle._prev_applied_amplitudes is not None:
                if len(self.solver.jet_obstacle._prev_applied_amplitudes) > 0:
                    current_amplitude = float(self.solver.jet_obstacle._prev_applied_amplitudes[0])
            
            # --- 记录数据 (每个 CFD step) ---
            cfd_step_num = cfd_step + 1
            if runner.cfg.force_coeffs.enabled:
                runner._record_force_coeffs()
            if runner.cfg.probes.enabled:
                runner._record_probes()
            
            # --- 记录 validation_curves 数据 (warmup 已在 reset 中完成) ---
            cd, cl = self._get_force_coeffs()
            self._val_cd.append(cd)
            self._val_cl.append(cl)
            self._val_amplitudes.append(float(current_amplitude))
            
            u, v = self._sample_target_velocity()
            self._val_velocities.append((float(u), float(v)))
            self._val_pressures.append(self._sample_target_pressure())
            
            # 计算即时 reward (每个 CFD step)
            reward_cfg = self.env.cfg.reward
            instant_reward = reward_cfg.cd_baseline - cd - reward_cfg.alpha_lift * abs(cl)
            reward_accumulator += instant_reward
            
            # 每个 DRL step 结束时记录累积 reward
            if cfd_step_in_drl == control_interval - 1:
                rewards.append(float(reward_accumulator))
                reward_accumulator = 0.0  # 重置累积器
            
            pbar.set_postfix_str(f"Cd={cd:.4f} |Cl|={abs(cl):.4f}")
            
            # 可视化和流场保存
            if runner.cfg.visuals.enabled and cfd_step_num % runner.cfg.visuals.interval == 0:
                runner._save_visual(cfd_step_num)
            if runner.cfg.fields.save.enabled and cfd_step_num % runner.cfg.fields.save.interval == 0:
                runner._save_fields_h5()
            
            # --- 每个 DRL step 结束时更新观测 ---
            if cfd_step_in_drl == control_interval - 1:
                obs = self.env._get_observation()
                self._cached_force_coeffs = None  # 清空缓存
        
        # ===== 7. 保存最终帧 =====
        if runner.cfg.visuals.enabled and runner.cfg.visuals.save_endpoints:
            runner._save_visual(val_cfd_steps, suffix='_final')
        
        # ===== 8. 保存 CSV、图表、视频 =====
        if runner.cfg.output.enabled:
            runner._save_csvs()
            if runner.force_coeffs:
                runner._plot_force_coeffs()
            runner._generate_videos()
        
        if runner._h5_file is not None:
            runner._h5_file.close()
            runner._h5_file = None
        
        # ===== 9. 保存 validation_curves =====
        results = self._save_results(is_drl=True, rewards=rewards)
        
        return results
    
    def _save_results(self, is_drl: bool = False, rewards: List[float] = None) -> Dict[str, Any]:
        """保存验证结果
        
        统一输出格式:
        - NPZ 和 CSV 使用相同键名
        - 必选字段: env_step, amplitude, cd, cl
        - 可选字段: loss, velocity_u, velocity_v, pressure, reward (根据模式)
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # 确定数据长度 (以 cd/cl 为准，它们在两种模式下都收集)
        n_steps = max(len(self._val_cd), len(self._val_cl), len(self._val_amplitudes))
        if n_steps == 0:
            LOGGER.warning("[Validator] 无有效数据，跳过保存")
            return {'output_dir': str(self.output_dir)}
        
        # 计算统计量
        mean_cd = float(np.mean(self._val_cd)) if self._val_cd else 0.0
        mean_cl = float(np.mean(self._val_cl)) if self._val_cl else 0.0
        mean_amp = float(np.mean(self._val_amplitudes)) if self._val_amplitudes else 0.0
        mean_loss = float(np.mean(self._val_losses)) if self._val_losses else 0.0
        
        results = {
            'warmup_steps': self.cfg.warmup_steps,
            'validation_steps': self.cfg.validation_steps,
            'control_interval': self.cfg.control_interval,
            'action_duration': self.cfg.action_duration,
            'mean_cd': mean_cd,
            'mean_cl': mean_cl,
            'mean_amplitude': mean_amp,
            'std_cd': float(np.std(self._val_cd)) if self._val_cd else 0.0,
            'std_cl': float(np.std(self._val_cl)) if self._val_cl else 0.0,
            'output_dir': str(self.output_dir),
        }
        
        if self._val_losses:
            results['mean_loss'] = mean_loss
        if is_drl and rewards:
            results['mean_reward'] = float(np.mean(rewards))
            results['total_reward'] = float(np.sum(rewards))
        
        # 获取目标速度 (用于元数据和绘图)
        loss_cfg = self.trainer_parser.get_loss_config() or {}
        target_u, target_v = loss_cfg.get('target_velocity', [0.0, 0.0])
        
        # 提取速度分量 (可能为空)
        velocities_u = [v[0] for v in self._val_velocities] if self._val_velocities else []
        velocities_v = [v[1] for v in self._val_velocities] if self._val_velocities else []
        
        # ===== 构建统一数据字典 =====
        # 使用与 CSV 相同的键名，这样 NPZ 和 CSV 完全一致
        data_dict = {
            'env_step': np.arange(n_steps),
            'amplitude': np.array(self._val_amplitudes) if self._val_amplitudes else np.zeros(n_steps),
            'cd': np.array(self._val_cd) if self._val_cd else np.zeros(n_steps),
            'cl': np.array(self._val_cl) if self._val_cl else np.zeros(n_steps),
        }
        
        # 可选字段
        if self._val_losses:
            data_dict['loss'] = np.array(self._val_losses)
        if velocities_u:
            data_dict['velocity_u'] = np.array(velocities_u)
            data_dict['velocity_v'] = np.array(velocities_v)
        data_dict['pressure'] = np.array(self._val_pressures)
        
        # DRL reward: 扩展到 CFD step 粒度 (每 control_interval 个 CFD step 共享同一 reward)
        if is_drl and rewards:
            control_interval = self.cfg.control_interval
            reward_expanded = []
            for r in rewards:
                reward_expanded.extend([r] * control_interval)
            # 截断或填充到正确长度
            if len(reward_expanded) < n_steps:
                reward_expanded.extend([reward_expanded[-1]] * (n_steps - len(reward_expanded)))
            reward_expanded = reward_expanded[:n_steps]
            data_dict['reward'] = np.array(reward_expanded)
        
        # ===== 保存 NPZ (包含元数据) =====
        npz_data = data_dict.copy()
        npz_data['target_velocity'] = np.array([target_u, target_v])
        if is_drl and rewards:
            # 同时保存原始 DRL step 级别的 reward (便于分析)
            npz_data['reward_per_drl_step'] = np.array(rewards)
        np.savez(self.output_dir / 'validation_curves.npz', **npz_data)
        
        # ===== 保存 CSV (不含元数据，仅时序数据) =====
        df = pd.DataFrame(data_dict)
        df.to_csv(self.output_dir / 'validation_curves.csv', index=False)
        
        # 绘制曲线
        self._plot_curves(velocities_u, velocities_v, target_u, target_v, 
                          pressures=self._val_pressures,
                          is_drl=is_drl, rewards=rewards)
        
        # 保存验证配置快照
        self._save_config_snapshot()
        
        LOGGER.info(f"[Validator] 完成 | mean_Cd={mean_cd:.4f} | mean_|Cl|={abs(mean_cl):.4f}"
                   + (f" | mean_Loss={mean_loss:.6f}" if self._val_losses else "")
                   + (f" | mean_Reward={results.get('mean_reward', 0):.4f}" if is_drl and rewards else ""))
        return results
    
    def _save_config_snapshot(self):
        """保存验证配置快照
        
        保存到 output_dir/configs/:
        - val_config.yaml: 验证参数配置
        - solver_config.yaml: 环境/solver 配置 (如果存在)
        - trainer_config.yaml: trainer 配置 (如果存在)
        """
        import yaml
        
        configs_dir = self.output_dir / 'configs'
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存验证配置
        val_cfg_dict = {
            'model_path': str(self.cfg.model_path),
            'model_type': self.model_type,
            'trainer_config_path': str(self.cfg.trainer_config_path),
            'warmup_steps': self.cfg.warmup_steps,
            'validation_steps': self.cfg.validation_steps,
            'deterministic': self.cfg.deterministic,
            'use_ema': self.cfg.use_ema,
            'device': str(self.device),
            'control_interval': self.cfg.control_interval,
            'action_duration': self.cfg.action_duration,
            'warmup_apply_jets': self.cfg.warmup_apply_jets,
            'warmup_update_controls': self.cfg.warmup_update_controls,
            'visuals': {
                'enabled': self.cfg.visuals.enabled,
                'interval': self.cfg.visuals.interval,
                'quantities': self.cfg.visuals.quantities,
                'save_fields': self.cfg.visuals.save_fields,
                'generate_video': self.cfg.visuals.generate_video,
            },
        }
        # 添加 DRL 专用配置
        if hasattr(self.cfg, 'env_config_path') and self.cfg.env_config_path:
            val_cfg_dict['env_config_path'] = str(self.cfg.env_config_path)
        if hasattr(self.cfg, 'input_features') and self.cfg.input_features:
            val_cfg_dict['input_features'] = self.cfg.input_features
        
        with open(configs_dir / 'val_config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(val_cfg_dict, f, sort_keys=False, allow_unicode=True)
        
        # 2. 复制 trainer_config (如果存在)
        if self.cfg.trainer_config_path and Path(self.cfg.trainer_config_path).exists():
            import shutil
            shutil.copy2(self.cfg.trainer_config_path, configs_dir / 'trainer_config.yaml')
        
        # 3. 复制 solver_config (如果存在)
        env_config_path = self.trainer_parser.get_env_config_path()
        if env_config_path and env_config_path.exists():
            import shutil
            shutil.copy2(env_config_path, configs_dir / 'solver_config.yaml')
        
        LOGGER.info(f"[Validator] 配置快照保存至: {configs_dir}")

    def _plot_curves(self, velocities_u: List[float], velocities_v: List[float],
                     target_u: float, target_v: float, 
                     pressures: List[float],
                     is_drl: bool = False, rewards: List[float] = None):
        """绘制验证曲线 (3x2 布局)
        
        布局:
        - (0,0): Loss (Supervision) 或 Reward (DRL)
        - (0,1): N/A
        - (1,0): Velocity & Pressure (如有)
        - (1,1): Amplitude
        - (2,0): Cd
        - (2,1): Cl
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # (0,0) Loss 或 Reward
        ax = axes[0, 0]
        if is_drl and rewards:
            ax.plot(rewards, 'b-', alpha=0.8, linewidth=0.8)
            ax.axhline(y=np.mean(rewards), color='r', linestyle='--', alpha=0.5, 
                       label=f'mean={np.mean(rewards):.3f}')
            ax.legend(fontsize=8)
            ax.set_title('Validation Reward')
        elif self._val_losses:
            ax.plot(self._val_losses, 'b-', alpha=0.8, linewidth=0.8)
            ax.set_title('Validation Primary Loss')
        else:
            ax.text(0.5, 0.5, 'No Loss/Reward Data', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('Loss/Reward (N/A)')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Loss' if not is_drl else 'Reward')
        ax.grid(True, alpha=0.3)
        
        # (0,1) 验证阶段无正则化
        ax = axes[0, 1]
        ax.text(0.5, 0.5, 'Validation: No Regularization', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Regularization Loss (N/A)')
        ax.grid(True, alpha=0.3)
        
        # (1,0) Velocity
        ax = axes[1, 0]
        if velocities_u:
            ax.plot(velocities_u, 'b-', label='u', alpha=0.8, linewidth=0.8)
            ax.plot(velocities_v, 'orange', label='v', alpha=0.8, linewidth=0.8)
            ax.axhline(y=target_u, color='blue', linestyle='--', alpha=0.5, label=f'u_target={target_u:.3f}')
            ax.axhline(y=target_v, color='orange', linestyle='--', alpha=0.5, label=f'v_target={target_v:.3f}')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Velocity')
        ax.set_title('Probe Velocity & Pressure')
        ax.grid(True, alpha=0.3)
        # 压力曲线（右轴）
        ax2 = ax.twinx()
        ax2.plot(pressures, 'g-', label='p', alpha=0.6, linewidth=0.8)
        ax2.set_ylabel('Pressure', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        
        # (1,1) Amplitude
        ax = axes[1, 1]
        if self._val_amplitudes:
            ax.plot(self._val_amplitudes, 'g-', alpha=0.8, linewidth=0.8)
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Amplitude')
        ax.set_title('Control Amplitude')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        # (2,0) Cd
        ax = axes[2, 0]
        if self._val_cd:
            valid_cd = [c if not np.isnan(c) else None for c in self._val_cd]
            ax.plot(valid_cd, 'b-', alpha=0.8, linewidth=0.8, label='Cd')
            ax.legend(fontsize=8)
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Cd')
        ax.set_title('Drag Coefficient (Cd)')
        ax.grid(True, alpha=0.3)
        
        # (2,1) Cl
        ax = axes[2, 1]
        if self._val_cl:
            valid_cl = [c if not np.isnan(c) else None for c in self._val_cl]
            ax.plot(valid_cl, 'r-', alpha=0.8, linewidth=0.8, label='Cl')
            ax.legend(fontsize=8)
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Cl')
        ax.set_title('Lift Coefficient (Cl)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'validation_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
