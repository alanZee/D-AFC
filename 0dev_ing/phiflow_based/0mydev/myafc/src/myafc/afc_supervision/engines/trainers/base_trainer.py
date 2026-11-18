"""
BaseTrainer - 训练器抽象基类

定义通用训练接口和基础设施:
- 随机种子、输出目录、日志
- 优化器、学习率调度、梯度裁剪 (YOLOv8 风格参数分组)
- EMA/AMP 支持 (YOLOv8 风格 ModelEMA)
- 检查点保存/加载 (YOLOv8 风格; model + ema 同时保存)
- TensorBoard 记录
"""
from __future__ import annotations
import io
import os
import re
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop, NAdam, RAdam

from myafc.utils import get_logger, setup_file_logging, set_log_level

LOGGER = get_logger()

# 版本信息
__version__ = "0.1.0"


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class TrainingBaseConfig:
    """训练基础配置
    
    Attributes:
        epochs: 训练轮数 (mini-episode 数量)
        learning_rate: 初始学习率
        lr_final: 最终学习率比例 (相对于初始 lr)
        lr_schedule: 学习率调度 ('constant' / 'linear' / 'cosine')
        warmup_updates: 学习率预热更新次数
        warmup_bias_lr: 预热初始 LR 比例
        optimizer: 优化器 ('adam' / 'adamw' / 'sgd')
        optimizer_momentum: 优化器动量 (Adam: beta1, SGD: momentum)
        weight_decay: 权重衰减
        grad_clip: 梯度裁剪阈值 (0 = 不裁剪)
        use_ema: 是否使用 EMA 模型平均
        ema_decay: EMA 衰减率
        ema_tau: EMA 衰减时间常数
        use_amp: 是否使用混合精度训练
        random_seed: 随机种子 (None = 不固定)
        checkpoint_interval: 检查点保存间隔 (epochs)
        checkpoint: 检查点路径 (null=不加载)
        resume: 断点续训模式 (true=恢复全部状态, false=若有checkpoint则仅加载权重)
        verbose: 日志级别 ('DEBUG' / 'INFO' / 'WARNING')
    """
    epochs: int = 100
    learning_rate: float = 1e-3
    lr_final: float = 0.01
    lr_schedule: str = 'cosine'
    warmup_updates: int = 10
    warmup_bias_lr: float = 0.1
    optimizer: str = 'adam'
    optimizer_momentum: float = 0.937
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_tau: int = 2000  # YOLOv8 默认值
    use_amp: bool = False
    random_seed: Optional[int] = 42
    checkpoint_interval: int = 50
    # 检查点与断点续训
    checkpoint: Optional[str] = None  # 检查点路径 (null=不加载)
    resume: bool = False  # true=断点续训, false=若有checkpoint则仅加载权重
    verbose: str = 'INFO'


@dataclass
class OutputBaseConfig:
    """输出配置"""
    project: str = 'runs/afc_train'
    name: str = 'exp'
    # 可视化配置
    scatter_enabled: bool = False   # 是否绘制散点标记（神经网络更新点）
    scatter_size: float = 20.0      # 散点尺寸
    vline_enabled: bool = True      # 是否绘制竖直虚线（标记神经网络更新位置）
    vline_width: float = 0.5        # 竖直虚线线宽


# ============================================================
# YOLOv8 风格 ModelEMA
# ============================================================
class ModelEMA:
    """指数移动平均模型 (YOLOv8 风格)
    
    对模型参数保持指数移动平均，用于:
    - 稳定训练过程
    - 提供更好的泛化性能
    - 验证评估使用 EMA 模型
    
    公式: ema = decay * ema + (1 - decay) * model
    衰减率动态调整: decay(x) = ema_decay * (1 - exp(-x / tau))
    
    Attributes:
        ema: EMA 模型副本 (eval 模式)
        updates: EMA 更新次数
        decay: 衰减率函数
        enabled: 是否启用
    
    Reference:
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        """初始化 EMA 模型
        
        使用 state_dict 方式复制模型，避免 deepcopy 在前向传播后失败。
        
        Args:
            model: 源模型
            decay: 最大 EMA 衰减率
            tau: 衰减时间常数 (用于动态调整)
            updates: 初始更新次数
        """
        # 使用 state_dict 方式复制，避免 deepcopy 问题
        try:
            self.ema = deepcopy(model).eval()
        except RuntimeError:
            # 如果 deepcopy 失败，使用 state_dict 方式
            self.ema = type(model)(**self._get_init_args(model))
            self.ema.load_state_dict(model.state_dict())
            self.ema.eval()
        
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # 动态衰减
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True
    
    @staticmethod
    def _get_init_args(model: nn.Module) -> Dict:
        """尝试获取模型初始化参数 (用于重建模型)"""
        # 对于简单 Sequential 模型，不需要参数
        if isinstance(model, nn.Sequential):
            return {}
        # 对于有 _config 属性的模型，尝试使用其配置
        if hasattr(model, '_config'):
            return {'config': model._config}
        return {}
    
    def update(self, model: nn.Module):
        """更新 EMA 参数
        
        Args:
            model: 当前模型
        """
        if not self.enabled:
            return
        
        self.updates += 1
        d = self.decay(self.updates)
        
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()
    
    def state_dict(self) -> Dict:
        """获取 EMA 状态字典"""
        return {
            'ema': self.ema.state_dict(),
            'updates': self.updates,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """加载 EMA 状态"""
        self.ema.load_state_dict(state_dict['ema'])
        self.updates = state_dict.get('updates', 0)


# ============================================================
# 训练器抽象基类
# ============================================================
class BaseTrainer(ABC):
    """训练器抽象基类
    
    提供通用基础设施，子类需实现:
    - _setup_env(): 创建/初始化环境
    - _setup_agent(): 创建/初始化 Agent
    - _build_state(): 构建 Agent 输入状态
    - _compute_loss(): 计算单步 loss
    - train(): 执行训练循环
    """
    
    def __init__(
        self,
        training_config: TrainingBaseConfig,
        output_config: OutputBaseConfig,
    ):
        """初始化基类
        
        Args:
            training_config: 训练配置
            output_config: 输出配置
        """
        self.training_cfg = training_config
        self.output_cfg = output_config
        
        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.update_count = 0
        self._total_updates = 1  # 由子类设置实际值
        self.best_fitness = float('inf')  # YOLOv8 风格: 最佳 fitness (用于保存 best.pt)
        self._is_resumed = False  # 是否为断点续训
        
        # 组件占位 (由子类初始化)
        self.agent = None
        self.optimizer = None
        self.device = 'cpu'
        
        # EMA 模型 (YOLOv8 风格 ModelEMA)
        self.ema: Optional[ModelEMA] = None
        
        # AMP GradScaler
        self._grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
        if training_config.use_amp and torch.cuda.is_available():
            self._grad_scaler = torch.cuda.amp.GradScaler()
        
        # TensorBoard
        self._tb_writer = None
    
    # ========== 抽象方法 (子类必须实现) ==========
    
    @abstractmethod
    def _setup_env(self) -> None:
        """创建/初始化环境"""
        pass
    
    @abstractmethod
    def _setup_agent(self) -> None:
        """创建/初始化 Agent"""
        pass
    
    @abstractmethod
    def _build_state(self) -> Dict:
        """构建 Agent 输入状态"""
        pass
    
    @abstractmethod
    def _compute_loss(self, **kwargs) -> torch.Tensor:
        """计算单步 loss"""
        pass
    
    @abstractmethod
    def train(self) -> Dict:
        """执行训练"""
        pass
    
    # ========== 基础设施方法 ==========
    
    def _setup_seed(self):
        """设置随机种子"""
        seed = self.training_cfg.random_seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            LOGGER.info(f"[Trainer] 随机种子: {seed}")
    
    def _setup_output_dir(self):
        """设置输出目录 (支持 Resume 模式)
        
        Resume 模式 (resume=true + checkpoint):
            - 从 checkpoint 路径推断实验目录
            - 复用原有目录
        
        新建模式:
            - YOLOv8 风格自增目录 (exp -> exp2 -> exp3 ...)
        """
        if self.training_cfg.resume:
            # Resume 模式: 复用原有目录
            ckpt_path = self._resolve_checkpoint_path()
            if ckpt_path and ckpt_path.exists():
                # 从 checkpoint 路径推断 exp_dir (典型结构: exp_dir/checkpoints/xxx.pt)
                self.exp_dir = ckpt_path.parent.parent
                if not self.exp_dir.exists():
                    raise FileNotFoundError(f"Resume 目录不存在: {self.exp_dir}")
                LOGGER.info(f"[Trainer] Resume 模式，复用目录: {self.exp_dir}")
            else:
                raise FileNotFoundError(
                    f"Resume 模式但找不到 checkpoint: {ckpt_path}\n"
                    f"请检查 checkpoint 配置"
                )
        else:
            # 新建模式: YOLOv8 风格自增
            base_path = Path(self.output_cfg.project) / self.output_cfg.name
            self.exp_dir = self._increment_path(base_path)
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'configs').mkdir(exist_ok=True)
        LOGGER.info(f"[Trainer] 实验目录: {self.exp_dir}")
    
    def _setup_logging(self):
        """设置日志"""
        set_log_level(self.training_cfg.verbose)
        log_path = self.exp_dir / 'train.log'
        # Resume 模式追加日志，新建模式覆盖
        setup_file_logging(str(log_path), append=self._is_resumed)
    
    # ========== Checkpoint / Resume 相关方法 ==========
    
    def _resolve_checkpoint_path(self) -> Optional[Path]:
        """解析 checkpoint 路径
        
        Returns:
            checkpoint 文件路径，或 None（未配置时）
        """
        ckpt_path = self.training_cfg.checkpoint
        if not ckpt_path:
            return None
        
        path = Path(ckpt_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    
    def _resume_training(self):
        """断点续训: 加载完整训练状态
        
        加载内容:
            - 模型权重
            - EMA 状态
            - 优化器状态
            - AMP scaler 状态
            - 训练进度 (epoch, step, update_count, best_fitness)
        """
        ckpt_path = self._resolve_checkpoint_path()
        if not ckpt_path or not ckpt_path.exists():
            LOGGER.warning(f"[Trainer] Resume checkpoint 不存在: {ckpt_path}")
            return
        
        LOGGER.info(f"[Trainer] 断点续训: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # 1. 加载模型权重
        model_state = ckpt.get('model')
        if model_state is not None:
            self._load_model_state_dict(model_state)
            LOGGER.info(f"  ✓ 模型权重已加载")
        
        # 2. 加载优化器状态
        optimizer_state = ckpt.get('optimizer')
        if self.optimizer and optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
            LOGGER.info(f"  ✓ 优化器状态已加载")
        
        # 3. 加载 EMA 状态
        ema_state = ckpt.get('ema')
        if self.ema and ema_state:
            self._load_ema_state_dict(ema_state)
            LOGGER.info(f"  ✓ EMA 状态已加载 (updates={self.ema.updates})")
        
        # 4. 加载 AMP scaler 状态
        if self._grad_scaler and ckpt.get('scaler'):
            self._grad_scaler.load_state_dict(ckpt['scaler'])
            LOGGER.info(f"  ✓ AMP scaler 状态已加载")
        
        # 5. 恢复训练进度
        # checkpoint 保存格式: epoch = 已完成轮数
        self.current_epoch = ckpt.get('epoch', 0)
        self.current_step = ckpt.get('step', 0)
        self.update_count = ckpt.get('update_count', 0)
        self.best_fitness = ckpt.get('best_fitness', float('inf'))
        
        # 标记为已恢复
        self._is_resumed = True
        
        LOGGER.info(
            f"  ✓ 训练进度已恢复: epoch={self.current_epoch}, "
            f"step={self.current_step}, updates={self.update_count}"
        )
    
    def _load_checkpoint_weights(self):
        """仅加载模型权重 (非断点续训，仅初始化网络)
        
        支持多种格式:
        1. AFCAgent.save_weights 格式
        2. Trainer checkpoint 格式
        3. 纯 state_dict 格式
        """
        ckpt_path = self._resolve_checkpoint_path()
        
        if not ckpt_path:
            return  # 未配置 checkpoint
        
        if not ckpt_path.exists():
            LOGGER.warning(f"[Trainer] checkpoint 文件不存在: {ckpt_path}")
            return
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            # 兼容多种格式
            if 'network_state_dict' in checkpoint:
                # AFCAgent.save_weights 格式
                state_dict = checkpoint['network_state_dict']
            elif 'model' in checkpoint:
                # Trainer checkpoint 格式
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and any(
                k.startswith(('mlp_extractor', 'action_net', 'log_std')) for k in checkpoint.keys()
            ):
                # 纯 state_dict
                state_dict = checkpoint
            else:
                LOGGER.warning(
                    f"[Trainer] 无法识别 checkpoint 格式: {list(checkpoint.keys())[:5]}..."
                )
                return
            
            self._load_model_state_dict(state_dict)
            LOGGER.info(f"[Trainer] 已加载模型权重: {ckpt_path}")
            
        except Exception as e:
            LOGGER.warning(f"[Trainer] 加载模型权重失败: {e}")
    
    def _setup_optimizer(self, parameters):
        """构建优化器 (YOLOv8 风格，支持多种优化器)
        
        支持的优化器 (对应 YOLOv8):
            - adam: 标准 Adam
            - adamw: AdamW (解耦权重衰减)
            - sgd: SGD with Nesterov momentum
            - nadam: NAdam (Nesterov-accelerated Adam)
            - radam: RAdam (Rectified Adam)
            - rmsprop: RMSprop
            - auto: 自动选择 (默认 AdamW)
        
        Args:
            parameters: 可训练参数 (通常是 agent.parameters())
        """
        opt_name = self.training_cfg.optimizer.lower()
        lr = self.training_cfg.learning_rate
        wd = self.training_cfg.weight_decay
        momentum = self.training_cfg.optimizer_momentum
        
        # 自动选择默认优化器
        if opt_name == 'auto':
            opt_name = 'adamw'
            LOGGER.info("[Trainer] 优化器 auto -> AdamW")
        
        if opt_name == 'adam':
            self.optimizer = Adam(parameters, lr=lr, betas=(momentum, 0.999), weight_decay=wd)
        elif opt_name == 'adamw':
            self.optimizer = AdamW(parameters, lr=lr, betas=(momentum, 0.999), weight_decay=wd)
        elif opt_name == 'sgd':
            self.optimizer = SGD(parameters, lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
        elif opt_name == 'nadam':
            self.optimizer = NAdam(parameters, lr=lr, betas=(momentum, 0.999), weight_decay=wd)
        elif opt_name == 'radam':
            self.optimizer = RAdam(parameters, lr=lr, betas=(momentum, 0.999), weight_decay=wd)
        elif opt_name == 'rmsprop':
            self.optimizer = RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=wd)
        else:
            LOGGER.warning(f"[Trainer] 未知优化器 '{opt_name}'，使用 AdamW")
            self.optimizer = AdamW(parameters, lr=lr, betas=(momentum, 0.999), weight_decay=wd)
        
        LOGGER.info(f"[Trainer] 优化器: {opt_name}, LR: {lr}")
    
    def _setup_optimizer_yolo(self, network: nn.Module):
        """构建优化器 (YOLOv8 风格参数分组)
        
        支持的优化器: adam, adamw, sgd, nadam, radam, rmsprop, auto
        
        参数分组:
        - g[0]: weights (with decay)
        - g[1]: weights (no decay) - BatchNorm 等
        - g[2]: bias (no decay)
        
        Args:
            network: 神经网络模块
        """
        cfg = self.training_cfg
        opt_name = cfg.optimizer.lower()
        lr = cfg.learning_rate
        wd = cfg.weight_decay
        momentum = cfg.optimizer_momentum
        
        # 自动选择默认优化器
        if opt_name == 'auto':
            opt_name = 'adamw'
            LOGGER.info("[Trainer] 优化器 auto -> AdamW")
        
        # YOLOv8 风格参数分组
        g = [[], [], []]
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, 
                    nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)
        
        for module_name, module in network.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if 'bias' in param_name:
                    g[2].append(param)
                elif isinstance(module, bn_types):
                    g[1].append(param)
                else:
                    g[0].append(param)
        
        # 优化器映射
        optimizer_map = {
            'adam': lambda p, **kw: Adam(p, lr=lr, betas=(momentum, 0.999), **kw),
            'adamw': lambda p, **kw: AdamW(p, lr=lr, betas=(momentum, 0.999), **kw),
            'nadam': lambda p, **kw: NAdam(p, lr=lr, betas=(momentum, 0.999), **kw),
            'radam': lambda p, **kw: RAdam(p, lr=lr, betas=(momentum, 0.999), **kw),
            'sgd': lambda p, **kw: SGD(p, lr=lr, momentum=momentum, nesterov=True, **kw),
            'rmsprop': lambda p, **kw: RMSprop(p, lr=lr, momentum=momentum, **kw),
        }
        
        # 创建优化器 (bias 组无权重衰减)
        opt_fn = optimizer_map.get(opt_name, optimizer_map['adamw'])
        self.optimizer = opt_fn(
            g[2] if g[2] else [{'params': []}],
            weight_decay=0.0
        )
        
        # 添加其他参数组
        if g[0]:
            self.optimizer.add_param_group({'params': g[0], 'weight_decay': wd})
        if g[1]:
            self.optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
        
        LOGGER.info(f"[Trainer] 优化器: {opt_name} (YOLOv8 分组) | lr={lr}")
        LOGGER.info(f"[Trainer] 参数分组: {len(g[2])} bias, {len(g[0])} weight, {len(g[1])} norm")
    
    def _setup_ema(self, network: nn.Module):
        """设置 EMA 模型 (YOLOv8 风格)
        
        Args:
            network: 要复制的网络
        """
        if not self.training_cfg.use_ema:
            return
        
        cfg = self.training_cfg
        self.ema = ModelEMA(
            network,
            decay=cfg.ema_decay,
            tau=getattr(cfg, 'ema_tau', 2000),
        )
        self.ema.ema.to(self.device)
        
        LOGGER.info(f"[Trainer] EMA 模型已创建 (YOLOv8 风格) | decay={cfg.ema_decay}, tau={getattr(cfg, 'ema_tau', 2000)}")
    
    def _update_ema(self, network: nn.Module):
        """更新 EMA 模型权重 (YOLOv8 风格)"""
        if self.ema is None:
            return
        self.ema.update(network)
    
    def _setup_tensorboard(self):
        """设置 TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(self.exp_dir / 'tensorboard'))
            LOGGER.info(f"[Trainer] TensorBoard: {self.exp_dir / 'tensorboard'}")
        except ImportError:
            LOGGER.warning("[Trainer] TensorBoard 不可用，跳过")
    
    def _update_lr(self) -> float:
        """更新学习率 (预热 + 调度)
        
        Returns:
            当前学习率
        """
        cfg = self.training_cfg
        progress = self.update_count / max(1, self._total_updates)
        progress = min(progress, 1.0)
        
        # 基础学习率
        lr_base = cfg.learning_rate
        lr_final = cfg.lr_final * lr_base
        
        # 调度
        if cfg.lr_schedule == 'constant':
            lr = lr_base
        elif cfg.lr_schedule == 'linear':
            lr = lr_base + (lr_final - lr_base) * progress
        else:  # cosine
            lr = lr_final + 0.5 * (lr_base - lr_final) * (1 + math.cos(math.pi * progress))
        
        # 预热覆盖
        if self.update_count < cfg.warmup_updates:
            warmup_progress = self.update_count / max(1, cfg.warmup_updates)
            lr_warmup_start = cfg.warmup_bias_lr * lr_base
            lr = lr_warmup_start + (lr - lr_warmup_start) * warmup_progress
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        return lr
    
    def _apply_grad_clip(self):
        """应用梯度裁剪"""
        if self.training_cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.training_cfg.grad_clip
            )
    
    def _save_checkpoint(self, name: str, fitness: float = None):
        """保存检查点 (YOLOv8 风格: 同时保存 model + ema)
        
        检查点包含:
        - epoch, step, update_count: 训练状态
        - model: 模型权重 (由 _get_model_state_dict 提供)
        - ema: EMA 模型权重 (若启用)
        - optimizer: 优化器状态
        - scaler: AMP scaler 状态 (若启用)
        - best_fitness: 最佳 fitness
        - date, version: 元信息
        
        Args:
            name: 检查点名称 ('last', 'best', 'epoch_10', 'final')
            fitness: 当前 fitness (用于更新 best)
        """
        ckpt_dir = self.exp_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f'{name}.pt'
        
        # 更新 best_fitness
        if fitness is not None and fitness < self.best_fitness:
            self.best_fitness = fitness
        
        # 构建检查点数据 (YOLOv8 风格)
        ckpt_data = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'update_count': self.update_count,
            'best_fitness': self.best_fitness,
            'model': self._get_model_state_dict(),
            'ema': self._get_ema_state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scaler': self._grad_scaler.state_dict() if self._grad_scaler else None,
            'date': datetime.now().isoformat(),
            'version': __version__,
        }
        
        # 序列化到 buffer 后写入 (更快)
        buffer = io.BytesIO()
        torch.save(ckpt_data, buffer)
        ckpt_path.write_bytes(buffer.getvalue())
        
        ema_info = f" (含 EMA, updates={self.ema.updates})" if self.ema else ""
        LOGGER.info(f"[Trainer] 保存检查点: {ckpt_path}{ema_info}")
    
    def _get_model_state_dict(self) -> Optional[Dict]:
        """获取模型状态字典 (子类可覆盖)
        
        默认实现: 尝试调用 agent.state_dict()
        子类可覆盖此方法以支持不同的模型接口
        """
        if self.agent is None:
            return None
        if hasattr(self.agent, 'state_dict'):
            return self.agent.state_dict()
        if hasattr(self.agent, 'network') and self.agent.network is not None:
            return self.agent.network.state_dict()
        return None
    
    def _get_ema_state_dict(self) -> Optional[Dict]:
        """获取 EMA 状态字典 (子类可覆盖)"""
        if self.ema is None:
            return None
        return self.ema.state_dict()
    
    def _load_model_state_dict(self, state_dict: Dict):
        """加载模型状态字典 (子类可覆盖)
        
        默认实现: 尝试调用 agent.load_state_dict()
        子类可覆盖此方法以支持不同的模型接口
        """
        if self.agent is None or state_dict is None:
            return
        if hasattr(self.agent, 'load_state_dict'):
            self.agent.load_state_dict(state_dict)
        elif hasattr(self.agent, 'network') and self.agent.network is not None:
            self.agent.network.load_state_dict(state_dict)
    
    def _load_ema_state_dict(self, state_dict: Dict):
        """加载 EMA 状态字典 (子类可覆盖)"""
        if self.ema is None or state_dict is None:
            return
        self.ema.load_state_dict(state_dict)
    
    def _load_checkpoint(self, path: str) -> Dict:
        """加载检查点 (YOLOv8 风格)
        
        Args:
            path: 检查点路径
        
        Returns:
            检查点字典
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        model_state = ckpt.get('model')
        self._load_model_state_dict(model_state)
        
        optimizer_state = ckpt.get('optimizer') or ckpt.get('optimizer_state_dict')
        if self.optimizer and optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        
        # 加载 EMA 状态
        self._load_ema_state_dict(ckpt.get('ema'))
        
        # 加载 AMP scaler 状态
        if self._grad_scaler and ckpt.get('scaler'):
            self._grad_scaler.load_state_dict(ckpt['scaler'])
        
        self.current_epoch = ckpt.get('epoch', 0)
        self.current_step = ckpt.get('step', 0)
        self.update_count = ckpt.get('update_count', 0)
        self.best_fitness = ckpt.get('best_fitness', float('inf'))
        
        LOGGER.info(f"[Trainer] 加载检查点: {path}")
        return ckpt
    
    def _log_metrics(self, metrics: Dict, step: int = None):
        """记录指标到 TensorBoard
        
        Args:
            metrics: 指标字典 {'name': value, ...}
            step: 步数 (默认使用 update_count)
        """
        if self._tb_writer is None:
            return
        
        step = step if step is not None else self.update_count
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self._tb_writer.add_scalar(name, value, step)
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict):
        """记录 epoch 摘要日志"""
        parts = [f"Epoch {epoch}"]
        for key, val in metrics.items():
            if isinstance(val, float):
                parts.append(f"{key}={val:.4f}")
            else:
                parts.append(f"{key}={val}")
        LOGGER.info(" | ".join(parts))
    
    # ========== 工具方法 ==========
    
    @staticmethod
    def _increment_path(path: Path, sep: str = '') -> Path:
        """自增路径 (exp -> exp2 -> exp3 ...)"""
        path = Path(path)
        if not path.exists():
            return path
        
        dirs = [d.name for d in path.parent.glob(f"{path.name}{sep}*") if d.is_dir()]
        matches = [re.search(rf"{path.name}{sep}(\d+)", d) for d in dirs]
        nums = [int(m.group(1)) for m in matches if m]
        n = max(nums) + 1 if nums else 2
        return path.parent / f"{path.name}{sep}{n}"
    
    def close(self):
        """清理资源"""
        if self._tb_writer:
            self._tb_writer.close()
