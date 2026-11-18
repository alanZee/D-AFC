"""
SimpleTrainer - 基于 PhiFlow 可微分仿真的 MLP 训练器

训练流程：
1. 在 PhiFlow Torch 后端使用 PyTorch 自动微分
2. 复用 MLPPolicy 作为控制策略网络
3. 简洁的训练流程：通过 solver.step(update_controls, apply_jets) 让 Solver 内部处理控制信号
4. 由 reward 计算 loss

关键配置参数（见 TrainingConfig）：
- action_duration: 一个控制信号持续施加多少个环境步
- control_interval: 多少个环境步更新一次控制信号（需 >= action_duration，支持 'equal2action_duration'）
- num_of_interaction_per_agent_update: 每次神经网络权重更新累积多少个 interaction

学习率调度器（YOLOv8 风格）：
- lr_schedule: 'constant' / 'linear' / 'cosine'（余弦衰减）
- warmup_updates: 前 N 次更新进行学习率预热
- lr_final: 最终学习率比例（相对于初始 lr）

重要性采样（PPO 风格，可选）：
- use_importance_sampling: 是否启用 PPO 式重要性采样
- importance_sampling_clip_range: 比率裁剪范围（默认 0.2）

使用方法：
    from myafc.afc_supervision.engines import SimpleTrainer
    
    # 方式1: 从 YAML 配置文件
    trainer = SimpleTrainer("configs/simple_trainer.yaml")
    
    # 方式2: 从配置字典
    trainer = SimpleTrainer(config_dict)
    
    # 训练
    results = trainer.train()
"""
from __future__ import annotations
import os
import re
import math as pymath
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field as dataclass_field, asdict
from pathlib import Path

import yaml
import numpy as np
import torch

# PhiFlow 导入
from phi.torch.flow import math, Point, vec

from myafc.utils import get_logger

LOGGER = get_logger()


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class MLPConfig:
    """MLP 网络配置
    
    Attributes:
        hidden_dims: 隐藏层维度列表
            示例: [32], [64, 32], [128, 64, 32]
        amplitude_max: 输出幅值上限（经 tanh 缩放后）
            范围: 正浮点数，如 1.0, 4.0
        angle_deg_fixed: 固定射流角度（度）
            90°=向上, 270°=向下, None=由网络学习
        shared_output: 所有喷嘴是否共用同一组输出
            True: 网络输出 2 个值，所有喷嘴共用
            False: 网络输出 2*num_nozzles 个值
        normalize_input: 是否对输入特征进行归一化
        action_scale: 动作缩放方式
            'tanh': tanh(x) * scale (SAC 风格，严格边界，梯度饱和快)
            'clamp': clamp(x, low, high) (PPO 风格，硬裁剪)
            'softsign': x/(1+|x|) * scale (梯度衰减慢)
        relative_action: 是否使用相对动作模式
            False: 绝对动作模式（默认），网络直接输出目标幅值
            True: 相对动作模式，action = prev_action + tanh(network_output) * action_range
        deterministic_policy: 是否使用确定性策略模式（不随机采样，直接取均值）
            False = 随机策略（默认），从高斯分布采样动作
            True = 确定性策略，直接使用策略网络输出的均值作为动作
            注意：该设置在训练和验证阶段保持一致
        agent_model_path: Agent 模型初始化权重文件路径
            None: 不加载权重文件，随机初始化模型参数
            str: 指定 .pt/.pth 文件路径，加载预训练权重用于 fine-tune 或继续训练
    """
    hidden_dims: List[int] = dataclass_field(default_factory=lambda: [64, 32])
    amplitude_max: float = 4.0
    angle_deg_fixed: Optional[float] = 90.0
    shared_output: bool = True
    normalize_input: bool = True
    action_scale: str = 'clamp'
    relative_action: bool = False
    deterministic_policy: bool = False  # 是否使用确定性策略（不随机采样，直接取均值）
    agent_model_path: Optional[str] = None  # Agent 权重文件路径，None 表示随机初始化


@dataclass
class InputFeatureConfig:
    """输入特征配置
    
    Attributes:
        main_input_type: 主要探针输入类型（作为神经网络输入特征）
            可选值:
            - None: 不使用任何探针数据作为输入（仅使用其他特征如目标速度、历史幅值等）
            - 'target_probe': 仅使用 target_probe（由 loss.target_probe_position 定义）
            - 'probes': 使用 base_config_path 中的所有 probes + target_probe
        probe_fields: 探针数据字段选择
            可选值组合: ['u', 'v', 'p'] 中的任意子集
            示例: ['u', 'v'] 只使用速度, ['p'] 只使用压力, ['u', 'v', 'p'] 全部使用
            注意: 仅当 main_input_type 不为 None 时生效
        include_target_velocity: 是否包含目标速度 [u, v] 作为输入
        include_prev_amplitude_num: 使用多少个历史环境步的 MLP 输出幅值作为输入特征
            0 = 不使用历史幅值
            1 = 使用上一步幅值
            n = 使用最近 n 步幅值（按时间顺序排列，最旧在前）
            注意: 对于未施加射流控制的环境步，幅值默认取 0
        include_force_coeffs: 是否包含力系数 [Cd, Cl] 作为输入
        envstep_state: 历史 state 堆叠的环境步数（别名 stack_envstep）
            用于构建 agent 输入特征张量时，stack 多少个历史环境步的 state
            1 = 仅使用当前 state（默认，参考 DRLinFluids 方柱 AFC 案例）
            n = 使用最近 n 个环境步的 state（按时间顺序堆叠，最旧在前）
            历史 state 会与当前 state 在特征维度上拼接，扩大输入维度
            示例: envstep_state=5, 探针特征维度=3 → 输入维度扩展为 3*5=15
    """
    main_input_type: Optional[str] = 'target_probe'
    probe_fields: List[str] = dataclass_field(default_factory=lambda: ['u', 'v', 'p'])
    include_target_velocity: bool = True
    include_prev_amplitude_num: int = 1
    include_force_coeffs: bool = False
    envstep_state: int = 1  # 历史 state 堆叠步数（别名 stack_envstep），默认 1（仅当前 state）


@dataclass
class LossConfig:
    """损失函数配置
    
    Attributes:
        loss_type: 损失函数类型
            可选值:
            - 'probe_mse': 探针速度 MSE（推荐，完全可微分）
            - 'probe_mae': 探针速度 MAE
            - 'force_coeffs': 力系数损失 Cd^2 + cl_weight * Cl^2
        target_probe_position: 目标探针相对位置 [x, y]
            取值范围 [0~1]，相对于 domain.size
        target_velocity: 目标速度 [u, v]
            用于 probe_mse/probe_mae loss，结合 target_probe_position 使用
        velocity_weight: 速度项权重
        cd_target: Cd 目标值（仅 force_coeffs 模式）
        cl_target: Cl 目标值（仅 force_coeffs 模式）
        cl_weight: Cl 项权重（仅 force_coeffs 模式）
        
        # 正则化项说明：每步都施加，数值仅在动作更新步刷新。

        # ===== 控制信号幅值 L2 正则化 =====
        amplitude_l2_weight: 控制信号幅值 L2 正则化权重（惩罚大幅值）
            loss += amplitude_l2_weight * mean(amplitude^2)
            目标：抑制极端动作，提升泛化性。

        # ===== 时序平滑正则化 =====
        temporal_smoothness_weight: 控制信号时序平滑正则化权重
            惩罚相邻控制信号的剧烈变化：|a_t - a_{t-1}|^2
            有助于缓解 loss 大起大落，提高训练稳定性。
        action_rate_limit: 动作变化率限制（软约束）
            当 |a_t - a_{t-1}| > action_rate_limit 时施加额外惩罚
            None 或 0 表示不限制

        # ===== 策略稳定性正则化（防过冲）=====
        policy_stability_weight: 策略稳定性正则化权重
            惩罚当前输出与历史移动平均的偏差：|a_t - ā_EMA|^2
            当策略变化过快时增加惩罚，防止过冲导致 loss 持续上升
            推荐范围：0.001 ~ 0.1
        policy_stability_momentum: 移动平均动量系数
            α ∈ (0, 1]，ā_EMA = α * a_t + (1-α) * ā_EMA
            较小的值使移动平均更平滑，对历史依赖更强
            推荐范围：0.05 ~ 0.2

        # ===== 熵正则化（避免策略过早收敛，PPO/SAC 风格）=====
        entropy_weight: 熵正则化权重
            使用高斯分布的真实熵（参考 PPO/SAC）：H(π) = Σ 0.5*log(2πe*σ²)
            鼓励策略保持探索性，避免标准差过小导致策略过早确定性化
            loss -= entropy_weight * H(π)
            推荐范围：0.001 ~ 0.1

        # ===== KL 散度正则化（PPO 系列风格，稳定策略更新）=====
        kl_weight: KL 散度正则化权重
            核心思想（参考 PPO/TRPO 等）：限制新旧策略分布的差异，避免更新过大导致训练崩溃
            
            实现细节：
            - 维护 policy_net_old 副本，它是**上一次更新前***的网络权重
            - 在动作更新步，缓存 state 副本，确保新旧策略使用相同 state
            - 用 policy_net_old 和缓存的 state 计算旧动作分布，与当前分布计算 KL 散度
            - 每次 optimizer.step() **前**，将当前网络权重同步到 policy_net_old
            
            KL(π_new || π_old) = Σ [log(σ_old/σ_new) + (σ_new² + (μ_new-μ_old)²)/(2σ_old²) - 0.5]
            loss += kl_weight * mean(KL)
            推荐范围：0.001 ~ 0.1
        
        # ===== 最后 N 个 epoch 禁用正则化 =====
        disable_regularization_last_epochs: 在最后 N 个 epoch 禁用所有正则化项
            0 = 不禁用（默认），n = 最后 n 个 epoch 禁用
            'all' 或 n >= epochs = 所有 epoch 都禁用正则化
    """
    loss_type: str = 'probe_mse'
    target_probe_position: List[float] = dataclass_field(default_factory=lambda: [0.5316, 0.532])
    target_velocity: List[float] = dataclass_field(default_factory=lambda: [0.0, 0.233])
    velocity_weight: float = 1.0
    cd_target: float = 0.0
    cl_target: float = 0.0
    cl_weight: float = 1.0
    
    # 控制信号幅值 L2 正则化
    amplitude_l2_weight: float = 0.0

    # 时序平滑正则化
    temporal_smoothness_weight: float = 0.0
    action_rate_limit: Optional[float] = None
    
    # 策略稳定性正则化（防过冲）
    policy_stability_weight: float = 0.0
    policy_stability_momentum: float = 0.1

    # 熵正则化（PPO 风格，使用高斯分布的真实熵）
    entropy_weight: float = 0.0
    
    # KL 散度正则化（PPO 系列风格，稳定策略更新）
    kl_weight: float = 0.0
    
    # ===== 最后 N 个 epoch 禁用正则化 =====
    # 在最后 N 个 epoch 禁用所有正则化项，让策略充分收敛
    # 取值: 0 = 不禁用（默认），n = 最后 n 个 epoch 禁用，'all' = 所有 epoch 禁用
    # 当 n >= epochs 时等效于 'all'
    disable_regularization_last_epochs: Union[int, str] = 0


@dataclass
class InletConfig:
    """入口边界配置
    
    Attributes:
        mode: 入口边界模式
            可选值:
            - 'zero_gradient': 零梯度边界
            - 'fixed': 固定速度入口
            - 'sinusoidal': 正弦变化速度入口
        fixed_velocity: 固定速度值 [Ux, Uy]（仅 fixed 模式）
        amplitude: 正弦振幅（仅 sinusoidal 模式）
        frequency: 正弦频率 Hz（仅 sinusoidal 模式）
        phase_deg: 正弦初始相位（度）（仅 sinusoidal 模式）
        offset: 正弦偏移量 [Ux0, Uy0]（仅 sinusoidal 模式）
    """
    mode: str = 'zero_gradient'
    fixed_velocity: List[float] = dataclass_field(default_factory=lambda: [0.0, 0.0])
    amplitude: float = 0.5
    frequency: float = 0.1
    phase_deg: float = 0.0
    offset: List[float] = dataclass_field(default_factory=lambda: [0.0, 0.0])


@dataclass
class TrainingConfig:
    """训练配置
    
    Attributes:
        num_interaction_per_mini_episode: 每个 mini episode 的交互次数（控制信号更新次数）
            总环境步数通过公式计算: env_warmup_envsteps + num_interaction_per_mini_episode * control_interval
        action_duration: 一个控制信号持续施加多少个环境步
            例如 action_duration=5 表示每个控制信号连续施加 5 步
        control_interval: 多少个环境步更新一次控制信号
            可选值:
            - 整数: 具体的环境步数，必须 >= action_duration
            - 'equal2action_duration': 自动设置为与 action_duration 相同
            例如 control_interval=5 表示每 5 步更新一次 MLP 输出
        num_of_interaction_per_agent_update: 每次神经网络权重更新累积多少个 interaction
            一个 interaction = control_interval 个环境步
            实际累积环境步数 (envstep_per_value) = num_of_interaction_per_agent_update * control_interval
            例如 num_of_interaction_per_agent_update=3, control_interval=10 → 累积 30 步 loss 后更新参数
        loss_weights: 每个环境步 loss 的权重
            可选值:
            - 'equal': 等权重（默认），每步权重 = 1/n
            - 'exponential': 基于 interaction 的指数衰减权重
                每个 interaction 内：射流施加阶段权重=1，停止后按 γ^i 衰减
                多个 interaction 在环境步维度对齐后求平均
            - [w1, w2, ...]: 自定义权重列表，长度需等于 envstep_per_value
        loss_gamma: 指数衰减因子（仅 loss_weights='exponential' 时生效）
            范围 (0, 1]，越小衰减越快
        learning_rate: 初始学习率
        lr_final: 最终学习率（学习率衰减的目标值）
            默认 0.01 表示最终 lr = 0.01 * learning_rate
            范围 (0, 1]，越小最终学习率越低
        lr_schedule: 学习率调度策略
            可选值:
            - 'constant': 恒定学习率（不衰减）
            - 'linear': 线性衰减，lr = lr0 * (1 - progress) + lr_final * lr0 * progress
            - 'cosine': 余弦衰减（YOLOv8 风格），lr = lr_final + 0.5*(1-lr_final)*(1+cos(π*progress))
        warmup_updates: 学习率预热的更新次数
            在前 warmup_updates 次参数更新中，学习率从 warmup_bias_lr 线性增长到 learning_rate
            0 表示不使用预热
        warmup_bias_lr: 预热阶段的初始学习率比例
            范围 [0, 1]，实际初始 lr = warmup_bias_lr * learning_rate
            YOLOv8 默认 0.1
        warmup_momentum: 预热阶段的动量初始值（仅 SGD 优化器有效）
            优化器动量从 warmup_momentum 线性增长到目标动量
            默认 0.8，YOLOv8 默认 0.8
        optimizer: 优化器类型
            可选值: 'adam', 'sgd', 'rmsprop', 'adamw'
        optimizer_momentum: 优化器动量（仅 SGD/RMSprop 有效）
            默认 0.937（YOLOv8 默认）
        optimizer_weight_decay: 权重衰减
            默认 0.0005（YOLOv8 默认）
        grad_clip: 梯度裁剪阈值
            None 或 0 表示不裁剪
        random_seed: 随机种子
            None 表示不固定
        env_warmup_envsteps: 环境预热的环境步数（只执行仿真，不更新神经网络权重）
        warmup_update_controls: 预热阶段是否更新控制信号
        warmup_apply_jets: 预热阶段是否施加射流
        validation_steps: 训练后验证步数
        verbose: 日志详细程度
            可选值: 'DEBUG', 'INFO', 'WARNING', 'ERROR'（默认 'INFO'）
            注意: Trainer 的 verbose 会覆盖 Runner 和 Solver 的 verbose 配置
        
        # ===== 重要性采样（PPO 风格）=====
        use_importance_sampling: 是否在 reward 上启用 PPO 式重要性采样
            启用后，reward 会根据新旧策略的概率比进行加权：
            weighted_reward = clip(ratio, 1-clip_range, 1+clip_range) * reward
            其中 ratio = π_new(a|s) / π_old(a|s)
        importance_sampling_clip_range: 重要性采样比率裁剪范围
            PPO 默认 0.2，即 ratio 被裁剪到 [0.8, 1.2]
            较小的值使策略更新更保守
        
        checkpoint_epoch_interval: Epoch 级别检查点保存间隔
            每隔多少个 epoch 保存一次检查点
            0 = 不按 epoch 保存（仅按环境步 checkpoint_interval 和最终保存）
            1 = 每个 epoch 结束后都保存
            n = 每 n 个 epoch 保存一次
        
        # ===== 多轮训练（Epoch / Mini-Episode）=====
        epoch: 训练轮数（别名: num_mini_episode_per_subenv）
            每轮（epoch）完成一个完整的 mini episode 训练循环
            轮与轮之间会进行全面 reset（除策略神经网络外）：
            - 重置 solver 流场状态
            - 清空 state history buffer
            - 清空 amplitude history
            - 重置其他统计数据
            
            典型用法：
            - epoch=1: 单轮训练（默认，向后兼容）
            - epoch>1: 多轮训练，策略网络在不同初始状态下累积学习
            
            注意: DRL 文献中有时称为 "number of episodes per environment"
            默认值 1（单轮训练）
        
    """
    epoch: int = 1  # 训练轮数（alias: num_mini_episode_per_subenv）
    num_interaction_per_mini_episode: int = 100  # 默认 100 次交互
    action_duration: int = 1
    control_interval: Union[int, str] = 'equal2action_duration'  # 支持 'equal2action_duration' 或整数
    num_of_interaction_per_agent_update: int = 3  # 每次权重更新累积的 interaction 数
    loss_weights: Union[str, List[float]] = 'exponential'
    loss_gamma: float = 0.99
    learning_rate: float = 1e-3
    lr_final: float = 0.01  # 最终学习率比例（相对于初始 lr）
    lr_schedule: str = 'cosine'  # 学习率调度策略: 'constant', 'linear', 'cosine'
    warmup_updates: int = 0  # 预热更新次数
    warmup_bias_lr: float = 0.1  # 预热初始学习率比例
    warmup_momentum: float = 0.8  # 预热动量初始值
    optimizer: str = 'adam'
    optimizer_momentum: float = 0.937  # 优化器动量
    optimizer_weight_decay: float = 0.0005  # 权重衰减
    grad_clip: Optional[float] = 1.0
    random_seed: Optional[int] = 42
    env_warmup_envsteps: int = 0
    warmup_update_controls: bool = False
    warmup_apply_jets: bool = False
    validation_steps: int = 50
    verbose: str = 'INFO'
    # 重要性采样配置
    use_importance_sampling: bool = False  # 是否启用 PPO 式重要性采样
    importance_sampling_clip_range: float = 0.2  # 重要性采样比率裁剪范围
    # Epoch 级别检查点保存
    checkpoint_epoch_interval: int = 0  # 每隔多少个 epoch 保存一次检查点，0 表示不按 epoch 保存
    # 覆盖 solver 环境配置
    smoothing_alpha: Optional[float] = None  # 射流时间平滑系数（覆盖 solver 配置），None=使用 solver 配置


@dataclass
class OutputConfig:
    """输出配置
    
    Attributes:
        project: 项目根目录
        name: 实验名称前缀（自动添加编号如 exp, exp2, exp3）
        checkpoint_interval: 检查点保存间隔（环境步）
    """
    project: str = 'runs/train'
    name: str = 'exp'
    checkpoint_interval: int = 100


@dataclass
class SimpleTrainerConfig:
    """SimpleTrainer 完整配置
    
    多环境训练支持:
        - num_envs: 总环境数量
          - 1: 仅使用基础环境，无派生
          - n (n>1): 1个基础环境 + (n-1)个派生环境 = n个环境
        - env_variants: 派生环境的配置变体列表，长度必须为 num_envs-1
          - 每个变体会覆盖 base_config_path 中的对应配置
          - 例如: num_envs=3, env_variants=[{...}, {...}]
            → Env 0: 基础配置
            → Env 1: 基础配置 + env_variants[0]
            → Env 2: 基础配置 + env_variants[1]
    """
    base_config_path: str = ''  # Solver 配置文件路径（必需）
    num_envs: int = 1  # 总环境数量（1=仅基础环境，n>1=1个基础 + n-1个派生）
    env_variants: Optional[List[Dict]] = None  # 派生环境配置变体，长度 = num_envs-1
    mlp: MLPConfig = dataclass_field(default_factory=MLPConfig)
    input_features: InputFeatureConfig = dataclass_field(default_factory=InputFeatureConfig)
    loss: LossConfig = dataclass_field(default_factory=LossConfig)
    inlet: InletConfig = dataclass_field(default_factory=InletConfig)
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)
    output: OutputConfig = dataclass_field(default_factory=OutputConfig)


def _convert_scientific_notation(obj):
    """递归转换配置中的科学计数法字符串为浮点数
    
    YAML 有时会将 "1e-3" 解析为字符串而非浮点数，
    此函数递归遍历配置字典/列表，将科学计数法字符串转换为浮点数。
    
    支持的格式: 1e-3, 1E-3, 1.5e-4, -2e3, +3.14e+2 等
    """
    import re
    # 匹配科学计数法的正则表达式
    scientific_pattern = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$')
    
    if isinstance(obj, dict):
        return {k: _convert_scientific_notation(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_scientific_notation(item) for item in obj]
    elif isinstance(obj, str):
        # 尝试匹配科学计数法
        if scientific_pattern.match(obj.strip()):
            try:
                return float(obj)
            except ValueError:
                return obj
        return obj
    else:
        return obj


def _parse_trainer_config(cfg_dict: Dict) -> SimpleTrainerConfig:
    """从字典解析配置
    
    自动处理：
    1. 科学计数法字符串转换为浮点数（如 "1e-3" -> 0.001）
    2. 嵌套配置解析
    3. 多环境配置解析
    """
    # 首先转换科学计数法
    cfg_dict = _convert_scientific_notation(cfg_dict)
    
    config = SimpleTrainerConfig()
    config.base_config_path = cfg_dict.get('base_config_path', '')
    config.num_envs = cfg_dict.get('num_envs', 1)
    config.env_variants = cfg_dict.get('env_variants', None)
    
    for key, cls in [('mlp', MLPConfig), ('input_features', InputFeatureConfig),
                     ('loss', LossConfig), ('inlet', InletConfig),
                     ('training', TrainingConfig), ('output', OutputConfig)]:
        if key in cfg_dict:
            setattr(config, key, cls(**cfg_dict[key]))
    return config


def _increment_path(path: Path, sep: str = '') -> Path:
    """自增路径，如 exp -> exp2 -> exp3 ..."""
    path = Path(path)
    if not path.exists():
        return path
    
    dirs = [d.name for d in path.parent.glob(f"{path.name}{sep}*") if d.is_dir()]
    matches = [re.search(rf"{path.name}{sep}(\d+)", d) for d in dirs]
    nums = [int(m.group(1)) for m in matches if m]
    n = max(nums) + 1 if nums else 2
    return path.parent / f"{path.name}{sep}{n}"


# ============================================================
# CustomMLPPolicy - 简洁的训练专用策略类（直接继承 MLPPolicy）
# ============================================================
from myafc.env_kernels.solvers.fluid_control.control_policy.mlp import MLPPolicy


class CustomMLPPolicy(MLPPolicy):
    """
    训练专用的 MLP 策略类（直接继承 MLPPolicy）
    
    相比基类 MLPPolicy，增加了可配置的输入特征：
    - target_velocity: (目标探针处的)目标速度 [u, v]（可选）
    - prev_amplitudes: 历史控制幅值序列（可选，长度由 include_prev_amplitude_num 控制）
    - probe_fields: 指定提取探针数据的哪些字段（u/v/p 的任意组合）
    - force_coeffs: 力系数 [Cd, Cl]（可选）
    
    使用示例:
        policy = CustomMLPPolicy(
            main_input_type='target_probe',  # None, 'target_probe', 或 'probes'
            probe_fields=['u', 'v'],    # 只使用速度分量
            hidden_dims=[64, 32],
            num_nozzles=2,
            include_target_velocity=True,
            include_prev_amplitude_num=3,  # 使用最近3步的幅值
            include_force_coeffs=True,
            envstep_state=1,  # 历史 state 堆叠步数（DRLinFluids 默认值）
            device='cuda'
        )
    """
    
    def __init__(
        self,
        main_input_type: Optional[str] = 'target_probe',
        probe_fields: Optional[List[str]] = None,
        hidden_dims: List[int] = None,
        num_nozzles: int = 1,
        shared_output: bool = True,
        amplitude_max: float = 1.0,
        angle_deg_fixed: Optional[float] = None,
        normalize_input: bool = False,
        device: str = 'cpu',
        action_scale: str = 'clamp',  # 动作缩放参数（传递给基类）
        relative_action: bool = False,
        # CustomMLPPolicy 特有参数
        include_target_velocity: bool = True,
        include_prev_amplitude_num: int = 1,
        include_force_coeffs: bool = True,
        envstep_state: int = 1,  # 历史 state 堆叠步数（别名 stack_envstep）
    ):
        """
        初始化 CustomMLPPolicy
        
        Args:
            main_input_type: 主要探针输入类型
                - None: 不使用任何探针数据作为输入
                - 'target_probe': 仅使用 target_probe
                - 'probes': 使用所有 probes + target_probe
            probe_fields: 探针数据字段选择 ['u', 'v', 'p'] 的任意子集（仅当 main_input_type 不为 None 时生效）
            hidden_dims: 隐藏层维度列表
            num_nozzles: 喷嘴数量
            shared_output: 是否共享输出层
            amplitude_max: 最大幅值
            angle_deg_fixed: 固定角度（度），None 表示动态
            normalize_input: 是否归一化输入
            device: 计算设备
            action_scale: 动作缩放方式 'tanh' / 'clamp' / 'softsign'
            relative_action: 是否使用相对动作模式（网络输出为增量）
            include_target_velocity: 是否包含目标速度输入
            include_prev_amplitude_num: 使用多少个历史幅值作为输入（0=不使用）
            include_force_coeffs: 是否包含力系数输入
            envstep_state: 历史 state 堆叠的环境步数（别名 stack_envstep）
                1 = 仅使用当前 state（默认，参考 DRLinFluids 方柱 AFC 案例）
                n = 使用最近 n 个环境步的 state 堆叠
        """
        # 适配基类的 input_type（基类使用 'probes'/'force_coeffs'/'both'）
        # _build_input 根据 main_input_type 选择输入特征。
        super().__init__(
            input_type='probes',  # 基类参数，实际逻辑在 _build_input 中重写
            hidden_dims=hidden_dims or [64, 32],
            num_nozzles=num_nozzles,
            shared_output=shared_output,
            amplitude_max=amplitude_max,
            angle_deg_fixed=angle_deg_fixed,
            action_scale=action_scale,
            normalize_input=normalize_input,
            relative_action=relative_action,
            device=device,
        )
        
        # CustomMLPPolicy 特有属性
        self._main_input_type = main_input_type  # None, 'target_probe', 或 'probes'
        self._probe_fields = probe_fields or ['u', 'v', 'p']
        self._include_target_velocity = include_target_velocity
        self._include_prev_amplitude_num = include_prev_amplitude_num
        self._include_force_coeffs = include_force_coeffs
        self._envstep_state = envstep_state  # 历史 state 堆叠步数（别名 stack_envstep）
        self._target_velocity = [0.0, 0.0]  # 由 Trainer 设置
        self._prev_amplitudes = []  # 历史幅值列表，由 Trainer 设置
    
    def set_target_velocity(self, velocity: List[float]):
        """设置目标速度（由 Trainer 调用）"""
        self._target_velocity = list(velocity) if velocity else [0.0, 0.0]
    
    def set_prev_amplitudes(self, amplitudes: List[float]):
        """设置历史幅值序列（由 Trainer 调用）
        
        Args:
            amplitudes: 历史幅值列表，按时间顺序排列（最旧在前）
                长度应等于 include_prev_amplitude_num
        """
        self._prev_amplitudes = list(amplitudes) if amplitudes else []
    
    @property
    def envstep_state(self) -> int:
        """获取历史 state 堆叠步数（别名 stack_envstep）"""
        return self._envstep_state
    
    def _extract_probe_data(self, probe: Dict) -> List[float]:
        """从探针数据字典中提取指定字段"""
        result = []
        for field in self._probe_fields:
            if field in probe:
                result.append(probe[field])
            else:
                result.append(0.0)
        return result
    
    def _extract_probe_features_from_state(self, state: Dict) -> List[float]:
        """从单个 state 中提取探针相关特征
        
        仅提取探针数据和力系数，不包含目标速度和历史幅值
        （这些在历史堆叠中不需要重复）
        """
        features = []
        
        # 探针数据
        if self._main_input_type is not None:
            probes = state.get('probes', {})
            
            if self._main_input_type == 'target_probe':
                if 'target' in probes:
                    p = probes['target']
                    if isinstance(p, dict):
                        features.extend(self._extract_probe_data(p))
                    elif isinstance(p, (list, tuple)):
                        features.extend(p[:len(self._probe_fields)])
            else:  # 'probes'
                for name, p in probes.items():
                    if isinstance(p, dict):
                        features.extend(self._extract_probe_data(p))
                    elif isinstance(p, (list, tuple)):
                        features.extend(p[:len(self._probe_fields)])
        
        # 力系数
        if self._include_force_coeffs:
            fc = state.get('force_coeffs', {})
            features.extend([fc.get('Cd', 0), fc.get('Cl', 0)])
        
        return features

    def _build_input(self, state: Dict, state_history: Optional[List[Dict]] = None) -> List[float]:
        """
        构建神经网络输入向量（重写基类方法）
        
        Args:
            state: 当前环境状态字典，可能包含 '_state_history' 字段
            state_history: 历史状态列表（外部显式传入时优先使用）
                如未传入，则尝试从 state['_state_history'] 获取
                长度应为 envstep_state - 1，不足时自动零填充
                按时间顺序排列（最旧在前）
        
        输入顺序：
        1. target_velocity [u, v] (可选，由 include_target_velocity 控制)
        2. prev_amplitudes (可选，由 include_prev_amplitude_num 控制，从 state 中获取)
        3. 历史 state 堆叠特征 (由 envstep_state 控制，包含探针数据和力系数)
           - 当 envstep_state=1 时，仅使用当前 state（默认行为，与 DRLinFluids 一致）
           - 当 envstep_state>1 时，堆叠最近 n 个环境步的探针+力系数特征
           - 堆叠顺序：最旧在前，当前在后（时间序列自然顺序）
        
        注意：
        - 历史 state 不包含 target_velocity 和 prev_amplitudes（这些是全局信息）
        - 历史 state 特征维度 = (探针特征数 + 力系数数) * envstep_state
        """
        inputs = []
        
        # 1. 目标速度（可选，优先从 state 获取，否则使用缓存值）
        if self._include_target_velocity:
            target_vel = state.get('target_velocity', self._target_velocity)
            if isinstance(target_vel, (list, tuple)) and len(target_vel) >= 2:
                inputs.extend([target_vel[0], target_vel[1]])
            else:
                inputs.extend(self._target_velocity)
        
        # 2. 历史幅值序列（可选，从 state 中获取）
        if self._include_prev_amplitude_num > 0:
            prev_amps = state.get('prev_amplitudes', [])
            # 确保长度为 include_prev_amplitude_num
            if len(prev_amps) < self._include_prev_amplitude_num:
                # 不足时在前面用 0 填充
                prev_amps = [0.0] * (self._include_prev_amplitude_num - len(prev_amps)) + list(prev_amps)
            inputs.extend(prev_amps[:self._include_prev_amplitude_num])
        
        # 3. 探针数据 + 力系数（支持历史 state 堆叠）
        if self._envstep_state <= 1:
            # 标准模式：仅使用当前 state（默认行为，与 DRLinFluids 一致）
            inputs.extend(self._extract_probe_features_from_state(state))
        else:
            # 历史堆叠模式：优先使用显式传入的 state_history，否则从 state 字典获取
            if state_history is not None:
                history_states = list(state_history)
            else:
                history_states = list(state.get('_state_history', []))
            
            # 确保长度等于 envstep_state - 1（历史步数）
            num_history_needed = self._envstep_state - 1
            current_features = self._extract_probe_features_from_state(state)
            feature_dim = len(current_features)
            
            if len(history_states) < num_history_needed:
                # 历史不足时，用零值填充（在前面填充）
                num_padding = num_history_needed - len(history_states)
                for _ in range(num_padding):
                    inputs.extend([0.0] * feature_dim)
            
            # 添加历史 state 特征（最旧在前，只取最近的 num_history_needed 个）
            for hist_state in history_states[-num_history_needed:]:
                inputs.extend(self._extract_probe_features_from_state(hist_state))
            
            # 最后添加当前 state 特征
            inputs.extend(current_features)
        
        return inputs


# ============================================================
# SimpleTrainer 主类
# ============================================================
class SimpleTrainer:
    """基于 PhiFlow 可微分仿真的 MLP 训练器
    
    核心特性：
    1. 使用 PyTorch 原生自动微分（端到端可微分）
    2. 复用 MLPPolicy 作为控制策略（与 JetObstacle 无缝对接）
    3. 通过 solver.step(update_controls, apply_jets) 简化训练流程
    
    Example:
        >>> trainer = SimpleTrainer("config.yaml")
        >>> results = trainer.train()
    """
    
    def __init__(self, config: Union[str, Path, Dict, SimpleTrainerConfig]):
        """初始化训练器
        
        Args:
            config: 配置来源，支持以下类型：
                - str/Path: YAML 配置文件路径
                - Dict: 配置字典
                - SimpleTrainerConfig: 配置对象
        """
        # 解析配置
        self.cfg = self._parse_config(config)
        
        # 初始化训练状态（在其他 setup 之前，因为 _build_state 可能引用）
        self.current_step = 0
        
        # 训练数据记录（每步记录）
        self.step_losses = []          # 每个 env_step 的 loss 值
        self.step_velocities = []      # 每个 env_step 的目标位置速度 [(u, v), ...]
        self.step_pressures = []       # 每个 env_step 的目标位置压力
        self.amplitude_history = []    # 每个 env_step 的控制幅值
        
        # 训练汇总数据（每次参数更新记录）
        self.loss_history = []         # 每次更新的加权 total_loss
        self.primary_loss_history = []  # 每次更新的加权 primary_loss（主损失，用于 fitness）
        self.reg_loss_history = []      # 每次更新的加权 reg_loss（正则化损失之和）
        
        # 初始化
        self._setup_exp_dir()
        self._setup_logging()
        self._setup_seed()
        self._setup_solver()
        
        # 目标探针位置（在 solver 初始化后，policy 初始化前设置）
        target_pos = self.cfg.loss.target_probe_position
        Lx, Ly = self.solver.cfg.domain.size
        self.target_point = Point(vec(x=target_pos[0] * Lx, y=target_pos[1] * Ly))
        
        # 加载 base_config 中的所有探针位置（用于 main_input_type='probes' 模式）
        self._setup_probe_points()
        
        # 预缓存目标张量（避免每次 _compute_loss 时重复创建）
        self._target_u = torch.tensor(self.cfg.loss.target_velocity[0], device=self.device)
        self._target_v = torch.tensor(self.cfg.loss.target_velocity[1], device=self.device)
        self._cd_target = torch.tensor(self.cfg.loss.cd_target, device=self.device)
        self._cl_target = torch.tensor(self.cfg.loss.cl_target, device=self.device)
        
        # 正则化项缓存（在动作更新步计算并缓存，非动作更新步复用）
        # 这些张量保持对计算图的引用，确保梯度链连通
        self._cached_reg_amp_l2: Optional[torch.Tensor] = None
        self._cached_reg_smoothness: Optional[torch.Tensor] = None
        self._cached_reg_stability: Optional[torch.Tensor] = None
        self._cached_reg_entropy: Optional[torch.Tensor] = None
        self._cached_reg_kl: Optional[torch.Tensor] = None
        
        # Epoch 追踪（用于最后 N 个 epoch 禁用正则化等功能）
        self.current_epoch: int = 0  # 当前 epoch 索引（0-based）
        self.num_epochs: int = self.cfg.training.epoch  # 总 epoch 数
        
        self._setup_policy()
        self._setup_optimizer()
        self._setup_tensorboard()
        self._setup_validation_runner()
    
    def _parse_config(self, config) -> SimpleTrainerConfig:
        """解析配置（支持多种输入类型）"""
        if isinstance(config, SimpleTrainerConfig):
            return config
        elif isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg_dict = yaml.safe_load(f)
            return _parse_trainer_config(cfg_dict)
        elif isinstance(config, dict):
            return _parse_trainer_config(config)
        else:
            raise TypeError(f"不支持的配置类型: {type(config)}")
    
    def _setup_exp_dir(self):
        """设置实验目录"""
        base_path = Path(self.cfg.output.project) / self.cfg.output.name
        self.exp_dir = _increment_path(base_path)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'configs').mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """设置日志"""
        from myafc.utils.logging import setup_file_logging, set_log_level
        
        # 设置日志级别
        verbose = getattr(self.cfg.training, 'verbose', 'INFO')
        set_log_level(verbose)
        
        # 设置文件日志
        log_path = self.exp_dir / 'train.log'
        setup_file_logging(str(log_path), append=False)
        LOGGER.info(f"[Trainer] 实验目录: {self.exp_dir}")
        LOGGER.info(f"[Trainer] 日志级别: {verbose}")
    
    def _setup_seed(self):
        """设置随机种子"""
        seed = self.cfg.training.random_seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            math.seed(seed)
            LOGGER.info(f"[Trainer] 随机种子: {seed}")
    
    def _setup_solver(self):
        """初始化 solver（支持单环境和多环境模式）"""
        base_cfg_path = self.cfg.base_config_path
        if not os.path.isabs(base_cfg_path):
            configs_dir = Path(__file__).parent.parent.parent / 'configs'
            base_cfg_path = str(configs_dir / base_cfg_path)
        
        if not os.path.exists(base_cfg_path):
            raise FileNotFoundError(f"Solver 配置文件不存在: {base_cfg_path}")
        
        with open(base_cfg_path, 'r', encoding='utf-8') as f:
            base_cfg = yaml.safe_load(f)
        
        solver_cfg = base_cfg.get('solver', base_cfg)
        self._override_solver_config(solver_cfg)
        
        # 保存修改后的配置
        modified_cfg_path = self.exp_dir / 'configs' / 'solver_config.yaml'
        with open(modified_cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(base_cfg, f, sort_keys=False, allow_unicode=True)
        
        # 根据 num_envs 选择单环境或并行多实例 solver
        n_envs = self.cfg.num_envs
        self.n_envs = n_envs
        
        if n_envs > 1:
            # 多环境模式：使用 ParallelSolverManager 创建多个独立 Solver 实例
            from myafc.env_kernels.solvers.parallel_solver_manager import ParallelSolverManager
            self.solver = ParallelSolverManager(
                base_cfg_path=str(modified_cfg_path),
                n_envs=n_envs,
                env_variants=self.cfg.env_variants,  # 每个环境的差异化配置
            )
            LOGGER.info(f"[Trainer] ParallelSolverManager 初始化 | 环境数: {n_envs}")
        else:
            # 单环境模式：使用原始 KarmanTorchSolver
            from myafc.env_kernels.solvers.karman_torch_solver import KarmanTorchSolver
            self.solver = KarmanTorchSolver(str(modified_cfg_path))
            LOGGER.info(f"[Trainer] Solver 初始化完成")
        
        self.device = self.solver.device if hasattr(self.solver, 'device') else self.solver._device
        LOGGER.info(f"[Trainer] 设备: {self.device}")
    
    def _override_solver_config(self, solver_cfg: Dict):
        """覆盖 solver 配置
        
        重要: Trainer 的 verbose 会覆盖 Solver 的 verbose 配置，
        确保日志级别由上层组件统一控制。
        """
        inlet_cfg = self.cfg.inlet
        
        # verbose 覆盖: trainer.verbose 覆盖 solver.runtime.verbose
        if 'runtime' not in solver_cfg:
            solver_cfg['runtime'] = {}
        solver_cfg['runtime']['verbose'] = self.cfg.training.verbose
        
        if 'boundary' not in solver_cfg:
            solver_cfg['boundary'] = {}
        if 'velocity' not in solver_cfg['boundary']:
            solver_cfg['boundary']['velocity'] = {}
        
        if inlet_cfg.mode == 'zero_gradient':
            solver_cfg['boundary']['velocity']['x-'] = {'type': 'zero-gradient'}
        elif inlet_cfg.mode == 'fixed':
            solver_cfg['boundary']['velocity']['x-'] = {'type': 'inflow', 'value': inlet_cfg.fixed_velocity}
        elif inlet_cfg.mode == 'sinusoidal':
            # sinusoidal 模式：初始使用 offset 值，运行时由 _update_sinusoidal_inlet() 更新
            solver_cfg['boundary']['velocity']['x-'] = {'type': 'inflow', 'value': inlet_cfg.offset}
            # 保存 sinusoidal 参数供运行时使用
            self._sinusoidal_params = {
                'amplitude': inlet_cfg.amplitude,
                'frequency': inlet_cfg.frequency,
                'phase_rad': pymath.radians(inlet_cfg.phase_deg),
                'offset': inlet_cfg.offset,
            }
            LOGGER.info(f"[Trainer] sinusoidal 入口模式: amp={inlet_cfg.amplitude}, "
                       f"freq={inlet_cfg.frequency}Hz, phase={inlet_cfg.phase_deg}°, "
                       f"offset={inlet_cfg.offset}")
        
        # 确保启用射流和可微分模式
        if 'jets' not in solver_cfg:
            solver_cfg['jets'] = {}
        solver_cfg['jets']['enabled'] = True
        
        if 'runtime' not in solver_cfg:
            solver_cfg['runtime'] = {}
        solver_cfg['runtime']['differentiable'] = True
    
    def _update_sinusoidal_inlet(self, time: float):
        """
        更新 sinusoidal 入口边界条件（每步调用）
        
        公式: U(t) = offset + amplitude * sin(2π * frequency * t + phase)
        
        Args:
            time: 当前物理时间
        """
        if not hasattr(self, '_sinusoidal_params'):
            return
        
        params = self._sinusoidal_params
        # 计算当前时刻的入口速度
        phase = 2 * pymath.pi * params['frequency'] * time + params['phase_rad']
        sin_factor = pymath.sin(phase)
        
        # 计算速度分量
        Ux = params['offset'][0] + params['amplitude'] * sin_factor
        Uy = params['offset'][1]  # y 分量通常保持为 offset
        
        # 更新 solver 的边界条件
        self.solver.vel_boundary['x-'] = vec(x=Ux, y=Uy)
    
    def _setup_policy(self):
        """初始化 CustomMLPPolicy（继承 MLPPolicy）并设置到 JetObstacle"""
        mlp_cfg = self.cfg.mlp
        feat_cfg = self.cfg.input_features
        
        # 使用简化的 CustomMLPPolicy（直接继承 MLPPolicy）
        self.policy = CustomMLPPolicy(
            main_input_type=feat_cfg.main_input_type,
            probe_fields=feat_cfg.probe_fields,
            hidden_dims=mlp_cfg.hidden_dims,
            num_nozzles=self.solver.jet_obstacle.num_nozzles if self.solver.jet_obstacle else 1,
            shared_output=mlp_cfg.shared_output,
            amplitude_max=mlp_cfg.amplitude_max,
            angle_deg_fixed=mlp_cfg.angle_deg_fixed,
            normalize_input=mlp_cfg.normalize_input,
            action_scale=mlp_cfg.action_scale,
            relative_action=mlp_cfg.relative_action,
            device=self.device,
            # CustomMLPPolicy 额外参数
            include_target_velocity=feat_cfg.include_target_velocity,
            include_prev_amplitude_num=feat_cfg.include_prev_amplitude_num,
            include_force_coeffs=feat_cfg.include_force_coeffs,
            envstep_state=feat_cfg.envstep_state,  # 历史 state 堆叠步数
        )
        
        # ========== 历史状态管理 ==========
        # 1. envstep_state 功能：用于堆叠多步历史状态作为神经网络输入
        self._state_history_buffer = []  # 连续更新的历史状态队列
        # 2. KL 散度计算：用于保存动作更新步的状态快照
        self._current_state_for_kl = None  # 包含 '_state_history' 字段的完整状态快照
        
        # 设置到 JetObstacle
        if self.solver.jet_obstacle is not None:
            self.solver.jet_obstacle.policy = self.policy
            # 设置确定性策略模式（不随机采样，直接取均值）
            self.solver.jet_obstacle.deterministic = self.cfg.mlp.deterministic_policy
            
            # 覆盖 smoothing_alpha（如果 trainer 配置中指定了）
            if hasattr(self.cfg.training, 'smoothing_alpha') and self.cfg.training.smoothing_alpha is not None:
                self.solver.jet_obstacle.jet_config.smoothing_alpha = self.cfg.training.smoothing_alpha
                LOGGER.info(f"[Trainer] 覆盖 jets.smoothing_alpha: {self.cfg.training.smoothing_alpha}")
        
        self.policy.train()
        
        # 加载预训练权重（如果指定了 agent_model_path）
        if mlp_cfg.agent_model_path is not None:
            self._load_agent_weights(mlp_cfg.agent_model_path)
        
        # 初始化 policy_net_old（用于 KL 散度正则化）
        # policy_net_old 保存的是**上一次更新前**的网络权重
        # 初始时与 policy 权重相同，之后每次 optimizer.step() 前同步
        self.policy_net_old = None  # 延迟初始化，在网络构建后创建
        
        LOGGER.info(f"[Trainer] CustomMLPPolicy: hidden={mlp_cfg.hidden_dims}, "
                   f"max_amp={mlp_cfg.amplitude_max}, shared={mlp_cfg.shared_output}")
    
    def _reset_state_history_buffer(self):
        """重置历史状态缓冲区和相关缓存（在环境 reset 或新仿真开始时调用）
        
        清空以下缓存，确保新的仿真运行从干净状态开始：
        1. _state_history_buffer: 用于 envstep_state 功能的历史状态队列
        2. _current_state_for_kl: 用于 KL 散度计算的状态快照
        
        这在以下情况下必须调用：
        1. solver.reset() 后
        2. 新的训练/验证 episode 开始时
        """
        # 重置 envstep_state 功能的历史缓冲区
        self._state_history_buffer = []
        # 重置 KL 散度计算的状态快照
        self._current_state_for_kl = None
        LOGGER.debug("[Trainer] 历史状态缓冲区已重置")
    
    def _reset_env(self, reason: str = 'epoch'):
        """进行全面环境重置（除策略神经网络及其训练状态外）
        
        通用的环境状态重置方法，用于以下场景：
        - 多轮训练时，每轮（epoch）开始前重置环境
        - 验证阶段开始前重置环境
        - 其他需要从干净状态开始的场景
        
        重置内容包括：
        1. solver 流场状态（重置到初始状态）
        2. 历史状态缓冲区（envstep_state 和 KL 散度相关）
        3. 策略稳定性 EMA（_policy_output_ema）
        4. 正则化项缓存（_cached_reg_*）
        5. 重要性采样比率缓存
        6. policy 内部状态（prev_amplitudes）
        7. 分离计算图（确保新阶段的梯度不会传播到旧阶段）
        
        不重置：
        - 策略神经网络权重（self.policy.network）
        - 优化器状态（self.optimizer）
        - policy_net_old（KL 散度参考网络）
        - 累积的训练历史数据（loss_history、step_losses 等，跨 epoch 保留用于分析）
        - TensorBoard writer
        
        Args:
            reason: 重置原因，用于日志记录（'epoch', 'validation', 'init' 等）
        """
        LOGGER.info(f"[Trainer] 开始环境重置 (原因: {reason})...")
        
        # 1. 重置 solver 流场状态（最重要，恢复初始流场）
        self.solver.reset()
        LOGGER.debug("[Trainer] Solver 流场状态已重置")
        
        # 2. 重置历史状态缓冲区（envstep_state 和 KL 散度相关）
        self._reset_state_history_buffer()
        
        # 3. 重置策略稳定性 EMA
        if hasattr(self, '_policy_output_ema'):
            self._policy_output_ema = None
        LOGGER.debug("[Trainer] 策略稳定性 EMA 已重置")
        
        # 4. 重置正则化项缓存
        self._cached_reg_amp_l2 = None
        self._cached_reg_smoothness = None
        self._cached_reg_stability = None
        self._cached_reg_entropy = None
        self._cached_reg_kl = None
        LOGGER.debug("[Trainer] 正则化项缓存已重置")
        
        # 5. 重置重要性采样比率缓存
        if hasattr(self, '_cached_importance_ratio'):
            self._cached_importance_ratio = torch.tensor(1.0, device=self.device)
        
        # 6. 重置 policy 内部历史幅值记录
        if hasattr(self.policy, '_prev_amplitudes'):
            self.policy._prev_amplitudes = []
        LOGGER.debug("[Trainer] Policy 历史幅值已重置")
        
        # 7. 分离计算图（确保新阶段的梯度不会传播到旧阶段）
        self._detach_solver_state()
        LOGGER.debug("[Trainer] 计算图已分离")
        
        LOGGER.info(f"[Trainer] 环境重置完成 (原因: {reason})")

    def _load_agent_weights(self, model_path: str):
        """加载预训练的 agent 模型权重
        
        支持的文件格式:
        - 完整模型文件 (.pt/.pth): 通过 torch.save(model, path) 保存的完整模型
        - 状态字典文件 (.pt/.pth): 通过 torch.save(model.state_dict(), path) 保存的权重字典
        
        Args:
            model_path: 模型权重文件路径（相对路径或绝对路径）
        
        Raises:
            FileNotFoundError: 文件不存在
            RuntimeError: 权重加载失败（如架构不匹配）
        """
        import os
        
        # 解析路径（支持相对路径）
        if not os.path.isabs(model_path):
            # 相对于配置文件目录
            config_dir = os.path.dirname(self.cfg_path) if hasattr(self, 'cfg_path') and self.cfg_path else os.getcwd()
            model_path = os.path.join(config_dir, model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Agent 权重文件不存在: {model_path}")
        
        LOGGER.info(f"[Trainer] 正在加载 agent 权重: {model_path}")
        
        # 首先确保网络已初始化
        if self.policy.network is None:
            dummy_state = self._build_state()
            _ = self.policy(dummy_state)
        
        # 加载权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 判断是完整模型还是状态字典
            if isinstance(checkpoint, dict):
                # 情况1: state_dict 或包含 'state_dict' 键的 checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'policy_state_dict' in checkpoint:
                    state_dict = checkpoint['policy_state_dict']
                else:
                    # 假设整个 dict 就是 state_dict
                    state_dict = checkpoint
                
                # 尝试加载（允许部分匹配）
                missing_keys, unexpected_keys = self.policy.network.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    LOGGER.warning(f"[Trainer] 权重加载缺失的键: {missing_keys}")
                if unexpected_keys:
                    LOGGER.warning(f"[Trainer] 权重加载多余的键: {unexpected_keys}")
                    
            elif hasattr(checkpoint, 'state_dict'):
                # 情况2: 完整模型对象
                self.policy.network.load_state_dict(checkpoint.state_dict(), strict=False)
            else:
                raise RuntimeError(f"无法识别的权重文件格式: {type(checkpoint)}")
            
            LOGGER.info(f"[Trainer] 成功加载 agent 权重: {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"加载 agent 权重失败: {model_path}, 错误: {e}")
    
    def _setup_optimizer(self):
        """初始化优化器和学习率调度器（YOLOv8 风格）
        
        YOLOv8 优化器特性:
        1. 参数分组: bias(无衰减)、normalization weights(无衰减)、其他 weights(带衰减)
        2. Adam/AdamW: betas=(momentum, 0.999)
        3. SGD: 启用 nesterov 加速
        
        学习率调度器支持 YOLOv8 风格的:
        - warmup: 前 warmup_updates 次更新中，lr 从 warmup_bias_lr * lr 线性增长到 lr
        - 衰减: constant / linear / cosine
        """
        # 确保 policy 网络已初始化
        if self.policy.network is None:
            # 用一个虚拟 state 初始化网络
            dummy_state = self._build_state()
            _ = self.policy(dummy_state)
        
        # 初始化 policy_net_old（通过 state_dict 复制，避免 deepcopy 兼容性问题）
        # 注意：当 use_importance_sampling 启用时也需要 policy_net_old
        if self.cfg.loss.kl_weight > 0 or self.cfg.training.use_importance_sampling:
            import io
            # 使用 torch.save/load + BytesIO 来实现可靠的网络复制
            buffer = io.BytesIO()
            torch.save(self.policy.network, buffer)
            buffer.seek(0)
            self.policy_net_old = torch.load(buffer, map_location=self.device, weights_only=False)
            self.policy_net_old.eval()  # 旧网络始终处于 eval 模式
            for param in self.policy_net_old.parameters():
                param.requires_grad = False  # 不需要梯度
            LOGGER.info("[Trainer] 已创建 policy_net_old 用于 KL 散度正则化/重要性采样")
        
        cfg = self.cfg.training
        opt_name = cfg.optimizer.lower()
        lr = cfg.learning_rate
        weight_decay = cfg.optimizer_weight_decay
        momentum = cfg.optimizer_momentum
        
        # ===== YOLOv8 风格参数分组 =====
        # g[0]: weights (with decay) - 普通权重，带权重衰减
        # g[1]: weights (no decay) - BatchNorm 等归一化层权重，无衰减
        # g[2]: bias (no decay) - 偏置项，无衰减
        g = [[], [], []]  # optimizer parameter groups
        
        # 归一化层类型（权重不衰减）
        bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm, 
                    torch.nn.GroupNorm, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d)
        
        for module_name, module in self.policy.network.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                
                if 'bias' in param_name:
                    # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn_types):
                    # normalization layer weights (no decay)
                    g[1].append(param)
                else:
                    # regular weights (with decay)
                    g[0].append(param)
        
        # ===== 创建优化器（YOLOv8 风格）=====
        # 先用 bias 组（g[2]）初始化优化器，然后添加其他参数组
        if opt_name in ('adam', 'adamw', 'nadam', 'radam'):
            # Adam 系列: betas=(momentum, 0.999)
            OptimizerClass = {
                'adam': torch.optim.Adam,
                'adamw': torch.optim.AdamW,
                'nadam': torch.optim.NAdam,
                'radam': torch.optim.RAdam,
            }.get(opt_name, torch.optim.Adam)
            
            # bias 组无衰减
            self.optimizer = OptimizerClass(
                g[2] if g[2] else [{'params': []}],  # bias group
                lr=lr, 
                betas=(momentum, 0.999),
                weight_decay=0.0
            )
        elif opt_name == 'sgd':
            # SGD: 启用 nesterov 加速
            self.optimizer = torch.optim.SGD(
                g[2] if g[2] else [{'params': []}],  # bias group
                lr=lr,
                momentum=momentum,
                nesterov=True,
                weight_decay=0.0
            )
        elif opt_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                g[2] if g[2] else [{'params': []}],  # bias group
                lr=lr,
                momentum=momentum,
                weight_decay=0.0
            )
        else:
            # 默认使用 Adam
            self.optimizer = torch.optim.Adam(
                g[2] if g[2] else [{'params': []}],
                lr=lr,
                betas=(momentum, 0.999),
                weight_decay=0.0
            )
        
        # 添加其他参数组
        if g[0]:  # weights with decay
            self.optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})
        if g[1]:  # normalization weights without decay
            self.optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
        
        # 计算总更新次数（用于学习率调度器）
        self._total_updates = self._compute_total_updates()
        
        # 初始化学习率调度器相关状态
        self._lr_scheduler_state = {
            'initial_lr': lr,
            'current_update': 0,
        }
        
        LOGGER.info(f"[Trainer] 优化器: {opt_name} | lr={lr} | momentum={momentum} | weight_decay={weight_decay}")
        LOGGER.info(f"[Trainer] 参数分组: {len(g[2])} bias(decay=0), {len(g[0])} weight(decay={weight_decay}), {len(g[1])} norm(decay=0)")
        LOGGER.info(f"[Trainer] LR调度: {cfg.lr_schedule} | warmup_updates={cfg.warmup_updates} | "
                   f"lr_final={cfg.lr_final} | 总更新次数: {self._total_updates}")
    
    def _compute_total_updates(self) -> int:
        """计算训练过程中的总参数更新次数"""
        cfg = self.cfg.training
        # 总交互次数
        total_interactions = cfg.num_interaction_per_mini_episode
        # 每次参数更新需要的交互次数
        interactions_per_update = cfg.num_of_interaction_per_agent_update
        # 总更新次数（向上取整）
        total_updates = (total_interactions + interactions_per_update - 1) // interactions_per_update
        return max(1, total_updates)
    
    def _get_lr(self, update_idx: int) -> float:
        """根据当前更新步计算学习率（YOLOv8 风格）
        
        实现:
        1. Warmup 阶段：lr 从 warmup_bias_lr * lr0 线性增长到 lr0
        2. 衰减阶段：根据 lr_schedule 进行 linear/cosine 衰减
        
        Args:
            update_idx: 当前参数更新步索引（从 0 开始）
            
        Returns:
            当前学习率
        """
        cfg = self.cfg.training
        lr0 = cfg.learning_rate
        lrf = cfg.lr_final  # 最终 lr 相对于初始 lr 的比例
        warmup_updates = cfg.warmup_updates
        total_updates = self._total_updates
        
        # Warmup 阶段
        if warmup_updates > 0 and update_idx < warmup_updates:
            # 线性从 warmup_bias_lr * lr0 增长到 lr0
            warmup_progress = update_idx / warmup_updates
            lr = cfg.warmup_bias_lr * lr0 + warmup_progress * (1 - cfg.warmup_bias_lr) * lr0
            return lr
        
        # 衰减阶段
        if cfg.lr_schedule == 'constant':
            return lr0
        
        # 计算衰减进度（从 warmup 结束开始）
        decay_steps = total_updates - warmup_updates
        if decay_steps <= 0:
            return lr0
        
        current_decay_step = update_idx - warmup_updates
        progress = min(1.0, current_decay_step / decay_steps)  # [0, 1]
        
        if cfg.lr_schedule == 'linear':
            # 线性衰减: lr = lr0 * (1 - progress * (1 - lrf))
            lr = lr0 * (1 - progress * (1 - lrf))
        elif cfg.lr_schedule == 'cosine':
            # 余弦衰减（YOLOv8 风格）: lr = lrf + 0.5 * (1 - lrf) * (1 + cos(π * progress))
            lr = lr0 * (lrf + 0.5 * (1 - lrf) * (1 + pymath.cos(pymath.pi * progress)))
        else:
            lr = lr0
        
        return lr
    
    def _update_lr(self, update_idx: int):
        """更新优化器学习率
        
        Args:
            update_idx: 当前参数更新步索引（从 0 开始）
        """
        new_lr = self._get_lr(update_idx)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 更新状态
        self._lr_scheduler_state['current_update'] = update_idx
        
        # 记录到 TensorBoard
        if self.writer:
            self.writer.add_scalar('LR/learning_rate', new_lr, update_idx)
    
    def _setup_tensorboard(self):
        """设置 TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.exp_dir / 'tensorboard'))
        except ImportError:
            self.writer = None
    
    def _setup_probe_points(self):
        """加载 base_config 中的所有探针位置（用于 main_input_type='probes' 模式）
        
        从 base_config_path 中读取 runner.probes.locations 配置，
        创建 Point 对象用于在 _build_state 中采样探针数据。
        
        探针位置支持两种格式:
        - fraction: [x, y] - 相对坐标（推荐）
        - units: [x, y] - 绝对坐标
        """
        self._probe_points = {}  # {name: Point}
        
        # 如果 main_input_type 为 None 或 'target_probe'，则无需加载额外探针
        if self.cfg.input_features.main_input_type in (None, 'target_probe'):
            LOGGER.debug(f"[Trainer] main_input_type={self.cfg.input_features.main_input_type!r}，跳过额外探针加载")
            return
        
        # 从 base_config 加载探针配置
        base_cfg_path = self.cfg.base_config_path
        if not os.path.isabs(base_cfg_path):
            configs_dir = Path(__file__).parent.parent.parent / 'configs'
            base_cfg_path = str(configs_dir / base_cfg_path)
        
        with open(base_cfg_path, 'r', encoding='utf-8') as f:
            base_cfg = yaml.safe_load(f)
        
        runner_cfg = base_cfg.get('runner', {})
        probes_cfg = runner_cfg.get('probes', {})
        locations = probes_cfg.get('locations', [])
        
        if not locations:
            LOGGER.debug("[Trainer] base_config 中未定义探针位置")
            return
        
        Lx, Ly = self.solver.cfg.domain.size
        
        for loc in locations:
            name = loc.get('name', f'probe_{len(self._probe_points)}')
            
            # 解析坐标（支持 fraction 和 units 两种格式）
            if 'fraction' in loc:
                frac = loc['fraction']
                x, y = frac[0] * Lx, frac[1] * Ly
            elif 'units' in loc:
                x, y = loc['units']
            else:
                LOGGER.warning(f"[Trainer] 探针 '{name}' 缺少坐标配置，跳过")
                continue
            
            self._probe_points[name] = Point(vec(x=x, y=y))
        
        LOGGER.info(f"[Trainer] 已加载 {len(self._probe_points)} 个探针点: {list(self._probe_points.keys())}")
    
    def _setup_validation_runner(self):
        """创建并缓存 validation runner 实例
        
        在构造函数中创建 SolverRunner 实例，缓存到 self.runners 列表。
        验证时重置 runner 缓存即可重复使用。
        
        配置来源: base_config_path 中的 runner 配置
        
        注意:
        - Trainer 的 verbose 会覆盖 Runner 的 verbose 配置
        - 验证输出将保存到 exp_dir/validation/ 目录
        """
        from myafc.env_kernels.solver_wrappers import SolverRunner
        
        # 初始化 runners 列表
        if not hasattr(self, 'runners'):
            self.runners = []
        
        # ---- 从 base_config_path 加载 runner 配置 ----
        base_cfg_path = self.cfg.base_config_path
        if not os.path.isabs(base_cfg_path):
            configs_dir = Path(__file__).parent.parent.parent / 'configs'
            base_cfg_path = str(configs_dir / base_cfg_path)
        
        with open(base_cfg_path, 'r', encoding='utf-8') as f:
            base_cfg = yaml.safe_load(f)
        
        # 获取基础 runner 配置（如果不存在则使用默认）
        runner_cfg = base_cfg.get('runner', {}).copy() if 'runner' in base_cfg else {}
        
        # 验证专用的输出目录配置（覆盖原有的 output 配置）
        val_output_dir = self.exp_dir / 'validation'
        val_output_dir.mkdir(parents=True, exist_ok=True)
        
        runner_cfg['output'] = {
            'enabled': True,
            'root': str(val_output_dir),
            'subdir': '',  # 直接使用 root 目录
            'skip_log_setup': True,  # 使用 trainer 的日志配置
            'skip_config_snapshot': True,  # 配置已保存在 configs/ 目录，无需重复保存
        }
        
        # verbose 覆盖: trainer.verbose 覆盖 runner.verbose
        runner_cfg['verbose'] = self.cfg.training.verbose
        
        # 确保探针配置包含目标位置探针
        if 'probes' not in runner_cfg:
            runner_cfg['probes'] = {'enabled': True, 'interval': 1, 'locations': []}
        if runner_cfg['probes'].get('enabled', False):
            # 检查是否已有 target 探针，没有则添加
            locations = runner_cfg['probes'].get('locations', [])
            has_target = any(loc.get('name') == 'target' for loc in locations)
            if not has_target:
                locations.append({
                    'name': 'target',
                    'fraction': [float(self.cfg.loss.target_probe_position[0]),
                                 float(self.cfg.loss.target_probe_position[1])]
                })
                runner_cfg['probes']['locations'] = locations
        
        # 创建 validation runner 实例
        self.validation_runner = SolverRunner(self.solver, {'runner': runner_cfg})
        self._validation_output_dir = val_output_dir
        
        # 添加到 runners 列表
        self.runners.append(self.validation_runner)
        
        LOGGER.debug(f"[Trainer] Validation runner 已创建 | 输出目录: {val_output_dir}")

    def _build_state(self) -> Dict[str, Any]:
        """构建传递给 policy/solver 的状态字典 (去梯度的 state)
        
        根据 input_features 配置构建状态字典，仅采样 probe_fields 中请求的字段:
        - main_input_type: 'target_probe'(仅目标探针) / 'probes'(全部探针)
        - probe_fields: ['u','v','p'] 的任意子集 — 与探针选择独立组合
        
        使用共享的 sample_probe_fields() 确保与所有 Trainer 采样一致。
        """
        from myafc.utils import sample_probe_fields
        
        feat_cfg = self.cfg.input_features
        probe_fields = feat_cfg.probe_fields
        
        state = {
            'time': self.solver.current_time,
            'step': self.solver.current_step,
        }
        
        # target 探针 (始终采样)
        state['probes'] = {'target': sample_probe_fields(
            self.solver.velocity, self.solver.pressure, self.target_point,
            probe_fields,
        )}
        
        # env 探针 (main_input_type='probes' 时)
        if feat_cfg.main_input_type == 'probes':
            for name, point in self._probe_points.items():
                state['probes'][name] = sample_probe_fields(
                    self.solver.velocity, self.solver.pressure, point,
                    probe_fields,
                )
        
        # 力系数
        Cd, Cl, _, _, _, _ = self.solver.compute_force_coeffs()
        state['force_coeffs'] = {
            'Cd': float(Cd.native()) if hasattr(Cd, 'native') else float(Cd),
            'Cl': float(Cl.native()) if hasattr(Cl, 'native') else float(Cl),
        }
        
        # 目标速度和历史幅值（闭环仅使用此两项）
        if feat_cfg.include_target_velocity:
            state['target_velocity'] = self.cfg.loss.target_velocity
        
        # 历史幅值序列（长度由 include_prev_amplitude_num 控制）
        num_prev = feat_cfg.include_prev_amplitude_num
        if num_prev > 0:
            # 获取最近 num_prev 个历史幅值，不足时用 0 填充
            if len(self.amplitude_history) >= num_prev:
                prev_amps = self.amplitude_history[-num_prev:]
            else:
                # 不足时在前面用 0 填充
                prev_amps = [0.0] * (num_prev - len(self.amplitude_history)) + self.amplitude_history
            state['prev_amplitudes'] = prev_amps
        
        return state
    
    def _compute_reward(self, is_net_forward_step: bool = False) -> torch.Tensor:
        """计算一个环境步的 reward（可微分实现）
        
        Reward 定义（与 DRL 对齐）:
        - probe_mse/probe_mae: reward = -velocity_weight * (u_error + v_error)
        - force_coeffs: reward = -((cd - cd_target)^2 + cl_weight * (cl - cl_target)^2)
        
        Reward 越大越好，loss = -reward
        
        Args:
            is_net_forward_step: 是否为神经网络 forward 步（即动作更新步）
            
        Returns:
            reward: 单步 reward（标量张量，可微分）
        """
        loss_type = self.cfg.loss.loss_type
        
        if loss_type.startswith('probe'):
            vel = self.solver.velocity.at(self.target_point)
            u = math.mean(vel.values.vector['x']).native()
            v = math.mean(vel.values.vector['y']).native()
            
            # 使用预先缓存的目标张量（在 __init__ 中创建）
            if 'mae' in loss_type:
                error = torch.abs(u - self._target_u) + torch.abs(v - self._target_v)
            else:  # mse
                error = (u - self._target_u) ** 2 + (v - self._target_v) ** 2
            
            # reward = -error (误差越小，reward 越大)
            reward = -self.cfg.loss.velocity_weight * error
            
        elif loss_type == 'force_coeffs':
            Cd, Cl, _, _, _, _ = self.solver.compute_force_coeffs()
            cd = Cd.native() if hasattr(Cd, 'native') else Cd
            cl = Cl.native() if hasattr(Cl, 'native') else Cl
            
            error = (cd - self._cd_target) ** 2 + self.cfg.loss.cl_weight * (cl - self._cl_target) ** 2
            reward = -error
        else:
            reward = torch.tensor(0.0, device=self.device)
        
        return reward
    
    def _compute_importance_sampling_ratio(self) -> torch.Tensor:
        """计算 PPO 式重要性采样比率 ratio = π_new(a|s) / π_old(a|s)
        
        使用当前策略和旧策略对同一动作的概率比。
        动作来自当前策略采样，但用旧策略评估其概率。
        
        Returns:
            ratio: 重要性采样比率（标量张量）
        """
        if self.policy_net_old is None or not self.policy.network:
            return torch.tensor(1.0, device=self.device)
        
        # 获取当前策略的输出
        current_output = self.policy.network.current_output
        if current_output is None or current_output.log_prob is None:
            return torch.tensor(1.0, device=self.device)
        
        log_prob_new = current_output.log_prob
        
        # 用旧策略评估同一动作的概率
        old_dist = self._get_old_policy_distribution()
        if old_dist is None:
            return torch.tensor(1.0, device=self.device)
        
        old_mean, old_log_std = old_dist
        old_std = torch.exp(old_log_std)
        
        # 获取当前动作
        action = current_output.action_raw if current_output.action_raw is not None else current_output.action
        
        # 计算旧策略对当前动作的 log_prob
        from torch.distributions import Normal
        old_dist_obj = Normal(old_mean, old_std)
        log_prob_old = old_dist_obj.log_prob(action)
        if log_prob_old.dim() > 0:
            log_prob_old = log_prob_old.sum(dim=-1)
        
        # 计算比率
        ratio = torch.exp(log_prob_new - log_prob_old)
        
        return ratio
    
    def _compute_loss(self, is_net_forward_step: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算一个环境步的 loss（基于 reward 的可微分实现）
        
        计算流程:
        1. 计算 reward（使用 _compute_reward）
        2. 可选：应用 PPO 式重要性采样（clipped ratio * reward）
        3. 计算 primary_loss = -reward（或 -weighted_reward）
        4. 添加正则化项得到 total_loss
        
        返回值:
            Tuple[primary_loss, reg_loss, total_loss, reward]:
            - primary_loss: 主损失（真正的训练目标，用于 fitness 计算和进度条显示）
            - reg_loss: 正则化损失之和（辅助项）
            - total_loss: 总损失 = primary_loss + reg_loss（用于 backward）
            - reward: 原始 reward 值（用于记录和监控，避免外部重复调用 _compute_reward）
        
        Loss 组成:
        1. 主损失 (从 reward 转换) - 每个环境步都计算
        2. 正则化项 - 每个环境步都施加，但正则化项的数值仅在动作更新步刷新：
           - 幅值 L2 正则化 (amplitude_l2_weight): 惩罚大幅值
           - 时序平滑正则化 (temporal_smoothness_weight): 惩罚控制信号剧烈变化
           - 策略稳定性正则化 (policy_stability_weight): 惩罚偏离 EMA，防过冲
           - 熵正则化 (entropy_weight): PPO 风格，使用高斯分布真实熵，鼓励探索
           - KL 散度正则化 (kl_weight): 限制新旧策略分布差异
        
        正则化项的生效机制：
        - 每个环境步都将正则化项加入 loss 
        - 但正则化项的数值仅在动作更新步（is_net_forward_step=True）时更新
        - 非动作更新步复用上一次的正则化项值（因为 current_controls/current_output 保持不变）
        - 由于正则化项保持对计算图的引用，梯度链得以保留
        
        Args:
            is_net_forward_step: 是否为神经网络 forward 步（即动作更新步，每个 control_interval 的第一步）
                True: 正则化项的数值会刷新（因为 PolicyNet.forward() 产生了新的输出）
                False: 正则化项复用上一次的数值（保持梯度链）
        """
        # ===== Step 1: 计算 reward =====
        reward = self._compute_reward(is_net_forward_step)
        
        # ===== Step 2: 可选的 PPO 式重要性采样 =====
        cfg = self.cfg.training
        if cfg.use_importance_sampling and is_net_forward_step:
            # 计算并缓存重要性采样比率
            ratio = self._compute_importance_sampling_ratio()
            clip_range = cfg.importance_sampling_clip_range
            
            # PPO 裁剪目标使用 reward 作为权重。
            ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            
            weighted_reward = torch.min(ratio * reward, ratio_clipped * reward)
            
            # 缓存比率用于记录
            self._cached_importance_ratio = ratio
        elif cfg.use_importance_sampling and not is_net_forward_step:
            # 非动作更新步：使用缓存的 ratio（如果有）
            if hasattr(self, '_cached_importance_ratio') and self._cached_importance_ratio is not None:
                ratio = self._cached_importance_ratio
                # 确保 ratio 是 tensor
                if not isinstance(ratio, torch.Tensor):
                    ratio = torch.tensor(ratio, device=self.device)
                clip_range = cfg.importance_sampling_clip_range
                ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                weighted_reward = torch.min(ratio * reward, ratio_clipped * reward)
            else:
                weighted_reward = reward
        else:
            weighted_reward = reward
        
        # ===== Step 3: 从 reward 转换为 loss =====
        # loss = -reward (最大化 reward 等价于最小化 loss)
        primary_loss = -weighted_reward
        
        # ===== Step 4: 正则化项 =====
        # 在动作更新步（is_net_forward_step=True）时计算并缓存正则化项张量
        # 非动作更新步直接使用缓存的张量（保持梯度链连通）
        
        # 初始化正则化损失
        reg_loss = torch.tensor(0.0, device=self.device)
        
        # 检查是否应禁用正则化（最后 N 个 epoch / 'all'）
        disable_reg_n = self.cfg.loss.disable_regularization_last_epochs
        if isinstance(disable_reg_n, str) and disable_reg_n.lower() == 'all':
            is_reg_disabled = True
        elif isinstance(disable_reg_n, int) and disable_reg_n > 0:
            is_reg_disabled = self.current_epoch >= self.num_epochs - disable_reg_n
        else:
            is_reg_disabled = False
        
        # 如果正则化被禁用，跳过所有正则化项计算
        if not is_reg_disabled:
            # ===== 4.1 幅值 L2 正则化 =====
            if self.cfg.loss.amplitude_l2_weight > 0 and self.solver.jet_obstacle:
                if is_net_forward_step:
                    # 动作更新步：重新计算并缓存
                    controls = self.solver.jet_obstacle.current_controls
                    if controls:
                        amp = controls[0][0]
                        self._cached_reg_amp_l2 = amp ** 2
                # 使用缓存值（动作更新步刚刚更新，非动作更新步复用上次的）
                if self._cached_reg_amp_l2 is not None:
                    reg_loss = reg_loss + self.cfg.loss.amplitude_l2_weight * self._cached_reg_amp_l2
        
            # ===== 4.2 时序平滑正则化 =====
            if self.cfg.loss.temporal_smoothness_weight > 0 and self.solver.jet_obstacle:
                if is_net_forward_step:
                    self._cached_reg_smoothness = self._compute_temporal_smoothness_loss()
                if self._cached_reg_smoothness is not None:
                    reg_loss = reg_loss + self.cfg.loss.temporal_smoothness_weight * self._cached_reg_smoothness
        
            # ===== 4.3 策略稳定性正则化（防过冲）=====
            if self.cfg.loss.policy_stability_weight > 0 and self.solver.jet_obstacle:
                if is_net_forward_step:
                    self._cached_reg_stability = self._compute_policy_stability_loss()
                if self._cached_reg_stability is not None:
                    reg_loss = reg_loss + self.cfg.loss.policy_stability_weight * self._cached_reg_stability
        
            # ===== 4.4 熵正则化（PPO 风格，使用高斯分布的熵）=====
            if self.cfg.loss.entropy_weight > 0:
                if is_net_forward_step:
                    self._cached_reg_entropy = self._compute_entropy_loss()
                # 熵越大越好，所以用减法（鼓励探索）
                if self._cached_reg_entropy is not None:
                    reg_loss = reg_loss - self.cfg.loss.entropy_weight * self._cached_reg_entropy
        
            # ===== 4.5 KL 散度正则化 =====
            if self.cfg.loss.kl_weight > 0:
                if is_net_forward_step:
                    self._cached_reg_kl = self._compute_kl_loss()
                if self._cached_reg_kl is not None:
                    reg_loss = reg_loss + self.cfg.loss.kl_weight * self._cached_reg_kl
        
        # 总损失 = 主损失 + 正则化损失
        total_loss = primary_loss + reg_loss
        
        return primary_loss, reg_loss, total_loss, reward
    
    def _compute_temporal_smoothness_loss(self) -> torch.Tensor:
        """计算时序平滑正则化损失
        
        惩罚相邻控制信号的剧烈变化：L_smooth = |a_t - a_{t-1}|^2
        
        如果设置了 action_rate_limit，则只惩罚超过限制的变化：
        L_smooth = max(0, |a_t - a_{t-1}| - limit)^2
        
        Returns:
            平滑损失值（标量张量）
        """
        if not self.solver.jet_obstacle:
            return torch.tensor(0.0, device=self.device)
        
        controls = self.solver.jet_obstacle.current_controls
        if not controls:
            return torch.tensor(0.0, device=self.device)
        
        current_amp = controls[0][0]
        
        # 获取上一步幅值
        if len(self.amplitude_history) > 0:
            prev_amp = self.amplitude_history[-1]
        else:
            prev_amp = 0.0
        
        # 计算幅值变化
        delta = current_amp - prev_amp
        
        # 如果设置了变化率限制，只惩罚超过限制的部分
        rate_limit = self.cfg.loss.action_rate_limit
        if rate_limit and rate_limit > 0:
            # Huber-like: 只惩罚超过限制的部分
            excess = torch.abs(delta) - rate_limit
            smoothness_loss = torch.relu(excess) ** 2
        else:
            # 标准 L2 平滑
            smoothness_loss = delta ** 2
        
        return smoothness_loss
    
    def _compute_entropy_loss(self) -> torch.Tensor:
        """计算熵正则化损失（PPO 风格，使用高斯分布的真实熵）
        
        高斯分布的熵公式:
            H(π) = Σ 0.5 * log(2πe * σ²) = Σ (0.5 + 0.5*log(2π) + log(σ))
        
        PolicyNet 在前向传播时会计算并存储 entropy，这里直接使用。
        熵越大，策略越具有探索性；熵越小，策略越确定性。
        
        Returns:
            entropy: 标量张量，高斯分布的熵值（越大表示探索性越强）
        """
        if not self.policy.network:
            return torch.tensor(0.0, device=self.device)
        
        # 获取 PolicyNet 最近一次前向传播的输出
        current_output = self.policy.network.current_output
        if current_output is None or current_output.entropy is None:
            # 如果没有 entropy（确定性模式），返回 0
            return torch.tensor(0.0, device=self.device)
        
        # current_output.entropy 是 PolicyNet.forward() 中计算的 dist.entropy().sum(dim=-1)
        return current_output.entropy
    
    def _compute_kl_loss(self) -> torch.Tensor:
        """计算 KL 散度正则化损失（PPO 系列风格，稳定策略更新）
        
        核心思想（参考 PPO/TRPO）：
        - 限制新旧策略分布的差异，避免更新过大导致训练崩溃
        - policy_net_old 保存的是**上一次更新前**的网络权重
        
        实现方式：
        1. 用 policy_net_old 和缓存的 state（_current_state_for_kl）计算旧分布
        2. 与当前策略的新分布计算 KL 散度
        3. 新旧策略使用相同的 state，确保比较的一致性
        
        高斯分布之间的 KL 散度:
            KL(π_new || π_old) = log(σ_old/σ_new) + (σ_new² + (μ_new - μ_old)²)/(2σ_old²) - 0.5
        
        Returns:
            kl_loss: 标量张量，当前 interaction 的 KL 散度值
        """
        if self.policy_net_old is None:
            return torch.tensor(0.0, device=self.device)
        
        # 获取当前策略的分布（新分布）
        if not self.policy.network or not self.policy.network.current_output:
            return torch.tensor(0.0, device=self.device)
        
        current_output = self.policy.network.current_output
        if current_output.mean is None or current_output.log_std is None:
            return torch.tensor(0.0, device=self.device)
        
        new_mean = current_output.mean
        new_log_std = current_output.log_std
        new_std = torch.exp(new_log_std)
        
        # 获取旧策略的分布（用 policy_net_old 计算）
        old_dist = self._get_old_policy_distribution()
        if old_dist is None:
            return torch.tensor(0.0, device=self.device)
        
        old_mean, old_log_std = old_dist
        old_std = torch.exp(old_log_std)
        
        # KL(π_new || π_old)
        kl = (old_log_std - new_log_std 
              + (new_std ** 2 + (new_mean - old_mean) ** 2) / (2 * old_std ** 2) 
              - 0.5)
        
        # 对动作维度求和
        if kl.dim() > 0:
            kl = kl.sum(dim=-1)
        if kl.dim() > 0:
            kl = kl.mean()
        
        return kl
    
    def _get_old_policy_distribution(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """用 policy_net_old 计算缓存的 state 的旧动作分布
        
        在动作更新步，用旧网络（无梯度、eval模式）计算旧动作分布。
        使用 _env_step 中缓存的 state，确保新旧策略使用相同的 state。
        
        Returns:
            (old_mean, old_log_std) 元组，或 None（如果无法计算）
        """
        if self.policy_net_old is None or not self.policy.network:
            return None
        
        # 使用缓存的 state（在 _env_step 中保存）
        if not hasattr(self, '_current_state_for_kl') or self._current_state_for_kl is None:
            return None
        
        state = self._current_state_for_kl
        
        # 用旧网络计算旧分布（state 中已包含 '_state_history' 字段）
        with torch.no_grad():
            inputs = self.policy._build_input(state)
            x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            
            # 构建 prev_action（如果使用相对动作模式）
            prev_action = None
            if self.policy.relative_action:
                prev_action = self.policy._build_prev_action(state)
            
            # 使用 policy_net_old 前向传播
            output = self.policy_net_old(x, deterministic=True, prev_action=prev_action)
            
            if output.mean is None or output.log_std is None:
                return None
            
            return (output.mean.detach(), output.log_std.detach())
    
    def _sync_policy_net_old(self):
        """将当前网络权重同步到 policy_net_old
        
        在每次 optimizer.step() **前** 调用，保存更新前的网络权重。
        这样 policy_net_old 始终保存的是“上一次更新前”的网络。
        """
        if self.policy_net_old is None or not self.policy.network:
            return
        
        self.policy_net_old.load_state_dict(self.policy.network.state_dict())
        LOGGER.debug("[Trainer] policy_net_old 已同步")
    
    def _compute_policy_stability_loss(self) -> torch.Tensor:
        """计算策略稳定性正则化损失（防过冲）
        
        惩罚当前输出与历史移动平均的偏差：
        L_stability = (a_t - ā_EMA)^2
        
        其中 ā_EMA 是指数移动平均：
        ā_EMA = α * a_t + (1-α) * ā_EMA_prev
        
        这可以防止策略变化过快导致的过冲问题：
        - 当策略输出突然大幅偏离历史均值时，增加惩罚
        - 使策略更新更加平滑，避免 loss 持续上升
        
        Returns:
            策略稳定性损失值（标量张量）
        """
        if not self.solver.jet_obstacle:
            return torch.tensor(0.0, device=self.device)
        
        controls = self.solver.jet_obstacle.current_controls
        if not controls:
            return torch.tensor(0.0, device=self.device)
        
        current_amp = controls[0][0]
        current_amp_val = float(current_amp.detach().cpu().numpy())
        
        # 初始化或更新 EMA
        if not hasattr(self, '_amplitude_ema'):
            self._amplitude_ema = current_amp_val
        
        # 计算与 EMA 的偏差
        deviation = current_amp - self._amplitude_ema
        stability_loss = deviation ** 2
        
        # 更新 EMA（使用 detach 后的值，不影响梯度）
        momentum = self.cfg.loss.policy_stability_momentum
        self._amplitude_ema = momentum * current_amp_val + (1 - momentum) * self._amplitude_ema
        
        return stability_loss
    
    def _detach_solver_state(self):
        """分离 solver 状态，防止计算图无限累积
        
        需要分离的状态：
        1. solver.velocity 和 solver.pressure
        2. jet_obstacle 的 current_controls 和 jet_velocity_field
        """
        if self.solver.velocity is not None:
            self.solver.velocity = math.stop_gradient(self.solver.velocity)
        if self.solver.pressure is not None:
            self.solver.pressure = math.stop_gradient(self.solver.pressure)
        
        # 分离 jet_obstacle 的状态
        if self.solver.jet_obstacle is not None:
            # 分离 current_controls（重要：防止旧计算图引用）
            if self.solver.jet_obstacle._current_controls is not None:
                detached_controls = []
                for amp, angle in self.solver.jet_obstacle._current_controls:
                    detached_controls.append((amp.detach(), angle.detach()))
                self.solver.jet_obstacle._current_controls = detached_controls
            
            # 清空 jet_velocity_field，下次 update_controls 时会重建
            self.solver.jet_obstacle._jet_velocity_field = None
        
        # 清空正则化项缓存（下一个累积周期需要重新计算）
        self._cached_reg_amp_l2 = None
        self._cached_reg_smoothness = None
        self._cached_reg_stability = None
        self._cached_reg_entropy = None
        self._cached_reg_kl = None
    
    def _env_step(
        self, 
        env_step: int, 
        steps_in_accumulation: int,
        is_warmup: bool = False
    ) -> Tuple[bool, bool]:
        """
        封装单步环境求解流程（训练和验证共用）
        
        流程:
        1. 更新 sinusoidal 入口边界（如果启用）
        2. 判断是否更新控制信号
        3. 判断是否施加射流
        4. 执行 solver.step
        
        Args:
            env_step: 当前环境步索引
            steps_in_accumulation: 当前累积步数（用于控制信号判断）
            is_warmup: 是否为预热阶段
            
        Returns:
            (should_update_control, should_apply_jets): 本步是否更新控制、是否施加射流
        """
        cfg = self.cfg.training
        
        # 1. sinusoidal 入口模式：更新边界条件
        self._update_sinusoidal_inlet(self.solver.current_time)
        
        # 2. 判断控制逻辑
        if is_warmup:
            should_update_control = cfg.warmup_update_controls
            should_apply_jets = cfg.warmup_apply_jets
        else:
            # 每 control_interval 步更新一次控制信号
            should_update_control = (steps_in_accumulation % cfg.control_interval == 0)
            # 每个 interaction 的前 action_duration 步施加射流
            should_apply_jets = (steps_in_accumulation % cfg.control_interval < cfg.action_duration)
        
        LOGGER.debug(f"[Trainer] Env Step: {env_step} | Update Control: {should_update_control} | Apply Jets: {should_apply_jets}")
        
        # 3. 构建状态并执行仿真步
        state = self._build_state()
        
        # 4. 注入历史状态到 state 字典（用于 policy._build_input 的 state stack 功能）
        feat_cfg = self.cfg.input_features
        if feat_cfg.envstep_state > 1:
            state['_state_history'] = list(self._state_history_buffer)
        
        # 5. 如果是动作更新步，缓存完整的 state 快照（用于 KL 散度计算）
        # 此时 state 已包含 '_state_history' 字段，无需额外变量
        if should_update_control:
            import copy
            self._current_state_for_kl = copy.deepcopy(state)
        
        self.solver.step(state=state, update_controls=should_update_control, apply_jets=should_apply_jets)
        
        # 6. 在仿真步完成后更新历史状态缓冲区（为下一步准备）
        if feat_cfg.envstep_state > 1:
            import copy
            # 添加当前状态的副本到历史缓冲区（不含 _state_history 字段）
            state_copy = {k: v for k, v in state.items() if not k.startswith('_')}
            self._state_history_buffer.append(copy.deepcopy(state_copy))
            # 保持缓冲区长度不超过 envstep_state - 1（历史步数）
            max_history_len = feat_cfg.envstep_state - 1
            if len(self._state_history_buffer) > max_history_len:
                self._state_history_buffer = self._state_history_buffer[-max_history_len:]
        
        return should_update_control, should_apply_jets
    
    def _preprocess_training_config(self) -> None:
        """训练配置预处理：科学计数法转换、类型强制、参数约束检查
        
        执行任务：
        1. 科学计数法字符串转浮点数（如 "1e-3" -> 0.001）
        2. 处理 control_interval='equal2action_duration' 特殊值
        3. 关键整数参数强制取整
        4. 参数约束检查与自动覆盖：
           - control_interval >= action_duration
           - num_of_interaction_per_agent_update >= 1
           - env_warmup_envsteps >= envstep_state - 1（历史状态缓冲区所需步数）
        5. 计算 envstep_per_value（内部变量）
        """
        # ================== 科学计数法支持 ==================
        # 转字典 -> 递归转换 -> 重建配置对象
        cfg_dict = _convert_scientific_notation(asdict(self.cfg))
        self.cfg = _parse_trainer_config(cfg_dict)
        
        cfg = self.cfg.training
        
        # ================== 参数类型检查与强制转换 ==================
        # 这些参数必须为整数（用于 range、tqdm 等）
        # 若超参数优化生成浮点数，则强制取整
        cfg.action_duration = int(cfg.action_duration)
        cfg.env_warmup_envsteps = int(cfg.env_warmup_envsteps)
        cfg.num_interaction_per_mini_episode = int(cfg.num_interaction_per_mini_episode)
        cfg.num_of_interaction_per_agent_update = int(cfg.num_of_interaction_per_agent_update)
        
        # ================== control_interval 特殊值处理 ==================
        # 支持 'equal2action_duration' 字符串，表示自动设置为 action_duration 的值
        if isinstance(cfg.control_interval, str):
            if cfg.control_interval.lower() == 'equal2action_duration':
                cfg.control_interval = cfg.action_duration
                LOGGER.info(f"[Trainer] control_interval='equal2action_duration' → 设置为 {cfg.action_duration}")
            else:
                # 尝试解析为整数
                try:
                    cfg.control_interval = int(cfg.control_interval)
                except ValueError:
                    raise ValueError(f"control_interval 值无效: '{cfg.control_interval}'，"
                                   f"可选值: 整数 或 'equal2action_duration'")
        else:
            cfg.control_interval = int(cfg.control_interval)
        
        # ================== 参数约束检查与自动覆盖 ==================
        # 重要硬性约束: control_interval >= action_duration
        if cfg.control_interval < cfg.action_duration:
            LOGGER.warning(f"[Trainer] control_interval ({cfg.control_interval}) < action_duration ({cfg.action_duration})，自动覆盖为 {cfg.action_duration}")
            cfg.control_interval = cfg.action_duration
        
        # 约束: num_of_interaction_per_agent_update >= 1
        if cfg.num_of_interaction_per_agent_update < 1:
            LOGGER.warning(f"[Trainer] num_of_interaction_per_agent_update ({cfg.num_of_interaction_per_agent_update}) < 1，自动覆盖为 1")
            cfg.num_of_interaction_per_agent_update = 1
        
        # ================== 计算内部变量 envstep_per_value ==================
        # envstep_per_value = num_of_interaction_per_agent_update * control_interval
        # 这是每次神经网络权重更新需要累积的环境步数
        self._envstep_per_value = cfg.num_of_interaction_per_agent_update * cfg.control_interval
        LOGGER.info(f"[Trainer] envstep_per_value = {cfg.num_of_interaction_per_agent_update} interactions × "
                   f"{cfg.control_interval} steps/interaction = {self._envstep_per_value} steps")
        
        # ================== envstep_state 约束检查与自动覆盖 ==================
        # 约束: env_warmup_envsteps >= envstep_state - 1
        # 历史状态缓冲区需要 envstep_state-1 步预热才能填满
        feat_cfg = self.cfg.input_features
        if feat_cfg.envstep_state > 1:
            min_warmup_required = feat_cfg.envstep_state - 1
            if cfg.env_warmup_envsteps < min_warmup_required:
                LOGGER.warning(
                    f"[Trainer] env_warmup_envsteps ({cfg.env_warmup_envsteps}) < 历史状态缓冲区所需步数 ({min_warmup_required})，"
                    f"自动覆盖为 {min_warmup_required}"
                )
                cfg.env_warmup_envsteps = min_warmup_required
    
    def _save_training_config(self) -> None:
        """保存训练配置文件
        
        该方法在 _preprocess_training_config() 之后、_train_impl() 之前调用，
        将经过预处理（类型转换、约束覆盖等）后的最终配置保存到实验目录。
        
        保存文件：
        - trainer_config.yaml: 最终使用的训练配置
        """
        with open(self.exp_dir / 'configs' / 'trainer_config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(asdict(self.cfg), f, sort_keys=False, allow_unicode=True)
        LOGGER.info(f"[Trainer] 训练配置已保存: {self.exp_dir / 'configs' / 'trainer_config.yaml'}")
    
    def train(self) -> Dict[str, Any]:
        """执行训练（入口方法）
        
        训练流程：
        1. 配置预处理（类型转换、科学计数法支持、参数约束检查）
        2. 保存最终训练配置
        3. 调用 _train_impl() 执行实际训练逻辑
        
        Returns:
            训练结果字典
        """
        # Step 1: 配置预处理
        self._preprocess_training_config()
        
        # Step 2: 保存最终配置（在参数检查与覆盖后）
        self._save_training_config()
        
        # Step 3: 执行实际训练
        return self._train_impl()
    
    def _train_impl(self) -> Dict[str, Any]:
        """执行训练实现（支持多轮 epoch 训练）
        
        训练流程：
        1. 循环执行 epoch 轮训练
        2. 每轮训练：预热阶段 + 训练阶段（累积 loss + 统一 backward）
        3. 每轮结束后重置环境状态（除策略网络外）
        4. 最后一轮完成后进行验证
        
        epoch 参数（别名 num_mini_episode_per_subenv）：
        - epoch=1: 单轮训练（默认，向后兼容）
        - epoch>1: 多轮训练，策略网络在不同初始状态下累积学习
        
        注意：调用此方法前需先调用 _preprocess_training_config() 进行配置预处理
        
        Returns:
            训练结果字典，包含所有 epoch 的累积历史数据
        """
        cfg = self.cfg.training
        num_epochs = cfg.epoch
        
        LOGGER.info(f"[Trainer] 开始训练 | 总轮数 (epochs): {num_epochs}")
        
        # 累积统计数据（跨 epoch 汇总）
        total_update_count = 0
        global_step_offset = 0  # 全局 step 偏移量（用于 TensorBoard 记录连续性）
        all_epoch_results = []
        
        checkpoint_epoch_interval = cfg.checkpoint_epoch_interval
        
        for epoch_idx in range(num_epochs):
            # ===== 更新当前 epoch 索引 =====
            self.current_epoch = epoch_idx
            
            # ===== 轮次初始化 =====
            if epoch_idx > 0:
                # 非首轮需要重置环境（策略网络不重置）
                self._reset_env(reason=f'epoch_{epoch_idx + 1}')
            
            # 检查是否进入"禁用正则化"阶段
            disable_reg_n = self.cfg.loss.disable_regularization_last_epochs
            if isinstance(disable_reg_n, str) and disable_reg_n.lower() == 'all':
                is_reg_disabled = True
                if epoch_idx == 0:
                    LOGGER.info("[Trainer] 所有 epoch 禁用正则化 (disable_regularization_last_epochs='all')")
            elif isinstance(disable_reg_n, int) and disable_reg_n > 0:
                is_reg_disabled = epoch_idx >= num_epochs - disable_reg_n
                if is_reg_disabled and epoch_idx == max(0, num_epochs - disable_reg_n):
                    LOGGER.info(f"[Trainer] 进入最后 {disable_reg_n} 个 epoch，禁用所有正则化项")
            else:
                is_reg_disabled = False
            
            LOGGER.info(f"[Trainer] ========== Epoch {epoch_idx + 1}/{num_epochs} ==========")
            
            # ===== 执行单轮训练 =====
            epoch_result = self._train_single_epoch(epoch_idx, num_epochs, global_step_offset)
            all_epoch_results.append(epoch_result)
            total_update_count += epoch_result.get('update_count', 0)
            global_step_offset += epoch_result.get('total_steps', 0)
            
            # ===== 按配置保存 epoch 级别检查点 =====
            if checkpoint_epoch_interval > 0 and (epoch_idx + 1) % checkpoint_epoch_interval == 0:
                self._save_checkpoint(f'epoch_{epoch_idx + 1}')
        
        # ===== 最终保存 =====
        self._save_checkpoint('final')
        self._save_results()
        
        # ===== 验证阶段（仅在所有 epoch 结束后进行一次）=====
        val_loss = None
        if cfg.validation_steps > 0:
            val_loss = self._validate()
        
        if self.writer:
            self.writer.close()
        
        LOGGER.info(f"[Trainer] 训练完成 | 总轮数: {num_epochs} | 总更新次数: {total_update_count} | 输出: {self.exp_dir}")
        
        return {
            'loss_history': self.loss_history,  # total_loss 历史（所有 epoch 累积）
            'primary_loss_history': self.primary_loss_history,  # 主损失历史
            'reg_loss_history': self.reg_loss_history,  # 正则化损失历史
            'amplitude_history': self.amplitude_history,  # 控制幅值历史（所有 epoch 累积）
            'final_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'final_primary_loss': self.primary_loss_history[-1] if self.primary_loss_history else 0.0,
            'final_reg_loss': self.reg_loss_history[-1] if self.reg_loss_history else 0.0,
            'validation_loss': val_loss,  # 验证 loss（可用于超参数优化 fitness 计算）
            'update_count': total_update_count,
            'num_epochs': num_epochs,
            'epoch_results': all_epoch_results,  # 每轮的详细结果
            'exp_dir': str(self.exp_dir),
        }
    
    def _train_single_epoch(self, epoch_idx: int = 0, num_epochs: int = 1, global_step_offset: int = 0) -> Dict[str, Any]:
        """执行单轮 epoch 训练（累积 loss + 统一 backward）
        
        训练流程：
        1. 预热阶段：只执行仿真，不更新参数
        2. 训练阶段：每步计算 loss 并存储，达到 envstep_per_value 后 backward
        
        Args:
            epoch_idx: 当前 epoch 索引（从 0 开始，用于日志和进度条）
            num_epochs: 总 epoch 数（用于进度条显示）
            global_step_offset: 全局 step 偏移量（用于 TensorBoard 记录连续性）
        
        Returns:
            单轮训练结果字典
        """
        cfg = self.cfg.training
        
        # 计算总环境步数: env_warmup_envsteps + num_interaction_per_mini_episode * control_interval
        total_steps = cfg.env_warmup_envsteps + cfg.num_interaction_per_mini_episode * cfg.control_interval
        env_warmup_envsteps = cfg.env_warmup_envsteps
        
        # 根据是否多轮训练调整日志格式
        epoch_info = f"[Epoch {epoch_idx + 1}/{num_epochs}] " if num_epochs > 1 else ""
        LOGGER.info(f"[Trainer] {epoch_info}开始训练 | 总步数: {total_steps} "
                   f"(预热: {env_warmup_envsteps} + 交互次数: {cfg.num_interaction_per_mini_episode} × 控制间隔: {cfg.control_interval})")
        if epoch_idx == 0:  # 仅在首轮打印详细配置
            LOGGER.info(f"[Trainer] action_duration: {cfg.action_duration} | "
                       f"control_interval: {cfg.control_interval} | "
                       f"num_of_interaction_per_agent_update: {cfg.num_of_interaction_per_agent_update} | "
                       f"envstep_per_value: {self._envstep_per_value}")
        
        from tqdm import tqdm
        env_step = 0
        update_count = 0
        
        # 预计算 reward 权重
        reward_weight_list = self._compute_reward_weights()
        
        # 进度条描述：单轮显示 "Training"，多轮显示 "Epoch X/Y"
        pbar_desc = f"Epoch {epoch_idx + 1}/{num_epochs}" if num_epochs > 1 else "Training"
        pbar = tqdm(total=total_steps, desc=pbar_desc, 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        # ================== 预热阶段 ==================
        # 预热阶段不计算 loss，不更新参数，不保留梯度
        if env_warmup_envsteps > 0:
            pbar.set_postfix_str("Warmup")
        while env_step < env_warmup_envsteps and env_step < total_steps:
            # 使用封装的 _env_step 方法（预热模式）
            _, warmup_apply_jets = self._env_step(env_step, steps_in_accumulation=0, is_warmup=True)
            self._record_amplitude(apply_jets=warmup_apply_jets)
            env_step += 1
            pbar.update(1)
        
        # 预热结束后切断梯度，从干净的状态开始训练
        if env_warmup_envsteps > 0:
            self._detach_solver_state()
            pbar.set_postfix_str(f"Warmup done")

        LOGGER.debug(f"[Trainer] 预热阶段完成 | 预热步数: {env_step} | 开始正式训练")
        
        # ================== 训练阶段 ==================
        # 累积 loss 列表（保持计算图连通）
        step_losses = []  # 存储每步的 total_loss（用于 backward）
        step_primary_losses = []  # 存储每步的 primary_loss（用于记录和显示）
        step_reg_losses = []  # 存储每步的 reg_loss（用于记录）
        steps_in_accumulation = 0
        last_primary_loss_val = 0.0  # 用于进度条显示的主损失
        last_total_loss_val = 0.0  # 用于日志的总损失
        
        while env_step < total_steps:
            # 使用封装的 _env_step 方法（训练模式）
            should_update_control, should_apply_jets = self._env_step(env_step, steps_in_accumulation, is_warmup=False)
            steps_in_accumulation += 1
            
            self._record_amplitude(apply_jets=should_apply_jets)
            
            # ---- 计算 loss（每步计算并存储，不 backward）----
            # _compute_loss 内部计算 reward 并一并返回，避免重复调用 _compute_reward
            primary_loss, reg_loss, total_loss, reward = self._compute_loss(is_net_forward_step=should_update_control)
            step_losses.append(total_loss)  # backward 使用 total_loss
            step_primary_losses.append(primary_loss)
            step_reg_losses.append(reg_loss)
            
            # ---- 记录每步数据（使用 primary_loss 和 reward 用于记录）----
            self._record_step_data(primary_loss, reward)
            
            # ---- 达到 envstep_per_value (or train loop 的最后一步)：加权求和后 backward ----
            if steps_in_accumulation >= self._envstep_per_value or env_step == total_steps - 1:
                LOGGER.debug(f"[Trainer] Env Step: {env_step} | Backward Step")

                # 末次累积步数不足时，按实际步数计算权重。
                if env_step == total_steps - 1 and steps_in_accumulation < self._envstep_per_value:
                    # 重新计算权重列表(累积的环境步总数可能小于 envstep_per_value)
                    reward_weight_list = self._compute_reward_weights(steps_in_accumulation)
                    LOGGER.debug(f"[Trainer] 最后一步训练，累计 loss 步数: {steps_in_accumulation}，小于 envstep_per_value: {self._envstep_per_value}，重新计算权重列表: {reward_weight_list}")
                
                # 加权求和（用于 backward 的是 total_loss）
                weighted_loss = torch.tensor(0.0, device=self.device)
                weighted_primary_loss = torch.tensor(0.0, device=self.device)
                weighted_reg_loss = torch.tensor(0.0, device=self.device)
                for i, loss in enumerate(step_losses):
                    w = reward_weight_list[i] if i < len(reward_weight_list) else reward_weight_list[-1]
                    weighted_loss = weighted_loss + w * loss
                    weighted_primary_loss = weighted_primary_loss + w * step_primary_losses[i]
                weighted_reg_loss = weighted_loss - weighted_primary_loss
                
                self.optimizer.zero_grad()
                weighted_loss.backward()
                
                # 记录梯度范数（用于诊断）
                grad_norm = 0.0
                has_grad = False
                for p in self.policy.network.parameters():
                    if p.grad is not None:
                        has_grad = True
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # 警告：梯度为零可能表示梯度断裂
                if grad_norm < 1e-10:
                    if has_grad:
                        LOGGER.warning(f"[Trainer] 警告: 梯度范数接近零 ({grad_norm:.2e})，可能存在梯度断裂问题")
                    else:
                        LOGGER.warning(f"[Trainer] 警告: 网络参数没有梯度，请检查计算图连通性")
                
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(), cfg.grad_clip)
                
                # ===== 同步 policy_net_old（在 optimizer.step() 前）=====
                # 保存当前（更新前）的网络权重到 policy_net_old
                # 或，当 use_importance_sampling 启用时也需要同步
                if self.cfg.loss.kl_weight > 0 or cfg.use_importance_sampling:
                    self._sync_policy_net_old()
                
                # ===== 更新学习率（在 optimizer.step() 前）=====
                self._update_lr(update_count)
                
                self.optimizer.step()
                
                # 记录（分离 primary_loss 和 reg_loss）
                last_primary_loss_val = float(weighted_primary_loss.detach().cpu().numpy())
                last_reg_loss_val = float(weighted_reg_loss.detach().cpu().numpy())
                last_total_loss_val = float(weighted_loss.detach().cpu().numpy())
                
                # loss_history 保持向后兼容，存储 total_loss
                self.loss_history.append(last_total_loss_val)
                self.primary_loss_history.append(last_primary_loss_val)
                self.reg_loss_history.append(last_reg_loss_val)
                update_count += 1
                
                # TensorBoard 记录：使用全局 step（local step + offset）保持跨 epoch 连续性
                global_step = env_step + global_step_offset
                if self.writer:
                    self.writer.add_scalar('Loss/total', last_total_loss_val, global_step)
                    self.writer.add_scalar('Loss/primary', last_primary_loss_val, global_step)
                    self.writer.add_scalar('Loss/regularization', last_reg_loss_val, global_step)
                    self.writer.add_scalar('Control/amplitude', self.amplitude_history[-1], global_step)
                    self.writer.add_scalar('Gradient/norm', grad_norm, global_step)
                    # 记录当前 epoch（便于在 TensorBoard 中区分）
                    self.writer.add_scalar('Training/epoch', epoch_idx + 1, global_step)
                    # 记录重要性采样比率（如果启用）
                    if cfg.use_importance_sampling and hasattr(self, '_cached_importance_ratio'):
                        self.writer.add_scalar('ImportanceSampling/ratio', self._cached_importance_ratio, global_step)
                
                # DEBUG 级别日志记录梯度信息
                LOGGER.debug(f"[Trainer] Update {update_count} | Primary: {last_primary_loss_val:.6f} | Reg: {last_reg_loss_val:.6f} | Total: {last_total_loss_val:.6f} | Grad: {grad_norm:.6f}")
                
                # 更新进度条
                pbar.set_postfix_str(f"Primary={last_primary_loss_val:.4f} Total={last_total_loss_val:.4f} Upd={update_count}")
                
                # 切断梯度，重置累积
                self._detach_solver_state()
                step_losses = []
                step_primary_losses = []
                step_reg_losses = []
                steps_in_accumulation = 0
            
            # 检查点（使用全局 env_step 计数器，考虑多轮 epoch）
            # 注意：epoch 级别的检查点在 _train_impl() 中处理
            if env_step > 0 and env_step % self.cfg.output.checkpoint_interval == 0:
                self._save_checkpoint(env_step)
            
            env_step += 1
            pbar.update(1)
        
        pbar.close()
        
        # 单轮训练结束日志
        epoch_info = f"[Epoch {epoch_idx + 1}/{num_epochs}] " if num_epochs > 1 else ""
        LOGGER.info(f"[Trainer] {epoch_info}训练完成 | 更新次数: {update_count}")
        
        # 返回单轮训练结果（用于汇总到 _train_impl()）
        epoch_loss_history_len = len(self.loss_history)  # 当前累积的历史长度
        return {
            'epoch_idx': epoch_idx,
            'update_count': update_count,
            'total_steps': total_steps,
            'final_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'final_primary_loss': self.primary_loss_history[-1] if self.primary_loss_history else 0.0,
            'final_reg_loss': self.reg_loss_history[-1] if self.reg_loss_history else 0.0,
            'amplitude_history': list(self.amplitude_history),  # 本轮的 amplitude 历史（每轮重置）
        }
    
    def _record_amplitude(self, apply_jets: bool = True):
        """记录当前控制幅值
        
        Args:
            apply_jets: 当前步是否施加射流。若为 False，记录 0（因为实际未施加控制）
        """
        if not apply_jets:
            # 射流未施加时，实际作用于流场的幅值为 0
            amp = 0.0
        elif self.solver.jet_obstacle and self.solver.jet_obstacle.current_controls:
            amp = float(self.solver.jet_obstacle.current_controls[0][0].detach().cpu())
        else:
            amp = 0.0
        self.amplitude_history.append(amp)
    
    def _record_step_data(self, step_loss: torch.Tensor, step_reward: Optional[torch.Tensor] = None):
        """记录每个 env_step 的训练数据
        
        记录内容：
        - loss 值（primary_loss）
        - reward 值（如果提供）
        - 目标位置的速度 (u, v)
        
        Args:
            step_loss: 当前步的 primary_loss
            step_reward: 当前步的 reward（可选，便于与 DRL 对比）
        """
        # 记录 loss（detach 后保存）
        loss_val = float(step_loss.detach().cpu().numpy())
        self.step_losses.append(loss_val)
        
        # 记录 reward（如果提供）
        if step_reward is not None:
            if not hasattr(self, 'step_rewards'):
                self.step_rewards = []
            reward_val = float(step_reward.detach().cpu().numpy())
            self.step_rewards.append(reward_val)
        
        # 记录目标位置速度
        vel = self.solver.velocity.at(self.target_point)
        u = float(math.mean(vel.values.vector['x']).native().detach().cpu())
        v = float(math.mean(vel.values.vector['y']).native().detach().cpu())
        self.step_velocities.append((u, v))
        
        # 记录目标位置压力
        p_at_target = self.solver.pressure.at(self.target_point)
        p = float(math.mean(p_at_target.values).native().detach().cpu())
        self.step_pressures.append(p)
    
    def _compute_reward_weights(self, n: int = None) -> List[float]:
        """计算每步 reward 累积的权重
        
        Returns:
            权重列表，长度为 envstep_per_value（或实际累积步数 n）
            
        权重模式：
        - 'equal': 等权重，每步 1/n（归一化）
        - 'exponential': 基于 interaction 的指数衰减权重
            每个 interaction：
            - 射流施加阶段（前 action_duration 步）：权重 = 1
            - 射流停止后：权重 = γ^(衰减步数)，持续衰减到更新为止
            多个 interaction 在 interaction 维度求平均
        - 列表: 直接使用给定的权重列表（不归一化）
        """
        cfg = self.cfg.training
        n = self._envstep_per_value if n is None else n  # 实际累积步数
        weights_cfg = cfg.loss_weights
        
        if isinstance(weights_cfg, list):
            # 自定义权重列表（不归一化）
            if len(weights_cfg) != n:
                LOGGER.warning(f"[Trainer] loss_weights 长度 {len(weights_cfg)} != envstep_per_value {n}，将自动截断/补齐")
            weights = list(weights_cfg)
            while len(weights) < n:
                weights.append(weights[-1] if weights else 1.0)
            weights = weights[:n]
        elif weights_cfg == 'exponential':
            # 基于 interaction 的指数衰减权重（不归一化）
            weights = self._compute_interaction_based_reward_weights(n)
        else:  # 'equal' 或默认
            # 等权重（归一化，确保权重和为 1）
            weights = [1.0 / n for _ in range(n)]
        
        return weights
    
    def _compute_interaction_based_reward_weights(self, total_steps: int) -> List[float]:
        """计算基于 interaction 的指数衰减 reward 权重
        
        权重计算逻辑：
        1. 每个 interaction：
           - 射流施加阶段（前 action_duration 步）：权重 = 1
           - 射流停止后：权重 = γ^(衰减步数)，持续衰减到神经网络参数更新为止
        2. 多个 interaction 在 interaction 维度求平均
        
        公式：
        对于第 n 个 interaction（n = 0, 1, ..., M-1）的权重序列：
        (t表示环境步数;
        N表示一段完整的神经网络参数更新周期中的 interaction 数量; 
        K表示每个 interaction 的环境步数;
        M表示从当前 interaction 的初始环境步起到下一次神经网络参数更新的环境步数)
        w^{(n)}_t = 
            0,                              if t < n * K                    (尚未开始)
            1,                              if n*K <= t < n*K + D           (射流施加中)
            γ^{t - n*K - D + 1},            if t >= n*K + D                 (持续衰减至神经网络参数更新)
        
        最终权重：w̄_t = (1/N) * Σ w^{(n)}_t
        
        Args:
            total_steps: 总环境步数（实际累积步数，可能 < envstep_per_value）
            
        Returns:
            权重列表，长度为 total_steps （实际累积步数）
        """
        cfg = self.cfg.training
        K = cfg.control_interval      # 每个 interaction 的步数
        D = cfg.action_duration       # 射流施加步数
        gamma = cfg.loss_gamma        # 衰减率
        
        # 计算 interaction 数量（向上取整以处理不满的情况）
        N = max(1, (total_steps + K - 1) // K)  # ceil(total_steps / K)
        
        # 构建权重矩阵 W[n, t]，N 个 interaction，每个 total_steps 步
        # W[n, t] 表示第 n 个 interaction 在第 t 个环境步的权重
        W = np.zeros((N, total_steps))
        
        for n in range(N):
            start = n * K  # 第 n 个 interaction 开始的环境步
            if start >= total_steps:
                break  # 该 interaction 尚未开始
            
            # 从 start 一直持续到 total_steps（而非 (n+1)*K）
            for t in range(start, total_steps):
                local_t = t - start  # 从该 interaction 开始算起的步数
                if local_t < D:
                    # 射流施加阶段，权重 = 1
                    W[n, t] = 1.0
                else:
                    # 射流停止后，持续衰减直到更新
                    decay_steps = local_t - D + 1  # 从 1 开始衰减
                    W[n, t] = gamma ** decay_steps
        
        # 在 interaction 维度求平均
        weights = np.mean(W, axis=0).tolist()
        
        LOGGER.debug(f"[Trainer] Reward weights: N={N}, K={K}, D={D}, γ={gamma}, T={total_steps}")
        LOGGER.debug(f"[Trainer] Weight matrix shape: {W.shape}")
        LOGGER.debug(f"[Trainer] Final weights (first 10): {weights[:10]}")
        
        return weights

    def _save_checkpoint(self, step_or_name):
        """保存检查点"""
        path = self.exp_dir / 'checkpoints' / f'policy_{step_or_name}.pt'
        self.policy.save_weights(str(path))
    
    def _save_results(self):
        """保存训练结果（包含每步详细数据和可视化）"""
        import matplotlib.pyplot as plt
        
        # ---- 保存详细训练数据 ----
        # 提取速度分量
        velocities_u = [v[0] for v in self.step_velocities] if self.step_velocities else []
        velocities_v = [v[1] for v in self.step_velocities] if self.step_velocities else []
        pressures = self.step_pressures if self.step_pressures else []
        target_u, target_v = self.cfg.loss.target_velocity
        
        # 提取 reward 数据（如果有）
        step_rewards = getattr(self, 'step_rewards', [])
        
        # 保存到 npz 文件
        save_data = {
            # 每次更新的汇总 loss
            'loss_history': np.array(self.loss_history),
            # 每步详细数据
            'step_losses': np.array(self.step_losses),
            'step_velocities_u': np.array(velocities_u),
            'step_velocities_v': np.array(velocities_v),
            'step_pressures': np.array(pressures),
            'amplitude_history': np.array(self.amplitude_history),
            # 目标值
            'target_velocity': np.array([target_u, target_v]),
        }
        # 如果有 reward 数据，也保存
        if step_rewards:
            save_data['step_rewards'] = np.array(step_rewards)
        
        np.savez(self.exp_dir / 'training_data.npz', **save_data)
        
        # 保存为 CSV（更易于分析）
        if self.step_losses:
            import pandas as pd
            csv_data = {
                'env_step': range(len(self.step_losses)),
                'loss': self.step_losses,
                'velocity_u': velocities_u,
                'velocity_v': velocities_v,
                'pressure': pressures,
                'amplitude': self.amplitude_history[:len(self.step_losses)],
            }
            # 如果有 reward 数据，也保存到 CSV
            if step_rewards and len(step_rewards) == len(self.step_losses):
                csv_data['reward'] = step_rewards
            df = pd.DataFrame(csv_data)
            df.to_csv(self.exp_dir / 'training_step_data.csv', index=False)
        
        # ---- 可视化 ----
        if not self.step_losses:
            return
        
        # 创建 2x2 子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # (0,0) 每步 Loss
        ax = axes[0, 0]
        ax.plot(self.step_losses, alpha=0.7, linewidth=0.8, label='Step Loss')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Loss')
        ax.set_title('Per-Step Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # (0,1) 参数更新时的加权 Loss
        ax = axes[0, 1]
        if self.loss_history:
            ax.plot(self.loss_history, 'o-', markersize=3, label='Weighted Loss (per update)')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Weighted Loss')
            ax.set_title('Training Loss per Update')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # (1,0) 目标位置速度
        ax = axes[1, 0]
        ax.plot(velocities_u, label='u (measured)', alpha=0.8)
        ax.plot(velocities_v, label='v (measured)', alpha=0.8)
        ax.axhline(y=target_u, color='blue', linestyle='--', alpha=0.5, label=f'u_target={target_u:.3f}')
        ax.axhline(y=target_v, color='orange', linestyle='--', alpha=0.5, label=f'v_target={target_v:.3f}')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Velocity')
        ax.set_title('Target Probe Velocity & Pressure')
        # 压力曲线（右轴）
        ax2 = ax.twinx()
        ax2.plot(pressures, 'g-', label='p', alpha=0.6, linewidth=0.8)
        ax2.set_ylabel('Pressure', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # (1,1) 控制信号幅值
        ax = axes[1, 1]
        ax.plot(self.amplitude_history, label='Amplitude', color='green', alpha=0.8)
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Amplitude')
        ax.set_title('Control Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(self.exp_dir / 'training_curves.png', dpi=150)
        plt.close(fig)
        
        LOGGER.info(f"[Trainer] 训练数据已保存: {self.exp_dir}")
    
    def _validate(self):
        """
        训练后验证（使用 SolverRunner 进行数据记录和可视化）
        
        验证流程：
        1. 调用 _reset_env 重置环境到初始状态
        2. 使用构造函数中创建的 validation_runner 实例
        3. 运行验证步数，记录力系数、探针、场数据
        4. SolverRunner 自动处理 CSV/图表/视频保存
        
        注意: runner 在构造函数的 _setup_validation_runner() 中创建并缓存到 self.runners 列表。
        """
        LOGGER.info(f"[Trainer] 开始验证 ({self.cfg.training.validation_steps} 步)...")
        
        cfg = self.cfg.training
        
        # 使用统一的环境重置方法
        self._reset_env(reason='validation')
        
        # 使用预创建的 validation_runner（重置缓存以清空旧数据）
        runner = self.validation_runner
        runner.reset_caches()
        val_output_dir = self._validation_output_dir
        
        # 验证数据记录
        val_losses = []
        val_velocities = []
        val_pressures = []
        val_amplitudes = []
        
        self.policy.eval()
        
        # 验证阶段使用确定性策略（不随机采样）
        prev_deterministic = False
        if self.solver.jet_obstacle:
            prev_deterministic = self.solver.jet_obstacle.deterministic
            self.solver.jet_obstacle.deterministic = True
        
        with torch.no_grad():
            from tqdm import tqdm
            steps_in_accumulation = 0
            for env_step in tqdm(range(cfg.validation_steps), desc='Validation'):
                is_warmup = (env_step < cfg.env_warmup_envsteps)
                
                # 使用封装的 _env_step 方法（验证模式）
                _, should_apply_jets = self._env_step(env_step, steps_in_accumulation, is_warmup=is_warmup)
                
                if is_warmup:
                    val_losses.append(np.nan)
                    val_velocities.append((np.nan, np.nan))
                    val_pressures.append(np.nan)
                    val_amplitudes.append(0.0)
                else:
                    steps_in_accumulation += 1
                    
                    # 记录 loss（使用 primary_loss 作为验证指标）
                    primary_loss, _, _, _ = self._compute_loss()
                    val_losses.append(float(primary_loss.cpu().numpy()))
                    
                    # 记录目标位置速度
                    vel = self.solver.velocity.at(self.target_point)
                    u = float(math.mean(vel.values.vector['x']).native().cpu())
                    v = float(math.mean(vel.values.vector['y']).native().cpu())
                    val_velocities.append((u, v))
                    
                    # 记录目标位置压力
                    p_at_target = self.solver.pressure.at(self.target_point)
                    p = float(math.mean(p_at_target.values).native().cpu())
                    val_pressures.append(p)
                    
                    # 记录控制幅值（当 should_apply_jets=False 时记录 0）
                    if not should_apply_jets:
                        amp = 0.0
                    elif self.solver.jet_obstacle and self.solver.jet_obstacle.current_controls:
                        amp = float(self.solver.jet_obstacle.current_controls[0][0].cpu())
                    else:
                        amp = 0.0
                    val_amplitudes.append(amp)
                
                # 使用 runner 记录数据（每步都记录）
                step_num = env_step + 1
                if runner.cfg.force_coeffs.enabled:
                    runner._record_force_coeffs()
                if runner.cfg.probes.enabled:
                    runner._record_probes()
                if runner.cfg.visuals.enabled and step_num % runner.cfg.visuals.interval == 0:
                    runner._save_visual(step_num)
                if runner.cfg.fields.save.enabled and step_num % runner.cfg.fields.save.interval == 0:
                    runner._save_fields_h5()
        
        self.policy.train()
        
        # 恢复验证前的 deterministic 设置
        if self.solver.jet_obstacle:
            self.solver.jet_obstacle.deterministic = prev_deterministic
        
        # 保存 runner 收集的 CSV 和图表
        if runner.cfg.output.enabled:
            runner._save_csvs()
            if runner.force_coeffs:
                runner._plot_force_coeffs()
            runner._generate_videos()
        
        # 关闭 h5 文件
        if runner._h5_file is not None:
            runner._h5_file.close()
        
        # 计算验证损失均值
        avg_loss = float(np.nanmean([l for l in val_losses if not np.isnan(l)]))
        LOGGER.info(f"[Trainer] 验证完成 | 平均 Loss: {avg_loss:.6f}")
        
        # ---- 保存额外的验证数据（loss、速度、幅值曲线）----
        self._save_validation_curves(val_losses, val_velocities, val_pressures, val_amplitudes, val_output_dir)
        
        return avg_loss
    
    def _save_validation_curves(
        self, 
        val_losses: List[float], 
        val_velocities: List[Tuple[float, float]], 
        val_pressures: List[float],
        val_amplitudes: List[float],
        output_dir: Path
    ):
        """保存验证阶段曲线数据并可视化
        
        统一格式:
        - NPZ 和 CSV 使用相同键名
        - 必选: env_step, amplitude
        - 可选: loss, velocity_u, velocity_v, pressure
        - 元数据: target_velocity (仅 NPZ)
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        n_steps = len(val_losses) if val_losses else len(val_amplitudes)
        
        # 提取速度分量
        velocities_u = [v[0] for v in val_velocities] if val_velocities else []
        velocities_v = [v[1] for v in val_velocities] if val_velocities else []
        target_u, target_v = self.cfg.loss.target_velocity
        
        # ===== 构建统一数据字典 =====
        data_dict = {
            'env_step': np.arange(n_steps),
            'amplitude': np.array(val_amplitudes),
        }
        if val_losses:
            data_dict['loss'] = np.array(val_losses)
        if velocities_u:
            data_dict['velocity_u'] = np.array(velocities_u)
            data_dict['velocity_v'] = np.array(velocities_v)
        data_dict['pressure'] = np.array(val_pressures)
        
        # ===== 保存 NPZ (含元数据) =====
        np.savez(
            output_dir / 'validation_curves.npz',
            **data_dict,
            target_velocity=np.array([target_u, target_v]),
        )
        
        # ===== 保存 CSV (不含元数据) =====
        df = pd.DataFrame(data_dict)
        df.to_csv(output_dir / 'validation_curves.csv', index=False)
        
        # ---- 可视化 ----
        if not val_losses and not val_amplitudes:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # (0) Loss
        ax = axes[0]
        if val_losses:
            ax.plot(val_losses, alpha=0.8, label='Validation Loss')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # (1) 目标位置速度
        ax = axes[1]
        if velocities_u:
            ax.plot(velocities_u, label='u (measured)', alpha=0.8)
            ax.plot(velocities_v, label='v (measured)', alpha=0.8)
            ax.axhline(y=target_u, color='blue', linestyle='--', alpha=0.5, label=f'u_target={target_u:.3f}')
            ax.axhline(y=target_v, color='orange', linestyle='--', alpha=0.5, label=f'v_target={target_v:.3f}')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Velocity')
        ax.set_title('Validation Probe Velocity & Pressure')
        ax.grid(True, alpha=0.3)
        # 压力曲线（右轴）
        ax2 = ax.twinx()
        ax2.plot(val_pressures, 'g-', label='p', alpha=0.6, linewidth=0.8)
        ax2.set_ylabel('Pressure', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        
        # (2) 控制信号幅值
        ax = axes[2]
        ax.plot(val_amplitudes, label='Amplitude', color='green', alpha=0.8)
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Amplitude')
        ax.set_title('Validation Control Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        # 禁用偏移量表示法，避免数据变化范围小但绝对值不小时出现混乱的坐标标记
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'validation_curves.png', dpi=150)
        plt.close(fig)
        
        LOGGER.info(f"[Trainer] 验证曲线已保存: {output_dir}")
