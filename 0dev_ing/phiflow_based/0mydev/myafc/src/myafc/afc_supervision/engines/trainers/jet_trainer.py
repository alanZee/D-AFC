"""
JetTrainer - 射流控制训练器

基于可微分模拟的 AFC 训练器，用于:
- 柱体绕流问题 (方柱/圆柱)
- 射流主动流动控制

核心特性:
1. 端到端可微分训练 (PhiFlow Torch 后端)
2. 多步 loss 累积 + 统一 backward
3. 梯度链路保持 → 分段释放计算图
4. 多环境并行训练 (ParallelSolverManager)
5. 重要性采样 (PPO 风格)
6. 历史状态堆叠 (envstep_state)
"""
from __future__ import annotations

# 设置 matplotlib 后端为 Agg (非交互式)，避免多线程 tkinter 问题
import matplotlib
matplotlib.use('Agg')

import io
import gc
import os
import re
import math as pymath
from contextlib import nullcontext
from collections import deque
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field as dataclass_field, asdict
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from phi.torch.flow import math, Point, vec, field

from .base_trainer import BaseTrainer, TrainingBaseConfig, OutputBaseConfig
from ....env_kernels.solver_wrappers import SolverRunner
from ...agents import AFCAgent
from ...agents.afc_agent import AFCAgentConfig
# === Loss 函数导入 ===
from ...losses import (
    # 探针 loss
    probe_mse_loss,
    probe_mae_loss,
    # 力系数 loss
    force_coeffs_loss,
    drl_reward_loss,
    compute_control_power,
    vortex_shedding_loss,
    # 流场 loss
    field_rms_loss,
    fft_mode_loss,
    fluctuation_sparse_loss,
    multi_field_loss,
    compute_vorticity,
    velocity_magnitude,
    extract_region,
    # 模态 loss
    pod_mode_loss,
    dmd_mode_loss,
    steady_state_residual_loss,
)
from ...losses.regularization import (
    amplitude_l2_loss, 
    temporal_smoothness_loss, 
    entropy_loss, 
    kl_divergence_loss,
    policy_stability_loss,
)
from myafc.utils import (
    get_logger,
    load_env_config,
    ensure_target_probe,
    build_training_runner_config,
    create_runner,
    tee_console_output,
)

LOGGER = get_logger()


# ============================================================
# 科学计数法转换 (处理 YAML 解析问题)
# ============================================================
def _convert_scientific_notation(cfg_dict: Dict, depth: int = 0) -> Dict:
    """递归转换科学计数法字符串为 float
    
    解决 YAML 解析器将 '1e-3' 读取为字符串的问题
    """
    if depth > 20:  # 防止无限递归
        return cfg_dict
    
    result = {}
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            result[key] = _convert_scientific_notation(value, depth + 1)
        elif isinstance(value, str):
            # 尝试转换科学计数法
            try:
                if 'e' in value.lower() or 'E' in value:
                    result[key] = float(value)
                else:
                    result[key] = value
            except ValueError:
                result[key] = value
        else:
            result[key] = value
    return result


def _resolve_resume_range_candidates(
    available_steps: List[int],
    resume_range: Union[List[int], Tuple[int, int]],
) -> List[int]:
    """将 `solver_initial_resume_step: [a, b]` 解析为可随机抽取的真实物理时间步列表。

    语义说明：
        - 非负端点：按“物理时间步值”解释（如 500、3000）
        - 负端点：按“从末尾倒数第 n 个物理时间步”解释
          例如：-1=最后一个物理时间步，-10=倒数第 10 个物理时间步
        - 端点顺序不限：自动归一化为 [min, max]

    注意：
        边界可落在未保存时间步上，最终会在边界区间内筛选“已保存帧”。
    """
    if not available_steps:
        raise ValueError("H5 文件中无可用时间步")

    if len(resume_range) != 2:
        raise ValueError(f"resume_range 长度必须为 2，当前: {resume_range}")

    start_raw = int(resume_range[0])
    end_raw = int(resume_range[1])
    last_physical_step = int(available_steps[-1])

    def _endpoint_to_step(endpoint: int) -> int:
        if endpoint >= 0:
            return endpoint

        # -1 -> last, -2 -> last-1, ...
        return last_physical_step + endpoint + 1

    start_step = _endpoint_to_step(start_raw)
    end_step = _endpoint_to_step(end_raw)
    min_step, max_step = sorted((start_step, end_step))

    return [s for s in available_steps if min_step <= s <= max_step]


def _resolve_resume_range_bounds(
    available_steps: List[int],
    resume_range: Union[List[int], Tuple[int, int]],
) -> Tuple[int, int]:
    """解析 `solver_initial_resume_step: [a, b]` 对应的物理时间步边界（归一化后）。"""
    if not available_steps:
        raise ValueError("H5 文件中无可用时间步")

    if len(resume_range) != 2:
        raise ValueError(f"resume_range 长度必须为 2，当前: {resume_range}")

    start_raw = int(resume_range[0])
    end_raw = int(resume_range[1])
    last_physical_step = int(available_steps[-1])

    def _endpoint_to_step(endpoint: int) -> int:
        if endpoint >= 0:
            return endpoint
        return last_physical_step + endpoint + 1

    start_step = _endpoint_to_step(start_raw)
    end_step = _endpoint_to_step(end_raw)
    return tuple(sorted((start_step, end_step)))


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class InputFeatureConfig:
    """输入特征配置
    
    控制神经网络接收哪些主要输入特征，两个独立维度可任意组合:
    
    1. 探针选择 (main_input_type):
       - None:           开环控制，仅使用 target_velocity
       - 'target_probe':  仅使用 loss.target_point 定义的单个目标探针
       - 'probes':        使用 env_config 中所有探针 + target 探针
    
    2. 字段选择 (probe_fields):
       - ['u', 'v', 'p'] 的任意子集，如 ['p'] 或 ['u', 'v']
       - 两个维度完全独立，任意组合均有效
    
    其他:
       - envstep_state: 历史状态堆叠步数 (1=仅当前, n=堆叠最近n步)
       - include_target_velocity: 是否将目标速度作为额外输入
       - include_prev_amplitude_num: 历史幅值输入数量
       - include_force_coeffs: 是否包含力系数 [Cd, Cl] 作为额外输入
    """
    main_input_type: Optional[str] = 'target_probe'
    probe_fields: List[str] = dataclass_field(default_factory=lambda: ['u', 'v', 'p'])
    include_target_velocity: bool = True
    include_prev_amplitude_num: int = 1
    include_force_coeffs: bool = False
    envstep_state: int = 1


@dataclass
class TrainingVisualsConfig:
    """训练阶段流场可视化配置
    
    控制训练过程中流场的保存和可视化
    输出目录结构: exp_dir/{folder}/
        ├── raw/          # 原始流场数据 (fields.h5)
        └── frames/       # 可视化图片
    
    Attributes:
        enabled: 是否启用训练可视化 (False = 禁用)
        interval: 每隔多少个 env-step 保存一帧 (10 = 每 10 步保存)
        quantities: 绘制的物理量列表 ['velocity_magnitude', 'pressure', 'vorticity']
        save_endpoints: 是否保存初始帧和最终帧 (_final)
        folder: 输出子目录名称 (默认 'training_fields')
        smooth: 是否平滑插值热力图
        video_fps: 合成视频帧率
        generate_video: 是否生成视频 (False = 仅保存图片，不合成视频)
        obstacle_style: 障碍物风格 'outline'/'fill'/'both'/'none'
    """
    enabled: bool = False  # 默认禁用，避免训练时占用 IO
    interval: int = 10     # 每 10 步保存一帧
    quantities: List[str] = dataclass_field(default_factory=lambda: ['velocity_magnitude', 'pressure'])
    save_endpoints: bool = True
    folder: str = 'training_fields'  # 输出目录 (raw + frames)
    smooth: bool = True
    video_fps: int = 10
    generate_video: bool = True  # 默认生成视频
    obstacle_style: str = 'fill'


@dataclass
class InletConfig:
    """入口边界配置
    
    控制入口边界条件模式:
    - mode: null (不覆盖，使用 env_config) / 'fixed' / 'zero_gradient' / 'sinusoidal'
    - fixed_velocity: 固定入口速度
    - amplitude/frequency/phase_deg/offset: sinusoidal 参数
    """
    mode: Optional[str] = None  # null = 不覆盖，使用 env_config_path 中的配置
    fixed_velocity: float = 0.233
    amplitude: float = 0.05
    frequency: float = 0.2
    phase_deg: float = 0.0
    offset: List[float] = dataclass_field(default_factory=lambda: [0.233, 0.0])


@dataclass
class LossConfig:
    """Loss 配置
    
    Loss 类型说明:
    ============
    1. 探针 Loss (probe_*)
       - 基于目标探针位置的速度 MSE/MAE
       
    2. 力系数 Loss (force_*)
       - 阻力/升力系数 MSE
       - DRLinFluids 风格 reward-based loss
       - 涡脱抑制 loss
       
    3. 流场 Loss (field_*)
       - RMS: 区域流场均方根
       - FFT: 傅里叶模态幅值
       - 波动稀疏: 自动定位高波动点
       
    4. 模态 Loss (modal_*)
       - POD: 最小化非稳态模态能量
       - DMD: 最小化非稳态模态幅值
       - 稳态残差: 原场减稳态模态
       
    5. 正则化 (*_weight 后缀)
    
    Attributes:
        # === 主 Loss 类型选择 ===
        # 通过设置权重 > 0 来启用相应的 loss 类型
        # 多个 loss 可同时启用，按权重加权求和
        
        # --- 探针 Loss ---
        probe_mse_weight: float = 1.0   # 探针 MSE 权重
        probe_mae_weight: float = 0.0   # 探针 MAE 权重
        
        # --- 力系数 Loss ---
        force_weight: float = 0.0       # 基础力系数 loss 权重
        force_drl_weight: float = 0.0   # DRLinFluids 风格 reward loss 权重
        force_vortex_weight: float = 0.0 # 涡脱抑制 loss 权重
        
        # --- 流场 Loss ---
        field_rms_weight: float = 0.0   # 流场 RMS 权重
        field_fft_weight: float = 0.0   # FFT 模态 loss 权重
        field_fluctuation_weight: float = 0.0  # 波动稀疏 loss 权重
        
        # --- 模态 Loss ---
        modal_pod_weight: float = 0.0   # POD 模态 loss 权重
        modal_dmd_weight: float = 0.0   # DMD 模态 loss 权重
        modal_residual_weight: float = 0.0  # 稳态残差 loss 权重
        
        # === 目标值 ===
        target_probe_position: 目标探针位置 (相对坐标)
        target_velocity: 目标速度 [u, v]
        cd_target: 目标阻力系数
        cl_target: 目标升力系数
        
        # === DRLinFluids Reward 参数 (论文原始公式) ===
        cd_baseline: 无控制基线 Cd (如 Re=100 方柱为 1.549)
        alpha_lift: 升力惩罚系数 (论文默认 0.1)
        
        # === 流场 Loss 配置 ===
        field_loss_region: 区域边界 [x_start, y_start, x_end, y_end] (相对坐标)
        field_velocity_weight: 速度场子权重
        field_vorticity_weight: 涡量场子权重
        field_pressure_weight: 压力场子权重
        field_fft_n_modes: FFT 考虑的模态数
        field_fft_exclude_dc: 是否排除 DC 分量
        field_fluctuation_top_k: 波动稀疏选取的 top-k 点
        
        # === 模态 Loss 配置 ===
        modal_n_modes: 模态分解的模态数
        modal_snapshot_buffer_size: 快照缓冲区大小
        modal_exclude_first_k: POD 排除前 k 个主模态
        
        # === 时间维度权重 ===
        loss_weights: 'equal'/'exponential'
        loss_gamma: 指数衰减因子(for 'exponential')
        
        # === 正则化 ===
        amplitude_l2_weight: 幅值 L2 正则化权重
        temporal_smoothness_weight: 时序平滑权重
        action_rate_limit: 动作变化率软约束阈值 (null/0 = 标准 L2)
        policy_stability_weight: 策略稳定性权重 (EMA 偏离惩罚)
        policy_stability_momentum: 策略稳定性 EMA 动量
        entropy_weight: 熵正则化权重
        kl_weight: KL 散度权重
        
        # === 最后 N 个 epoch 禁用正则化 ===
        disable_regularization_last_epochs: 在最后 N 个 epoch 禁用所有正则化项
            0 = 不禁用（默认），n = 最后 n 个 epoch 禁用
            'all' 或 n >= epochs = 所有 epoch 都禁用正则化
    """
    # ===== 探针 Loss =====
    probe_mse_weight: float = 1.0
    probe_mae_weight: float = 0.0
    
    # ===== 力系数 Loss =====
    force_weight: float = 0.0
    force_drl_weight: float = 0.0      # DRLinFluids reward-based
    force_vortex_weight: float = 0.0   # 涡脱抑制
    
    # ===== 流场 Loss =====
    field_rms_weight: float = 0.0
    field_fft_weight: float = 0.0
    field_fluctuation_weight: float = 0.0
    
    # ===== 模态 Loss =====
    modal_pod_weight: float = 0.0
    modal_dmd_weight: float = 0.0
    modal_residual_weight: float = 0.0
    
    # ===== 目标值 =====
    target_probe_position: List[float] = dataclass_field(default_factory=lambda: [0.53, 0.53])
    target_velocity: List[float] = dataclass_field(default_factory=lambda: [0.0, 0.233])
    cd_target: float = 0.0
    cl_target: float = 0.0
    
    # ===== DRLinFluids Reward 参数 (论文原始公式) =====
    cd_baseline: float = 1.549   # Re=100 方柱无控制基线 Cd
    alpha_lift: float = 0.1      # 升力惩罚系数 (论文默认 0.1)
    
    # ===== DRLinFluids 时间平均配置 =====
    # 对 DRL loss 中的 Cd/Cl 进行时间平均 (类似论文中的处理)
    # 取值:
    #   null 或 1: 不进行时间平均，使用瞬时值 (默认)
    #   正整数 N: 对最近 N 个 env-step 的 Cd/Cl 进行滑动平均
    #   'action_duration': 自动设为 action_duration 的值
    #   'control_interval': 自动设为 control_interval 的值
    force_drl_avg_window: Optional[Union[int, str]] = None
    
    # 是否对 Cl 取绝对值后再进行时间平均
    # True: 平均 |Cl| (即 mean(|Cl_t|))，适合抑制升力振荡
    # False: 平均 Cl (即 mean(Cl_t))，可能因正负抵消趋近零 (默认)
    force_drl_cl_abs: bool = False
    
    # ===== 涡脱抑制配置 =====
    vortex_cl_history_len: int = 50      # Cl 历史长度 (用于 FFT 分析)
    
    # ===== 流场 Loss 配置 =====
    field_loss_region: Optional[List[float]] = None  # None = 下游默认区域
    field_velocity_weight: float = 1.0   # 多场加权中速度场权重
    field_vorticity_weight: float = 0.0  # 多场加权中涡量场权重
    field_pressure_weight: float = 0.0   # 多场加权中压力场权重
    field_fft_n_modes: int = 5           # FFT 考虑的模态数
    field_fft_exclude_dc: bool = True    # 排除 DC 分量
    field_fluctuation_top_k: int = 100   # 波动稀疏选取点数
    field_fluctuation_agg: str = 'mean'  # 波动稀疏聚合方式: 'mean'/'sum'/'max'
    
    # ===== 模态 Loss 配置 =====
    modal_n_modes: int = 5               # 模态分解的模态数
    modal_snapshot_buffer_size: int = 50 # 快照缓冲区大小
    modal_exclude_first_k: int = 1       # POD 排除前 k 个主模态
    
    # ===== 时间维度权重 =====
    loss_weights: str = 'exponential'
    loss_gamma: float = 0.99
    
    # ===== 正则化 =====
    amplitude_l2_weight: float = 0.0
    temporal_smoothness_weight: float = 0.01
    action_rate_limit: Optional[float] = None
    policy_stability_weight: float = 0.0
    policy_stability_momentum: float = 0.95
    entropy_weight: float = 0.01
    kl_weight: float = 0.0
    
    # ===== 最后 N 个 epoch 禁用正则化 =====
    # 在最后 N 个 epoch 禁用所有正则化项，让策略充分收敛
    # 取值: 0 = 不禁用（默认），n = 最后 n 个 epoch 禁用，'all' = 所有 epoch 禁用
    # 当 n >= epochs 时等效于 'all'
    disable_regularization_last_epochs: Union[int, str] = 0


@dataclass
class JetTrainingConfig(TrainingBaseConfig):
    """射流控制训练配置
    
    继承 TrainingBaseConfig，添加 AFC 特有参数
    
    控制周期参数说明:
        action_duration: 控制信号持续施加的环境步数
        control_interval: 多少个环境步更新一次控制信号
            - 整数: 具体步数
            - 'equal2action_duration': 自动与 action_duration 相同 (推荐)
        
    Attributes:
        warmup_momentum: 预热阶段动量初始值 (仅 SGD 有效)
            优化器动量从 warmup_momentum 线性增长到 optimizer_momentum
        
        warmup_apply_jets: 预热阶段是否施加射流边界条件
            **重要**: 默认 True，避免从 apply_jets=False 切换到 True 时压力场剧变导致 Cd 尖峰。
            - True（默认）: 预热阶段使用 JetObstacle 进行压力投影（控制信号仍为 0）
            - False: 预热阶段使用普通障碍物（可能导致切换时 Cd 尖峰）
        
        env_warmup_envsteps: 预热阶段步数
            **建议**: 当从 h5 加载无射流稳态流场时，建议设置 >=50 步以让流场
            适应 JetObstacle 的边界条件。流场在前几步可能出现过渡性的力系数波动。
    """
    # 训练循环
    env_warmup_envsteps: int = 50  # 建议 >=50 步以让流场适应 JetObstacle 边界条件
    warmup_update_controls: bool = True
    warmup_apply_jets: bool = True  # 默认 True，避免 apply_jets 切换导致的 Cd 尖峰
    warmup_momentum: float = 0.8  # 预热动量初始值
    action_duration: int = 26
    control_interval_raw: Union[int, str] = 26  # 原始配置值，支持 'equal2action_duration'
    num_of_interaction_per_agent_update: int = 2
    num_interaction_per_mini_episode: int = 100

    # 长时序反传显存优化（不改变梯度语义）
    # 原理: 使用 torch.autograd.graph.save_on_cpu() 将反传所需中间张量暂存到 CPU，
    #      backward 时再回传到 GPU 计算梯度。相比 TBPTT，不会截断跨时间步梯度。
    # 建议: 长更新窗口（如 >= 500 env-steps / update）时启用。
    autograd_offload_to_cpu: bool = True
    autograd_offload_pin_memory: bool = True
    
    # 多环境配置
    # - num_envs: 总环境数量
    #   - 1: 仅使用基础环境（无派生）
    #   - n (n>1): 1个基础环境 + (n-1)个派生环境 = n个环境
    # - env_variants: 派生环境配置变体列表，长度必须为 num_envs-1
    num_envs: int = 1
    env_variants: Optional[List[Dict]] = None  # 长度 = num_envs-1
    
    # 随机初始帧配置 (覆盖 env_config 中的 solver.initial.resume_step)
    # 用于增加训练多样性，每个 epoch 使用不同的初始流场状态
    # - null: 使用 env_config 中的配置
    # - int: 固定帧索引
    # - [min, max]: 每个 epoch 从 [min, max] 范围内随机选择
    #   注1: 负索引表示“从最后物理时间步向前倒数”（-1=最后一个物理时间步）
    #   注2: 边界可落在未保存步上，内部会在该物理区间筛选已保存帧
    #   注3: 支持混合/逆序端点（如 [100, -100]），内部会自动归一化
    solver_initial_resume_step: Optional[Union[int, List[int]]] = None
    
    # 多环境是否同步使用相同的初始帧索引
    # - true: 所有环境使用同一随机帧（epoch 间不同，env 间相同）
    # - false: 每个环境独立随机选择帧（epoch 和 env 间均不同）
    sync_resume_step_across_envs: bool = True
    
    # 重要性采样 (PPO 风格)
    use_importance_sampling: bool = False
    importance_sampling_clip_range: float = 0.2
    
    # [Global 全局参数] 控制平滑 - 同时作用于训练和验证
    smoothing_alpha: Optional[float] = None  # 覆盖 solver jets.smoothing_alpha
    
    # 噪声增强（训练阶段）
    noise_envs: bool = False           # 是否对初始速度场施加噪声
    noise_observation: bool = False    # 是否对观测施加噪声
    noise_std: float = 0.01            # 噪声标准差（相对于参考速度）
    
    # 噪声增强（验证阶段）
    # 启用后，验证阶段也会施加与训练相同的噪声增强
    # 用于评估策略对噪声的鲁棒性
    validation_noise_enabled: bool = False
    
    # 验证
    validation_steps: int = 100
    
    # 检查点保存间隔 (epoch)
    checkpoint_epoch_interval: int = 50
    
    # 注: checkpoint 和 resume 参数继承自 TrainingBaseConfig
    
    @property
    def control_interval(self) -> int:
        """获取实际的 control_interval 值
        
        支持语法糖:
          - 'action_duration' 或 'equal2action_duration': 等于 action_duration
        """
        if isinstance(self.control_interval_raw, str):
            if self.control_interval_raw.lower() in ('action_duration', 'equal2action_duration'):
                return self.action_duration
            raise ValueError(f"不支持的 control_interval 值: {self.control_interval_raw}")
        return self.control_interval_raw


@dataclass
class JetTrainerConfig:
    """JetTrainer 完整配置"""
    env_config_path: str = ''
    training: JetTrainingConfig = dataclass_field(default_factory=JetTrainingConfig)
    agent: AFCAgentConfig = dataclass_field(default_factory=AFCAgentConfig)
    input_features: InputFeatureConfig = dataclass_field(default_factory=InputFeatureConfig)
    inlet: InletConfig = dataclass_field(default_factory=InletConfig)
    loss: LossConfig = dataclass_field(default_factory=LossConfig)
    output: OutputBaseConfig = dataclass_field(default_factory=OutputBaseConfig)
    training_visuals: TrainingVisualsConfig = dataclass_field(default_factory=TrainingVisualsConfig)


def _parse_jet_trainer_config(cfg_dict: Dict) -> JetTrainerConfig:
    """从字典解析配置"""
    # 首先处理科学计数法
    cfg_dict = _convert_scientific_notation(cfg_dict)
    
    config = JetTrainerConfig()
    config.env_config_path = cfg_dict.get('env_config_path', '')
    
    if 'training' in cfg_dict:
        training_dict = cfg_dict['training'].copy()
        # 处理 control_interval -> control_interval_raw 的映射
        if 'control_interval' in training_dict:
            training_dict['control_interval_raw'] = training_dict.pop('control_interval')
        # 过滤掉不属于 JetTrainingConfig 的字段
        training_dict = {k: v for k, v in training_dict.items() 
                        if hasattr(JetTrainingConfig, k)}
        config.training = JetTrainingConfig(**training_dict)
    if 'agent' in cfg_dict:
        config.agent = AFCAgentConfig(**cfg_dict['agent'])
    if 'input_features' in cfg_dict:
        config.input_features = InputFeatureConfig(**cfg_dict['input_features'])
    if 'inlet' in cfg_dict:
        config.inlet = InletConfig(**cfg_dict['inlet'])
    if 'loss' in cfg_dict:
        config.loss = LossConfig(**cfg_dict['loss'])
    if 'output' in cfg_dict:
        config.output = OutputBaseConfig(**cfg_dict['output'])
    if 'training_visuals' in cfg_dict:
        config.training_visuals = TrainingVisualsConfig(**cfg_dict['training_visuals'])
    
    return config


# ============================================================
# JetTrainer 主类
# ============================================================
class JetTrainer(BaseTrainer):
    """射流控制训练器
    
    训练流程:
    1. Warmup: 预热 env_warmup_envsteps 步 (不控制, 填充历史)
    2. Training: 多个 interaction, 每个 = control_interval 环境步
    3. 累积 envstep_per_value 步 loss 后 backward + step
    
    核心特性:
    - 多环境并行训练 (ParallelSolverManager)
    - 重要性采样 (PPO 风格)
    - 历史状态堆叠 (envstep_state)
    - 策略稳定性 EMA
    - 完整正则化项支持
    
    Example:
        trainer = JetTrainer("configs/jet_trainer.yaml")
        results = trainer.train()
    """
    
    def __init__(self, config: Union[str, Path, Dict, JetTrainerConfig]):
        """初始化 JetTrainer
        
        Args:
            config: 配置来源 (YAML 路径 / Dict / JetTrainerConfig)
        """
        # 解析配置
        self.cfg = self._parse_config(config)
        
        # 初始化基类
        super().__init__(self.cfg.training, self.cfg.output)
        
        # 多环境支持
        self.n_envs = self.cfg.training.num_envs
        
        # Epoch 追踪 (用于正则化禁用)
        self.current_epoch: int = 0
        self.num_epochs: int = self.cfg.training.epochs
        
        # AFC 训练状态
        self._prev_amplitude = torch.tensor(0.0)
        self._current_amplitude = torch.tensor(0.0)
        self._current_policy_output = None
        self._step_losses: List[torch.Tensor] = []
        
        # 力系数缓存 (每步清空，避免重复计算)
        self._cached_force_coeffs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # 策略稳定性 EMA
        self._policy_output_ema: Optional[torch.Tensor] = None
        
        # 历史状态缓冲 (用于 envstep_state 功能)
        self._state_history_buffer: List[Dict] = []
        
        # KL 散度/重要性采样的旧网络
        self.policy_net_old: Optional[nn.Module] = None
        self._current_state_for_kl: Optional[Dict] = None
        
        # 正则化项缓存 (保持梯度连通)
        self._cached_reg_amp_l2: Optional[torch.Tensor] = None
        self._cached_reg_smoothness: Optional[torch.Tensor] = None
        self._cached_reg_stability: Optional[torch.Tensor] = None
        self._cached_reg_entropy: Optional[torch.Tensor] = None
        self._cached_reg_kl: Optional[torch.Tensor] = None
        
        # 重要性采样缓存
        self._cached_importance_ratio: Optional[torch.Tensor] = None
        
        # 历史记录 (用于日志/分析)
        self.loss_history: List[float] = []
        self.primary_loss_history: List[float] = []
        self.reg_loss_history: List[float] = []
        self.amplitude_history: List[float] = []

        # CUDA 显存监控：记录上一个 epoch 结束时的当前显存（用于 delta）
        self._last_epoch_end_alloc_mb: Optional[float] = None
        self._last_epoch_end_reserved_mb: Optional[float] = None
        
        # 模态 loss 需求标志 (避免每步重复判断)
        loss_cfg = self.cfg.loss
        self._need_modal_loss = (loss_cfg.modal_pod_weight > 0 or
                                 loss_cfg.modal_dmd_weight > 0 or
                                 loss_cfg.modal_residual_weight > 0)
        
        # 初始化组件 (使用 BaseTrainer 的 resume 支持)
        self._setup_output_dir()  # resume 模式会复用目录
        self._setup_logging()
        self._setup_seed()
        self._setup_env()
        self._setup_agent()
        self._setup_optimizer_for_jet()  # 先创建优化器，再恢复状态
        self._setup_tensorboard()
        self._setup_validation_runner() # 验证阶段可视化 runner
        self._setup_training_runner()  # 训练阶段可视化 runner
        
        # 检查点加载 (使用 BaseTrainer 提供的方法)
        if self.training_cfg.resume:
            self._resume_training()      # BaseTrainer 方法
        else:
            self._load_checkpoint_weights()  # BaseTrainer 方法
        
        # 保存配置
        self._save_config()
    
    def _parse_config(self, config) -> JetTrainerConfig:
        """解析配置"""
        if isinstance(config, JetTrainerConfig):
            return config
        elif isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg_dict = yaml.safe_load(f)
            return _parse_jet_trainer_config(cfg_dict)
        elif isinstance(config, dict):
            return _parse_jet_trainer_config(config)
        else:
            raise TypeError(f"不支持的配置类型: {type(config)}")
    
    # ========== 抽象方法实现 ==========
    
    def _setup_env(self):
        """初始化环境 (Solver)
        
        支持单环境和多环境模式:
        - num_envs=1: 使用 KarmanTorchSolver
        - num_envs>1: 使用 ParallelSolverManager
        """
        # 使用统一的配置加载方法
        base_cfg = self._load_env_config()
        
        solver_cfg = base_cfg.get('solver', base_cfg)
        self._override_solver_config(solver_cfg)
        
        # 保存修改后的配置
        modified_cfg_path = self.exp_dir / 'configs' / 'solver_config.yaml'
        with open(modified_cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(base_cfg, f, sort_keys=False, allow_unicode=True)
        
        # 根据 num_envs 选择 solver 类型
        if self.n_envs > 1:
            from myafc.env_kernels.solvers.parallel_solver_manager import ParallelSolverManager
            self.solver = ParallelSolverManager(
                base_cfg_path=str(modified_cfg_path),
                n_envs=self.n_envs,
                env_variants=self.cfg.training.env_variants,
            )
            LOGGER.info(f"[JetTrainer] ParallelSolverManager 初始化 | 环境数: {self.n_envs}")
        else:
            from myafc.env_kernels.solvers.karman_torch_solver import KarmanTorchSolver
            self.solver = KarmanTorchSolver(str(modified_cfg_path))
            LOGGER.info(f"[JetTrainer] Solver 初始化完成")
        
        # 获取设备
        self.device = self.solver.device if hasattr(self.solver, 'device') else self.solver._device
        
        # 计算目标探针位置
        target_pos = self.cfg.loss.target_probe_position
        Lx, Ly = self.solver.cfg.domain.size
        self.target_point = Point(vec(x=target_pos[0] * Lx, y=target_pos[1] * Ly))
        
        # 加载额外探针点 (用于 main_input_type='probes' 模式)
        self._setup_probe_points()
        
        # 预缓存目标张量
        self._target_u = torch.tensor(self.cfg.loss.target_velocity[0], device=self.device)
        self._target_v = torch.tensor(self.cfg.loss.target_velocity[1], device=self.device)
        self._cd_target = torch.tensor(self.cfg.loss.cd_target, device=self.device)
        self._cl_target = torch.tensor(self.cfg.loss.cl_target, device=self.device)
        
        # 预计算默认 field_loss_region (障碍物下游区域)
        self._default_field_loss_region = self._compute_default_field_loss_region()
        
        LOGGER.info(f"[JetTrainer] 设备: {self.device}")
    
    def _compute_default_field_loss_region(self) -> List[float]:
        """计算默认的流场 Loss 区域 (障碍物下游)
        
        自动根据障碍物配置计算下游区域边界:
        - x_start: 障碍物右边缘 (相对坐标)
        - y_start: 0.0
        - x_end: 1.0 (域右边界)
        - y_end: 1.0 (域上边界)
        
        对于方形障碍物: x_right = center_x + size_x / (2 * Lx)
        对于圆形障碍物: x_right = center_x + radius / Lx
        
        Returns:
            [x_start, y_start, x_end, y_end] 相对坐标 (0~1)
        """
        obs_cfg = self.solver.cfg.obstacle
        Lx, Ly = self.solver.cfg.domain.size
        
        if not obs_cfg.enabled:
            # 无障碍物时使用域后半部分
            LOGGER.debug("[JetTrainer] 无障碍物，默认 field_loss_region 为域后半部分")
            return [0.5, 0.0, 1.0, 1.0]
        
        # 障碍物中心 (相对坐标)
        cx_rel = obs_cfg.center[0]
        
        # 根据形状计算右边缘相对坐标
        shape_type = obs_cfg.shape.lower()
        if shape_type == 'box':
            # 方形障碍物: size 是绝对尺寸
            sx = obs_cfg.size[0]
            half_width_rel = sx / (2 * Lx)
            x_right_rel = cx_rel + half_width_rel
        elif shape_type in ('sphere', 'cylinder', 'circle'):
            # 圆形障碍物: radius 是绝对尺寸
            r = obs_cfg.radius
            x_right_rel = cx_rel + r / Lx
        else:
            # 未知形状，使用中心点 + 小偏移
            LOGGER.warning(f"[JetTrainer] 未知障碍物形状 '{shape_type}'，使用中心点估计下游区域")
            x_right_rel = cx_rel + 0.05
        
        # 添加小间隙，避免边界效应
        gap_rel = 0.02
        x_start = min(x_right_rel + gap_rel, 0.9)  # 确保不会超出太多
        
        region = [x_start, 0.0, 1.0, 1.0]
        LOGGER.info(f"[JetTrainer] 自动计算 field_loss_region: {region} (障碍物右边缘: x={x_right_rel:.3f})")
        return region

    def _setup_probe_points(self):
        """加载 env_config 中的所有探针位置 (用于 main_input_type='probes' 模式)
        
        从 env_config_path 中读取 runner.probes.locations 配置，
        创建 Point 对象用于在 _build_state 中采样探针数据。
        
        探针位置支持两种格式:
        - fraction: [x, y] - 相对坐标 (推荐)
        - units: [x, y] - 绝对坐标
        """
        self._probe_points: Dict[str, Point] = {}
        
        # 如果 main_input_type 为 None 或 'target_probe'，则无需加载额外探针
        if self.cfg.input_features.main_input_type in (None, 'target_probe'):
            LOGGER.debug(f"[JetTrainer] main_input_type={self.cfg.input_features.main_input_type!r}, 跳过额外探针加载")
            return
        
        # 使用统一的配置加载方法
        env_cfg = self._load_env_config()
        
        runner_cfg = env_cfg.get('runner', {})
        probes_cfg = runner_cfg.get('probes', {})
        locations = probes_cfg.get('locations', [])
        
        if not locations:
            LOGGER.debug("[JetTrainer] env_config 中未定义探针位置")
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
                LOGGER.warning(f"[JetTrainer] 探针 '{name}' 缺少坐标配置, 跳过")
                continue
            
            self._probe_points[name] = Point(vec(x=x, y=y))
        
        LOGGER.info(f"[JetTrainer] 已加载 {len(self._probe_points)} 个探针点: {list(self._probe_points.keys())}")
    
    def _override_solver_config(self, solver_cfg: Dict):
        """覆盖 solver 配置
        
        应用 Trainer 级别的配置覆盖:
        - verbose 级别
        - 入口边界条件
        - 射流和可微分模式
        - smoothing_alpha
        """
        inlet_cfg = self.cfg.inlet
        
        # verbose 覆盖
        if 'runtime' not in solver_cfg:
            solver_cfg['runtime'] = {}
        solver_cfg['runtime']['verbose'] = self.cfg.training.verbose
        
        # 边界条件 (mode=None 时不覆盖，使用 env_config 中的配置)
        if inlet_cfg.mode is not None:
            if 'boundary' not in solver_cfg:
                solver_cfg['boundary'] = {}
            if 'velocity' not in solver_cfg['boundary']:
                solver_cfg['boundary']['velocity'] = {}
            
            if inlet_cfg.mode == 'zero_gradient':
                solver_cfg['boundary']['velocity']['x-'] = {'type': 'zero-gradient'}
            elif inlet_cfg.mode == 'fixed':
                solver_cfg['boundary']['velocity']['x-'] = {'type': 'inflow', 'value': inlet_cfg.fixed_velocity}
            elif inlet_cfg.mode == 'sinusoidal':
                solver_cfg['boundary']['velocity']['x-'] = {'type': 'inflow', 'value': inlet_cfg.offset}
                self._sinusoidal_params = {
                    'amplitude': inlet_cfg.amplitude,
                    'frequency': inlet_cfg.frequency,
                    'phase_rad': pymath.radians(inlet_cfg.phase_deg),
                    'offset': inlet_cfg.offset,
                }
        
        # 启用射流和可微分模式
        if 'jets' not in solver_cfg:
            solver_cfg['jets'] = {}
        solver_cfg['jets']['enabled'] = True
        
        # smoothing_alpha 覆盖（全局参数，影响训练和验证）
        if self.cfg.training.smoothing_alpha is not None:
            solver_cfg['jets']['smoothing_alpha'] = self.cfg.training.smoothing_alpha
            LOGGER.info(f"[JetTrainer] 覆盖 jets.smoothing_alpha: {self.cfg.training.smoothing_alpha}")
        else:
            LOGGER.debug(f"[JetTrainer] 使用 env_config 中的 smoothing_alpha: {solver_cfg.get('jets', {}).get('smoothing_alpha', '未设置')}")
        
        if 'runtime' not in solver_cfg:
            solver_cfg['runtime'] = {}
        solver_cfg['runtime']['differentiable'] = True
    
    def _update_sinusoidal_inlet(self, time: float):
        """更新 sinusoidal 入口边界条件 (每步调用)
        
        公式: U(t) = offset + amplitude * sin(2π * frequency * t + phase)
        
        Args:
            time: 当前物理时间
        """
        if not hasattr(self, '_sinusoidal_params'):
            return
        
        params = self._sinusoidal_params
        phase = 2 * pymath.pi * params['frequency'] * time + params['phase_rad']
        sin_factor = pymath.sin(phase)
        
        # 计算速度分量
        Ux = params['offset'][0] + params['amplitude'] * sin_factor
        Uy = params['offset'][1]  # y 分量保持为 offset
        
        # 更新 solver 边界条件
        self.solver.vel_boundary['x-'] = vec(x=Ux, y=Uy)
    
    def _setup_agent(self):
        """初始化 Agent
        
        包含 amplitude_max 与 jet saturation 的一致性检查：
        当 amplitude_max > saturation 时，Agent 输出超出饱和范围的部分
        会被 torch.clamp 截断，导致梯度为零，反传链断裂。
        此处自动对齐 amplitude_max = saturation 以确保梯度畅通。
        """
        num_nozzles = len(self.solver.cfg.jets.nozzles) if self.solver.cfg.jets.enabled else 1
        
        # 合并 input_features 配置到 agent config
        agent_cfg = self.cfg.agent
        feat_cfg = self.cfg.input_features
        agent_cfg.main_input_type = feat_cfg.main_input_type  # 传递主输入类型
        agent_cfg.envstep_state = feat_cfg.envstep_state
        agent_cfg.include_target_velocity = feat_cfg.include_target_velocity
        agent_cfg.include_prev_amplitude_num = feat_cfg.include_prev_amplitude_num
        agent_cfg.include_force_coeffs = feat_cfg.include_force_coeffs
        agent_cfg.probe_fields = feat_cfg.probe_fields
        
        # ★ amplitude_max 与 jet saturation 一致性检查
        # 当 amplitude_max > saturation 时，JetObstacle.apply_controls() 中的
        # torch.clamp(amp, -sat, sat) 会截断超出部分，导致该区间梯度为零。
        # 例如: amplitude_max=4.0 但 saturation=1.5 时，输出 > 1.5 的梯度全部丢失。
        if self.solver.jet_obstacle is not None:
            jet_sat = self.solver.jet_obstacle.jet_config.saturation
            if agent_cfg.amplitude_max > jet_sat + 1e-6:
                LOGGER.warning(
                    f"[JetTrainer] ⚠️ amplitude_max ({agent_cfg.amplitude_max:.2f}) > "
                    f"jet saturation ({jet_sat:.2f}). "
                    f"超出 saturation 的 Agent 输出会被 torch.clamp 截断，梯度为零！"
                    f"自动对齐: amplitude_max = {jet_sat:.2f}"
                )
                agent_cfg.amplitude_max = jet_sat
        
        self.agent = AFCAgent(
            config=agent_cfg,
            num_nozzles=num_nozzles,
            device=str(self.device),
        )
        self.agent.set_target_velocity(self.cfg.loss.target_velocity)
        
        # 设置到 JetObstacle
        if self.solver.jet_obstacle is not None:
            self.solver.jet_obstacle.policy = self.agent
            self.solver.jet_obstacle.deterministic = agent_cfg.deterministic
        
        # 强制初始化网络
        dummy_state = self._build_state()
        with torch.no_grad():
            _ = self.agent.compute_action(dummy_state)
        
        LOGGER.info(f"[JetTrainer] Agent 初始化完成, 喷嘴数: {num_nozzles}, "
                     f"amplitude_max: {agent_cfg.amplitude_max:.2f}")
    
    
    def _setup_optimizer_for_jet(self):
        """构建优化器 (使用 BaseTrainer 的 YOLOv8 风格方法)"""
        # 确保网络已初始化
        if self.agent.network is None:
            dummy_state = self._build_state()
            _ = self.agent.compute_action(dummy_state)
        
        # 初始化 policy_net_old (用于 KL 散度/重要性采样)
        if self.cfg.loss.kl_weight > 0 or self.cfg.training.use_importance_sampling:
            buffer = io.BytesIO()
            torch.save(self.agent.network, buffer)
            buffer.seek(0)
            self.policy_net_old = torch.load(buffer, map_location=self.device, weights_only=False)
            self.policy_net_old.eval()
            for param in self.policy_net_old.parameters():
                param.requires_grad = False
            LOGGER.info("[JetTrainer] 已创建 policy_net_old")
        
        # 使用 BaseTrainer 提供的 YOLOv8 风格优化器
        self._setup_optimizer_yolo(self.agent.network)
        
        # 设置 EMA 模型 (使用 BaseTrainer 提供的 YOLOv8 风格 ModelEMA)
        self._setup_ema(self.agent.network)
        
        # 计算总更新次数
        self._total_updates = self._compute_total_updates()
    
    def _compute_total_updates(self) -> int:
        """计算总参数更新次数"""
        cfg = self.cfg.training
        total_interactions = cfg.num_interaction_per_mini_episode
        interactions_per_update = cfg.num_of_interaction_per_agent_update
        updates_per_epoch = (total_interactions + interactions_per_update - 1) // interactions_per_update
        return max(1, cfg.epochs * updates_per_epoch)
    
    def _build_state(self) -> Dict:
        """构建 Agent 输入状态
        
        根据 input_features 配置构建状态字典，仅采样 probe_fields 中请求的字段:
        - main_input_type: 'target_probe'(仅目标探针) / 'probes'(全部探针)
        - probe_fields: ['u','v','p'] 的任意子集 — 与探针选择独立组合
        - include_force_coeffs: 是否包含力系数
        
        噪声仅在训练模式 (agent.network.training=True) 时注入，
        验证阶段 (agent.eval() 后) 自动关闭。
        """
        from myafc.utils import sample_probe_fields, build_probe_batch, sample_probe_fields_batch
        
        state = {
            'time': self.solver.current_time,
            'step': self.solver.current_step,
        }
        
        probe_fields = self.cfg.input_features.probe_fields
        # 噪声仅在训练模式启用，验证阶段 (agent.eval()) 自动关闭
        is_training = self.agent.network.training if (self.agent and self.agent.network) else False
        noise_std = self.cfg.training.noise_std if (self.cfg.training.noise_observation and is_training) else 0.0
        
        # target 探针 (始终采样)
        state['probes'] = {'target': sample_probe_fields(
            self.solver.velocity, self.solver.pressure, self.target_point,
            probe_fields, noise_std=noise_std,
        )}
        
        # env 探针 (main_input_type='probes' 时)
        if self.cfg.input_features.main_input_type == 'probes' and self._probe_points:
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
                noise_std=noise_std,
            )
            state['probes'].update(batch_data)
        
        # 力系数
        if self.cfg.input_features.include_force_coeffs:
            cd, cl, _, _, _, _ = self.solver.compute_force_coeffs()
            state['force_coeffs'] = {
                'Cd': cd.item() if isinstance(cd, torch.Tensor) else cd,
                'Cl': cl.item() if isinstance(cl, torch.Tensor) else cl,
            }
        
        return state
    
    def _compute_loss(
        self,
        current_velocity: torch.Tensor,
        amplitude: torch.Tensor,
        policy_output=None,
        is_update_step: bool = True,
    ) -> tuple:
        """计算单步 loss
        
        通过设置各 loss 权重 > 0 来启用相应的 loss 类型，
        多个 loss 可同时启用，按权重加权求和。
        
        支持的 Loss 类型:
        ================
        1. 探针 Loss: probe_mse, probe_mae
        2. 力系数 Loss: force (MSE), force_drl (reward), force_vortex (涡脱)
        3. 流场 Loss: field_rms, field_fft, field_fluctuation
        4. 模态 Loss: modal_pod, modal_dmd, modal_residual
        
        Args:
            current_velocity: 当前探针速度 [u, v]
            amplitude: 当前控制幅值
            policy_output: PolicyNetOutput (用于正则化)
            is_update_step: 是否是动作更新步 (影响正则化缓存)
        
        Returns:
            (total_loss, primary_loss, reg_loss) 三元组
        """
        loss_cfg = self.cfg.loss
        total_loss = torch.tensor(0.0, device=self.device)
        reg_loss = torch.tensor(0.0, device=self.device)
        
        # ===== Primary Loss (使用共享方法) =====
        # 将 [u, v] tensor 转换为 (u, v) tuple
        u = current_velocity[0] if current_velocity.dim() == 1 else current_velocity[:, 0]
        v = current_velocity[1] if current_velocity.dim() == 1 else current_velocity[:, 1]
        primary_loss = self._compute_primary_loss(current_velocity=(u, v))
        
        # ===== PPO 式重要性采样 (可选) =====
        cfg = self.cfg.training
        if cfg.use_importance_sampling and is_update_step:
            # 计算并缓存重要性采样比率
            ratio = self._compute_importance_sampling_ratio(policy_output)
            clip_range = cfg.importance_sampling_clip_range
            ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            # 使用 clipped surrogate objective
            primary_loss = torch.max(ratio * primary_loss, ratio_clipped * primary_loss)
            self._cached_importance_ratio = ratio
        elif cfg.use_importance_sampling and not is_update_step:
            # 非动作更新步：使用缓存的 ratio
            if self._cached_importance_ratio is not None:
                ratio = self._cached_importance_ratio
                if not isinstance(ratio, torch.Tensor):
                    ratio = torch.tensor(ratio, device=self.device)
                clip_range = cfg.importance_sampling_clip_range
                ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                primary_loss = torch.max(ratio * primary_loss, ratio_clipped * primary_loss)
        
        total_loss = total_loss + primary_loss
        
        # ===== 检查是否禁用正则化 (最后 N 个 epoch / 'all') =====
        disable_last_n = self.cfg.loss.disable_regularization_last_epochs
        if isinstance(disable_last_n, str) and disable_last_n.lower() == 'all':
            is_reg_disabled = True
        elif isinstance(disable_last_n, int) and disable_last_n > 0:
            is_reg_disabled = self.current_epoch >= self.num_epochs - disable_last_n
        else:
            is_reg_disabled = False
        
        # ===== 正则化 Loss (仅在动作更新步计算新值，否则复用缓存) =====
        if not is_reg_disabled:
            if is_update_step:
                # 幅值 L2
                if loss_cfg.amplitude_l2_weight > 0:
                    self._cached_reg_amp_l2 = amplitude_l2_loss(amplitude, loss_cfg.amplitude_l2_weight)
                
                # 时序平滑 (支持 action_rate_limit 软约束)
                if loss_cfg.temporal_smoothness_weight > 0:
                    rate_limit = loss_cfg.action_rate_limit
                    if rate_limit is not None and rate_limit > 0:
                        # 软约束: 仅惩罚超过 rate_limit 的部分
                        diff = amplitude - self._prev_amplitude
                        excess = torch.relu(diff.abs() - rate_limit)
                        self._cached_reg_smoothness = loss_cfg.temporal_smoothness_weight * (excess ** 2).mean()
                    else:
                        # 标准 L2 平滑
                        self._cached_reg_smoothness = temporal_smoothness_loss(
                            amplitude, self._prev_amplitude, loss_cfg.temporal_smoothness_weight
                        )
                
                # 策略稳定性 (EMA)
                if loss_cfg.policy_stability_weight > 0 and policy_output is not None:
                    amp_tensor = amplitude.detach() if isinstance(amplitude, torch.Tensor) else torch.tensor(amplitude)
                    if self._policy_output_ema is None:
                        self._policy_output_ema = amp_tensor.clone()
                    else:
                        momentum = loss_cfg.policy_stability_momentum
                        self._policy_output_ema = momentum * self._policy_output_ema + (1 - momentum) * amp_tensor
                    self._cached_reg_stability = policy_stability_loss(
                        amplitude, self._policy_output_ema, loss_cfg.policy_stability_weight
                    )
                
                # 熵正则化
                if loss_cfg.entropy_weight > 0 and policy_output is not None:
                    log_std = getattr(policy_output, 'log_std', None)
                    if log_std is not None:
                        self._cached_reg_entropy = entropy_loss(log_std, loss_cfg.entropy_weight)
                
                # KL 散度
                if loss_cfg.kl_weight > 0 and policy_output is not None and self.policy_net_old is not None:
                    self._cached_reg_kl = self._compute_kl_loss(policy_output)
            
            # 累加缓存的正则化项
            if self._cached_reg_amp_l2 is not None:
                reg_loss = reg_loss + self._cached_reg_amp_l2
            if self._cached_reg_smoothness is not None:
                reg_loss = reg_loss + self._cached_reg_smoothness
            if self._cached_reg_stability is not None:
                reg_loss = reg_loss + self._cached_reg_stability
            if self._cached_reg_entropy is not None:
                reg_loss = reg_loss + self._cached_reg_entropy
            if self._cached_reg_kl is not None:
                reg_loss = reg_loss + self._cached_reg_kl
        
        total_loss = total_loss + reg_loss
        
        return total_loss, primary_loss, reg_loss
    
    def _compute_kl_loss(self, policy_output) -> torch.Tensor:
        """计算 KL 散度 loss
        
        使用保存的旧网络参数计算 KL(π_new || π_old)
        
        Args:
            policy_output: 当前策略输出 (包含 mean, log_std)
        
        Returns:
            KL 散度 loss 张量
        """
        if self._current_state_for_kl is None or self.policy_net_old is None:
            return torch.tensor(0.0, device=self.device)
        
        # 获取新策略的均值和对数标准差
        mean_new = getattr(policy_output, 'mean', None)
        log_std_new = getattr(policy_output, 'log_std', None)
        
        if mean_new is None or log_std_new is None:
            return torch.tensor(0.0, device=self.device)
        
        # 使用旧网络计算旧策略输出
        with torch.no_grad():
            # 构建输入
            state = self.agent._prepare_state(self._current_state_for_kl)
            input_vec = self.agent._build_input(state)
            input_tensor = torch.tensor(input_vec, dtype=torch.float32, device=self.device)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            # 前向传播旧网络
            old_output = self.policy_net_old(input_tensor)
            mean_old = getattr(old_output, 'mean', mean_new.detach())
            log_std_old = getattr(old_output, 'log_std', log_std_new.detach())
        
        return kl_divergence_loss(
            mean_new, log_std_new, 
            mean_old, log_std_old,
            self.cfg.loss.kl_weight
        )
    
    def _compute_primary_loss(
        self,
        current_velocity: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """计算 primary loss (训练和验证统一使用)
        
        此方法实现所有 primary loss 的计算逻辑（probe/force/field/modal），
        不包含正则化项和 PPO 重要性采样。
        
        **设计原则**：训练和验证使用完全相同的计算路径，确保 loss 值可比。
        验证阶段在外部调用 .detach().item() 转为 float。
        
        注：需要历史 buffer 的 loss（如 DRL 时间平均、涡脱 FFT）在 buffer 为空时
        会使用瞬时值作为替代，确保验证阶段也能正常计算。
        
        Args:
            current_velocity: (u, v) 探针速度张量，None 则从 solver 采样
            
        Returns:
            primary_loss: tensor
        """
        loss_cfg = self.cfg.loss
        primary_loss = torch.tensor(0.0, device=self.device)
        
        # ===== 1. 探针 Loss =====
        if current_velocity is None:
            vel_at_target = field.sample(self.solver.velocity, self.target_point)
            u = vel_at_target.vector['x'].native().squeeze()
            v = vel_at_target.vector['y'].native().squeeze()
        else:
            u, v = current_velocity
            # 确保是 tensor
            if not isinstance(u, torch.Tensor):
                u = torch.tensor(u, device=self.device)
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=self.device)
        
        target_u, target_v = loss_cfg.target_velocity
        
        if loss_cfg.probe_mse_weight > 0:
            current_vel = torch.stack([u, v])
            target_vel = torch.stack([
                torch.tensor(target_u, device=self.device),
                torch.tensor(target_v, device=self.device)
            ])
            primary_loss = primary_loss + loss_cfg.probe_mse_weight * probe_mse_loss(
                current_vel, target_vel
            )
        
        if loss_cfg.probe_mae_weight > 0:
            current_vel = torch.stack([u, v])
            target_vel = torch.stack([
                torch.tensor(target_u, device=self.device),
                torch.tensor(target_v, device=self.device)
            ])
            primary_loss = primary_loss + loss_cfg.probe_mae_weight * probe_mae_loss(
                current_vel, target_vel
            )
        
        # ===== 2. 力系数 Loss =====
        _need_force = (loss_cfg.force_weight > 0 or 
                       loss_cfg.force_drl_weight > 0 or 
                       loss_cfg.force_vortex_weight > 0)
        if _need_force:
            cd, cl = self._get_force_coeffs()
            
            # 基础 MSE loss
            if loss_cfg.force_weight > 0:
                primary_loss = primary_loss + loss_cfg.force_weight * force_coeffs_loss(
                    cd, cl, loss_cfg.cd_target, loss_cfg.cl_target
                )
            
            # DRLinFluids reward-based loss (统一使用时间平均)
            if loss_cfg.force_drl_weight > 0:
                cd_for_drl, cl_for_drl = self._get_averaged_drl_forces(cd, cl)
                drl_loss = drl_reward_loss(
                    cd_for_drl, cl_for_drl,
                    cd_baseline=loss_cfg.cd_baseline,
                    alpha_lift=loss_cfg.alpha_lift,
                )
                primary_loss = primary_loss + loss_cfg.force_drl_weight * drl_loss
            
            # 涡脱抑制 loss (需要足够的历史数据)
            if loss_cfg.force_vortex_weight > 0:
                if hasattr(self, '_cl_history_buffer') and len(self._cl_history_buffer) > 3:
                    cl_history = torch.stack(list(self._cl_history_buffer))
                    vortex_loss = vortex_shedding_loss(cl_history)
                    primary_loss = primary_loss + loss_cfg.force_vortex_weight * vortex_loss
        
        # ===== 3. 流场 Loss =====
        _need_field = (loss_cfg.field_rms_weight > 0 or 
                       loss_cfg.field_fft_weight > 0 or 
                       loss_cfg.field_fluctuation_weight > 0)
        if _need_field:
            velocity_field = self._get_velocity_field_tensor()
            pressure_field = self._get_pressure_field_tensor() if loss_cfg.field_pressure_weight > 0 else None
            region = loss_cfg.field_loss_region or self._default_field_loss_region
            
            if loss_cfg.field_rms_weight > 0:
                field_loss = multi_field_loss(
                    velocity=velocity_field, pressure=pressure_field, region=region,
                    velocity_weight=loss_cfg.field_velocity_weight,
                    vorticity_weight=loss_cfg.field_vorticity_weight,
                    pressure_weight=loss_cfg.field_pressure_weight,
                    loss_type='rms',
                )
                primary_loss = primary_loss + loss_cfg.field_rms_weight * field_loss
            
            if loss_cfg.field_fft_weight > 0:
                field_loss = multi_field_loss(
                    velocity=velocity_field, pressure=pressure_field, region=region,
                    velocity_weight=loss_cfg.field_velocity_weight,
                    vorticity_weight=loss_cfg.field_vorticity_weight,
                    pressure_weight=loss_cfg.field_pressure_weight,
                    loss_type='fft',
                    n_modes=loss_cfg.field_fft_n_modes,
                    exclude_dc=loss_cfg.field_fft_exclude_dc,
                )
                primary_loss = primary_loss + loss_cfg.field_fft_weight * field_loss
            
            if loss_cfg.field_fluctuation_weight > 0:
                field_loss = multi_field_loss(
                    velocity=velocity_field, pressure=pressure_field, region=region,
                    velocity_weight=loss_cfg.field_velocity_weight,
                    vorticity_weight=loss_cfg.field_vorticity_weight,
                    pressure_weight=loss_cfg.field_pressure_weight,
                    loss_type='fluctuation',
                    top_k=loss_cfg.field_fluctuation_top_k,
                )
                primary_loss = primary_loss + loss_cfg.field_fluctuation_weight * field_loss
        
        # ===== 4. 模态 Loss (需要足够的快照数据) =====
        _need_modal = (loss_cfg.modal_pod_weight > 0 or 
                       loss_cfg.modal_dmd_weight > 0 or 
                       loss_cfg.modal_residual_weight > 0)
        if _need_modal and hasattr(self, '_snapshot_buffer') and len(self._snapshot_buffer) >= loss_cfg.modal_n_modes:
            snapshots = torch.stack(list(self._snapshot_buffer))
            
            if loss_cfg.modal_pod_weight > 0:
                modal_loss = pod_mode_loss(snapshots, n_modes=loss_cfg.modal_n_modes,
                                           exclude_first_k=loss_cfg.modal_exclude_first_k)
                primary_loss = primary_loss + loss_cfg.modal_pod_weight * modal_loss
            
            if loss_cfg.modal_dmd_weight > 0:
                modal_loss = dmd_mode_loss(snapshots, n_modes=loss_cfg.modal_n_modes)
                primary_loss = primary_loss + loss_cfg.modal_dmd_weight * modal_loss
            
            if loss_cfg.modal_residual_weight > 0:
                modal_loss = steady_state_residual_loss(snapshots, n_modes=loss_cfg.modal_n_modes)
                primary_loss = primary_loss + loss_cfg.modal_residual_weight * modal_loss
        
        return primary_loss
    
    def _compute_importance_sampling_ratio(self, policy_output) -> torch.Tensor:
        """计算 PPO 式重要性采样比率 ratio = π_new(a|s) / π_old(a|s)
        
        使用当前策略和旧策略对同一动作的概率比。
        
        Args:
            policy_output: 当前策略输出 (包含 mean, log_std, action)
        
        Returns:
            ratio: 重要性采样比率（标量张量）
        """
        if self.policy_net_old is None or policy_output is None:
            return torch.tensor(1.0, device=self.device)
        
        # 获取当前策略的 log_prob
        log_prob_new = getattr(policy_output, 'log_prob', None)
        if log_prob_new is None:
            return torch.tensor(1.0, device=self.device)
        
        # 获取当前动作
        action = getattr(policy_output, 'action_raw', None)
        if action is None:
            action = getattr(policy_output, 'action', None)
        if action is None:
            return torch.tensor(1.0, device=self.device)
        
        # 使用旧网络计算旧策略的 log_prob
        if self._current_state_for_kl is None:
            return torch.tensor(1.0, device=self.device)
        
        with torch.no_grad():
            state = self.agent._prepare_state(self._current_state_for_kl)
            input_vec = self.agent._build_input(state)
            input_tensor = torch.tensor(input_vec, dtype=torch.float32, device=self.device)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            old_output = self.policy_net_old(input_tensor)
            mean_old = getattr(old_output, 'mean', None)
            log_std_old = getattr(old_output, 'log_std', None)
            
            if mean_old is None or log_std_old is None:
                return torch.tensor(1.0, device=self.device)
            
            from torch.distributions import Normal
            old_std = torch.exp(log_std_old)
            old_dist = Normal(mean_old, old_std)
            log_prob_old = old_dist.log_prob(action)
            if log_prob_old.dim() > 0:
                log_prob_old = log_prob_old.sum(dim=-1)
        
        # 计算比率
        ratio = torch.exp(log_prob_new - log_prob_old)
        return ratio
    
    # ============================================================
    # 流场数据获取辅助方法 (用于 field/modal loss)
    # ============================================================
    def _get_velocity_field_tensor(self) -> torch.Tensor:
        """从 solver 获取速度场张量
        
        将 StaggeredGrid 速度场插值到网格中心（CenteredGrid），
        然后导出为 PyTorch 张量。
        
        Returns:
            velocity: (2, H, W) 张量，[u, v] 分量，H=y维度, W=x维度
        """
        # StaggeredGrid → CenteredGrid (插值到网格中心)
        vel_centered = self.solver.velocity.at_centers()
        # 导出为张量，指定维度顺序
        u = vel_centered.vector['x'].values.native(('y', 'x'))
        v = vel_centered.vector['y'].values.native(('y', 'x'))
        # 去掉可能的 batch 维度
        if u.dim() > 2:
            u = u.squeeze(0)
        if v.dim() > 2:
            v = v.squeeze(0)
        return torch.stack([u, v], dim=0)
    
    def _get_pressure_field_tensor(self) -> torch.Tensor:
        """从 solver 获取压力场张量
        
        Returns:
            pressure: (H, W) 张量，H=y维度, W=x维度
        """
        p_field = self.solver.pressure
        p = p_field.values.native(('y', 'x'))
        if p.dim() > 2:
            p = p.squeeze(0)
        return p
    
    def _update_snapshot_buffer(self, velocity_field: torch.Tensor = None):
        """更新快照缓冲区 (用于模态 loss)
        
        快照会应用 field_loss_region 裁剪，确保模态分析只在指定区域进行。
        注意: 快照保留梯度 (不 detach)，以支持可微分 DMD/POD loss。
        梯度在 _detach_graph() 中统一释放。
        
        Args:
            velocity_field: 速度场张量 (2, H, W)，None 则自动获取
        """
        from myafc.afc_supervision.losses.field_loss import extract_region
        
        loss_cfg = self.cfg.loss
        buffer_size = loss_cfg.modal_snapshot_buffer_size
        
        # 初始化缓冲区
        if not hasattr(self, '_snapshot_buffer'):
            self._snapshot_buffer = deque(maxlen=buffer_size)
        
        # 获取速度场
        if velocity_field is None:
            velocity_field = self._get_velocity_field_tensor()
        
        # 计算速度模值
        vel_mag = velocity_magnitude(velocity_field)
        
        # 应用 field_loss_region 裁剪 (与流场 Loss 保持一致)
        region = loss_cfg.field_loss_region
        if region is None:
            region = self._default_field_loss_region
        vel_mag_cropped = extract_region(vel_mag, region)
        
        # 保留梯度: 不调用 .detach()，支持可微分模态 loss
        self._snapshot_buffer.append(vel_mag_cropped)
    
    def _update_cl_history_buffer(self, cl: torch.Tensor):
        """更新 Cl 历史缓冲区 (用于涡脱抑制 loss)
        
        Args:
            cl: 当前升力系数
        """
        buffer_size = 100  # 固定缓冲区大小
        
        if not hasattr(self, '_cl_history_buffer'):
            self._cl_history_buffer = deque(maxlen=buffer_size)
        
        cl_val = cl.detach() if isinstance(cl, torch.Tensor) else torch.tensor(cl)
        self._cl_history_buffer.append(cl_val)
    
    def _update_force_buffers(self, cd: torch.Tensor = None, cl: torch.Tensor = None):
        """统一更新所有力系数相关的 history buffer
        
        此方法用于训练和验证阶段，确保两者行为一致。
        在 _compute_primary_loss 调用前更新，为 DRL 时间平均和涡脱 FFT 提供数据。
        
        Args:
            cd: 阻力系数，None 则自动计算
            cl: 升力系数，None 则自动计算
        """
        loss_cfg = self.cfg.loss
        need_update = (loss_cfg.force_drl_weight > 0 or loss_cfg.force_vortex_weight > 0)
        
        if not need_update:
            return
        
        # 按需计算力系数（使用缓存方法）
        if cd is None or cl is None:
            cd, cl = self._get_force_coeffs()
        
        # DRL 时间平均 buffer
        if loss_cfg.force_drl_weight > 0:
            self._update_drl_force_history(cd, cl)
        
        # 涡脱 FFT buffer  
        if loss_cfg.force_vortex_weight > 0:
            self._update_cl_history_buffer(cl)

    def _update_drl_force_history(self, cd: torch.Tensor, cl: torch.Tensor):
        """更新 DRL loss 用的 Cd/Cl 历史缓冲区 (用于时间平均)
        
        与 _cl_history_buffer 分开，因为:
        1. 用途不同 (DRL reward vs 涡脱 FFT)
        2. 窗口大小由 force_drl_avg_window 配置
        3. 使用带梯度的张量
        
        Args:
            cd: 当前阻力系数 (保持梯度)
            cl: 当前升力系数 (保持梯度)
        """
        # 获取配置的窗口大小
        window_size = self._get_drl_avg_window_size()
        
        # 如果不需要平均 (window_size <= 1)，跳过
        if window_size is None or window_size <= 1:
            return
        
        # 初始化缓冲区
        if not hasattr(self, '_drl_cd_history_buffer'):
            self._drl_cd_history_buffer = deque(maxlen=window_size)
            self._drl_cl_history_buffer = deque(maxlen=window_size)
        
        # 更新缓冲区大小 (如果配置改变)
        if self._drl_cd_history_buffer.maxlen != window_size:
            old_cd = list(self._drl_cd_history_buffer)
            old_cl = list(self._drl_cl_history_buffer)
            self._drl_cd_history_buffer = deque(old_cd[-window_size:], maxlen=window_size)
            self._drl_cl_history_buffer = deque(old_cl[-window_size:], maxlen=window_size)
        
        # 注意：保持梯度以支持可微分训练
        # 如果配置了 force_drl_cl_abs，则存储 |Cl|
        cl_to_store = torch.abs(cl) if self.cfg.loss.force_drl_cl_abs else cl
        self._drl_cd_history_buffer.append(cd)
        self._drl_cl_history_buffer.append(cl_to_store)
    
    def _get_drl_avg_window_size(self) -> Optional[int]:
        """获取 DRL loss 时间平均的实际窗口大小
        
        Returns:
            None 或 1: 不进行平均
            > 1: 平均窗口大小
        """
        loss_cfg = self.cfg.loss
        raw_value = loss_cfg.force_drl_avg_window
        
        if raw_value is None:
            return None
        
        if isinstance(raw_value, int):
            return raw_value if raw_value > 1 else None
        
        if isinstance(raw_value, str):
            raw_lower = raw_value.lower()
            if raw_lower == 'action_duration':
                return self.cfg.training.action_duration
            elif raw_lower == 'control_interval':
                return self.cfg.training.control_interval
            else:
                raise ValueError(f"不支持的 force_drl_avg_window 值: {raw_value}")
        
        return None
    
    def _get_averaged_drl_forces(self, cd: torch.Tensor, cl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取时间平均的 Cd/Cl (用于 DRL loss)
        
        如果配置了 force_drl_avg_window，则返回滑动平均值。
        否则直接返回输入的瞬时值。
        
        如果 force_drl_cl_abs=True，则 Cl 使用绝对值。
        
        Args:
            cd: 当前阻力系数
            cl: 当前升力系数
        
        Returns:
            (cd_avg, cl_avg): 平均后的力系数 (如果不需要平均则返回原值/|Cl|)
        """
        window_size = self._get_drl_avg_window_size()
        use_cl_abs = self.cfg.loss.force_drl_cl_abs
        
        # 不需要时间平均
        if window_size is None or window_size <= 1:
            cl_out = torch.abs(cl) if use_cl_abs else cl
            return cd, cl_out
        
        # 先更新历史
        self._update_drl_force_history(cd, cl)
        
        # 缓冲区为空 (不应该发生，因为刚刚更新了)
        if not hasattr(self, '_drl_cd_history_buffer') or len(self._drl_cd_history_buffer) == 0:
            cl_out = torch.abs(cl) if use_cl_abs else cl
            return cd, cl_out
        
        # 计算平均 (使用 torch.stack 保持梯度)
        cd_stack = torch.stack(list(self._drl_cd_history_buffer))
        cl_stack = torch.stack(list(self._drl_cl_history_buffer))  # 已在存储时处理了 abs
        
        cd_avg = cd_stack.mean()
        cl_avg = cl_stack.mean()
        
        return cd_avg, cl_avg
    
    def _reset_loss_buffers(self):
        """重置 loss 相关缓冲区"""
        if hasattr(self, '_snapshot_buffer'):
            self._snapshot_buffer.clear()
        if hasattr(self, '_cl_history_buffer'):
            self._cl_history_buffer.clear()
        if hasattr(self, '_drl_cd_history_buffer'):
            self._drl_cd_history_buffer.clear()
        if hasattr(self, '_drl_cl_history_buffer'):
            self._drl_cl_history_buffer.clear()

    def _get_model_state_dict(self) -> Dict:
        """获取模型状态字典 (覆盖 BaseTrainer)
        
        AFCAgent 使用 network.state_dict() 而非 agent.state_dict()
        """
        if self.agent is None or self.agent.network is None:
            return None
        return self.agent.network.state_dict()
    
    def _get_ema_state_dict(self) -> Dict:
        """获取 EMA 状态字典 (覆盖 BaseTrainer)"""
        if self.ema is None:
            return None
        return self.ema.state_dict()
    
    def _load_model_state_dict(self, state_dict: Dict):
        """加载模型状态字典 (覆盖 BaseTrainer)
        
        AFCAgent 使用 network.load_state_dict()
        """
        if self.agent is None or self.agent.network is None or state_dict is None:
            return
        self.agent.network.load_state_dict(state_dict)
    
    def _load_ema_state_dict(self, state_dict: Dict):
        """加载 EMA 状态字典 (覆盖 BaseTrainer)"""
        if self.ema is None or state_dict is None:
            return
        self.ema.load_state_dict(state_dict)

    def _reset_env(self, reason: str = 'epoch'):
        """重置环境状态
        
        Args:
            reason: 重置原因，'epoch' (训练) 或 'validation' (验证)
        
        流场初始化策略:
            - 训练模式 (reason='epoch'):
                若 solver_initial_resume_step 配置有效，使用随机帧初始化；
                否则使用 env_config 中的 resume_step。
            - 验证模式 (reason='validation'):
                始终使用 env_config 中的 resume_step（若 null 则用 -1，即最后一帧）
        
        执行顺序:
            流场初始化 → 噪声增强 → agent重置 → 缓存清理
        """
        LOGGER.debug(f"[JetTrainer] 环境重置 (原因: {reason})")
        
        cfg = self.training_cfg
        
        # 判断是否使用随机帧初始化（仅训练阶段）
        use_random_resume = (
            reason == 'epoch' and 
            cfg.solver_initial_resume_step is not None and
            self._has_valid_resume_from()
        )
        
        if use_random_resume:
            # 训练模式 + 配置了随机帧范围 → 使用随机帧
            self._reset_with_random_resume_step()
        else:
            # 验证模式 或 未配置随机帧 → 使用 env_config 中的 resume_step
            try:
                self.solver.reset(use_h5=True)  # 默认 use_h5=True
            except Exception as e:
                LOGGER.warning(f"[JetTrainer] 从 h5 初始化失败，改用配置参数进行简单初始化: {e}")
                self.solver.reset(use_h5=False)
        
        # 噪声增强：根据配置决定是否施加
        # - 训练阶段：由 noise_envs 控制
        # - 验证阶段：由 validation_noise_enabled 控制
        apply_noise = (
            cfg.noise_std > 0 and (
                (reason == 'epoch' and cfg.noise_envs) or
                (reason == 'validation' and getattr(cfg, 'validation_noise_enabled', False))
            )
        )
        if apply_noise:
            self._apply_noise_to_velocity_field()
        
        self.agent.reset()
        
        self._state_history_buffer.clear()
        self._current_state_for_kl = None
        
        self._policy_output_ema = None
        
        # 清理 loss 相关的 history buffer（确保每个 epoch/validation 从干净状态开始）
        if hasattr(self, '_drl_cd_history_buffer'):
            self._drl_cd_history_buffer.clear()
        if hasattr(self, '_drl_cl_history_buffer'):
            self._drl_cl_history_buffer.clear()
        if hasattr(self, '_cl_history_buffer'):
            self._cl_history_buffer.clear()
        if hasattr(self, '_snapshot_buffer'):
            self._snapshot_buffer.clear()
        
        self._cached_reg_amp_l2 = None
        self._cached_reg_smoothness = None
        self._cached_reg_stability = None
        self._cached_reg_entropy = None
        self._cached_reg_kl = None
        
        # 重置重要性采样缓存
        self._cached_importance_ratio = torch.tensor(1.0, device=self.device)
        
        self._prev_amplitude = torch.tensor(0.0, device=self.device)
        
        self._detach_graph()
    
    def _get_h5_path(self) -> str:
        """获取首个环境配置的 h5 文件路径"""
        if self.n_envs == 1:
            return self.solver.cfg.initial.resume_from
        return self.solver.solvers[0].cfg.initial.resume_from

    def _has_valid_resume_from(self) -> bool:
        """检查 solver 配置中是否有有效的 resume_from 路径"""
        import os
        h5_path = self._get_h5_path()
        return h5_path is not None and os.path.exists(h5_path)
    
    def _reset_with_random_resume_step(self):
        """使用随机帧索引重置环境
        
        根据 solver_initial_resume_step 配置选择帧索引：
        - int: 使用固定帧
        - [min, max]: 从范围内随机选择 (将自适应 H5 文件内实际保存的时间步间隔)
        
        多环境时根据 sync_resume_step_across_envs 决定是否同步。
        """
        import random
        from myafc.utils.phiflow_io import get_h5_steps
        cfg = self.training_cfg
        resume_cfg = cfg.solver_initial_resume_step
        
        # 解析帧索引配置
        if isinstance(resume_cfg, int):
            # 固定帧
            step = resume_cfg
            steps_factory = lambda: step
        elif isinstance(resume_cfg, (list, tuple)) and len(resume_cfg) == 2:
            # 范围 [min, max]
            h5_path = self._get_h5_path()
            try:
                available_steps = get_h5_steps(h5_path)
                min_step, max_step = _resolve_resume_range_bounds(available_steps, resume_cfg)
                valid_steps = _resolve_resume_range_candidates(available_steps, resume_cfg)

                if valid_steps and (min_step not in available_steps or max_step not in available_steps):
                    LOGGER.info(
                        f"[JetTrainer] 随机抽取策略 | 指定物理区间 [{min_step}, {max_step}] 含未保存时间步，"
                        f"已自适应为已保存区间 [{valid_steps[0]}, {valid_steps[-1]}]"
                    )
                
                # 区间内没有已保存的帧时，使用首个可用帧。
                if not valid_steps:
                    LOGGER.warning(
                        f"[JetTrainer] 随机重置范围 [{resume_cfg[0]}, {resume_cfg[1]}] 中无有效数据。"
                        f"后退使用第一个可用物理帧: {available_steps[0]}"
                    )
                    valid_steps = [available_steps[0]]
                    
                # 检查保存的间隔
                if len(available_steps) > 1:
                    interval = available_steps[1] - available_steps[0]
                    if interval > 1 and not getattr(self, '_has_printed_h5_interval', False):
                        LOGGER.info(f"[JetTrainer] 随机抽取策略 | 检测到 h5 数据具有间隔保存 ({interval} steps), 已自适应调整抽取步长")
                        self._has_printed_h5_interval = True
                        
                steps_factory = lambda: random.choice(valid_steps)
            except Exception as e:
                LOGGER.warning(f"[JetTrainer] 获取 h5 时间步列表发生异常: {e}。将在配置范围内均匀抽取整数步。")
                lo, hi = sorted((int(resume_cfg[0]), int(resume_cfg[1])))
                steps_factory = lambda: random.randint(lo, hi)
        else:
            raise ValueError(f"solver_initial_resume_step 格式无效: {resume_cfg}")
        
        if self.n_envs == 1:
            # 单环境
            step = steps_factory()
            resume_info = self.solver.reset(use_h5=True, step=step)
            LOGGER.info(f"[JetTrainer] 随机帧初始化 | 帧: {resume_info['step']}")
        else:
            # 多环境
            if cfg.sync_resume_step_across_envs:
                # 所有环境使用相同帧
                steps = steps_factory()
            else:
                # 每个环境独立随机
                steps = [steps_factory() for _ in range(self.n_envs)]
            
            resume_infos = self.solver.reset(use_h5=True, steps=steps)
            loaded_steps = [info['step'] for info in resume_infos]
            LOGGER.info(f"[JetTrainer] 随机帧初始化 | 帧: {loaded_steps}")
    
    def _apply_noise_to_velocity_field(self):
        """向初始速度场添加高斯噪声
        
        用于增强训练泛化能力，每次 reset 后施加不同的噪声扰动。
        噪声标准差由 training.noise_std 控制。
        
        实现方式:
        - 获取 solver 的速度场张量
        - 添加均值为 0、标准差为 noise_std 的高斯噪声
        - 噪声相对于速度场的参考速度进行缩放
        """
        noise_std = self.training_cfg.noise_std
        
        try:
            # 获取速度场的内部张量
            vel_x = self.solver.velocity['x'].values
            vel_y = self.solver.velocity['y'].values
            
            # 估算参考速度幅值（使用入口速度或平均速度）
            ref_velocity = 1.0  # 默认参考速度
            if hasattr(self.solver, '_inflow_velocity') and self.solver._inflow_velocity is not None:
                ref_velocity = max(abs(self.solver._inflow_velocity[0]), 0.1)
            
            # 生成并添加高斯噪声
            noise_x = torch.randn_like(vel_x) * noise_std * ref_velocity
            noise_y = torch.randn_like(vel_y) * noise_std * ref_velocity
            
            # 原地添加噪声
            vel_x.add_(noise_x)
            vel_y.add_(noise_y)
            
            LOGGER.debug(f"[JetTrainer] noise_envs: 添加高斯噪声 (std={noise_std * ref_velocity:.4f})")
            
        except Exception as e:
            LOGGER.warning(f"[JetTrainer] noise_envs 失败: {e}")
    
    def _get_current_amplitude(self, apply_jets: bool = True) -> float:
        """获取当前控制幅值 - 使用实际施加值
        
        统一训练和验证阶段的幅值获取逻辑，减少重复代码。
        
        Args:
            apply_jets: 当前步是否施加射流。若为 False，返回 0（因为实际未施加控制）
        
        Returns:
            当前实际施加的控制幅值
        """
        if not apply_jets:
            # 射流未施加时，实际作用于流场的幅值为 0
            return 0.0
        
        if self.solver.jet_obstacle and self.solver.jet_obstacle.current_controls:
            # 使用实际施加值（经过 apply_controls 处理后的值）
            return float(self.solver.jet_obstacle.current_controls[0][0].detach().cpu())
        
        if hasattr(self, '_current_amplitude') and self._current_amplitude is not None:
            # fallback: 使用网络原始输出
            return float(self._current_amplitude.detach().cpu())
        
        return 0.0
    
    def _get_actual_solver(self):
        """获取实际的 solver 实例
        
        多环境时 (ParallelSolverManager) 返回第一个子环境的 solver。
        单环境时直接返回 self.solver。
        
        用于创建 runner 等需要单个 solver 实例的场景。
        
        Returns:
            KarmanTorchSolver 实例
        """
        if hasattr(self.solver, 'solvers') and len(self.solver.solvers) > 0:
            return self.solver.solvers[0]
        return self.solver
    
    def _record_amplitude(self, apply_jets: bool = True):
        """记录当前控制幅值到历史列表
        
        Args:
            apply_jets: 当前步是否施加射流
        """
        amp = self._get_current_amplitude(apply_jets)
        self.amplitude_history.append(amp)

    def _get_training_phase_amplitude_history(self) -> list:
        """获取训练阶段（去除 warmup）的幅值历史

        说明:
            `amplitude_history` 记录了 warmup + training 全部步（跨 epoch 累积）。
            step 级 loss/velocity/force 只记录 training 段。
            因此应优先按“已记录的训练步数”对齐(裁剪每段 warmup)，而不是只裁剪首段 warmup。
        """
        # 优先按 epoch 结构对齐（逐 epoch 去除 warmup）
        if hasattr(self, '_step_losses_record') and self._step_losses_record is not None:
            n_training_steps = len(self._step_losses_record)
            if n_training_steps <= 0:
                return []

            warmup_steps = int(getattr(self.cfg.training, 'env_warmup_envsteps', 0))
            train_per_epoch = int(getattr(self.cfg.training, 'num_interaction_per_mini_episode', 0)) * int(getattr(self.cfg.training, 'control_interval', 0))

            if warmup_steps > 0 and train_per_epoch > 0:
                full_epochs = n_training_steps // train_per_epoch
                remainder_steps = n_training_steps % train_per_epoch

                # 需要的最小历史长度（含每个 epoch 的 warmup）
                min_required = full_epochs * (warmup_steps + train_per_epoch)
                if remainder_steps > 0:
                    min_required += warmup_steps + remainder_steps

                if len(self.amplitude_history) >= min_required:
                    aligned = []
                    cursor = 0

                    for _ in range(full_epochs):
                        cursor += warmup_steps
                        aligned.extend(self.amplitude_history[cursor:cursor + train_per_epoch])
                        cursor += train_per_epoch

                    if remainder_steps > 0:
                        cursor += warmup_steps
                        aligned.extend(self.amplitude_history[cursor:cursor + remainder_steps])

                    if len(aligned) == n_training_steps:
                        return aligned

            # 兜底：无法可靠按 epoch 分段时，回退为尾部等长对齐
            if len(self.amplitude_history) <= n_training_steps:
                return list(self.amplitude_history)
            return self.amplitude_history[-n_training_steps:]

        # 无逐步记录时，仅裁剪首段 warmup。
        warmup_steps = int(getattr(self.cfg.training, 'env_warmup_envsteps', 0))
        if warmup_steps <= 0:
            return list(self.amplitude_history)
        if len(self.amplitude_history) <= warmup_steps:
            return []
        return self.amplitude_history[warmup_steps:]
    
    def _sample_target_velocity(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样目标探针位置的速度
        
        统一训练和验证阶段的速度采样逻辑，减少重复代码。
        
        Returns:
            (u, v): 目标点的 x 和 y 方向速度张量
        """
        vel_at_target = field.sample(self.solver.velocity, self.target_point)
        u = vel_at_target.vector['x'].native().squeeze()
        v = vel_at_target.vector['y'].native().squeeze()
        return u, v
    
    def _sample_target_pressure(self) -> float:
        """采样目标探针位置的压力
        
        统一训练和验证阶段的压力采样逻辑。
        
        Returns:
            p: 目标点的压力标量值 (float)
        """
        p_at_target = field.sample(self.solver.pressure, self.target_point)
        p = float(p_at_target.native().squeeze().detach().cpu())
        return p
    
    def _env_step(
        self, 
        steps_in_accumulation: int,
        is_warmup: bool = False,
    ) -> Tuple[bool, bool, Dict]:
        """
        封装单步环境求解的准备阶段（不执行 solver.step）
        
        执行流程:
        1. 更新 sinusoidal 入口边界（如果启用）
        2. 判断控制信号更新时机和射流施加时机
        3. 构建状态并注入历史信息
        4. 缓存 KL 散度计算所需的状态快照
        
        调用方使用模式:
        ```python
        is_update, apply_jets, state = self._env_step(steps, is_warmup)
        if is_update:
            amplitude, policy_output = self.agent.compute_action(state)
            self.solver.jet_obstacle.set_controls_external(amplitudes=amplitude, ...)
        self._do_solver_step(apply_jets, state)
        ```
        
        Args:
            steps_in_accumulation: 当前累积步数（用于判断控制时机）
            is_warmup: 是否为预热阶段
            
        Returns:
            (should_update_control, should_apply_jets, state):
            - should_update_control: 是否应该更新控制信号
            - should_apply_jets: 是否应该施加射流
            - state: 当前状态字典
        """
        import copy
        cfg = self.cfg.training
        
        # 1. sinusoidal 入口模式：更新边界条件
        self._update_sinusoidal_inlet(self.solver.current_time)
        
        # 2. 判断控制逻辑
        if is_warmup:
            should_update_control = cfg.warmup_update_controls
            should_apply_jets = cfg.warmup_apply_jets
        else:
            should_update_control = (steps_in_accumulation % cfg.control_interval == 0)
            should_apply_jets = (steps_in_accumulation % cfg.control_interval < cfg.action_duration)
        
        # 3. 构建状态并注入历史
        state = self._build_state()
        feat_cfg = self.cfg.input_features
        if feat_cfg.envstep_state > 1:
            state['_state_history'] = list(self._state_history_buffer)
        
        # 4. 缓存 KL 散度计算所需的状态快照
        if should_update_control and not is_warmup:
            self._current_state_for_kl = copy.deepcopy(state)
        
        return should_update_control, should_apply_jets, state
    
    def _do_solver_step(self, apply_jets: bool, state: Dict = None):
        """执行 solver.step 并更新历史缓冲区
        
        Args:
            apply_jets: 是否施加射流
            state: 当前状态（用于更新历史缓冲区）
        """
        import copy
        self.solver.step(update_controls=False, apply_jets=apply_jets)
        
        # 清空力系数缓存（每步求解后流场变化，需重新计算）
        self._cached_force_coeffs = None
        
        # 更新历史状态缓冲区
        feat_cfg = self.cfg.input_features
        if feat_cfg.envstep_state > 1 and state is not None:
            state_copy = {k: v for k, v in state.items() if not k.startswith('_')}
            self._state_history_buffer.append(copy.deepcopy(state_copy))
            max_history_len = feat_cfg.envstep_state - 1
            if len(self._state_history_buffer) > max_history_len:
                self._state_history_buffer = self._state_history_buffer[-max_history_len:]
    
    def _get_force_coeffs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前力系数（带缓存）
        
        在同一 env-step 内多次调用时复用缓存，避免重复计算。
        缓存在 _do_solver_step() 后自动清空。
        
        Returns:
            (cd, cl): 阻力系数和升力系数
        """
        if self._cached_force_coeffs is None:
            cd, cl, _, _, _, _ = self.solver.compute_force_coeffs()
            self._cached_force_coeffs = (cd, cl)
        return self._cached_force_coeffs
    
    def _get_env_config_path(self) -> str:
        """获取环境配置文件的绝对路径（统一路径解析）
        
        处理相对路径：相对于 myafc/configs/ 目录
        
        Returns:
            环境配置文件的绝对路径
        """
        env_cfg_path = self.cfg.env_config_path
        if not os.path.isabs(env_cfg_path):
            configs_dir = Path(__file__).parent.parent.parent.parent / 'configs'
            env_cfg_path = str(configs_dir / env_cfg_path)
        return env_cfg_path
    
    def _load_env_config(self) -> Dict[str, Any]:
        """加载环境配置文件（委托给 utils.load_env_config）
        
        Returns:
            解析后的配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
        """
        env_cfg_path = self._get_env_config_path()
        return load_env_config(env_cfg_path)
    
    def _create_runner(self, output_dir: Path, runner_cfg: Dict[str, Any], 
                       name: str = "runner") -> 'SolverRunner':
        """创建 SolverRunner 实例（委托给 utils.create_runner）
        
        Args:
            output_dir: 输出目录路径
            runner_cfg: runner 配置字典
            name: runner 名称（用于日志）
            
        Returns:
            创建的 SolverRunner 实例
        """
        # 初始化 runners 列表
        if not hasattr(self, 'runners'):
            self.runners = []
        
        # 获取实际的 solver（使用公共方法）
        actual_solver = self._get_actual_solver()
        
        # 使用共用函数创建 runner
        runner = create_runner(actual_solver, runner_cfg, output_dir, name)
        self.runners.append(runner)
        
        return runner

    # ========== 配置预处理 ==========
    
    def _preprocess_config(self):
        """配置预处理：类型转换、约束检查与自动覆盖
        
        执行任务：
        1. 关键整数参数强制取整
        2. 参数约束检查与自动覆盖：
           - control_interval >= action_duration
           - num_of_interaction_per_agent_update >= 1
           - env_warmup_envsteps >= envstep_state - 1
        3. 计算 envstep_per_value
        """
        cfg = self.cfg.training
        
        # ===== 整数参数强制转换 =====
        cfg.action_duration = int(cfg.action_duration)
        cfg.env_warmup_envsteps = int(cfg.env_warmup_envsteps)
        cfg.num_interaction_per_mini_episode = int(cfg.num_interaction_per_mini_episode)
        cfg.num_of_interaction_per_agent_update = int(cfg.num_of_interaction_per_agent_update)
        
        # ===== control_interval_raw 处理 =====
        # 支持别名: 'action_duration' 和 'equal2action_duration' 均表示等于 action_duration
        if isinstance(cfg.control_interval_raw, str):
            if cfg.control_interval_raw.lower() not in ('action_duration', 'equal2action_duration'):
                try:
                    cfg.control_interval_raw = int(cfg.control_interval_raw)
                except ValueError:
                    raise ValueError(f"control_interval 值无效: '{cfg.control_interval_raw}'")
        elif isinstance(cfg.control_interval_raw, float):
            cfg.control_interval_raw = int(cfg.control_interval_raw)
        
        # ===== 约束检查: control_interval >= action_duration =====
        if cfg.control_interval < cfg.action_duration:
            LOGGER.warning(f"[JetTrainer] control_interval ({cfg.control_interval}) < action_duration ({cfg.action_duration})，自动覆盖为 {cfg.action_duration}")
            cfg.control_interval_raw = cfg.action_duration
        
        # ===== 约束检查: num_of_interaction_per_agent_update >= 1 =====
        if cfg.num_of_interaction_per_agent_update < 1:
            LOGGER.warning(f"[JetTrainer] num_of_interaction_per_agent_update < 1，自动覆盖为 1")
            cfg.num_of_interaction_per_agent_update = 1
        
        # ===== 约束检查: env_warmup_envsteps >= envstep_state - 1 =====
        envstep_state = self.cfg.input_features.envstep_state
        if envstep_state > 1:
            min_warmup = envstep_state - 1
            if cfg.env_warmup_envsteps < min_warmup:
                LOGGER.warning(f"[JetTrainer] env_warmup_envsteps ({cfg.env_warmup_envsteps}) < 历史缓冲所需步数 ({min_warmup})，自动覆盖")
                cfg.env_warmup_envsteps = min_warmup
        
        # ===== 计算 envstep_per_value =====
        self._envstep_per_value = cfg.num_of_interaction_per_agent_update * cfg.control_interval
        LOGGER.info(f"[JetTrainer] envstep_per_value = {cfg.num_of_interaction_per_agent_update} × {cfg.control_interval} = {self._envstep_per_value}")

        # ===== 检查 epoch 内更新窗口是否整除 =====
        # 若不整除，最后会存在一个未满窗口（当前实现不会触发更新），
        # 这会导致有效梯度更新次数低于预期。
        if cfg.num_interaction_per_mini_episode % cfg.num_of_interaction_per_agent_update != 0:
            remainder = cfg.num_interaction_per_mini_episode % cfg.num_of_interaction_per_agent_update
            LOGGER.warning(
                f"[JetTrainer] num_interaction_per_mini_episode ({cfg.num_interaction_per_mini_episode}) "
                f"不能被 num_of_interaction_per_agent_update ({cfg.num_of_interaction_per_agent_update}) 整除，"
                f"每个 epoch 末尾将有 {remainder} 个 interaction 不参与参数更新。"
            )

        # ===== 检查更新窗口长度是否较大 =====
        if cfg.num_of_interaction_per_agent_update >= 800 and not getattr(cfg, 'autograd_offload_to_cpu', False):
            LOGGER.warning(f"[JetTrainer] num_of_interaction_per_agent_update ({cfg.num_of_interaction_per_agent_update}) 较大，建议开启 autograd_offload_to_cpu。")

    def _get_autograd_offload_context(self):
        """返回反传中间张量 CPU offload 上下文。

        该优化不改变 loss 与梯度语义，仅改变中间张量存储位置（GPU -> CPU）。
        """
        cfg = self.cfg.training
        if not getattr(cfg, 'autograd_offload_to_cpu', False):
            return nullcontext()

        if not torch.cuda.is_available():
            return nullcontext()

        # 旧版本 torch 可能不存在 graph.save_on_cpu
        graph_mod = getattr(torch.autograd, 'graph', None)
        save_on_cpu = getattr(graph_mod, 'save_on_cpu', None) if graph_mod is not None else None
        if save_on_cpu is None:
            LOGGER.warning("[JetTrainer] 当前 PyTorch 不支持 autograd.graph.save_on_cpu，已自动禁用 autograd_offload_to_cpu")
            return nullcontext()

        pin_memory = bool(getattr(cfg, 'autograd_offload_pin_memory', True))
        return save_on_cpu(pin_memory=pin_memory)
    
    # ========== 可视化辅助方法（训练/验证共用） ==========
    
    def _plot_velocity_panel(self, ax, velocities_u: list, velocities_v: list, 
                             target_u: float, target_v: float, 
                             pressures: list,
                             add_markers_fn=None, add_vlines_fn=None):
        """绘制速度子图（训练/验证共用），含压力曲线（右轴）
        
        Args:
            ax: matplotlib axes
            velocities_u: u 分量列表
            velocities_v: v 分量列表
            target_u: u 目标值
            target_v: v 目标值
            pressures: 压力值列表（绘制在右轴）
            add_markers_fn: 可选，添加更新标记的回调 (ax, x_data, y_data, color)
            add_vlines_fn: 可选，添加竖直虚线的回调 (ax)
        """
        if not velocities_u:
            ax.text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=ax.transAxes)
            return
        ax.plot(velocities_u, 'b-', label='u', alpha=0.8, linewidth=0.8)
        ax.plot(velocities_v, 'orange', label='v', alpha=0.8, linewidth=0.8)
        ax.axhline(y=target_u, color='blue', linestyle='--', alpha=0.5, label=f'u_target={target_u:.3f}')
        ax.axhline(y=target_v, color='orange', linestyle='--', alpha=0.5, label=f'v_target={target_v:.3f}')
        if add_vlines_fn:
            add_vlines_fn(ax)
        if add_markers_fn:
            add_markers_fn(ax, range(len(velocities_u)), velocities_u, color='purple')
            add_markers_fn(ax, range(len(velocities_v)), velocities_v, color='darkorange')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Velocity')
        ax.set_title('Probe Velocity & Pressure')
        
        # 压力曲线（右轴）
        ax2 = ax.twinx()
        ax2.plot(pressures, 'g-', label='p', alpha=0.6, linewidth=0.8)
        ax2.set_ylabel('Pressure', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_amplitude_panel(self, ax, amplitudes: list, 
                               add_markers_fn=None, add_vlines_fn=None):
        """绘制控制幅值子图（训练/验证共用）"""
        if not amplitudes:
            ax.text(0.5, 0.5, 'No amplitude data', ha='center', va='center', transform=ax.transAxes)
            return
        ax.plot(amplitudes, 'g-', alpha=0.8, linewidth=0.8)
        if add_vlines_fn:
            add_vlines_fn(ax)
        if add_markers_fn:
            add_markers_fn(ax, range(len(amplitudes)), amplitudes, color='darkgreen')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Amplitude')
        ax.set_title('Control Amplitude')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    
    def _plot_cd_panel(self, ax, cd_data: list, 
                        add_markers_fn=None, add_vlines_fn=None):
        """绘制阻力系数 Cd 子图（训练/验证共用）"""
        if not cd_data:
            ax.text(0.5, 0.5, 'No Cd data', ha='center', va='center', transform=ax.transAxes)
            return
        # 处理 NaN
        valid_cd = [c if not np.isnan(c) else None for c in cd_data]
        ax.plot(valid_cd, 'b-', alpha=0.8, linewidth=0.8, label='Cd')
        if add_vlines_fn:
            add_vlines_fn(ax)
        if add_markers_fn:
            add_markers_fn(ax, range(len(cd_data)), cd_data, color='navy')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Cd')
        ax.set_title('Drag Coefficient (Cd)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_cl_panel(self, ax, cl_data: list, 
                        add_markers_fn=None, add_vlines_fn=None):
        """绘制升力系数 Cl 子图（训练/验证共用）"""
        if not cl_data:
            ax.text(0.5, 0.5, 'No Cl data', ha='center', va='center', transform=ax.transAxes)
            return
        # 处理 NaN
        valid_cl = [c if not np.isnan(c) else None for c in cl_data]
        ax.plot(valid_cl, 'orange', alpha=0.8, linewidth=0.8, label='Cl')
        if add_vlines_fn:
            add_vlines_fn(ax)
        if add_markers_fn:
            add_markers_fn(ax, range(len(cl_data)), cl_data, color='darkorange')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Cl')
        ax.set_title('Lift Coefficient (Cl)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_loss_panel(self, ax, loss_data: list, title: str = 'Loss',
                          label: str = 'Loss', color: str = 'g-',
                          add_markers_fn=None, add_vlines_fn=None):
        """绘制 Loss 子图（训练/验证共用）"""
        if not loss_data:
            ax.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax.transAxes)
            return
        # 处理 NaN
        valid_losses = [l if not np.isnan(l) else None for l in loss_data]
        ax.plot(valid_losses, color, alpha=0.8, linewidth=0.8, label=label)
        if add_vlines_fn:
            add_vlines_fn(ax)
        if add_markers_fn:
            add_markers_fn(ax, range(len(loss_data)), loss_data, color='green')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # ========== 训练数据增量加载/保存 ==========
    
    def _load_existing_training_data(self):
        """加载已有训练数据 (用于 resume 增量保存)
        
        从 training_data.npz 加载数据并填充到历史记录列表，
        使后续保存时能保持数据连续性。
        """
        npz_path = self.exp_dir / 'training_data.npz'
        if not npz_path.exists():
            LOGGER.info("[JetTrainer] 无已有训练数据，从空开始")
            return
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Epoch 级别数据
            if 'loss_history' in data and data['loss_history'].size > 0:
                self.loss_history = data['loss_history'].tolist()
            if 'primary_loss_history' in data and data['primary_loss_history'].size > 0:
                self.primary_loss_history = data['primary_loss_history'].tolist()
            if 'reg_loss_history' in data and data['reg_loss_history'].size > 0:
                self.reg_loss_history = data['reg_loss_history'].tolist()
            
            # Step 级别数据
            if 'step_losses' in data and data['step_losses'].size > 0:
                self._step_losses_record = data['step_losses'].tolist()
            if 'step_total_losses' in data and data['step_total_losses'].size > 0:
                self._step_total_losses_record = data['step_total_losses'].tolist()
            if 'step_reg_losses' in data and data['step_reg_losses'].size > 0:
                self._step_reg_losses_record = data['step_reg_losses'].tolist()
            if 'step_velocities_u' in data and data['step_velocities_u'].size > 0:
                self._step_velocities_u = data['step_velocities_u'].tolist()
            if 'step_velocities_v' in data and data['step_velocities_v'].size > 0:
                self._step_velocities_v = data['step_velocities_v'].tolist()
            if 'step_pressures' in data and data['step_pressures'].size > 0:
                self._step_pressures = data['step_pressures'].tolist()
            if 'amplitude_history' in data and data['amplitude_history'].size > 0:
                self.amplitude_history = data['amplitude_history'].tolist()
            if 'step_cd' in data and data['step_cd'].size > 0:
                self._step_cd_record = data['step_cd'].tolist()
            if 'step_cl' in data and data['step_cl'].size > 0:
                self._step_cl_record = data['step_cl'].tolist()
            
            # NN 更新级别数据
            if 'acc_total_loss' in data and data['acc_total_loss'].size > 0:
                self._acc_total_loss_record = data['acc_total_loss'].tolist()
            if 'update_step_indices' in data and data['update_step_indices'].size > 0:
                self._update_step_indices = data['update_step_indices'].tolist()
            
            LOGGER.info(f"[JetTrainer] 已加载已有训练数据: {len(self.loss_history)} epochs, {len(self._step_losses_record)} steps")
            
        except Exception as e:
            LOGGER.warning(f"[JetTrainer] 加载已有训练数据失败: {e}，从空开始")
    
    def _save_training_results(self, final: bool = True):
        """保存训练结果 (npz, csv, 曲线图)
        
        保存内容与 training_curves.png 可视化数据一致:
        - loss_history/primary_loss_history/reg_loss_history: epoch 级别
        - step_*: env-step 级别 (用于曲线图绘制)
        - acc_total_loss: 每次 NN 更新的加权累积 loss
        - final=False 时只做增量保存，不生成视频
        - 场数据 h5 的写入由 SolverRunner 内部按次打开/关闭完成，不依赖长期句柄
        
        注意: Resume 时会先加载已有数据，保存时包含完整历史。
        """
        import matplotlib.pyplot as plt

        # 训练阶段幅值（去除 warmup），用于与 step 级曲线对齐
        amplitude_history_training = self._get_training_phase_amplitude_history()
        
        # ===== 保存 npz (与 training_curves.png 可视化数据对齐) =====
        np.savez(
            self.exp_dir / 'training_data.npz',
            # Epoch 级别数据
            loss_history=np.array(self.loss_history),
            primary_loss_history=np.array(self.primary_loss_history),
            reg_loss_history=np.array(self.reg_loss_history),
            # Step 级别数据 (用于可视化)
            step_losses=np.array(self._step_losses_record) if hasattr(self, '_step_losses_record') else np.array([]),
            step_total_losses=np.array(self._step_total_losses_record) if hasattr(self, '_step_total_losses_record') else np.array([]),
            step_reg_losses=np.array(self._step_reg_losses_record) if hasattr(self, '_step_reg_losses_record') else np.array([]),
            step_velocities_u=np.array(self._step_velocities_u) if hasattr(self, '_step_velocities_u') else np.array([]),
            step_velocities_v=np.array(self._step_velocities_v) if hasattr(self, '_step_velocities_v') else np.array([]),
            step_pressures=np.array(self._step_pressures),
            amplitude_history=np.array(amplitude_history_training),
            amplitude_history_full=np.array(self.amplitude_history),
            step_cd=np.array(self._step_cd_record) if hasattr(self, '_step_cd_record') else np.array([]),
            step_cl=np.array(self._step_cl_record) if hasattr(self, '_step_cl_record') else np.array([]),
            # NN 更新级别数据
            acc_total_loss=np.array(self._acc_total_loss_record) if hasattr(self, '_acc_total_loss_record') else np.array([]),
            update_step_indices=np.array(self._update_step_indices) if hasattr(self, '_update_step_indices') else np.array([]),
            # 配置参考
            target_velocity=np.array(self.cfg.loss.target_velocity),
        )
        
        # ===== 保存 CSV (与 training_curves.png 可视化数据对齐) =====
        if hasattr(self, '_step_losses_record') and self._step_losses_record:
            import pandas as pd
            n_steps = len(self._step_losses_record)
            df_data = {
                'env_step': range(n_steps),
                'primary_loss': self._step_losses_record,
                'total_loss': self._step_total_losses_record if hasattr(self, '_step_total_losses_record') else [0.0] * n_steps,
                'reg_loss': self._step_reg_losses_record if hasattr(self, '_step_reg_losses_record') else [0.0] * n_steps,
                'velocity_u': self._step_velocities_u if hasattr(self, '_step_velocities_u') else [0.0] * n_steps,
                'velocity_v': self._step_velocities_v if hasattr(self, '_step_velocities_v') else [0.0] * n_steps,
                'pressure': self._step_pressures,
                'amplitude': (amplitude_history_training[:n_steps]
                              if len(amplitude_history_training) >= n_steps
                              else amplitude_history_training + [0.0] * (n_steps - len(amplitude_history_training))),
            }
            # 添加力系数数据（如果有）
            if hasattr(self, '_step_cd_record') and self._step_cd_record:
                df_data['cd'] = self._step_cd_record
                df_data['cl'] = self._step_cl_record
            df = pd.DataFrame(df_data)
            df.to_csv(self.exp_dir / 'training_step_data.csv', index=False)
        
        # ===== 绘制训练曲线 (4行2列布局) =====
        if not self.loss_history:
            return
        
        fig, axes = plt.subplots(4, 2, figsize=(14, 18))
        
        # 获取 update step 索引（用于可视化标记）
        update_indices = getattr(self, '_update_step_indices', [])
        
        # 可视化配置：散点 (曲线上的更新点标记)
        scatter_enabled = getattr(self.cfg.output, 'scatter_enabled', True)
        scatter_size = getattr(self.cfg.output, 'scatter_size', 20.0)
        
        # 可视化配置：竖直虚线 (神经网络更新位置)
        vline_enabled = getattr(self.cfg.output, 'vline_enabled', False)
        vline_width = getattr(self.cfg.output, 'vline_width', 0.5)
        
        # 辅助函数：在曲线上添加更新点散点
        def add_update_markers(ax, x_data, y_data, color='red', marker='o', size=None, alpha=0.6):
            """在神经网络更新时刻添加散点标记"""
            if not scatter_enabled or not update_indices:
                return
            if size is None:
                size = scatter_size
            valid_indices = [i for i in update_indices if i < len(y_data)]
            if valid_indices:
                ax.scatter(valid_indices, [y_data[i] for i in valid_indices],
                          c=color, marker=marker, s=size, alpha=alpha, 
                          zorder=5, label='NN Update')
        
        # 辅助函数：绘制神经网络更新位置的竖直虚线
        def add_update_vlines(ax, color='red', alpha=0.4):
            """在神经网络更新时刻绘制竖直虚线"""
            if not vline_enabled or not update_indices:
                return
            for idx in update_indices:
                ax.axvline(x=idx, color=color, linestyle='--', 
                          linewidth=vline_width, alpha=alpha, zorder=1)
        
        # ========== 第0行: 加权累积 Loss 和 权重分布 ==========
        
        # (0,0) 加权累积 Loss (acc_total_loss)
        ax = axes[0, 0]
        if hasattr(self, '_acc_total_loss_record') and self._acc_total_loss_record:
            ax.plot(self._acc_total_loss_record, 'purple', alpha=0.8, linewidth=1.0, marker='o', markersize=4)
            ax.set_xlabel('NN Update')
            ax.set_ylabel('Accumulated Total Loss')
            ax.set_title('Weighted Accumulated Total Loss (per NN Update)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No acc_total_loss data', ha='center', va='center', transform=ax.transAxes)
        
        # (0,1) Loss 权重分布 (一个更新周期内)
        ax = axes[0, 1]
        cfg = self.cfg.training
        envstep_per_value = cfg.num_of_interaction_per_agent_update * cfg.control_interval
        weights = self._compute_loss_weights(envstep_per_value)
        ax.bar(range(len(weights)), weights, color='teal', alpha=0.7)
        ax.set_xlabel('Step (within one NN Update cycle)')
        ax.set_ylabel('Weight')
        # 标题：显示模式 + 关键参数信息
        weight_mode = self.cfg.loss.loss_weights
        if weight_mode == 'exponential':
            # 提示 D=K 时权重递增的特殊行为
            dk_note = " [D=K → ascending]" if cfg.action_duration == cfg.control_interval else ""
            ax.set_title(f'Loss Weights ({weight_mode}, γ={self.cfg.loss.loss_gamma}{dk_note})')
        else:
            ax.set_title(f'Loss Weights per Step ({weight_mode} mode)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # ========== 第1行: Per-Step Loss ==========
        
        # (1,0) Per-Step Loss: Total Loss + Primary Loss
        ax = axes[1, 0]
        if hasattr(self, '_step_total_losses_record') and self._step_total_losses_record:
            ax.plot(self._step_total_losses_record, 'b-', alpha=0.7, linewidth=0.8, label='Total Loss')
            ax.plot(self._step_losses_record, 'g--', alpha=0.7, linewidth=0.8, label='Primary Loss')
            add_update_vlines(ax)
            add_update_markers(ax, range(len(self._step_total_losses_record)), self._step_total_losses_record)
            ax.set_xlabel('Env Step')
            ax.set_ylabel('Loss')
            ax.set_title('Per-Step Loss (Total & Primary)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No step data', ha='center', va='center', transform=ax.transAxes)
        
        # (1,1) Per-Step Regularization Loss
        ax = axes[1, 1]
        if hasattr(self, '_step_reg_losses_record') and self._step_reg_losses_record:
            ax.plot(self._step_reg_losses_record, 'r-', alpha=0.7, linewidth=0.8, label='Reg Loss')
            add_update_vlines(ax)
            add_update_markers(ax, range(len(self._step_reg_losses_record)), self._step_reg_losses_record, color='darkred')
            ax.set_xlabel('Env Step')
            ax.set_ylabel('Reg Loss')
            ax.set_title('Per-Step Regularization Loss')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No reg loss data', ha='center', va='center', transform=ax.transAxes)
        
        # ========== 第2行: Velocity & Amplitude ==========
        
        # (2,0) Velocity - 使用共用辅助方法
        target_u, target_v = self.cfg.loss.target_velocity
        self._plot_velocity_panel(
            axes[2, 0], 
            self._step_velocities_u, self._step_velocities_v,
            target_u, target_v,
            pressures=self._step_pressures,
            add_markers_fn=add_update_markers, add_vlines_fn=add_update_vlines,
        )
        
        # (2,1) Amplitude - 使用共用辅助方法
        self._plot_amplitude_panel(
            axes[2, 1], amplitude_history_training,
            add_markers_fn=add_update_markers, add_vlines_fn=add_update_vlines
        )
        
        # ========== 第3行: Force Coefficients ==========
        
        # (3,0) Cd - 使用共用辅助方法
        self._plot_cd_panel(
            axes[3, 0],
            self._step_cd_record if hasattr(self, '_step_cd_record') else [],
            add_markers_fn=add_update_markers, add_vlines_fn=add_update_vlines
        )
        
        # (3,1) Cl - 使用共用辅助方法
        self._plot_cl_panel(
            axes[3, 1],
            self._step_cl_record if hasattr(self, '_step_cl_record') else [],
            add_markers_fn=add_update_markers, add_vlines_fn=add_update_vlines
        )
        
        plt.tight_layout()
        fig.savefig(self.exp_dir / 'training_curves.png', dpi=150)
        plt.close(fig)
        
        if final:
            # ===== 生成训练阶段流场视频（如果启用） =====
            if (self.training_runner is not None 
                and self.cfg.training_visuals.enabled 
                and self.cfg.training_visuals.generate_video):
                self.training_runner._generate_videos()

            # ===== 关闭训练阶段 h5 文件 =====
            if self.training_runner is not None and self.training_runner._h5_file is not None:
                self.training_runner._h5_file.close()
                self.training_runner._h5_file = None
        
        LOGGER.info(f"[JetTrainer] 训练数据已保存: {self.exp_dir}")
    
    # ========== 核心训练循环 ==========
    
    def train(self) -> Dict:
        """执行训练
        
        Returns:
            训练结果字典
        """
        # 日志和进度条统一写入 train.log，同时保留控制台输出
        with tee_console_output(str(self.exp_dir / 'train.log'), append=True):
            # ===== 配置预处理 =====
            self._preprocess_config()

            cfg = self.cfg.training
            envstep_per_value = cfg.num_of_interaction_per_agent_update * cfg.control_interval

            # 初始化详细 step 记录列表
            self._step_losses_record: List[float] = []       # Primary Loss (每步)
            self._step_total_losses_record: List[float] = [] # Total Loss = Primary + Reg (每步)
            self._step_reg_losses_record: List[float] = []   # Reg Loss (每步)
            self._acc_total_loss_record: List[float] = []    # 加权累积后的 Total Loss (每次网络更新)
            self._step_velocities_u: List[float] = []
            self._step_velocities_v: List[float] = []
            self._step_pressures: List[float] = []
            
            # 力系数记录 (每步)
            self._step_cd_record: List[float] = []
            self._step_cl_record: List[float] = []
            
            # 神经网络更新时刻记录 (update step 索引列表)
            self._update_step_indices: List[int] = []
            
            # best loss 追踪 (YOLOv8 风格)
            self._best_loss = float('inf')
            
            # Resume 时加载已有训练数据 (增量保存支持)
            if self.training_cfg.resume and self.current_epoch > 0:
                self._load_existing_training_data()
                LOGGER.info(f"[JetTrainer] 已加载原有训练数据用于增量保存")
            
            LOGGER.info(f"[JetTrainer] 开始训练")
            LOGGER.info(f"  - Epochs: {cfg.epochs}")
            LOGGER.info(f"  - Interactions/Epoch: {cfg.num_interaction_per_mini_episode}")
            LOGGER.info(f"  - Steps/Update: {envstep_per_value}")
            LOGGER.info(f"  - 多环境数: {self.n_envs}")
            
            # resume 后从已完成轮次开始（若已全部完成则跳过训练循环）
            start_epoch = self.current_epoch
            if start_epoch >= cfg.epochs:
                LOGGER.info(f"[JetTrainer] 已完成所有 {cfg.epochs} 轮训练 (resume from epoch {start_epoch})，跳过训练阶段")
            elif start_epoch > 0:
                LOGGER.info(f"[JetTrainer] Resume (断点续训): 从 epoch {start_epoch}/{cfg.epochs} 开始")
            
            for epoch in range(start_epoch, cfg.epochs):
                self.current_epoch = epoch
                # 检查是否进入"无正则化"阶段并日志记录
                disable_last_n = self.cfg.loss.disable_regularization_last_epochs
                if isinstance(disable_last_n, str) and disable_last_n.lower() == 'all':
                    if epoch == 0:
                        LOGGER.info("[JetTrainer] 所有 epoch 禁用正则化 (disable_regularization_last_epochs='all')")
                elif isinstance(disable_last_n, int) and disable_last_n > 0:
                    if epoch == max(0, self.num_epochs - disable_last_n):
                        LOGGER.info(f"[JetTrainer] 进入无正则化阶段: 最后 {disable_last_n} 个 epoch 将禁用所有正则化项")

                epoch_loss, epoch_primary, epoch_reg = self._train_single_epoch(epoch, cfg.epochs)

                # 记录
                self.loss_history.append(epoch_loss)
                self.primary_loss_history.append(epoch_primary)
                self.reg_loss_history.append(epoch_reg)

                self._log_metrics({
                    'loss/epoch_total': epoch_loss,
                    'loss/epoch_primary': epoch_primary,
                    'loss/epoch_reg': epoch_reg,
                }, epoch)

                # ===== 检查点保存 (YOLOv8 风格: last / best / epoch_N) =====
                # 更新 current_epoch 为已完成轮数 (epoch + 1)，供 checkpoint 保存
                self.current_epoch = epoch + 1

                # 每个 epoch 都保存 last (传入 fitness 用于更新 best_fitness)
                self._save_checkpoint('last', fitness=epoch_loss)

                # 如果是最优，保存 best
                if epoch_loss <= self._best_loss:
                    self._best_loss = epoch_loss
                    self._save_checkpoint('best', fitness=epoch_loss)

                # 按间隔保存 epoch_N
                checkpoint_interval = cfg.checkpoint_epoch_interval
                if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(f'epoch_{epoch + 1}', fitness=epoch_loss)

                # 日志
                self._log_epoch_summary(epoch + 1, {
                    'loss': epoch_loss,
                    'primary': epoch_primary,
                    'reg': epoch_reg,
                })

                # 增量保存训练结果，避免中断后丢失历史 epoch
                self._save_training_results(final=False)
            
            # 保存训练结果 (最终落盘 + 视频生成)
            if start_epoch < cfg.epochs: #  判断以上训练循环是否被执行了
                self._save_training_results(final=True)
            
            # 验证阶段 (双模型: 原模型 + EMA模型)
            val_loss = 0.0
            val_loss_ema = None
            if cfg.validation_steps > 0:
                val_loss, val_loss_ema = self._run_validation()
            
            return {
                'loss_history': self.loss_history,
                'primary_loss_history': self.primary_loss_history,
                'reg_loss_history': self.reg_loss_history,
                'final_loss': self.loss_history[-1] if self.loss_history else 0.0,
                'validation_loss': val_loss,
                'validation_loss_ema': val_loss_ema,
                'amplitude_history': self.amplitude_history,
                'exp_dir': str(self.exp_dir),
            }
    
    def _train_single_epoch(self, epoch_idx: int = 0, num_epochs: int = 1) -> tuple:
        """训练单个 epoch
        
        Args:
            epoch_idx: 当前 epoch 索引（从 0 开始）
            num_epochs: 总 epoch 数
        
        Returns:
            (epoch_total_loss, epoch_primary_loss, epoch_reg_loss) 三元组
        """
        from tqdm import tqdm
        
        cfg = self.cfg.training
        envstep_per_value = cfg.num_of_interaction_per_agent_update * cfg.control_interval
        
        # 重置环境和 agent
        self._reset_env('epoch')
        self._step_losses.clear()
        # 注意：amplitude_history 不清空，保持跨 epoch 累积
        self._memory_cleanup('epoch_start', full=True)

        # 每个 epoch 开始时重置 CUDA 峰值统计（放在 reset 之后，聚焦训练过程）
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        epoch_total_loss = 0.0
        epoch_primary_loss = 0.0
        epoch_reg_loss = 0.0
        num_updates = 0
        
        # 计算总步数 (用于进度条)
        total_steps = cfg.env_warmup_envsteps + cfg.num_interaction_per_mini_episode * cfg.control_interval
        
        # 创建进度条
        pbar_desc = f"Epoch {epoch_idx + 1}/{num_epochs}" if num_epochs > 1 else "Training"
        pbar = tqdm(total=total_steps, desc=pbar_desc,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        # ===== Warmup Phase =====
        # 使用 _env_step 统一处理预热阶段
        self._memory_cleanup('warmup_start', full=False)
        if cfg.env_warmup_envsteps > 0:
            pbar.set_postfix_str("Warmup")
        
        # 保存训练初始帧（如果启用）
        if self.training_runner is not None and self.cfg.training_visuals.save_endpoints:
            self.training_runner._save_visual(0)
            if self.training_runner.cfg.fields.save.enabled:
                self.training_runner._save_fields_h5()
        
        for step in range(cfg.env_warmup_envsteps):
            # 预热阶段：直接执行仿真步（不更新控制信号）
            is_update, apply_jets, state = self._env_step(
                steps_in_accumulation=step, 
                is_warmup=True
            )
            self._do_solver_step(apply_jets, state)
            self.agent.update_history(state)
            self._update_force_buffers()  # 与验证阶段一致：预热期也填充 buffer
            self._record_amplitude(apply_jets=apply_jets)
            pbar.update(1)
        
        # 预热后全面清理显存
        self._memory_cleanup('warmup_end', full=False)
        
        if cfg.env_warmup_envsteps > 0:
            pbar.set_postfix_str("Training")
        
        # ===== Training Phase =====
        steps_in_accumulation = 0
        step_losses_total = []
        step_losses_primary = []
        step_losses_reg = []
        update_window_active = False
        
        for interaction_idx in range(cfg.num_interaction_per_mini_episode):
            if not update_window_active:
                self._memory_cleanup('update_window_start', full=False)
                update_window_active = True
            for local_step in range(cfg.control_interval):
                self.current_step += 1
                with self._get_autograd_offload_context():
                    # 1. 准备阶段：判断控制时机 + 构建状态
                    is_update, apply_jets, state = self._env_step(
                        steps_in_accumulation=steps_in_accumulation,
                        is_warmup=False
                    )

                    # 2. 更新控制信号（如果需要）
                    if is_update:
                        amplitude, policy_output = self.agent.compute_action(state)
                        self._current_amplitude = amplitude
                        self._current_policy_output = policy_output
                        if self.solver.jet_obstacle is not None:
                            self.solver.jet_obstacle.set_controls_external(
                                amplitudes=amplitude,
                                angles=self.cfg.agent.angle_deg_fixed or 90.0,
                            )

                    # 3. 执行仿真步
                    self._do_solver_step(apply_jets, state)

                    steps_in_accumulation += 1
                    self._record_amplitude(apply_jets=apply_jets)
                    self.agent.update_history(state)
                    pbar.update(1)

                    # 训练阶段可视化（如果启用）
                    self._record_training_visual(self.current_step)

                    # 4. 更新力系数 history buffer（确保 DRL 时间平均和涡脱 FFT 正常工作）
                    self._update_force_buffers()

                    # 4.5 更新快照缓冲区 (用于模态 loss: DMD/POD/稳态残差)
                    if self._need_modal_loss:
                        self._update_snapshot_buffer()

                    # 5. 采样当前速度（使用统一方法）
                    u, v = self._sample_target_velocity()
                    current_vel = torch.stack([u, v])

                    # 6. 计算 loss (返回三元组)
                    total_loss, primary_loss, reg_loss = self._compute_loss(
                        current_vel,
                        self._current_amplitude,
                        self._current_policy_output,
                        is_update_step=is_update,
                    )
                    
                step_losses_total.append(total_loss)
                step_losses_primary.append(primary_loss)
                step_losses_reg.append(reg_loss)
                
                # ===== 记录详细 step 数据 =====
                self._step_losses_record.append(float(primary_loss.detach().cpu()))
                self._step_total_losses_record.append(float(total_loss.detach().cpu()))
                self._step_reg_losses_record.append(float(reg_loss.detach().cpu()))
                u_val = float(current_vel[0].detach().cpu())
                v_val = float(current_vel[1].detach().cpu())
                self._step_velocities_u.append(u_val)
                self._step_velocities_v.append(v_val)
                self._step_pressures.append(self._sample_target_pressure())
                
                # 记录力系数（使用缓存方法）
                try:
                    cd, cl = self._get_force_coeffs()
                    self._step_cd_record.append(float(cd.detach().cpu()) if hasattr(cd, 'detach') else float(cd))
                    self._step_cl_record.append(float(cl.detach().cpu()) if hasattr(cl, 'detach') else float(cl))
                except Exception:
                    self._step_cd_record.append(float('nan'))
                    self._step_cl_record.append(float('nan'))
                
                # 更新幅值历史到 agent
                if is_update:
                    actual_amp = self._current_amplitude.item() if apply_jets else 0.0
                    self.agent.update_amplitude_history(actual_amp)
                    self._prev_amplitude = self._current_amplitude.detach()
            
            # ===== 累积足够步数后更新 =====
            if len(step_losses_total) >= envstep_per_value:
                # 加权求和
                weights = self._compute_loss_weights(len(step_losses_total))
                weighted_total = sum(w * l for w, l in zip(weights, step_losses_total))
                weighted_primary = sum(w * l for w, l in zip(weights, step_losses_primary))
                weighted_reg = sum(w * l for w, l in zip(weights, step_losses_reg))
                
                # 多环境平均
                if self.n_envs > 1:
                    weighted_total = weighted_total / self.n_envs
                
                # 同步旧网络 (在 backward 前)
                if self.policy_net_old is not None:
                    self._sync_old_policy()
                
                # 更新学习率 (使用 BaseTrainer 的方法)
                self._update_lr()
                
                # 反向传播 (YOLOv8 风格: 支持 AMP)
                # 注意: CFD 模拟中 AMP 可能导致压力求解数值问题，默认禁用
                self.optimizer.zero_grad()
                if self._grad_scaler is not None:
                    # AMP 模式: 使用 GradScaler
                    self._grad_scaler.scale(weighted_total).backward()
                    self._grad_scaler.unscale_(self.optimizer)
                    self._apply_grad_clip()
                    self._grad_scaler.step(self.optimizer)
                    self._grad_scaler.update()
                else:
                    # 标准模式
                    weighted_total.backward()
                    self._apply_grad_clip()
                    self.optimizer.step()
                
                # 更新 EMA 模型 (YOLOv8 风格: 每次 optimizer.step() 后更新)
                self._update_ema(self.agent.network)
                
                # 记录神经网络更新时刻 (当前 env step 索引)
                current_global_step = len(self._step_losses_record) - 1
                self._update_step_indices.append(current_global_step)
                
                # 记录加权累积 loss
                self._acc_total_loss_record.append(weighted_total.item())
                
                # 记录
                epoch_total_loss += weighted_total.item()
                epoch_primary_loss += weighted_primary.item()
                epoch_reg_loss += weighted_reg.item()
                num_updates += 1
                self.update_count += 1
                
                self._log_metrics({
                    'loss/step_total': weighted_total.item(),
                    'loss/step_primary': weighted_primary.item(),
                    'loss/step_reg': weighted_reg.item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                
                # 更新进度条显示
                pbar.set_postfix_str(f"acc_total_loss={weighted_total.item():.4f}")
                
                # 分离计算图，清空累积
                self._memory_cleanup('update_window_end', full=False)
                step_losses_total.clear()
                step_losses_primary.clear()
                step_losses_reg.clear()
                update_window_active = False

        # 若 epoch 结束时存在未满更新窗口，也执行窗口结束清理
        if update_window_active:
            self._memory_cleanup('update_window_end_partial', full=False)
        
        # 关闭进度条
        pbar.close()

        # 打印本 epoch 的 CUDA 显存峰值
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
            curr_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            curr_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            mem_msg = (
                f"[JetTrainer] Epoch {epoch_idx + 1}/{num_epochs} CUDA 显存 | "
                f"peak_allocated={peak_alloc_mb:.2f} MB, peak_reserved={peak_reserved_mb:.2f} MB, "
                f"current_allocated={curr_alloc_mb:.2f} MB, current_reserved={curr_reserved_mb:.2f} MB"
            )

            # 打印与上一个 epoch 结束时相比的净增量
            if self._last_epoch_end_alloc_mb is not None and self._last_epoch_end_reserved_mb is not None:
                delta_alloc_mb = curr_alloc_mb - self._last_epoch_end_alloc_mb
                delta_reserved_mb = curr_reserved_mb - self._last_epoch_end_reserved_mb
                mem_msg += (
                    f"; epoch 间 CUDA 净增量: delta_allocated={delta_alloc_mb:+.2f} MB, "
                    f"delta_reserved={delta_reserved_mb:+.2f} MB"
                )

            LOGGER.info(mem_msg)

            # 更新上一个 epoch 的结束显存基线
            self._last_epoch_end_alloc_mb = curr_alloc_mb
            self._last_epoch_end_reserved_mb = curr_reserved_mb
        
        # 保存训练最终帧（如果启用）
        if self.training_runner is not None and self.cfg.training_visuals.save_endpoints:
            self.training_runner._save_visual(self.current_step, suffix='_final')
            if self.training_runner.cfg.fields.save.enabled:
                self.training_runner._save_fields_h5()

        self._memory_cleanup('epoch_end', full=True)
        
        return (
            epoch_total_loss / max(1, num_updates),
            epoch_primary_loss / max(1, num_updates),
            epoch_reg_loss / max(1, num_updates),
        )
    
    def _sync_old_policy(self):
        """同步旧网络权重 (用于 KL 散度/重要性采样)"""
        if self.policy_net_old is not None:
            self.policy_net_old.load_state_dict(self.agent.network.state_dict())
    
    def _compute_loss_weights(self, n_steps: int = None) -> List[float]:
        """计算时间维度 loss 权重
        
        实现：
        - 'equal': 等权重（归一化，总和为1）
        - 'exponential': 基于 interaction 的指数衰减（不归一化）
        
        Args:
            n_steps: 步数（None 则使用 envstep_per_value）
        
        Returns:
            权重列表
        """
        n = self._envstep_per_value if n_steps is None else n_steps
        loss_cfg = self.cfg.loss
        
        if loss_cfg.loss_weights == 'equal':
            # 等权重，归一化
            return [1.0 / n] * n
        elif loss_cfg.loss_weights == 'exponential':
            # 基于 interaction 的指数衰减权重
            return self._compute_interaction_based_loss_weights(n)
        else:
            return [1.0 / n] * n
    
    def _compute_interaction_based_loss_weights(self, total_steps: int) -> List[float]:
        """计算基于 interaction 的指数衰减 loss 权重
        
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
            total_steps: 总环境步数
            
        Returns:
            权重列表（不归一化）
        """
        cfg = self.cfg.training
        K = cfg.control_interval       # 每个 interaction 的步数
        D = cfg.action_duration        # 射流施加步数
        gamma = self.cfg.loss.loss_gamma  # 衰减率
        
        # 计算 interaction 数量
        N = max(1, (total_steps + K - 1) // K)  # ceil(total_steps / K)
        
        # 构建权重矩阵 W[n, t]
        W = np.zeros((N, total_steps))
        
        for n in range(N):
            start = n * K  # 第 n 个 interaction 开始的环境步
            if start >= total_steps:
                break
            
            for t in range(start, total_steps):
                local_t = t - start  # 从该 interaction 开始算起的步数
                if local_t < D:
                    # 射流施加阶段，权重 = 1
                    W[n, t] = 1.0
                else:
                    # 射流停止后，持续衰减
                    decay_steps = local_t - D + 1
                    W[n, t] = gamma ** decay_steps
        
        # 在 interaction 维度求平均（不归一化）
        weights = np.mean(W, axis=0).tolist()
        
        return weights
    
    def _detach_graph(self, clear_jet_velocity_field: bool = True):
        """分离计算图 (释放显存)
        
        需要分离的状态：
        1. solver.velocity 和 solver.pressure
        2. jet_obstacle 的 _current_controls 和 _jet_velocity_field
        3. 快照缓冲区 (_snapshot_buffer, 用于模态 loss)
        4. DRL 力系数历史缓冲区 (_drl_cd/cl_history_buffer)
        
        PhiFlow 使用 math.stop_gradient 来阻断梯度流
        """
        from phi.torch.flow import math as phi_math
        
        # 1. 分离 solver 的流场状态
        if hasattr(self.solver, 'velocity') and self.solver.velocity is not None:
            self.solver.velocity = phi_math.stop_gradient(self.solver.velocity)
        if hasattr(self.solver, 'pressure') and self.solver.pressure is not None:
            self.solver.pressure = phi_math.stop_gradient(self.solver.pressure)
        
        # 2. 分离 jet_obstacle 的状态（防止旧计算图引用）
        if hasattr(self.solver, 'jet_obstacle') and self.solver.jet_obstacle is not None:
            jet_obs = self.solver.jet_obstacle
            # 分离 current_controls
            if hasattr(jet_obs, '_current_controls') and jet_obs._current_controls is not None:
                detached_controls = []
                for amp, angle in jet_obs._current_controls:
                    detached_controls.append((amp.detach(), angle.detach()))
                jet_obs._current_controls = detached_controls
            # 清空 jet_velocity_field（可选），下次 update_controls 时会重建
            if clear_jet_velocity_field and hasattr(jet_obs, '_jet_velocity_field'):
                jet_obs._jet_velocity_field = None
        
        # 3. 分离快照缓冲区 (用于模态 loss)
        #    快照保留梯度以支持可微分 DMD/POD，backward 后需要分离
        if hasattr(self, '_snapshot_buffer') and self._snapshot_buffer:
            self._snapshot_buffer = deque(
                (t.detach() for t in self._snapshot_buffer),
                maxlen=self._snapshot_buffer.maxlen
            )
        
        # 4. 分离 DRL 力系数历史缓冲区 (用于时间平均)
        #    这些缓冲区保存带梯度的张量，backward 后需要分离
        if hasattr(self, '_drl_cd_history_buffer') and self._drl_cd_history_buffer:
            detached_cd = deque(
                (t.detach() for t in self._drl_cd_history_buffer),
                maxlen=self._drl_cd_history_buffer.maxlen
            )
            self._drl_cd_history_buffer = detached_cd
        if hasattr(self, '_drl_cl_history_buffer') and self._drl_cl_history_buffer:
            detached_cl = deque(
                (t.detach() for t in self._drl_cl_history_buffer),
                maxlen=self._drl_cl_history_buffer.maxlen
            )
            self._drl_cl_history_buffer = detached_cl

    def _memory_cleanup(self, stage: str = '', full: bool = False):
        """统一显存/计算图清理入口

        Args:
            stage: 清理触发阶段标识（用于日志）
            full:
                - True: 全面清理（epoch 开始/结束）
                  断图 + gc + CUDA cache/sync，并允许清空 jet_velocity_field。
                - False: 轻量清理（epoch 内部边界）
                  仅断图，不做 gc/CUDA cache 级清理，保留运行时缓存。

        说明：
        - 不会清空输入特征缓存（如 _state_history_buffer、agent history）；
        - 不会清空用于保存/可视化的记录列表（loss/amplitude/Cd/Cl 等）。
        """
        self._detach_graph(clear_jet_velocity_field=full)

        if full:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

                torch.cuda.empty_cache()

                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

        if stage:
            mode = 'full' if full else 'light'
            LOGGER.debug(f"[JetTrainer] 清理完成 | mode={mode}, stage={stage}")
    
    def _save_config(self):
        """保存配置到实验目录"""
        config_path = self.exp_dir / 'configs' / 'trainer_config.yaml'
        config_dict = {
            'env_config_path': self.cfg.env_config_path,
            'training': asdict(self.cfg.training),
            'agent': asdict(self.cfg.agent),
            'input_features': asdict(self.cfg.input_features),
            'inlet': asdict(self.cfg.inlet),
            'loss': asdict(self.cfg.loss),
            'output': asdict(self.cfg.output),
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)
    
    def _setup_validation_runner(self):
        """创建并缓存 validation runner 实例
        
        在构造函数中创建 SolverRunner 实例，缓存到 self.runners 列表。
        验证时重置 runner 缓存即可重复使用。
        
        配置来源: env_config_path 中的 runner 配置
        
        注意:
        - Trainer 的 verbose 会覆盖 Runner 的 verbose 配置
        - 验证输出将保存到 exp_dir/validation/ 目录
        """
        # 使用统一的配置加载方法
        base_cfg = self._load_env_config()
        
        # 获取基础 runner 配置（如果不存在则使用默认）
        runner_cfg = base_cfg.get('runner', {}).copy() if 'runner' in base_cfg else {}
        
        # 验证专用的输出目录配置
        val_output_dir = self.exp_dir / 'validation'
        
        runner_cfg['output'] = {
            'enabled': True,
            'root': str(val_output_dir),
            'subdir': '',  # 直接使用 root 目录
            'skip_log_setup': True,  # 使用 trainer 的日志配置
            'skip_config_snapshot': True,  # 配置已保存在 configs/ 目录，无需重复保存
        }
        
        # verbose 覆盖: trainer.verbose 覆盖 runner.verbose
        runner_cfg['verbose'] = self.cfg.training.verbose
        
        # 确保探针配置包含目标位置探针（使用共用函数）
        if 'probes' not in runner_cfg:
            runner_cfg['probes'] = {'enabled': True, 'interval': 1, 'locations': []}
        if runner_cfg['probes'].get('enabled', False):
            target_pos = self.cfg.loss.target_probe_position
            ensure_target_probe(runner_cfg, target_pos, 'target')
        
        # 使用公共方法创建 runner
        self.validation_runner = self._create_runner(val_output_dir, runner_cfg, "Validation")
        self._validation_output_dir = val_output_dir

    def _setup_training_runner(self):
        """创建训练阶段可视化 runner
        
        根据 training_visuals 配置创建用于训练阶段流场可视化的 runner。
        仅当 training_visuals.enabled=True 时创建。
        
        输出目录结构:
            exp_dir/{training_visuals.folder}/
                ├── raw/          # 原始流场数据 (fields.h5)
                └── frames/       # 可视化图片
        """
        vis_cfg = self.cfg.training_visuals
        if not vis_cfg.enabled:
            self.training_runner = None
            self._training_output_dir = None
            return
        
        # 训练可视化输出目录
        train_output_dir = self.exp_dir / vis_cfg.folder
        
        # 读取 env_config 中基础 runner 配置（用于继承 solver 侧 verbose）
        base_cfg = self._load_env_config()
        base_runner_cfg = base_cfg.get('runner', {}).copy() if 'runner' in base_cfg else {}
        
        # 构建 visuals 配置字典（用于 build_training_runner_config）
        visuals_dict = {
            'interval': vis_cfg.interval,
            'quantities': vis_cfg.quantities,
            'save_endpoints': vis_cfg.save_endpoints,
            'smooth': vis_cfg.smooth,
            'video_fps': vis_cfg.video_fps,
            'obstacle_style': vis_cfg.obstacle_style,
        }
        
        # 使用共用函数构建 runner 配置
        runner_cfg = build_training_runner_config(
            output_dir=train_output_dir,
            visuals_config=visuals_dict,
            base_runner_config=base_runner_cfg,
            # 全局日志级别优先级：trainer.training.verbose 覆盖 solver/env runner.verbose
            verbose=self.cfg.training.verbose,
        )
        
        # 使用公共方法创建 runner
        self.training_runner = self._create_runner(train_output_dir, runner_cfg, "Training")
        self._training_output_dir = train_output_dir
        
        LOGGER.info(f"[JetTrainer] Training visuals enabled | interval={vis_cfg.interval}")

    def _record_training_visual(self, global_step: int):
        """记录训练阶段流场可视化与场数据
        
        根据 training_visuals 配置，按 interval 间隔保存流场图片和 H5 场数据。
        
        Args:
            global_step: 当前全局环境步数
        """
        if self.training_runner is None:
            return
        
        runner = self.training_runner
        vis_cfg = self.cfg.training_visuals
        if global_step % vis_cfg.interval == 0:
            runner._save_visual(global_step)
        if runner.cfg.fields.save.enabled and global_step % runner.cfg.fields.save.interval == 0:
            runner._save_fields_h5()

    # ============ 验证方法 ============
    
    def _run_validation(self) -> Tuple[float, Optional[float]]:
        """运行验证阶段（双模型验证: 原模型 + EMA模型）
        
        验证流程：
        1. 使用原始模型进行验证
        2. 若 EMA 启用，额外使用 EMA 模型进行验证（含独立可视化）
        3. 返回两个 loss 值
        
        Returns:
            (原模型验证 loss, EMA 模型验证 loss 或 None)
        """
        # 1. 原模型验证
        model_loss = self._run_single_validation(use_ema=False)
        
        # 2. EMA 模型额外验证
        ema_loss = None
        if self.ema is not None:
            ema_loss = self._run_single_validation(use_ema=True)
        
        return model_loss, ema_loss
    
    def _run_single_validation(self, use_ema: bool = False) -> float:
        """运行单次验证（使用指定模型）
        
        Args:
            use_ema: 是否使用 EMA 模型
        
        Returns:
            平均验证 loss
        """
        import numpy as np
        from tqdm import tqdm
        
        cfg = self.cfg.training
        total_val_steps = cfg.env_warmup_envsteps + cfg.validation_steps
        
        # 确定使用的模型
        if use_ema and self.ema is not None:
            LOGGER.info(f"[JetTrainer] 开始验证 (warmup {cfg.env_warmup_envsteps} + 正式 {cfg.validation_steps} = {total_val_steps} 步) | 使用 EMA 模型")
            original_network = self.agent.network
            self.agent.network = self.ema.ema
            val_suffix = "_ema"
        else:
            LOGGER.info(f"[JetTrainer] 开始验证 (warmup {cfg.env_warmup_envsteps} + 正式 {cfg.validation_steps} = {total_val_steps} 步)...")
            original_network = None
            val_suffix = ""
        
        # 重置环境（使用 env_config 中的 resume_step 初始化流场）
        self._reset_env(reason='validation')
        
        # 获取 runner 和输出目录
        # 注意: reset_solver=False 保留 _reset_env 中 reset_from_h5 的效果
        runner = self.validation_runner
        runner.reset_caches(reset_solver=False)
        val_output_dir = Path(str(self._validation_output_dir) + val_suffix)
        val_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证数据记录（与训练阶段对齐）
        val_losses = []       # Primary Loss
        val_velocities = []   # (u, v) 探针速度
        val_pressures = []    # 探针压力
        val_amplitudes = []   # 控制幅值
        val_cd = []           # 阻力系数
        val_cl = []           # 升力系数
        
        # 设置验证模式（确定性策略）
        self.agent.eval()
        prev_deterministic = False
        if self.solver.jet_obstacle:
            prev_deterministic = self.solver.jet_obstacle.deterministic
            self.solver.jet_obstacle.deterministic = True
        
        # 手动保存初始帧 (step 0)
        if runner.cfg.visuals.enabled and runner.cfg.visuals.save_endpoints:
            runner._save_visual(0)
        
        # ===== Warmup Phase (与训练阶段完全一致) =====
        # 预热阶段：填充 history buffer，不记录到 validation_curves
        # 注意：warmup 数据仅记录到 runner (raw/force_coeffs.csv)，不记录到 validation_curves.csv
        #       这与 training 阶段保持一致（training_step_data.csv 不含 warmup 数据）
        self._memory_cleanup('validation_warmup_start', full=False)
        for step in range(cfg.env_warmup_envsteps):
            is_update, apply_jets, state = self._env_step(
                steps_in_accumulation=step, 
                is_warmup=True
            )
            self._do_solver_step(apply_jets, state)
            self.agent.update_history(state)
            
            # 使用 runner 记录数据（预热阶段记录到 raw/，保持时序完整）
            # ⚠️ runner 记录必须在 _update_force_buffers 之前执行：
            #   runner._record_force_coeffs() 调用 solver.compute_force_coeffs() 并返回 Cd/Cl，
            #   填充 _cached_force_coeffs 后，_update_force_buffers → _get_force_coeffs 直接复用缓存，
            #   避免 solver.compute_force_coeffs() 重复计算
            step_num = step + 1
            if runner.cfg.force_coeffs.enabled:
                _Cd, _Cl = runner._record_force_coeffs()
                if _Cd is not None and _Cl is not None:
                    self._cached_force_coeffs = (_Cd, _Cl)
            if runner.cfg.probes.enabled:
                runner._record_probes()
            
            self._update_force_buffers()  # 统一方法：使用已填充的缓存
        
        # 预热后全面清理显存（与训练一致）
        self._memory_cleanup('validation_warmup_end', full=False)
        
        # ===== Validation Phase (正式验证) =====
        # 注意: validation_steps 为正式验证步数，不含 warmup
        # 总步数 = env_warmup_envsteps + validation_steps
        steps_in_accumulation = 0  # 重置计数器（与训练一致）
        num_val_steps = cfg.validation_steps  # 直接使用，不减去 warmup
        
        with torch.no_grad():
            for local_step in tqdm(range(num_val_steps), desc=f'Validation{val_suffix}'):
                env_step = cfg.env_warmup_envsteps + local_step
                
                # 1. 准备阶段
                is_update, apply_jets, state = self._env_step(
                    steps_in_accumulation, is_warmup=False
                )
                
                # 2. 更新控制信号
                if is_update:
                    amplitude, _ = self.agent.compute_action(state)
                    self._current_amplitude = amplitude
                    if self.solver.jet_obstacle:
                        self.solver.jet_obstacle.set_controls_external(
                            amplitudes=amplitude,
                            angles=self.cfg.agent.angle_deg_fixed or 90.0,
                        )
                
                # 3. 执行仿真步
                self._do_solver_step(apply_jets, state)
                steps_in_accumulation += 1
                self.agent.update_history(state)
                
                # 4. Runner 记录（source of truth，每步记录）
                # ⚠️ 必须在 _update_force_buffers / 探针提取之前执行，消除重复计算：
                #   - runner._record_force_coeffs() 返回 Cd/Cl → 填充 _cached_force_coeffs
                #   - runner._record_probes() 返回 {name: (u,v,p)} → 提取 'target' 探针数据
                #   后续 _update_force_buffers / _compute_primary_loss 直接复用，不再重复调用
                #   solver.compute_force_coeffs() 或 field.sample(target_point)
                step_num = env_step + 1
                if runner.cfg.force_coeffs.enabled:
                    _Cd, _Cl = runner._record_force_coeffs()
                    if _Cd is not None and _Cl is not None:
                        self._cached_force_coeffs = (_Cd, _Cl)
                probes_result = {}
                if runner.cfg.probes.enabled:
                    probes_result = runner._record_probes()
                
                # 5. 更新力系数 history buffer（使用已填充的缓存，不再重复计算）
                self._update_force_buffers()
                
                # 6. 记录探针速度（优先从 runner 结果提取，避免重复插值）
                if 'target' in probes_result:
                    u, v, p_val = probes_result['target']
                    val_velocities.append((
                        float(u.item() if hasattr(u, 'item') else u),
                        float(v.item() if hasattr(v, 'item') else v),
                    ))
                    val_pressures.append(float(p_val.item() if hasattr(p_val, 'item') else p_val))
                else:
                    # Fallback: probes 未启用或不含 target 探针时直接采样
                    u, v = self._sample_target_velocity()
                    val_velocities.append((float(u.cpu()), float(v.cpu())))
                    val_pressures.append(self._sample_target_pressure())
                
                # 7. 计算 loss (与训练完全相同的 primary loss)
                loss = self._compute_primary_loss(current_velocity=(u, v))
                val_losses.append(float(loss.detach().cpu()))
                
                # 8. 记录幅值 - 使用统一的公共方法
                amp = self._get_current_amplitude(apply_jets)
                val_amplitudes.append(amp)
                
                # 9. 记录力系数（使用已缓存的值，不再重复计算）
                cd, cl = self._get_force_coeffs()
                val_cd.append(float(cd.detach().cpu()) if hasattr(cd, 'detach') else float(cd))
                val_cl.append(float(cl.detach().cpu()) if hasattr(cl, 'detach') else float(cl))
                
                # 10. 可视化与场数据保存（按 interval）
                if runner.cfg.visuals.enabled and step_num % runner.cfg.visuals.interval == 0:
                    runner._save_visual(step_num)
                if runner.cfg.fields.save.enabled and step_num % runner.cfg.fields.save.interval == 0:
                    runner._save_fields_h5()
        
        # 手动保存最终帧 (step N, _final 后缀)
        # 总步数 = warmup + validation_steps
        total_val_steps = cfg.env_warmup_envsteps + cfg.validation_steps
        if runner.cfg.visuals.enabled and runner.cfg.visuals.save_endpoints:
            runner._save_visual(total_val_steps, suffix='_final')
        
        # 恢复训练模式和原始网络
        self.agent.train()
        if original_network is not None:
            self.agent.network = original_network  # 恢复原始网络
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
            runner._h5_file = None
        
        # 计算平均 loss
        valid_losses = [l for l in val_losses if not np.isnan(l)]
        avg_loss = float(np.mean(valid_losses)) if valid_losses else 0.0
        ema_suffix = " (EMA)" if use_ema else ""
        LOGGER.info(f"[JetTrainer] 验证完成{ema_suffix} | 平均 Loss: {avg_loss:.6f}")
        
        # 保存验证曲线（与训练阶段对齐）
        self._save_validation_curves(
            val_losses, val_velocities, val_pressures,
            val_amplitudes, val_cd, val_cl, val_output_dir
        )
        
        return avg_loss
    
    def _save_validation_curves(
        self, 
        val_losses: List[float], 
        val_velocities: List, 
        val_pressures: List[float],
        val_amplitudes: List[float],
        val_cd: List[float],
        val_cl: List[float],
        output_dir: Path
    ):
        """保存验证曲线数据
        
        统一格式:
        - NPZ 和 CSV 使用相同键名
        - 必选: env_step, amplitude, cd, cl
        - 可选: loss, velocity_u, velocity_v, pressure
        - 元数据: target_velocity (仅 NPZ)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
        output_dir.mkdir(parents=True, exist_ok=True)
        n_steps = len(val_losses) if val_losses else len(val_amplitudes)
        
        # 提取速度分量
        velocities_u = [v[0] for v in val_velocities] if val_velocities else []
        velocities_v = [v[1] for v in val_velocities] if val_velocities else []
        target_u, target_v = self.cfg.loss.target_velocity
        
        # ===== 构建统一数据字典 (NPZ 和 CSV 共用) =====
        data_dict = {
            'env_step': np.arange(n_steps),
            'amplitude': np.array(val_amplitudes),
            'cd': np.array(val_cd),
            'cl': np.array(val_cl),
        }
        # 可选字段
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
        
        # ===== 可视化 (3行2列) =====
        if not val_losses and not val_amplitudes:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # (0,0) Primary Loss - 使用共用辅助方法
        self._plot_loss_panel(
            axes[0, 0], val_losses, 
            title='Validation Primary Loss', label='Primary Loss'
        )
        
        # (0,1) 验证阶段无正则化，显示提示信息
        ax = axes[0, 1]
        ax.text(0.5, 0.5, 'Validation: No Regularization', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Regularization Loss (N/A)')
        ax.grid(True, alpha=0.3)
        
        # (1,0) Velocity - 使用共用辅助方法
        self._plot_velocity_panel(axes[1, 0], velocities_u, velocities_v, target_u, target_v,
                                  pressures=val_pressures or [])
        
        # (1,1) Amplitude - 使用共用辅助方法
        self._plot_amplitude_panel(axes[1, 1], val_amplitudes)
        
        # (2,0) Cd - 使用共用辅助方法
        self._plot_cd_panel(axes[2, 0], val_cd)
        
        # (2,1) Cl - 使用共用辅助方法
        self._plot_cl_panel(axes[2, 1], val_cl)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_curves.png', dpi=150)
        plt.close(fig)
        
        LOGGER.info(f"[JetTrainer] 验证曲线已保存到: {output_dir}")
