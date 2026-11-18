"""
afc_supervision 模块

可微分模拟主动流动控制训练框架。
使用 PyTorch 原生自动微分 (PhiFlow Torch 后端支持端到端可微分)。

主要组件:
- engines: BaseTrainer, JetTrainer, SimpleTrainer
- agents: AFCAgent
- losses: probe_loss, force_loss, regularization, field_loss, modal_loss
"""
# 训练引擎
from .engines import (
    # JetTrainer
    BaseTrainer,
    TrainingBaseConfig,
    OutputBaseConfig,
    JetTrainer,
    JetTrainerConfig,
    JetTrainingConfig,
    # SimpleTrainer
    SimpleTrainer,
    SimpleTrainerConfig,
    MLPConfig,
    InputFeatureConfig,
    LossConfig,
    InletConfig,
    TrainingConfig,
    OutputConfig,
)

# agents
from .agents import AFCAgent
from .agents.afc_agent import AFCAgentConfig

# losses
from .losses import (
    # 基础
    probe_mse_loss,
    probe_mae_loss,
    force_coeffs_loss,
    # 正则化
    amplitude_l2_loss,
    temporal_smoothness_loss,
    entropy_loss,
    kl_divergence_loss,
    policy_stability_loss,
    # 流场
    RegionConfig,
    field_rms_loss,
    fft_mode_loss,
    fluctuation_sparse_loss,
    region_velocity_loss,
    downstream_velocity_loss,
    # 模态分解
    differentiable_pod,
    differentiable_dmd,
    pod_mode_loss,
    dmd_mode_loss,
    steady_state_residual_loss,
    # 自定义
    load_custom_loss,
)

__all__ = [
    # JetTrainer
    'BaseTrainer',
    'TrainingBaseConfig',
    'OutputBaseConfig',
    'JetTrainer',
    'JetTrainerConfig',
    'JetTrainingConfig',
    # SimpleTrainer
    'SimpleTrainer',
    'SimpleTrainerConfig',
    'MLPConfig',
    'InputFeatureConfig',
    'LossConfig',
    'InletConfig',
    'TrainingConfig',
    'OutputConfig',
    # agents
    'AFCAgent',
    'AFCAgentConfig',
    # losses
    'probe_mse_loss',
    'probe_mae_loss',
    'force_coeffs_loss',
    'amplitude_l2_loss',
    'temporal_smoothness_loss',
    'entropy_loss',
    'kl_divergence_loss',
    'policy_stability_loss',
    'RegionConfig',
    'field_rms_loss',
    'fft_mode_loss',
    'fluctuation_sparse_loss',
    'region_velocity_loss',
    'downstream_velocity_loss',
    'differentiable_pod',
    'differentiable_dmd',
    'pod_mode_loss',
    'dmd_mode_loss',
    'steady_state_residual_loss',
    'load_custom_loss',
]
