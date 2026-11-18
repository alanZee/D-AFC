"""
AFC Losses 模块

包含:
- probe_loss: 探针速度 MSE/MAE loss
- force_loss: 力系数 loss (含 DRLinFluids 风格)
- regularization: 正则化项 (L2, 平滑, 熵, KL)
- field_loss: 流场数据 loss (RMS, FFT, 波动稀疏, 多场加权)
- modal_loss: 可微分 DMD/POD 模态 loss
- custom_loss: 自定义 loss 加载接口
"""
# 基础 loss
from .probe_loss import probe_mse_loss, probe_mae_loss
from .force_loss import (
    force_coeffs_loss,
    drag_reduction_loss,
    lift_suppression_loss,
    drl_reward_loss,
    compute_control_power,
    vortex_shedding_loss,
    cl_oscillation_loss,
    combined_force_loss,
    ForceLossConfig,
)

# 正则化
from .regularization import (
    amplitude_l2_loss,
    temporal_smoothness_loss,
    entropy_loss,
    kl_divergence_loss,
    policy_stability_loss,
)

# 流场 loss
from .field_loss import (
    RegionConfig,
    FieldLossConfig,
    extract_region,
    compute_vorticity,
    velocity_magnitude,
    field_rms_loss,
    fft_mode_loss,
    fluctuation_sparse_loss,
    multi_field_loss,
    region_velocity_loss,
    downstream_velocity_loss,
    get_default_downstream_region,
)

# 模态分解 loss
from .modal_loss import (
    differentiable_pod,
    differentiable_dmd,
    pod_mode_loss,
    dmd_mode_loss,
    steady_state_residual_loss,
)

# 自定义 loss
from .custom_loss import load_custom_loss, example_custom_loss

__all__ = [
    # 基础 loss
    'probe_mse_loss',
    'probe_mae_loss', 
    'force_coeffs_loss',
    'drag_reduction_loss',
    'lift_suppression_loss',
    'drl_reward_loss',
    'compute_control_power',
    'vortex_shedding_loss',
    'cl_oscillation_loss',
    'combined_force_loss',
    'ForceLossConfig',
    # 正则化
    'amplitude_l2_loss',
    'temporal_smoothness_loss',
    'entropy_loss',
    'kl_divergence_loss',
    'policy_stability_loss',
    # 流场 loss
    'RegionConfig',
    'FieldLossConfig',
    'extract_region',
    'compute_vorticity',
    'velocity_magnitude',
    'field_rms_loss',
    'fft_mode_loss',
    'fluctuation_sparse_loss',
    'multi_field_loss',
    'region_velocity_loss',
    'downstream_velocity_loss',
    'get_default_downstream_region',
    # 模态分解
    'differentiable_pod',
    'differentiable_dmd',
    'pod_mode_loss',
    'dmd_mode_loss',
    'steady_state_residual_loss',
    # 自定义
    'load_custom_loss',
    'example_custom_loss',
]
