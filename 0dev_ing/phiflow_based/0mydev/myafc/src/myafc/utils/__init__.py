# -*- coding: utf-8 -*-
"""
myafc.utils - 工具模块

提供：
- logging: 统一日志配置
- phiflow_io: PhiFlow 数据 I/O 工具（场数据读写、Scene 管理）
- differentiable_ops: 可微分数学操作（FFT、POD、DMD）
- runner_factory: SolverRunner 工厂与配置工具
- obs_builder: 统一的观测空间/输入特征构建器（AFCAgent 和 PhiFlowEnv 共用）
- path_utils: 路径和目录管理工具
- plot_utils: 曲线数据保存与绘图工具
- config_utils: 配置解析与处理工具
- probe_sampling: 统一的探针采样工具函数（所有 _build_state 共用）
"""
from .logging import get_logger, setup_file_logging, set_log_level, tee_console_output
from .obs_builder import (
    ObservationBuilder,
    ObservationConfig,
    compute_obs_dim_from_config,
)
from .phiflow_io import (
    save_field_npz,
    load_field_npz,
    save_fields_dict,
    load_fields_dict,
    save_fields_h5,
    load_fields_h5,
    FieldScene,
    velocity_to_centered,
    centered_to_staggered,
)
from .differentiable_ops import (
    # FFT
    differentiable_fft,
    fft_get_top_modes,
    # POD
    differentiable_pod,
    pod_reconstruct,
    # DMD
    differentiable_dmd,
    dmd_get_frequencies,
    dmd_get_growth_rates,
    # Vorticity
    compute_vorticity,
    velocity_magnitude,
)
from .runner_factory import (
    build_validation_runner_config,
    build_training_runner_config,
    ensure_target_probe,
    create_runner,
    load_env_config,
    get_target_probe_position_from_config,
)
from .path_utils import (
    create_timestamped_dir,
    save_config_snapshot,
    resolve_relative_path,
    ensure_dir,
    get_output_subdirs,
)
from .plot_utils import (
    save_validation_curves,
    plot_validation_curves_3x2,
    plot_validation_curves_3panels,
    plot_force_coeffs,
    plot_amplitude_history,
    generate_video_from_frames,
)
from .config_utils import (
    convert_scientific_notation,
    load_yaml_config,
    merge_configs,
    resolve_config_path,
    get_nested,
    set_nested,
    flatten_config,
)
from .probe_sampling import sample_probe_fields, build_probe_batch, sample_probe_fields_batch

__all__ = [
    # logging
    'get_logger',
    'setup_file_logging',
    'set_log_level',
    'tee_console_output',
    
    # obs_builder
    'ObservationBuilder',
    'ObservationConfig',
    'compute_obs_dim_from_config',
    
    # phiflow_io
    'save_field_npz',
    'load_field_npz',
    'save_fields_dict',
    'load_fields_dict',
    'save_fields_h5',
    'load_fields_h5',
    'FieldScene',
    'velocity_to_centered',
    'centered_to_staggered',
    
    # differentiable_ops
    'differentiable_fft',
    'fft_get_top_modes',
    'differentiable_pod',
    'pod_reconstruct',
    'differentiable_dmd',
    'dmd_get_frequencies',
    'dmd_get_growth_rates',
    'compute_vorticity',
    'velocity_magnitude',
    
    # runner_factory
    'build_validation_runner_config',
    'build_training_runner_config',
    'ensure_target_probe',
    'create_runner',
    'load_env_config',
    'get_target_probe_position_from_config',
    
    # path_utils
    'create_timestamped_dir',
    'save_config_snapshot',
    'resolve_relative_path',
    'ensure_dir',
    'get_output_subdirs',
    
    # plot_utils
    'save_validation_curves',
    'plot_validation_curves_3x2',
    'plot_validation_curves_3panels',
    'plot_force_coeffs',
    'plot_amplitude_history',
    'generate_video_from_frames',
    
    # config_utils
    'convert_scientific_notation',
    'load_yaml_config',
    'merge_configs',
    'resolve_config_path',
    'get_nested',
    'set_nested',
    'flatten_config',
    
    # probe_sampling
    'sample_probe_fields',
    'build_probe_batch',
    'sample_probe_fields_batch',
]
