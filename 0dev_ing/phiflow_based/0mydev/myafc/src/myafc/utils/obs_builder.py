"""
ObservationBuilder - 统一的观测空间/输入特征构建器

为 AFCAgent 和 PhiFlowEnv (DRL) 提供一致的输入特征构建逻辑，
确保两者使用完全相同的观测维度和特征顺序。

核心功能:
1. 计算 obs_dim（观测空间维度）
2. 从 state 字典构建输入向量
3. 管理历史状态缓冲

特征构建顺序 (与 AFCAgent._build_input 一致):
1. target_velocity [u_target, v_target]  (如果 include_target_velocity=True)
2. prev_amplitudes [amp1, amp2, ...]     (如果 include_prev_amplitude_num > 0)
3. 历史+当前 state 的探针/力系数特征   (支持 envstep_state 堆叠)

使用方法:
    from myafc.utils import ObservationBuilder
    
    builder = ObservationBuilder(
        probe_locations=[...],
        probe_fields=['u', 'v', 'p'],
        main_input_type='probes',
        envstep_state=2,
        include_prev_amplitude_num=1,
        include_target_velocity=False,
        include_force_coeffs=False,
    )
    
    obs_dim = builder.obs_dim
    obs = builder.build(state)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field as dataclass_field
from collections import deque

import numpy as np

from myafc.utils import get_logger

LOGGER = get_logger()


@dataclass
class ObservationConfig:
    """观测空间配置
    
    与 JetTrainer InputFeatureConfig 和 DRL EnvConfig 对齐。
    
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
    probe_fields: List[str] = dataclass_field(default_factory=lambda: ['u', 'v'])
    envstep_state: int = 1
    include_target_velocity: bool = False
    include_prev_amplitude_num: int = 0
    include_force_coeffs: bool = False
    target_velocity: List[float] = dataclass_field(default_factory=lambda: [0.0, 0.0])


class ObservationBuilder:
    """统一的观测空间构建器
    
    提供与 AFCAgent 完全一致的输入特征构建逻辑。
    
    Example:
        # 初始化
        builder = ObservationBuilder(
            n_env_probes=133,  # env_config 中的探针数量
            config=ObservationConfig(
                main_input_type='probes',
                probe_fields=['u', 'v', 'p'],
                envstep_state=2,
                include_prev_amplitude_num=1,
            )
        )
        
        # 获取维度
        print(f"obs_dim: {builder.obs_dim}")  # 805
        
        # 构建观测
        obs = builder.build(state)
        
        # 更新历史
        builder.update_history(state)
        builder.update_amplitude(0.5)
    """
    
    def __init__(
        self,
        n_env_probes: int = 0,
        config: ObservationConfig = None,
        domain_size: Tuple[float, float] = (1.0, 1.0),
    ):
        """初始化 ObservationBuilder
        
        Args:
            n_env_probes: env_config 中定义的探针数量
            config: 观测配置
            domain_size: 域尺寸 (Lx, Ly)，用于坐标转换
        """
        self._config = config or ObservationConfig()
        self._n_env_probes = n_env_probes
        self._domain_size = domain_size
        
        # 历史缓冲
        self._state_history: deque = deque(maxlen=max(1, self._config.envstep_state - 1))
        self._amplitude_history: deque = deque(maxlen=max(1, self._config.include_prev_amplitude_num))
        
        # 预计算维度
        self._obs_dim = self._compute_obs_dim()
        
        LOGGER.debug(f"[ObservationBuilder] 初始化 | obs_dim={self._obs_dim}")
    
    @property
    def obs_dim(self) -> int:
        """观测空间维度"""
        return self._obs_dim
    
    @property
    def config(self) -> ObservationConfig:
        """获取配置"""
        return self._config
    
    def _compute_obs_dim(self) -> int:
        """计算观测空间维度
        
        特征维度计算公式 (与 AFCAgent._build_input 对齐):
        1. n_fields = len(probe_fields)  # 每个探针的特征数
        2. n_probes = 
           - main_input_type='probes': n_env_probes + 1 (target_probe)
           - main_input_type='target_probe': 1
           - 其他: 0
        3. probe_dim_per_state = n_probes × n_fields + (2 if include_force_coeffs else 0)
        4. total_probe_dim = probe_dim_per_state × envstep_state
        5. obs_dim = total_probe_dim 
                   + (2 if include_target_velocity else 0)
                   + include_prev_amplitude_num
        """
        cfg = self._config
        n_fields = len(cfg.probe_fields)
        
        # 探针数量
        if cfg.main_input_type == 'probes':
            n_probes = self._n_env_probes + 1  # +1 for target_probe
        elif cfg.main_input_type == 'target_probe':
            n_probes = 1
        else:
            n_probes = 0
        
        # 每个 state 的探针/力系数特征维度
        probe_dim_per_state = n_probes * n_fields
        if cfg.include_force_coeffs:
            probe_dim_per_state += 2  # Cd, Cl
        
        # 历史堆叠
        total_probe_dim = probe_dim_per_state * cfg.envstep_state
        
        # 全局特征
        obs_dim = total_probe_dim
        if cfg.include_target_velocity:
            obs_dim += 2  # target [u, v]
        if cfg.include_prev_amplitude_num > 0:
            obs_dim += cfg.include_prev_amplitude_num
        
        return obs_dim
    
    def build(self, state: Dict) -> np.ndarray:
        """从 state 字典构建观测向量
        
        特征构建顺序 (与 AFCAgent._build_input 一致):
        1. target_velocity [u_target, v_target]
        2. prev_amplitudes [amp1, amp2, ...]
        3. 历史+当前 state 的探针/力系数特征
        
        Args:
            state: 状态字典，需包含:
                - probes: {probe_name: {'u': ..., 'v': ..., 'p': ...}, ...}
                - force_coeffs: {'Cd': ..., 'Cl': ...} (如果 include_force_coeffs)
        
        Returns:
            观测向量 np.ndarray, shape=(obs_dim,)
        """
        cfg = self._config
        inputs = []
        
        # 1. target_velocity (全局信息，不随历史堆叠)
        if cfg.include_target_velocity:
            inputs.extend(cfg.target_velocity)
        
        # 2. 历史幅值 (全局信息，不随历史堆叠)
        if cfg.include_prev_amplitude_num > 0:
            prev_amps = list(self._amplitude_history)
            # 不足时在前面用 0 填充
            while len(prev_amps) < cfg.include_prev_amplitude_num:
                prev_amps = [0.0] + prev_amps
            inputs.extend(prev_amps[-cfg.include_prev_amplitude_num:])
        
        # 3. 探针+力系数特征 (支持历史堆叠)
        if cfg.envstep_state <= 1:
            # 标准模式：仅当前 state
            inputs.extend(self._extract_probe_features(state))
        else:
            # 历史堆叠模式
            history_states = list(self._state_history)
            current_features = self._extract_probe_features(state)
            feature_dim = len(current_features)
            num_history_needed = cfg.envstep_state - 1
            
            # 历史不足时用零值填充 (在前面填充)
            if len(history_states) < num_history_needed:
                num_padding = num_history_needed - len(history_states)
                for _ in range(num_padding):
                    inputs.extend([0.0] * feature_dim)
            
            # 添加历史 state 特征 (最旧在前)
            for hist_state in history_states[-num_history_needed:]:
                inputs.extend(self._extract_probe_features(hist_state))
            
            # 最后添加当前 state 特征
            inputs.extend(current_features)
        
        return np.array(inputs, dtype=np.float32)
    
    def _extract_probe_features(self, state: Dict) -> List[float]:
        """从单个 state 提取探针和力系数特征
        
        Args:
            state: 状态字典
        
        Returns:
            探针+力系数特征列表
        """
        features = []
        cfg = self._config
        
        # 探针数据
        if cfg.main_input_type in ('probes', 'target_probe'):
            probes = state.get('probes', {})
            
            if cfg.main_input_type == 'target_probe':
                # 仅使用 target 探针
                if 'target' in probes:
                    features.extend(self._extract_single_probe(probes['target']))
            else:
                # 使用所有探针 (按名称排序确保顺序一致)
                for name in sorted(probes.keys()):
                    features.extend(self._extract_single_probe(probes[name]))
        
        # 力系数
        if cfg.include_force_coeffs:
            fc = state.get('force_coeffs', {})
            features.extend([fc.get('Cd', 0.0), fc.get('Cl', 0.0)])
        
        return features
    
    def _extract_single_probe(self, probe_data) -> List[float]:
        """从单个探针提取特征
        
        Args:
            probe_data: 探针数据，可以是:
                - dict: {'u': ..., 'v': ..., 'p': ...}
                - list/tuple: [u, v, p]
        
        Returns:
            按 probe_fields 顺序的特征列表
        """
        cfg = self._config
        
        if isinstance(probe_data, dict):
            return [probe_data.get(field, 0.0) for field in cfg.probe_fields]
        elif isinstance(probe_data, (list, tuple)):
            # 假设顺序为 [u, v, p]
            field_to_idx = {'u': 0, 'v': 1, 'p': 2}
            return [
                probe_data[field_to_idx.get(field, 0)] 
                if field_to_idx.get(field, 0) < len(probe_data) 
                else 0.0
                for field in cfg.probe_fields
            ]
        else:
            return [0.0] * len(cfg.probe_fields)
    
    def update_history(self, state: Dict):
        """更新历史状态缓冲
        
        Args:
            state: 当前环境状态
        """
        if self._config.envstep_state > 1:
            self._state_history.append(state.copy())
    
    def update_amplitude(self, amplitude: float):
        """更新幅值历史
        
        Args:
            amplitude: 当前控制幅值
        """
        if self._config.include_prev_amplitude_num > 0:
            self._amplitude_history.append(amplitude)
    
    def reset(self):
        """重置历史缓冲 (新 episode)"""
        self._state_history.clear()
        self._amplitude_history.clear()
    
    def set_target_velocity(self, velocity: List[float]):
        """设置目标速度"""
        self._config.target_velocity = list(velocity) if velocity else [0.0, 0.0]


def compute_obs_dim_from_config(
    n_env_probes: int,
    main_input_type: Optional[str],
    probe_fields: List[str],
    envstep_state: int = 1,
    include_target_velocity: bool = False,
    include_prev_amplitude_num: int = 0,
    include_force_coeffs: bool = False,
) -> int:
    """便捷函数：从配置参数计算 obs_dim
    
    Args:
        n_env_probes: env_config 中定义的探针数量
        main_input_type: 主输入类型 ('probes'/'target_probe'/None)
        probe_fields: 探针字段列表
        envstep_state: 历史状态堆叠步数
        include_target_velocity: 是否包含目标速度
        include_prev_amplitude_num: 历史幅值数量
        include_force_coeffs: 是否包含力系数
    
    Returns:
        obs_dim: 观测空间维度
    """
    config = ObservationConfig(
        main_input_type=main_input_type,
        probe_fields=probe_fields,
        envstep_state=envstep_state,
        include_target_velocity=include_target_velocity,
        include_prev_amplitude_num=include_prev_amplitude_num,
        include_force_coeffs=include_force_coeffs,
    )
    builder = ObservationBuilder(n_env_probes=n_env_probes, config=config)
    return builder.obs_dim
