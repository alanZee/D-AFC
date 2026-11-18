"""
AFCAgent - AFC 专用控制策略 Agent

继承 MLPPolicy，添加:
1. 历史状态管理 (state_history buffer)
2. 权重保存/加载功能
3. 与 Trainer 的简洁接口
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field as dataclass_field

import torch
import torch.nn as nn

from myafc.env_kernels.solvers.fluid_control.control_policy.mlp import MLPPolicy, PolicyNetOutput
from myafc.utils import get_logger

LOGGER = get_logger()


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class AFCAgentConfig:
    """AFCAgent 配置
    
    网络结构参数:
        hidden_dims: 隐藏层维度列表，如 [512, 512] / [256, 128]
        amplitude_max: 输出幅值上限（正数），网络输出映射到 [-max, +max]
        angle_deg_fixed: 固定射流角度（度），None 表示由网络学习
        shared_output: 所有喷嘴是否共用输出
            True: 网络输出 1 组值，复制到所有喷嘴
            False: 每个喷嘴独立输出
        normalize_input: 是否对输入特征进行 EMA 归一化
        action_scale: 动作缩放方式
            'tanh': tanh(x) * scale（SAC 风格，严格边界）
            'clamp': clamp(x, low, high)（PPO 风格，硬裁剪）
            'softsign': x/(1+|x|) * scale（梯度衰减慢，推荐）
        relative_action: 相对动作模式（增量控制）
            False: 绝对模式，网络直接输出目标幅值
            True: 增量模式，action = prev_action + delta
        deterministic: 确定性策略（不随机采样，直接取均值）
    
    输入特征参数:
        main_input_type: 主输入类型（探针数据来源）
            None: 开环控制，仅使用 target_velocity
            'target_probe': 仅使用 loss.target_point 定义的单个目标探针
            'probes': 使用 env_config 中所有探针 + target 探针
        envstep_state: 历史状态堆叠步数（参考 DRLinFluids）
            1: 仅当前 state（默认）
            n: 堆叠最近 n 个 state 的探针/力系数特征
        include_target_velocity: 是否包含目标速度 [u, v]
        include_prev_amplitude_num: 历史幅值输入数量
            0: 不使用历史幅值
            n: 使用最近 n 步幅值（按时间顺序，最旧在前）
        include_force_coeffs: 是否包含力系数 [Cd, Cl]
        probe_fields: 探针字段列表，['u', 'v', 'p'] 的子集
    
    """
    # 网络结构
    hidden_dims: List[int] = dataclass_field(default_factory=lambda: [512, 512])
    amplitude_max: float = 4.0
    angle_deg_fixed: Optional[float] = 90.0
    shared_output: bool = True
    normalize_input: bool = True
    action_scale: str = 'clamp'
    relative_action: bool = False
    deterministic: bool = False
    
    # 输入特征
    main_input_type: Optional[str] = None  # None = 仅 target_velocity（开环）
    envstep_state: int = 1
    include_target_velocity: bool = True
    include_prev_amplitude_num: int = 1
    include_force_coeffs: bool = False
    probe_fields: List[str] = dataclass_field(default_factory=lambda: ['u', 'v', 'p'])


class AFCAgent(MLPPolicy):
    """AFC 专用 Agent
    
    基于 MLPPolicy，添加:
    1. 历史状态缓冲管理
    2. 权重保存/加载
    3. 与 JetObstacle 的无缝对接
    
    Example:
        agent = AFCAgent(config)
        agent.reset()
        
        # 训练循环
        action, output = agent.compute_action(state)
        agent.update_history(state)
    """
    
    def __init__(
        self,
        config: AFCAgentConfig = None,
        num_nozzles: int = 1,
        device: str = 'cpu',
        **kwargs,
    ):
        """初始化 AFCAgent
        
        Args:
            config: Agent 配置
            num_nozzles: 喷嘴数量 (由 Solver 决定)
            device: 计算设备
            **kwargs: 额外参数覆盖 config
        """
        config = config or AFCAgentConfig()
        
        # 合并 kwargs 到 config
        for key, val in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, val)
        
        self._config = config
        self._envstep_state = config.envstep_state
        self._device_str = device
        
        # 确定基类的 input_type
        # main_input_type=None 表示开环（仅 target_velocity），否则使用探针/力系数
        base_input_type = config.main_input_type or 'target_velocity'
        if base_input_type in ('target_velocity', None):
            base_input_type = 'target_velocity'  # 特殊标记，由 _build_input 处理
        
        # 初始化基类 MLPPolicy
        super().__init__(
            input_type=base_input_type,
            hidden_dims=config.hidden_dims,
            num_nozzles=num_nozzles,
            shared_output=config.shared_output,
            amplitude_max=config.amplitude_max,
            angle_deg_fixed=config.angle_deg_fixed,
            action_scale=config.action_scale,
            normalize_input=config.normalize_input,
            relative_action=config.relative_action,  # 传递 relative_action 到基类
            device=device,
        )
        
        # 历史状态缓冲
        self._state_history: List[Dict] = []
        self._amplitude_history: List[float] = []
        self._target_velocity: List[float] = [0.0, 0.0]
        
        # 注意: pretrained_path 的加载由 Trainer 负责
        # 因为网络使用 lazy init，此时 self.network 为 None
        # 见 JetTrainer._load_pretrained_weights()
    
    # ========== 公开接口 ==========
    
    def compute_action(
        self,
        state: Dict,
        deterministic: bool = None,
    ) -> Tuple[torch.Tensor, Optional[PolicyNetOutput]]:
        """计算控制信号
        
        Args:
            state: 环境状态字典
            deterministic: 是否确定性 (None = 使用 config)
        
        Returns:
            action: 控制幅值 tensor (所有喷嘴的第一个幅值)
            policy_output: PolicyNetOutput (如果 network 可用)
        """
        if deterministic is None:
            deterministic = self._config.deterministic
        
        # 构建完整输入 (含历史)
        state_with_history = self._prepare_state(state)
        
        # 调用父类 __call__ 获取 [(amp, angle), ...] 列表
        result_list = self(state_with_history, deterministic=deterministic)
        
        # 提取第一个喷嘴的幅值作为主控制信号
        if result_list and len(result_list) > 0:
            amplitude, angle = result_list[0]
        else:
            amplitude = torch.tensor(0.0, device=self._device_str)
        
        # 获取 PolicyNetOutput (用于正则化 loss)
        policy_output = None
        if self.network is not None and hasattr(self.network, 'current_output'):
            policy_output = self.network.current_output
        
        return amplitude, policy_output
    
    def update_history(self, state: Dict):
        """更新历史状态缓冲
        
        Args:
            state: 当前环境状态
        """
        if self._envstep_state > 1:
            self._state_history.append(state.copy())
            max_history = self._envstep_state - 1
            if len(self._state_history) > max_history:
                self._state_history.pop(0)
    
    def update_amplitude_history(self, amplitude: float):
        """更新幅值历史
        
        Args:
            amplitude: 当前控制幅值
        """
        max_len = self._config.include_prev_amplitude_num
        if max_len > 0:
            self._amplitude_history.append(amplitude)
            if len(self._amplitude_history) > max_len:
                self._amplitude_history.pop(0)
    
    def set_target_velocity(self, velocity: List[float]):
        """设置目标速度 (由 Trainer 调用)"""
        self._target_velocity = list(velocity) if velocity else [0.0, 0.0]
    
    def reset(self):
        """重置 Agent 状态 (新 episode)"""
        self._state_history.clear()
        self._amplitude_history.clear()
    
    # ========== 网络接口 ==========
    
    def parameters(self):
        """获取网络参数（用于优化器）
        
        Returns:
            网络参数迭代器
        
        Raises:
            RuntimeError: 如果网络未初始化
        """
        if self.network is None:
            raise RuntimeError("网络尚未初始化，请先调用 compute_action 或手动初始化")
        return self.network.parameters()
    
    def train(self, mode: bool = True):
        """设置训练/推理模式"""
        super().train(mode)
        return self
    
    def eval(self):
        """设置为推理模式"""
        return self.train(False)
    
    # ========== 权重管理 ==========
    
    def save_weights(self, path: str):
        """保存模型权重
        
        Args:
            path: 保存路径 (.pt)
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'config': self._config,
        }, path)
        LOGGER.debug(f"[AFCAgent] 保存权重: {path}")
    
    def load_weights(self, path: str):
        """加载模型权重
        
        支持多种格式:
        1. AFCAgent.save_weights 格式: {'network_state_dict': ..., 'config': ...}
        2. JetTrainer checkpoint 格式: {'model': state_dict, ...}
        3. 纯 state_dict 格式
        
        Args:
            path: 权重文件路径 (.pt)
        
        Raises:
            RuntimeError: 网络未初始化时调用
        """
        if self.network is None:
            raise RuntimeError(
                "网络未初始化，无法加载权重。"
                "请先调用 compute_action() 初始化网络，或由 Trainer 加载。"
            )
        
        checkpoint = torch.load(path, map_location=self._device_str, weights_only=False)
        
        # 兼容多种格式
        if 'network_state_dict' in checkpoint:
            # AFCAgent.save_weights 格式
            state_dict = checkpoint['network_state_dict']
        elif 'model' in checkpoint:
            # JetTrainer checkpoint 格式
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and any(
            k.startswith(('mlp_extractor', 'action_net', 'log_std')) for k in checkpoint.keys()
        ):
            # 纯 state_dict 格式
            state_dict = checkpoint
        else:
            raise ValueError(
                f"无法识别的权重格式。支持: network_state_dict, model, 或纯 state_dict。"
                f"实际 keys: {list(checkpoint.keys())[:5]}..."
            )
        
        self.network.load_state_dict(state_dict)
        LOGGER.debug(f"[AFCAgent] 加载权重: {path}")
    
    # ========== 内部方法 ==========
    
    def _extract_probe_features(self, state: Dict) -> List[float]:
        """从单个 state 提取探针和力系数特征
        
        仅提取探针数据和力系数，不包含 target_velocity 和 prev_amplitudes。
        用于历史状态堆叠时避免重复添加全局信息。
        
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
                    p = probes['target']
                    if isinstance(p, dict):
                        for field in cfg.probe_fields:
                            features.append(p.get(field, 0.0))
                    elif isinstance(p, (list, tuple)):
                        features.extend(p[:len(cfg.probe_fields)])
            else:
                # 使用所有探针 (sorted 确保顺序一致，与 ObservationBuilder 对齐)
                for name in sorted(probes.keys()):
                    probe_data = probes[name]
                    if isinstance(probe_data, dict):
                        for field in cfg.probe_fields:
                            features.append(probe_data.get(field, 0.0))
                    elif isinstance(probe_data, (list, tuple)):
                        features.extend(probe_data[:len(cfg.probe_fields)])
        
        # 力系数
        if cfg.include_force_coeffs:
            fc = state.get('force_coeffs', {})
            features.extend([fc.get('Cd', 0.0), fc.get('Cl', 0.0)])
        
        return features
    
    def _build_input(self, state: Dict) -> List[float]:
        """构建神经网络输入向量
        
        重写基类方法以支持 target_velocity、历史状态堆叠等额外输入特征。
        与 SimpleTrainer 的 CustomMLPPolicy._build_input 保持一致。
        
        输入构建顺序（与 CustomMLPPolicy 一致）:
        1. target_velocity [u_target, v_target]  (如果 include_target_velocity=True)
        2. prev_amplitudes [amp1, amp2, ...]     (如果 include_prev_amplitude_num > 0)
        3. 历史+当前 state 的探针/力系数特征     (支持 envstep_state 堆叠)
           - envstep_state=1: 仅当前 state（默认，与 DRLinFluids 一致）
           - envstep_state>1: 堆叠最近 N 个 state（最旧在前，当前在后）
        
        Args:
            state: 状态字典（由 _prepare_state 准备，含 _state_history）
        
        Returns:
            输入特征列表
        """
        inputs = []
        cfg = self._config
        
        # 1. target_velocity（全局信息，不随历史堆叠）
        if cfg.include_target_velocity:
            target_vel = state.get('target_velocity', self._target_velocity)
            if target_vel is not None:
                inputs.extend(target_vel)
        
        # 2. 历史幅值（全局信息，不随历史堆叠）
        if cfg.include_prev_amplitude_num > 0:
            prev_amps = list(state.get('prev_amplitudes', []))
            # 不足时在前面用 0 填充
            while len(prev_amps) < cfg.include_prev_amplitude_num:
                prev_amps = [0.0] + prev_amps
            inputs.extend(prev_amps[-cfg.include_prev_amplitude_num:])
        
        # 3. 探针+力系数特征（支持历史堆叠）
        if cfg.envstep_state <= 1:
            # 标准模式：仅当前 state（默认，与 DRLinFluids 一致）
            inputs.extend(self._extract_probe_features(state))
        else:
            # 历史堆叠模式
            history_states = list(state.get('_state_history', []))
            current_features = self._extract_probe_features(state)
            feature_dim = len(current_features)
            num_history_needed = cfg.envstep_state - 1
            
            # 历史不足时用零值填充（在前面填充）
            if len(history_states) < num_history_needed:
                num_padding = num_history_needed - len(history_states)
                for _ in range(num_padding):
                    inputs.extend([0.0] * feature_dim)
            
            # 添加历史 state 特征（最旧在前）
            for hist_state in history_states[-num_history_needed:]:
                inputs.extend(self._extract_probe_features(hist_state))
            
            # 最后添加当前 state 特征
            inputs.extend(current_features)
        
        return inputs
    
    def _prepare_state(self, state: Dict) -> Dict:
        """准备完整状态 (含历史)"""
        prepared = state.copy()
        
        # 添加历史状态
        if self._envstep_state > 1:
            prepared['_state_history'] = self._state_history.copy()
        
        # 添加历史幅值
        if self._config.include_prev_amplitude_num > 0:
            prepared['prev_amplitudes'] = self._amplitude_history.copy()
        
        # 添加目标速度
        if self._config.include_target_velocity:
            prepared['target_velocity'] = self._target_velocity
        
        return prepared
    
    @property
    def config(self) -> AFCAgentConfig:
        """获取配置"""
        return self._config
