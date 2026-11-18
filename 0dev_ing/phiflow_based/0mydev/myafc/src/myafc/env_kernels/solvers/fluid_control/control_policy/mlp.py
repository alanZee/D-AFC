"""
MLP 神经网络控制策略

使用 MLP 从观测数据预测控制信号，支持 PyTorch 可微分训练。

模块包含:
    - PolicyNet: 策略网络（带输入归一化、多种动作缩放、可选随机采样）
    - MLPPolicy: 高层策略封装（输入构建、输出解析、对接 JetObstacle ）

PolicyNet 输出模式:
    - shared_output=True: 输出 1 组 (amp, [angle])，所有喷嘴共用
    - shared_output=False: 输出 n 组 (amp, [angle])，分别对应 n 个喷嘴
    - angle_deg_fixed 非 None: 每组只输出 1 个值 (amp)，角度固定
    - angle_deg_fixed 为 None: 每组输出 2 个值 (amp, angle)
    # 每个 amp 或 angle 都包含 mean, log_std, action, log_prob, action_raw

动作缩放方式:
    - tanh: tanh(x) * scale (SAC 风格，严格边界)
    - clamp: clamp(x, low, high) (PPO 风格，直接裁剪到边界)
    - softsign: x/(1+|x|) * scale (梯度衰减慢，推荐)

动作边界 (action_bounds):
    - 支持列表形式，每个动作维度可单独设置范围
    - 支持 float('-inf') / float('inf') 或字符串 '-inf' / 'inf' 表示无界
    - 无界维度使用 action_scale='none' (直接输出原始值)

相对动作模式:
    当 relative_action=True 时，网络输出解释为增量：
    - 需要通过 state['prev_amplitudes'] 和 state['prev_angles'] 提供上一步动作
    - 增量范围通过 tanh 缩放到 [-(max-min), (max-min)]
    - 最终动作 = prev_action + delta （在裁剪到合法范围的前一步进行）
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from .base import BasePolicy
from myafc.utils import get_logger

LOGGER = get_logger()


# ============================================================
# PolicyNet 输出结构
# ============================================================
@dataclass
class PolicyNetOutput:
    """PolicyNet 前向传播输出
    
    Attributes:
        mean: 动作均值 (batch_size, action_dim) 或 (action_dim,)
        log_std: 动作对数标准差
        action: 最终动作（经缩放后）
        log_prob: 动作对数概率（仅随机采样时有效）
        entropy: 分布熵（仅随机采样时有效）
        action_raw: 缩放前的原始动作
    """
    mean: torch.Tensor
    log_std: torch.Tensor
    action: torch.Tensor
    log_prob: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    action_raw: Optional[torch.Tensor] = None


# ============================================================
# PolicyNet: 策略网络（整合输入归一化、动作缩放）
# ============================================================
class PolicyNet(nn.Module):
    """
    策略网络，支持确定性输出或随机采样（高斯分布）
    
    特性:
    - 内置 EMA 输入归一化
    - 多种动作缩放: tanh / clamp / softsign
    - 可选高斯随机采样（重参数化技巧，由 log_std_mode 控制）
    - 相对动作模式：输出增量（tanh 缩放）叠加到上一步动作
    - 兼容 SB3 PPO 权重格式
    
    网络结构:
        Input -> [Normalize] -> [Linear -> Activation] * n -> Linear -> Output
    
    采样流程 (log_std_mode='global' 时):
        features = mlp_extractor(normalize(obs))
        mean = action_net(features)
        if not deterministic:
            action_raw = mean + exp(log_std) * N(0, 1)
        action = scale(action_raw)  # tanh / clamp / softsign
    
    与 SB3 PPO 的对应关系:
    ┌─────────────────────┬──────────────────────────────────┐
    │ 本实现               │ SB3 PPO (ActorCriticPolicy)      │
    ├─────────────────────┼──────────────────────────────────┤
    │ self.mlp_extractor  │ policy.mlp_extractor             │
    │ self.action_net     │ policy.action_net                │
    │ self.log_std        │ policy.log_std                   │
    │ forward() -> mean   │ _get_action_dist_from_latent()   │
    └─────────────────────┴──────────────────────────────────┘
    """
    
    # 动作缩放方式
    SCALE_TANH = 'tanh'
    SCALE_CLAMP = 'clamp'
    SCALE_SOFTSIGN = 'softsign'
    SCALE_NONE = 'none'  # 无缩放，直接输出原始值
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = 'tanh',  # 隐藏层激活函数: 'tanh' | 'relu'
        action_scale: str = 'clamp',  # 输出缩放: 'tanh' | 'clamp' | 'softsign' | 'none'
        action_bounds: Any = (-1.0, 1.0),  # 单一元组或列表形式（每维独立）; 支持字符串 '-inf' / 'inf' 表示无界
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        log_std_mode: str = 'global',  # 'none' | 'global' | 'state_dependent'
        normalize_input: bool = True,
        normalize_momentum: float = 0.01,
        relative_action: bool = False,
    ):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度（动作维度）
            hidden_dims: 隐藏层维度，默认 [64, 64]
            activation: 隐藏层激活函数 'tanh' 或 'relu'
            action_scale: 动作缩放方式
                - 'tanh': tanh(x) * range/2 + center (SAC 风格，梯度饱和快)
                - 'clamp': clamp(x, low, high) 直接裁剪到边界 (PPO 风格)
                - 'softsign': x/(1+|x|) * range/2 + center (梯度饱和慢)
                - 'none': 无缩放，直接输出原始值（用于无界动作）
            action_bounds: 动作边界，支持两种形式:
                - (low, high): 单一元组，所有维度共用
                - [(low1, high1), (low2, high2), ...]: 列表形式，每维独立
                支持 float('-inf') / float('inf') 或字符串 '-inf' / 'inf' 表示无界
            log_std_init: log_std 初始值（SB3 默认 0.0）
            log_std_min: log_std 下限（避免 σ 过小）
            log_std_max: log_std 上限（避免 σ 过大）
            log_std_mode: log_std 模式
                - 'none': 确定性策略，不使用随机采样
                - 'global': 全局参数，与状态无关（SB3 PPO 默认）
                - 'state_dependent': 从隐藏层预测，与状态相关
            normalize_input: 是否归一化输入
            normalize_momentum: 归一化 EMA 动量 (0~1)
            relative_action: 是否使用相对动作模式
                - False: 绝对动作模式，网络输出直接作为动作
                - True: 相对动作模式，网络输出为增量，增量通过 tanh 缩放到 
                        [-action_range, action_range]，最终动作 = prev + delta
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [64, 64]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.action_scale = action_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std_mode = log_std_mode
        self.normalize_input = normalize_input
        self._norm_momentum = normalize_momentum
        self._norm_eps = 1e-5
        self.relative_action = relative_action
        
        # 解析 action_bounds（支持列表形式，每维独立）
        self._parse_action_bounds(action_bounds, output_dim)
        
        # 输入归一化 buffer
        if normalize_input:
            self.register_buffer('running_mean', torch.zeros(input_dim))
            self.register_buffer('running_var', torch.ones(input_dim))
            self.register_buffer('norm_count', torch.tensor(0, dtype=torch.long))
        
        # 选择激活函数
        act_fn = nn.Tanh if activation == 'tanh' else nn.ReLU
        
        # MLP 特征提取器
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_fn())
        self.mlp_extractor = nn.Sequential(*layers)
        
        # 动作输出层
        self.action_net = nn.Linear(hidden_dims[-1], output_dim)
        
        # 随机策略的 log_std 参数
        if log_std_mode == 'global':
            self.log_std = nn.Parameter(torch.ones(output_dim) * log_std_init)
        elif log_std_mode == 'state_dependent':
            self.log_std_net = nn.Linear(hidden_dims[-1], output_dim)
        # log_std_mode == 'none' 时不创建任何 log_std 参数
        
        self._init_weights()
    
    def _parse_action_bounds(self, action_bounds: Any, output_dim: int):
        """解析动作边界，支持单一元组或列表形式
        
        Args:
            action_bounds: (low, high) 或 [(low1, high1), (low2, high2), ...]
                           支持 '-inf' / 'inf' 字符串表示无界
            output_dim: 输出维度
        
        设置属性:
            self.action_low: Tensor (output_dim,) 下界
            self.action_high: Tensor (output_dim,) 上界
            self._action_range: Tensor (output_dim,) 范围
            self._action_center: Tensor (output_dim,) 中心
            self._is_bounded: Tensor (output_dim,) 是否有界
        """
        def parse_bound(val):
            """解析单个边界值，支持字符串 '-inf'/'inf'"""
            if isinstance(val, str):
                if val.lower() == '-inf':
                    return float('-inf')
                elif val.lower() == 'inf':
                    return float('inf')
            return float(val)
        
        # 判断是单一元组还是列表形式
        if isinstance(action_bounds, (list, tuple)) and len(action_bounds) > 0:
            if isinstance(action_bounds[0], (list, tuple)):
                # 列表形式: [(low1, high1), (low2, high2), ...]
                if len(action_bounds) != output_dim:
                    raise ValueError(
                        f"action_bounds 长度 ({len(action_bounds)}) 必须等于 output_dim ({output_dim})"
                    )
                lows = [parse_bound(b[0]) for b in action_bounds]
                highs = [parse_bound(b[1]) for b in action_bounds]
            else:
                # 单一元组: (low, high)，扩展到所有维度
                low = parse_bound(action_bounds[0])
                high = parse_bound(action_bounds[1])
                lows = [low] * output_dim
                highs = [high] * output_dim
        else:
            raise ValueError(f"无效的 action_bounds 格式: {action_bounds}")
        
        # 转为 Tensor 并注册为 buffer
        self.register_buffer('action_low', torch.tensor(lows, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(highs, dtype=torch.float32))
        
        # 计算范围和中心（处理无界情况）
        action_range = []
        action_center = []
        is_bounded = []
        for low, high in zip(lows, highs):
            if math.isinf(low) or math.isinf(high):
                # 无界：范围设为 1（无意义但避免 NaN），中心设为 0
                action_range.append(1.0)
                action_center.append(0.0)
                is_bounded.append(False)
            else:
                action_range.append(high - low)
                action_center.append((high + low) / 2)
                is_bounded.append(True)
        
        self.register_buffer('_action_range', torch.tensor(action_range, dtype=torch.float32))
        self.register_buffer('_action_center', torch.tensor(action_center, dtype=torch.float32))
        self.register_buffer('_is_bounded', torch.tensor(is_bounded, dtype=torch.bool))
    
    def _init_weights(self):
        """权重初始化（SB3 兼容）"""
        for module in self.mlp_extractor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        # 输出层使用较小初始化
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.zeros_(self.action_net.bias)
        
        if self.log_std_mode == 'state_dependent':
            nn.init.orthogonal_(self.log_std_net.weight, gain=0.01)
            nn.init.zeros_(self.log_std_net.bias)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """输入归一化 (EMA)"""
        if not self.normalize_input:
            return x
        
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            if self.norm_count == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var + self._norm_eps)
            else:
                m = self._norm_momentum
                self.running_mean.mul_(1 - m).add_(batch_mean * m)
                self.running_var.mul_(1 - m).add_(batch_var * m)
            self.norm_count.add_(1)
        
        std = torch.sqrt(self.running_var + self._norm_eps)
        x_norm = (x - self.running_mean) / std
        return x_norm.squeeze(0) if squeeze else x_norm
    
    def _scale_action(self, action_raw: torch.Tensor) -> torch.Tensor:
        """缩放动作到目标范围
        
        对于有界维度：根据 action_scale 方式缩放到 [low, high]
        对于无界维度：直接输出原始值
        
        缩放方式:
        - clamp: 直接将网络输出裁剪到 [low, high]（PPO 风格）
        - tanh: 将网络输出通过 tanh 压缩到 (-1, 1)，再映射到 [low, high]
        - softsign: 将网络输出通过 softsign 压缩到 (-1, 1)，再映射到 [low, high]
        """
        if self.action_scale == self.SCALE_NONE:
            # 无缩放，直接返回原始值
            return action_raw
        elif self.action_scale == self.SCALE_CLAMP:
            # PPO 风格：直接裁剪到边界
            return torch.clamp(action_raw, self.action_low, self.action_high)
        elif self.action_scale == self.SCALE_TANH:
            scaled = torch.tanh(action_raw)
        else:  # softsign
            scaled = action_raw / (1 + torch.abs(action_raw))
        
        # 缩放到目标范围：scaled ∈ (-1, 1) → result ∈ (low, high)
        # 公式: result = (scaled + 1) / 2 * range + low = scaled * range/2 + center
        result = scaled * self._action_range / 2.0 + self._action_center
        
        # 对于无界维度，使用原始值
        if not self._is_bounded.all():
            unbounded_mask = ~self._is_bounded
            result = torch.where(unbounded_mask, action_raw, result)
        
        return result
    
    def _log_prob_correction(self, action_raw: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """
        对非线性变换进行对数概率修正
        
        根据变量替换公式: p(y) = p(x) / |dy/dx|
        log p(y) = log p(x) - log |dy/dx|
        
        对于 tanh: y = tanh(x), dy/dx = 1 - tanh²(x) > 0
            log p(y) = log p(x) - Σ log(1 - tanh²(x))
        
        对于 softsign: y = x/(1+|x|), dy/dx = 1/(1+|x|)² > 0
            log |dy/dx| = -2 * log(1 + |x|)
            log p(y) = log p(x) + 2 * Σ log(1 + |x|)
        
        注意: 修正项应该是 -log|dy/dx|，因为 p(y) = p(x) / |dy/dx|
        """
        if self.action_scale == self.SCALE_TANH:
            # log(1 - tanh²(x)) 是 log |dy/dx|，需要减去
            correction = torch.sum(
                torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-6),
                dim=-1
            )
            return log_prob - correction
        elif self.action_scale == self.SCALE_SOFTSIGN:
            # log |dy/dx| = -2 * log(1 + |x|)，需要减去
            # 所以 log p(y) = log p(x) - (-2 * log(1 + |x|)) = log p(x) + 2 * log(1 + |x|)
            correction = 2 * torch.sum(torch.log(1 + torch.abs(action_raw) + 1e-6), dim=-1)
            return log_prob + correction
        return log_prob
    
    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        prev_action: Optional[torch.Tensor] = None,
    ) -> PolicyNetOutput:
        """
        前向传播
        
        Args:
            x: 输入 (batch_size, input_dim) 或 (input_dim,)
            deterministic: 确定性输出（忽略随机采样）
            prev_action: 上一步动作（相对动作模式时使用）
        
        Returns:
            PolicyNetOutput: 包含 mean, log_std, action, log_prob, entropy 等
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
            if prev_action is not None and prev_action.dim() == 1:
                prev_action = prev_action.unsqueeze(0)
        
        # 归一化 → 特征提取 → 动作均值
        x_norm = self._normalize(x)
        features = self.mlp_extractor(x_norm)
        mean = self.action_net(features)
        
        # 确定 log_std
        if self.log_std_mode == 'global':
            log_std = torch.clamp(self.log_std.expand_as(mean), self.log_std_min, self.log_std_max)
        elif self.log_std_mode == 'state_dependent':
            log_std = torch.clamp(self.log_std_net(features), self.log_std_min, self.log_std_max)
        else:  # 'none'
            log_std = torch.zeros_like(mean)
        
        # 随机采样（仅当 log_std_mode != 'none' 且 not deterministic）
        log_prob = None
        entropy = None
        if self.log_std_mode != 'none' and not deterministic:
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action_raw = dist.rsample()
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            # 计算熵：高斯分布的熵 = sum(0.5 * log(2 * pi * e * sigma^2))
            entropy = dist.entropy().sum(dim=-1)
        else:
            action_raw = mean
        
        # 相对动作模式
        if self.relative_action and prev_action is not None:
            # 增量通过 tanh 缩放到 [-(max-min), (max-min)]
            # 这样增量可以让动作从最小值变到最大值（或反向）
            delta = torch.tanh(action_raw) * self._action_range
            action_before_clip = prev_action + delta
            # 最终缩放到合法范围
            action = self._scale_action(action_before_clip)
        else:
            # 绝对动作模式：直接缩放
            action = self._scale_action(action_raw)
        
        # 概率修正（仅对非相对动作模式且 tanh/softsign 缩放）
        if log_prob is not None and not self.relative_action:
            log_prob = self._log_prob_correction(action_raw, log_prob)
        
        # 压缩 batch 维度
        if squeeze:
            mean, action, action_raw = mean.squeeze(0), action.squeeze(0), action_raw.squeeze(0)
            log_std = log_std.squeeze(0) if log_std.dim() > 1 else log_std
            if log_prob is not None:
                log_prob = log_prob.squeeze(0)
            if entropy is not None:
                entropy = entropy.squeeze(0)
        
        self.current_output = PolicyNetOutput(mean=mean, log_std=log_std, action=action, 
                                              log_prob=log_prob, entropy=entropy, action_raw=action_raw)
        return self.current_output
    
    def _inverse_scale_action(self, actions: torch.Tensor) -> torch.Tensor:
        """逆变换：将缩放后的动作还原为网络原始输出
        
        用于 evaluate_actions 中从给定动作计算对数概率。
        
        Args:
            actions: 缩放后的动作 (batch_size, output_dim)
        
        Returns:
            action_raw: 网络原始输出（缩放前）
        """
        if self.action_scale == self.SCALE_CLAMP or self.action_scale == self.SCALE_NONE:
            # clamp 和 none 模式下，动作就是原始值（无法精确逆变换）
            return actions
        
        # 先归一化到 (-1, 1)
        a_norm = (actions - self._action_center) / (self._action_range / 2.0)
        a_norm = torch.clamp(a_norm, -0.999, 0.999)  # 避免边界问题
        
        if self.action_scale == self.SCALE_TANH:
            # 逆 tanh: atanh(x) = 0.5 * log((1+x)/(1-x))
            action_raw = 0.5 * torch.log((1 + a_norm) / (1 - a_norm))
        else:  # softsign
            # 逆 softsign: x = y / (1 - |y|)
            action_raw = a_norm / (1 - torch.abs(a_norm))
        
        return action_raw
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作的对数概率和熵（PPO 训练用）
        
        Args:
            obs: 观测 (batch_size, input_dim)
            actions: 动作 (batch_size, output_dim)
        
        Returns:
            log_prob: (batch_size,)
            entropy: (batch_size,)
        """
        if self.log_std_mode == 'none':
            raise ValueError("evaluate_actions 需要 log_std_mode != 'none'")
        
        # 获取均值和标准差
        out = self.forward(obs, deterministic=True)
        
        if self.log_std_mode == 'global':
            std = torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))
        else:
            std = torch.exp(out.log_std)
        
        # 构建分布
        dist = Normal(out.mean, std)
        
        # 逆变换动作到原始空间
        action_raw = self._inverse_scale_action(actions)
        
        # 计算对数概率和熵
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        log_prob = self._log_prob_correction(action_raw, log_prob)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy
    
    def reset_normalizer(self):
        """重置归一化统计量"""
        if self.normalize_input:
            self.running_mean.zero_()
            self.running_var.fill_(1.0)
            self.norm_count.zero_()
    
    def get_normalizer_stats(self) -> Optional[Dict[str, Any]]:
        """获取归一化统计量"""
        if not self.normalize_input:
            return None
        return {
            'mean': self.running_mean.clone(),
            'std': torch.sqrt(self.running_var + self._norm_eps),
            'count': self.norm_count.item(),
        }
    
    # ============ SB3 权重兼容 ============
    def to_sb3_state_dict(self) -> Dict[str, torch.Tensor]:
        """导出为 SB3 兼容格式"""
        state = {}
        for i, layer in enumerate(self.mlp_extractor):
            if isinstance(layer, nn.Linear):
                idx = i // 2
                state[f'mlp_extractor.policy_net.{idx}.weight'] = layer.weight.data
                state[f'mlp_extractor.policy_net.{idx}.bias'] = layer.bias.data
        state['action_net.weight'] = self.action_net.weight.data
        state['action_net.bias'] = self.action_net.bias.data
        if self.log_std_mode == 'global':
            state['log_std'] = self.log_std.data
        return state
    
    def load_sb3_state_dict(self, state: Dict[str, torch.Tensor], strict: bool = False):
        """从 SB3 格式加载权重"""
        own_state = self.state_dict()
        for sb3_key, tensor in state.items():
            if 'mlp_extractor.policy_net' in sb3_key:
                parts = sb3_key.split('.')
                idx = int(parts[2]) * 2
                own_key = f'mlp_extractor.{idx}.{parts[3]}'
            else:
                own_key = sb3_key
            if own_key in own_state:
                own_state[own_key].copy_(tensor)
            elif strict:
                raise KeyError(f"权重键 {own_key} 不存在")


# ============================================================
# MLPPolicy: 高层策略封装
# ============================================================
class MLPPolicy(BasePolicy):
    """
    MLP 策略封装（PyTorch，可微分）
    
    职责:
    - 根据配置初始化合适的 PolicyNet（输出维度由 shared_output 和 angle_deg_fixed 决定）
    - 输入构建：从 state dict 提取特征向量
    - 网络调用：调用 PolicyNet 进行推理
    - 输出解析：将网络输出转为 (amplitude, angle) 列表，对接 JetObstacle
    
    重要接口约定:
    - __call__(state) 接口不可变更，确保与 JetObstacle、Solver、Runner、Trainer 对接
    - 相对动作模式下，prev_action 通过 state['prev_amplitudes'] 和 state['prev_angles'] 传递
    
    输出维度设计（PolicyNet 无差别对待所有输出，MLPPolicy 负责解析）:
    ┌────────────────┬────────────────────┬──────────────────────────────────┐
    │ shared_output  │ angle_deg_fixed    │ 网络输出维度 (output_dim)        │
    ├────────────────┼────────────────────┼──────────────────────────────────┤
    │ True           │ 设置了固定角度      │ 1 (只有 amplitude)               │
    │ True           │ None (动态角度)    │ 2 (amplitude + angle)            │
    │ False          │ 设置了固定角度      │ num_nozzles (每个喷嘴各 1 个 amp)│
    │ False          │ None (动态角度)    │ 2 * num_nozzles (每个喷嘴 2 个)  │
    └────────────────┴────────────────────┴──────────────────────────────────┘
    
    动作边界设计（每个输出维度独立设置边界）:
    - 幅值维度: [-amplitude_max, amplitude_max]
    - 角度维度: angle_range_deg (仅当 angle_deg_fixed=None 时)
    - 所有缩放由 PolicyNet 统一完成，MLPPolicy 只负责解析和分发
    """
    
    def __init__(
        self,
        input_type: str = 'probes',
        probe_names: Optional[List[str]] = None,
        hidden_dims: Optional[List[int]] = None,
        num_nozzles: int = 1,
        shared_output: bool = True,
        amplitude_max: float = 1.0,
        angle_range_deg: Tuple[float, float] = (0.0, 180.0),
        angle_deg_fixed: Optional[float] = None,
        action_scale: str = 'clamp',
        normalize_input: bool = True,
        normalize_momentum: float = 0.01,
        relative_action: bool = False,
        weights_path: Optional[str] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            input_type: 输入类型 'probes' / 'force_coeffs' / 'both'
            probe_names: 探针名称列表（None=全部）
            hidden_dims: 隐藏层维度，默认 [64, 32]
            num_nozzles: 喷嘴数量（由 JetObstacle 自动设置）
            shared_output: True=所有喷嘴共用一组输出，False=每个喷嘴独立输出
            amplitude_max: 幅值上限，输出映射到 [-amplitude_max, amplitude_max]
            angle_range_deg: 角度范围（度），仅当 angle_deg_fixed=None 时生效
            angle_deg_fixed: 固定角度（度），设置后网络不输出角度，所有喷嘴使用固定值
            action_scale: 动作缩放方式 'tanh' / 'clamp' / 'softsign'
            normalize_input: 是否对输入特征进行归一化（默认 True）
            normalize_momentum: 归一化 EMA 动量系数（默认 0.01）
            relative_action: 是否使用相对动作模式
                - False: 绝对动作模式，网络输出直接作为动作
                - True: 相对动作模式，网络输出为增量，最终动作 = prev + delta
            weights_path: 预训练权重路径
            device: 计算设备 'cpu' / 'cuda'（由 solver 统一设置）
        """
        super().__init__(num_nozzles)
        self.input_type = input_type.lower()
        self.probe_names = probe_names
        self.hidden_dims = hidden_dims or [64, 32]
        self.shared_output = shared_output
        self.amplitude_max = amplitude_max
        self.angle_range_deg = angle_range_deg
        self.angle_deg_fixed = angle_deg_fixed
        self.action_scale = action_scale
        self.normalize_input = normalize_input
        self.normalize_momentum = normalize_momentum
        self.relative_action = relative_action
        self.device = device
        
        # 计算输出维度：根据 shared_output 和 angle_deg_fixed 决定
        # PolicyNet 无差别对待所有输出维度，MLPPolicy 负责解析
        self._has_dynamic_angle = (angle_deg_fixed is None)
        self._outputs_per_nozzle = 2 if self._has_dynamic_angle else 1
        self._num_output_groups = 1 if shared_output else num_nozzles
        self._output_dim = self._outputs_per_nozzle * self._num_output_groups
        
        # 网络（延迟初始化）
        self.network: Optional[PolicyNet] = None
        
        if weights_path:
            self.load_weights(weights_path)
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        if self.network is not None:
            self.network.train(mode)
        return self
    
    def eval(self):
        """设置推理模式"""
        return self.train(False)
    
    def _init_network(self, input_dim: int):
        """初始化网络
        
        根据配置创建 PolicyNet，输出维度由 _output_dim 决定。
        PolicyNet 将所有输出视为无差别的动作，MLPPolicy 负责解析为 (amp, angle)。
        
        动作边界 (action_bounds) 设计:
        - 每个输出维度可以有独立的边界
        - 幅值使用 [-amplitude_max, amplitude_max]
        - 角度使用 angle_range_deg（当 angle_deg_fixed=None 时）
        """
        # 构建每个输出维度的边界
        action_bounds = []
        for i in range(self._num_output_groups):
            # 幅值边界
            action_bounds.append((-self.amplitude_max, self.amplitude_max))
            # 角度边界（仅当动态角度时）
            if self._has_dynamic_angle:
                action_bounds.append((self.angle_range_deg[0], self.angle_range_deg[1]))
        
        self.network = PolicyNet(
            input_dim=input_dim,
            output_dim=self._output_dim,
            hidden_dims=self.hidden_dims,
            activation='tanh',
            action_scale=self.action_scale,
            action_bounds=action_bounds,
            log_std_mode='global',
            normalize_input=self.normalize_input,
            normalize_momentum=self.normalize_momentum,
            relative_action=self.relative_action,
        ).to(self.device)
        LOGGER.info(
            f"MLPPolicy: input={input_dim}, hidden={self.hidden_dims}, "
            f"output={self._output_dim} (per_nozzle={self._outputs_per_nozzle}, "
            f"groups={self._num_output_groups}), scale={self.action_scale}, "
            f"relative={self.relative_action}, angle_fixed={self.angle_deg_fixed}"
        )
    
    def __call__(self, state: Optional[Dict[str, Any]] = None, deterministic: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算控制信号
        
        接口说明:
        - 此接口不可变更，确保与 JetObstacle、Solver、Runner、Trainer 正确对接
        - 相对动作模式 (relative_action=True) 时，必须通过 state 提供上一步动作:
          - state['prev_amplitudes']: 上一步幅值列表（相对动作模式时必需）
          - state['prev_angles']: 上一步角度列表（仅相对动作模式 + 动态角度时必需）
        
        Args:
            state: 状态字典，可包含:
                - probes: 探针数据
                - force_coeffs: 力系数
                - prev_amplitudes: 上一步幅值列表（相对动作模式时必需）
                - prev_angles: 上一步角度列表（仅相对动作模式 + 动态角度时必需）
        
        Returns:
            List[(amplitude, angle_deg)] for each nozzle
        """
        default_angle = torch.tensor(
            self.angle_deg_fixed if self.angle_deg_fixed is not None else self.angle_range_deg[0],
            dtype=torch.float32, device=self.device
        )
        default_amp = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        default = [(default_amp, default_angle)] * self.num_nozzles
        
        if state is None:
            return default
        
        inputs = self._build_input(state)
        if not inputs:
            return default
        
        # 延迟初始化
        if self.network is None:
            self._init_network(len(inputs))
        
        # 构建 prev_action（用于相对动作模式）
        # 注：_build_prev_action 会在缺少必要参数时抛出异常
        prev_action = None
        if self.relative_action:
            prev_action = self._build_prev_action(state)
        
        # 前向传播
        x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        out = self.network(x, deterministic=deterministic, prev_action=prev_action)
        
        return self._parse_output(out.action)
    
    def _build_prev_action(self, state: Dict) -> Optional[torch.Tensor]:
        """从 state 构建 prev_action 向量（用于相对动作模式）
        
        Args:
            state: 状态字典，需要包含:
                - prev_amplitudes: 上一步幅值列表（必需）
                - prev_angles: 上一步角度列表（仅动态角度模式需要）
        
        Returns:
            prev_action: Tensor (output_dim,)
        
        Raises:
            ValueError: 缺少必需的 prev_amplitudes 或 prev_angles（动态角度模式）
        """
        prev_amps = state.get('prev_amplitudes')
        if prev_amps is None:
            raise ValueError(
                "relative_action=True 时，state 必须包含 'prev_amplitudes'"
            )
        
        # 动态角度模式下必须有 prev_angles
        if self._has_dynamic_angle:
            prev_angles = state.get('prev_angles')
            assert prev_angles is not None, (
                "relative_action=True 且 angle_deg_fixed=None（动态角度模式）时，"
                "state 必须包含 'prev_angles'"
            )
        
        # 确保 prev_amps 是列表
        if not isinstance(prev_amps, (list, tuple)):
            prev_amps = [prev_amps]
        
        # shared_output 模式下只需要一个值
        if self.shared_output:
            prev_amps = prev_amps[:1] if len(prev_amps) >= 1 else [0.0]
        else:
            # 每个喷嘴一个值
            while len(prev_amps) < self._num_output_groups:
                prev_amps.append(0.0)
        
        # 构建 prev_action 向量
        prev_action_list = []
        for i in range(self._num_output_groups):
            prev_action_list.append(prev_amps[i] if i < len(prev_amps) else 0.0)
            
            # 动态角度模式下还需要角度
            if self._has_dynamic_angle:
                prev_angles = state.get('prev_angles')
                if not isinstance(prev_angles, (list, tuple)):
                    prev_angles = [prev_angles]
                prev_action_list.append(prev_angles[i] if i < len(prev_angles) else self.angle_range_deg[0])
        
        return torch.tensor(prev_action_list, dtype=torch.float32, device=self.device)
    
    def _parse_output(self, output: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        解析网络输出为 (amplitude, angle) 列表
        
        根据 _outputs_per_nozzle 和 _num_output_groups 解析：
        - _outputs_per_nozzle=1 (angle 固定): 每组只有 1 个值 (amp)
        - _outputs_per_nozzle=2 (angle 动态): 每组有 2 个值 (amp, angle)
        
        注意: 幅值和角度的范围缩放已由 PolicyNet 完成，这里只负责解析和分发。
        """
        results = []
        
        for i in range(self._num_output_groups):
            idx = i * self._outputs_per_nozzle
            amp = output[idx]
            
            # 角度处理
            if self.angle_deg_fixed is not None:
                # 固定角度模式：网络不输出角度
                angle = torch.tensor(self.angle_deg_fixed, dtype=torch.float32, device=self.device)
            else:
                # 动态角度模式：直接使用网络输出（已由 PolicyNet 缩放到 angle_range_deg）
                angle = output[idx + 1]
            
            results.append((amp, angle))
        
        # shared_output=True 时，复制给所有喷嘴
        if self.shared_output:
            results = results * self.num_nozzles
        return results
    
    def _build_input(self, state: Dict) -> List[float]:
        """
        从状态构建输入向量
        
        子类可重写此方法以支持不同的输入构建逻辑。
        """
        inputs = []
        if self.input_type in ('probes', 'both'):
            probes = state.get('probes', {})
            for name in (self.probe_names or probes.keys()):
                if name in probes:
                    p = probes[name]
                    if isinstance(p, dict):
                        inputs.extend([p.get('u', 0), p.get('v', 0), p.get('p', 0)])
                    elif isinstance(p, (list, tuple)):
                        inputs.extend(p[:3])
        if self.input_type in ('force_coeffs', 'both'):
            fc = state.get('force_coeffs', {})
            inputs.extend([fc.get('Cd', 0), fc.get('Cl', 0)])
        return inputs
    
    def load_weights(self, path: str):
        """加载权重"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if 'network' in checkpoint:
                network_state = checkpoint['network']
                input_dim = checkpoint.get('input_dim')
                if checkpoint.get('hidden_dims'):
                    self.hidden_dims = checkpoint['hidden_dims']
                if checkpoint.get('output_dim'):
                    self._output_dim = checkpoint['output_dim']
            else:
                network_state = checkpoint
                first_key = [k for k in network_state.keys() if 'weight' in k][0]
                input_dim = network_state[first_key].shape[1]
            
            self._init_network(input_dim)
            self.network.load_state_dict(network_state)
            LOGGER.info(f"加载权重: {path}")
        except Exception as e:
            LOGGER.warning(f"加载权重失败: {e}")
    
    def save_weights(self, path: str):
        """保存权重"""
        if self.network is None:
            LOGGER.warning("网络未初始化，无法保存权重")
            return
        checkpoint = {
            'network': self.network.state_dict(),
            'input_dim': self.network.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self._output_dim,
            'outputs_per_nozzle': self._outputs_per_nozzle,
            'num_output_groups': self._num_output_groups,
            'angle_deg_fixed': self.angle_deg_fixed,
            'shared_output': self.shared_output,
        }
        torch.save(checkpoint, path)
        LOGGER.debug(f"保存权重: {path}")
    
    def reset_normalizer(self):
        """重置归一化统计量"""
        if self.network is not None:
            self.network.reset_normalizer()
    
    def get_normalizer_stats(self) -> Optional[Dict[str, Any]]:
        """获取归一化统计量"""
        return self.network.get_normalizer_stats() if self.network else None
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'MLPPolicy':
        """从配置创建实例"""
        return cls(
            input_type=cfg.get('input_type', 'probes'),
            probe_names=cfg.get('probe_names'),
            hidden_dims=cfg.get('hidden_dims', [64, 32]),
            num_nozzles=cfg.get('num_nozzles', 1),
            shared_output=cfg.get('shared_output', True),
            amplitude_max=cfg.get('amplitude_max', 1.0),
            angle_range_deg=tuple(cfg.get('angle_range_deg', [0.0, 180.0])),
            angle_deg_fixed=cfg.get('angle_deg_fixed'),
            action_scale=cfg.get('action_scale', 'softsign'),
            normalize_input=cfg.get('normalize_input', True),
            normalize_momentum=cfg.get('normalize_momentum', 0.01),
            relative_action=cfg.get('relative_action', False),
            weights_path=cfg.get('weights_path'),
            device=cfg.get('device', 'cuda'),
        )
