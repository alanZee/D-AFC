# PolicyNet 与 MLPPolicy 模块说明

## 概述

本模块实现了用于主动流动控制（AFC）的神经网络策略，采用两层架构：

- **PolicyNet**: 底层 MLP 网络，支持 PPO/SAC 风格的高斯策略
- **MLPPolicy**: 高层策略封装，解析 PolicyNet 输出为 JetObstacle 所需的 `(amplitude, angle)` 控制信号

**主要特性**：
- EMA 输入归一化：自动跟踪输入统计量，稳定训练
- 多种动作缩放：tanh / clamp / softsign / none
- 重参数化采样：支持可微分的高斯采样
- 概率修正：tanh/softsign 变换的 Jacobian 修正
- 相对动作模式：输出增量叠加到上一步动作
- SB3 兼容：可导入/导出 SB3 PPO 权重格式

---

## 架构设计

```
┌───────────────────────────────────────────────────┐
│                   JetObstacle                      │
│  需要: List[(amplitude, angle)] 控制信号           │
└───────────────────────────────────────────────────┘
                        ▲
                        │
┌───────────────────────────────────────────────────┐
│                    MLPPolicy                       │
│  1. _build_input(): 从 state 提取特征向量          │
│  2. network(x): 调用 PolicyNet                     │
│  3. _parse_output(): 解析为 (amp, angle) 列表      │
└───────────────────────────────────────────────────┘
                        ▲
                        │
┌───────────────────────────────────────────────────┐
│                    PolicyNet                       │
│  - EMA 输入归一化                                  │
│  - 多种动作缩放: tanh / clamp / softsign           │
│  - 支持每维独立的动作边界                           │
│  - 可选高斯随机采样 (log_std_mode)                 │
└───────────────────────────────────────────────────┘
```

---

## 核心公式

### 1. 重参数化采样技巧 (Reparameterization Trick)

将随机采样过程转化为可微分操作：

$$a_{\text{raw}} = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\mu$ 为网络输出均值，$\sigma = \exp(\log\sigma)$ 为标准差。

```python
dist = Normal(mean, std)
action_raw = dist.rsample()  # 重参数化采样，保持梯度
```

### 2. 动作缩放

将网络原始输出 $a_{\text{raw}}$ 映射到动作空间 $[a_{\min}, a_{\max}]$：

| 方式 | 公式 | 特点 |
|:---|:---|:---|
| `tanh` | $a = \tanh(a_{\text{raw}}) \cdot \frac{r}{2} + c$ | SAC 风格，严格边界，梯度饱和快 |
| `clamp` | $a = \text{clamp}(a_{\text{raw}}, a_{\min}, a_{\max})$ | PPO 风格，硬裁剪 |
| `softsign` | $a = \frac{a_{\text{raw}}}{1+\|a_{\text{raw}}\|} \cdot \frac{r}{2} + c$ | 梯度饱和慢|
| `none` | $a = a_{\text{raw}}$ | 无缩放（无界动作）|

其中 $r = a_{\max} - a_{\min}$，$c = (a_{\max} + a_{\min})/2$。

### 3. 概率修正 (Log Probability Correction)

对于 tanh/softsign 缩放，需要修正对数概率（变量替换 Jacobian）：

| 缩放方式 | 修正公式 |
|:---|:---|
| tanh | $\log p(a) = \log p(a_{\text{raw}}) - \sum_i \log(1 - \tanh^2(a_{\text{raw}, i}))$ |
| softsign | $\log p(a) = \log p(a_{\text{raw}}) + 2 \sum_i \log(1 + \|a_{\text{raw}, i}\|)$ |
| clamp | 无修正 |

### 4. 高斯分布熵

用于熵正则化（鼓励探索）：

$$H(\pi) = \sum_i \frac{1}{2}\log(2\pi e \sigma_i^2)$$

---

## 动作向量维度

| shared_output | angle_deg_fixed | 每喷嘴输出 | 输出组数 | **总维度** |
|:-------------:|:---------------:|:----------:|:--------:|:----------:|
| True          | 设置值          | 1 (幅值)   | 1        | **1**      |
| True          | None            | 2 (幅值+角度) | 1     | **2**      |
| False         | 设置值          | 1 (幅值)   | num_nozzles | **N**   |
| False         | None            | 2          | num_nozzles | **2N**  |

---

## 动作边界 (action_bounds)

PolicyNet 支持**每维独立**的动作边界：

```python
# 列表形式：每维独立设置
action_bounds = [
    (-1.5, 1.5),      # dim 0: amplitude
    (0.0, 180.0),     # dim 1: angle (deg)
]

# 单一元组：所有维度共用
action_bounds = (-1.0, 1.0)

# 支持无界
action_bounds = [('-inf', 'inf'), (0.0, 360.0)]
```

MLPPolicy 自动根据 `amplitude_max` 和 `angle_range_deg` 构建每维边界。

---

## 相对动作模式

当 `relative_action=True` 时，网络输出为**增量**：

$$a_t = a_{t-1} + \text{tanh}(a_{\text{raw}}) \cdot (a_{\max} - a_{\min})$$

**必须**通过 state 提供上一步动作：
- `state['prev_amplitudes']`: 上一步幅值列表（必需）
- `state['prev_angles']`: 上一步角度列表（仅动态角度模式需要）

---

## 配置示例

### PolicyNet 配置

```python
PolicyNet(
    input_dim=8,                # 输入维度
    output_dim=2,               # 输出维度
    hidden_dims=[64, 64],       # 隐藏层
    activation='tanh',          # 隐藏层激活: 'tanh' | 'relu'
    action_scale='clamp',       # 动作缩放: 'tanh' | 'clamp' | 'softsign' | 'none'
    action_bounds=(-1.0, 1.0),  # 动作边界
    log_std_init=0.0,           # log_std 初始值
    log_std_mode='global',      # 'none' | 'global' | 'state_dependent'
    normalize_input=True,       # 输入归一化
    relative_action=False,      # 相对动作模式
)
```

### log_std_mode 说明

| 模式 | 说明 |
|:---|:---|
| `none` | 确定性策略，不使用随机采样 |
| `global` | 全局 log_std 参数，与状态无关（SB3 PPO 默认） |
| `state_dependent` | 从隐藏层预测 log_std |

### MLPPolicy 配置 (YAML)

```yaml
jets:
  policy:
    type: mlp
    input_type: probes        # probes / force_coeffs / both
    hidden_dims: [64, 32]
    shared_output: true
    amplitude_max: 1.5
    angle_range_deg: [0, 180]
    angle_deg_fixed: 90.0     # null = 动态角度
    action_scale: softsign
    relative_action: false
    normalize_input: true
```

---

## 关键方法

### forward()

```python
output = policy_net(obs, deterministic=False, prev_action=None)
# output.mean: 动作均值
# output.action: 最终动作（经缩放）
# output.log_prob: 对数概率（含修正）
# output.entropy: 分布熵
```

### evaluate_actions()

PPO 训练时评估给定动作的概率（用于重要性采样）：

```python
log_prob, entropy = policy_net.evaluate_actions(obs, actions)
```

---

## 参数说明

### PolicyNet

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_dim` | int | - | 输入维度 |
| `output_dim` | int | - | 输出维度 |
| `hidden_dims` | List[int] | [64, 64] | 隐藏层维度 |
| `action_scale` | str | 'clamp' | 动作缩放方式 |
| `action_bounds` | tuple/list | (-1, 1) | 动作边界（支持每维独立）|
| `log_std_mode` | str | 'global' | 'none'/'global'/'state_dependent' |
| `relative_action` | bool | False | 相对动作模式 |
| `normalize_input` | bool | True | EMA 输入归一化 |

### MLPPolicy

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_type` | str | 'probes' | 输入类型 |
| `hidden_dims` | List[int] | [64, 32] | 隐藏层维度 |
| `num_nozzles` | int | 1 | 喷嘴数量 |
| `shared_output` | bool | True | 所有喷嘴共享输出 |
| `amplitude_max` | float | 1.0 | 幅值上限 |
| `angle_range_deg` | tuple | (0, 180) | 角度范围（度）|
| `angle_deg_fixed` | float/None | None | 固定角度 |
| `action_scale` | str | 'clamp' | 动作缩放方式 |
| `relative_action` | bool | False | 相对动作模式 |

---

## SB3 PPO 权重兼容

### 对应关系

| PolicyNet | SB3 PPO |
|:---|:---|
| `self.mlp_extractor` | `policy.mlp_extractor.policy_net` |
| `self.action_net` | `policy.action_net` |
| `self.log_std` | `policy.log_std` |

### 导出为 SB3 格式

```python
sb3_state = policy_net.to_sb3_state_dict()
# 返回格式:
# {
#   'mlp_extractor.policy_net.0.weight': ...,
#   'mlp_extractor.policy_net.0.bias': ...,
#   'action_net.weight': ...,
#   'action_net.bias': ...,
#   'log_std': ...
# }
```

### 从 SB3 格式加载

```python
policy_net.load_sb3_state_dict(sb3_state)
```

---

## 正则化机制（SimpleTrainer）

SimpleTrainer 提供多种正则化机制确保策略神经网络的稳定性以及训练的稳定性。

**重要**：所有正则化项在每个环境步都施加，但正则化项的数值仅在动作更新步刷新（每个 control_interval 的第一步），其他步复用上一次的值以保持梯度链连通。

| 类型 | 参数 | 作用 |
|------|------|------|
| 熵正则化 | `entropy_weight` | 鼓励探索，防止过早收敛 |
| KL 散度 | `kl_weight` | 限制策略更新幅度 |
| 幅值 L2 | `amplitude_l2_weight` | 惩罚大幅值控制信号 |
| 时序平滑 | `temporal_smoothness_weight` | 惩罚控制信号剧烈变化 |
| EMA 稳定性 | `policy_stability_weight` | 惩罚偏离历史移动平均 |

详见 [JetTrainer 手册的正则化说明](architecture-afc_supervision-JetTrainer_技术手册.md#45-正则化模块)。
