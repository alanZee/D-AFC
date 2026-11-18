# architecture-afc_supervision-JetTrainer 技术手册

> 基于可微分模拟的射流主动流动控制 (AFC) 训练框架

---

## 目录

1. [概述与核心理念](#1-概述与核心理念)
2. [快速入门](#2-快速入门)
3. [系统架构](#3-系统架构)
4. [核心组件](#4-核心组件)(类定义、方法说明)
5. [训练流程详解](#5-训练流程详解)(流程 + 机制 + 梯度流)
6. [配置参数详解](#6-配置参数详解)
7. [API 参考](#7-api-参考)

**附录**
- [A. JetTrainer 功能清单](#附录a-jettrainer-功能清单)
- [B. 符号表与公式推导](#附录b-符号表与公式推导)
- [C. 常见问题与解决方案](#附录c-常见问题与解决方案)
- [D. 涡脱周期](#附录d-涡脱周期)

---

## 1. 概述与核心理念

### 1.1 什么是基于可微分模拟的 AFC？

**主动流动控制 (Active Flow Control, AFC)** 通过智能控制（如表面射流）改变流场特性，实现减阻、抑涡、提升稳定性等目标。

**本框架的核心创新**：使用完全可微分的 CFD 求解器 (PhiFlow)，实现端到端梯度反传，将传统 RL 问题转化为类监督学习问题。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    基于可微分模拟的 AFC 训练                          │
├─────────────────────────────────────────────────────────────────────┤
│  • 环境（CFD求解器）完全可微分                                        │
│  • 梯度可从 Loss 直接反向传播到 Agent 参数                            │
│  • 类似监督学习的端到端训练，但解决的是 RL 类型的决策问题               │
│  • 无需大量采样、无需 Critic 网络、无需复杂的 RL 算法                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 与传统 DRL 方法的对比

| 特性 | 传统 DRL (如 DRLinFluids) | **JetTrainer (可微分)** |
|------|--------------------------|------------------------|
| 环境类型 | 黑盒 (OpenFOAM) | 可微分 (PhiFlow Torch) |
| 梯度获取 | 策略梯度估计 | **直接反向传播** |
| 采样效率 | 低（需大量交互） | **高（直接优化）** |
| Critic 网络 | 需要 (2 个 Q 网络) | **不需要** |
| 经验回放 | 需要 (Buffer=20k) | **不需要** |
| 收敛速度 | 3000+ epochs | **~50-100 epochs** |

### 1.3 核心设计原则

```
┌────────────────────────────────────────────────────────────────┐
│         累积(多步) Loss → 统一 backward → 释放计算图            │
└────────────────────────────────────────────────────────────────┘
```

| 原则 | 说明 |
|------|------|
| **端到端可微分** | 控制信号 → 多步求解 → 采样数据全流程保持梯度 |
| **分段释放** | 每次 backward 后释放计算图，控制显存 |
| **配置驱动** | 全部参数通过 YAML 配置，无需修改代码 |
| **模块化** | BaseTrainer / JetTrainer / AFCAgent 松耦合 |

---

## 2. 快速入门

### 2.1 最小示例

```python
from myafc.afc_supervision import JetTrainer

# 从配置文件初始化
trainer = JetTrainer("configs/jet_trainer_benchmark.yaml")

# 训练
results = trainer.train()

# 结果
print(f"Final Loss: {results['final_loss']:.6f}")
print(f"Exp Dir: {results['exp_dir']}")

# 清理
trainer.close()
```

### 2.2 最小配置

```yaml
env_config_path: cfgs4env/karman_torch_drl_square.yaml

training:
  epochs: 50
  learning_rate: 1e-3

agent:
  hidden_dims: [256, 256]

loss:
  target_velocity: [0.0, 0.233]
```

### 2.3 输出目录结构

```
runs/afc_jet_train/exp/
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   └── epoch_10.pt
├── configs/
│   ├── trainer_config.yaml
│   └── solver_config.yaml
├── validation/
├── tensorboard/
├── train.log
├── training_curves.png       # 训练曲线可视化 (4×2 子图)
├── training_data.npz         # 训练数据 (与可视化对齐)
└── training_step_data.csv    # 逐步数据 CSV 格式
```

#### 训练数据保存格式

**NPZ 文件** (`training_data.npz`)：

| 字段 | 维度 | 说明 |
|------|------|------|
| `loss_history` | (epochs,) | 每 epoch 总 loss |
| `primary_loss_history` | (epochs,) | 每 epoch 主 loss |
| `reg_loss_history` | (epochs,) | 每 epoch 正则化 loss |
| `step_losses` | (env_steps,) | 每步 Primary Loss |
| `step_total_losses` | (env_steps,) | 每步 Total Loss |
| `step_reg_losses` | (env_steps,) | 每步 Reg Loss |
| `step_velocities_u/v` | (env_steps,) | 探针速度 u/v 分量 |
| `step_pressures` | (env_steps,) | 探针压力 |
| `amplitude_history` | (env_steps,) | 射流幅值 |
| `step_cd/cl` | (env_steps,) | 力系数 Cd/Cl |
| `acc_total_loss` | (nn_updates,) | 每次 NN 更新的加权累积 loss |
| `update_step_indices` | (nn_updates,) | NN 更新时刻索引 |
| `target_velocity` | (2,) | 目标速度 [u, v] |

**CSV 文件** (`training_step_data.csv`)：

| 列名 | 说明 |
|------|------|
| `env_step` | 环境步索引 |
| `primary_loss` | Primary Loss |
| `total_loss` | Total Loss = Primary + Reg |
| `reg_loss` | Regularization Loss |
| `velocity_u/v` | 探针速度 |
| `pressure` | 探针压力 |
| `amplitude` | 射流幅值 |
| `cd/cl` | 力系数 (可选) |

> **Resume 增量保存**：当从 checkpoint 恢复训练时，会先加载已有的 `training_data.npz` 数据到内存历史记录中，新训练数据追加到原有数据后面，保存时写入完整历史。这确保训练曲线的连续性，无论训练被中断几次。

```
训练中断 → 保存 checkpoint + training_data.npz
    │
Resume 启动 →  加载 checkpoint (权重、epoch 等)
    │
    └→ _load_existing_training_data() 
           └→ 从 npz 加载历史数据到内存列表
    │
继续训练 → 新数据追加到历史列表
    │
训练结束 → _save_training_results() 保存完整历史
```

#### 训练曲线可视化 (`training_curves.png`)

训练结束后自动生成 4×2 布局的可视化图：

```
┌───────────────────────────────────────────────┐
│  加权累积 Loss        │  Loss 权重分布          │
├───────────────────────────────────────────────┤
│  每步 Loss 曲线       │  每步正则化 Loss 曲线   │
├───────────────────────────────────────────────┤
│  探针速度分量+压力    │  射流幅值 (V)           │
├───────────────────────────────────────────────┤
│  力系数 Cd 曲线       │  力系数 Cl 曲线         │
└───────────────────────────────────────────────┘
```

**子图说明**：

| 位置 | 内容 | 说明 |
|------|------|------|
| (0,0) | **Weighted Accumulated Total Loss** | 每次 NN 更新时的加权累积总 loss（用于实际 backward） |
| (0,1) | **Loss Weights per Step** | 一个 NN 更新周期内各时间步的权重分布（柱状图） |
| (1,0) | Per-Step Loss | 每步的 Total Loss 和 Primary Loss |
| (1,1) | Per-Step Reg Loss | 每步的正则化 Loss |
| (2,0) | Probe Velocity & Pressure | 探针位置的速度分量 (u, v) 及压力 p（右轴） |
| (2,1) | Control Amplitude | 各喷嘴的射流幅值 |
| (3,0) | Cd | 阻力系数随时间变化 |
| (3,1) | Cl | 升力系数随时间变化 |

**关键特性**：
- **加权累积 Loss 曲线**：显示实际用于神经网络更新的累积 loss，反映真实优化目标
- **权重分布图**：可视化 `exponential` 或 `equal` 权重模式的时间步权重
- **Cd/Cl 力系数曲线**：显示阻力和升力系数随时间的变化，便于评估控制效果
- **NN 更新点标记**：可选功能，散点标记神经网络更新时刻，直观显示控制策略何时进行学习（配置参数控制）
- **竖直虚线标记**：可选功能，在神经网络更新位置绘制竖直虚线（配置参数控制）
- **多喷嘴支持**：射流幅值图显示各喷嘴的独立控制信号

**可视化配置参数** (`output` 节):

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scatter_enabled` | bool | false | 是否绘制散点标记（曲线上的更新点） |
| `scatter_size` | float | 20.0 | 散点尺寸 |
| `vline_enabled` | bool | true | 是否绘制竖直虚线（标记更新位置） |
| `vline_width` | float | 0.5 | 竖直虚线线宽 |

#### 训练阶段流场可视化 (`training_visuals` 节)

训练过程中可实时保存流场图片和原始数据，用于监控和分析。

**输出目录结构:**
```
exp_dir/{folder}/
├── raw/          # 原始流场数据 (fields.h5)
└── frames/       # 可视化图片
```

```yaml
training_visuals:
  enabled: false              # 是否启用 (默认禁用，避免 IO 开销)
  interval: 10                # 每隔多少 env-step 保存一帧
  quantities:                 # 绘制的物理量
    - velocity_magnitude
    - pressure
  save_endpoints: true        # 是否保存初始帧和最终帧
  folder: training_fields     # 输出子目录名 (raw + frames)
  smooth: true                # 是否平滑插值热力图
  video_fps: 10               # 合成视频帧率
  generate_video: true        # 是否生成视频 (false = 仅保存图片)
  obstacle_style: fill        # 障碍物风格: outline/fill/both/none
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用训练流场可视化 |
| `interval` | int | 10 | 保存间隔 (env-step) |
| `quantities` | list | ['velocity_magnitude', 'pressure'] | 物理量列表 |
| `save_endpoints` | bool | true | 保存初始帧(step=0)和最终帧(_final) |
| `folder` | str | 'training_fields' | 输出子目录（含 raw/ 和 frames/ 子目录） |
| `smooth` | bool | true | 热力图平滑插值 |
| `video_fps` | int | 10 | 视频帧率 |
| `generate_video` | bool | true | 是否生成视频（false=仅保存图片，不合成视频） |
| `obstacle_style` | str | 'fill' | 障碍物显示风格 |

### 2.4 命令行运行

```bash
cd myafc/examples/run_jet_trainer
python train_jet.py path/to/config.yaml

# 覆盖参数
python train_jet.py config.yaml --epochs 200 --lr 0.001
```

---

## 3. 系统架构

### 3.1 模块层次结构

```
myafc/src/myafc/
├── afc_supervision/                    # 训练框架
│   ├── engines/
│   │   ├── trainers/
│   │   │   ├── base_trainer.py        # BaseTrainer 抽象基类
│   │   │   └── jet_trainer.py         # JetTrainer 核心实现
│   │   └── simple_trainer.py          # SimpleTrainer (参考实现)
│   ├── agents/
│   │   └── afc_agent.py               # AFCAgent (策略网络)
│   └── losses/                         # 损失函数模块
│
├── env_kernels/                        # 环境核心
│   ├── solvers/
│   │   ├── karman_torch_solver.py     # 可微分 CFD 求解器
│   │   ├── parallel_solver_manager.py # 多环境并行管理
│   │   └── fluid_control/             # 射流控制模块
│   │       ├── jet_obstacle.py        # JetObstacle
│   │       └── control_policy/        # 策略网络基类
│   └── solver_wrappers/
│       └── solver_runner.py           # SolverRunner 封装
│
└── configs/                            # 配置文件
    ├── cfgs4env/                       # 环境配置
    └── cfgs4trainer/                   # 训练器配置
```

### 3.2 组件依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JetTrainer                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         训练循环调度                                  │   │
│  │  • 初始化 Solver(s) / Agent / Optimizer / Scheduler                 │   │
│  │  • 协调仿真推进与神经网络训练                                          │   │
│  │  • 管理日志、检查点、TensorBoard                                      │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│    ┌─────────────────────────┼─────────────────────────┐                   │
│    ▼                         ▼                         ▼                   │
│  ┌───────────────────┐ ┌───────────────┐ ┌─────────────────────────┐       │
│  │ KarmanTorchSolver │ │   AFCAgent    │ │   Loss Functions        │       │
│  │ (PhiFlow Torch)   │ │  (torch.nn)   │ │  (probe/force/reg)      │       │
│  └─────────┬─────────┘ └───────────────┘ └─────────────────────────┘       │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        JetObstacle                                   │   │
│  │  • 障碍物几何定义                                                     │   │
│  │  • 多喷嘴射流边界条件                                                 │   │
│  │  • set_controls_external() 接口                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Solver 内置策略 vs Agent 策略

> **重要说明**：日志中显示的 `内置策略: constant` 是 Solver 的默认策略，JetTrainer 训练时由 Agent 网络覆盖。

```
solver_config.yaml:
  jets:
    policy:
      type: constant       ← Solver 内置策略 (日志中显示)
      amplitude: 4.0
      angle_deg: 90.0

训练时:
  Agent (MLP) → jet_obstacle.set_controls_external(action) → 覆盖内置策略
```

| 场景 | 控制来源 | 日志显示 |
|------|----------|----------|
| 纯 CFD 仿真 (无 Agent) | Solver 内置策略 | `内置策略: constant` |
| JetTrainer 训练 | Agent 网络输出 | `内置策略: constant (训练时由 Agent 覆盖)` |
| AFCValidator 验证 | 加载的 Agent 网络 | `内置策略: constant (训练时由 Agent 覆盖)` |

**关键接口**：`jet_obstacle.set_controls_external(amplitudes, angles)`

---

## 4. 核心组件

### 4.1 BaseTrainer 抽象基类

**文件**: `afc_supervision/engines/trainers/base_trainer.py`  
**代码量**: ~432 行

BaseTrainer 定义了所有训练器的标准接口与公共功能，采用 **YOLOv8 风格** 的优化器配置。

#### 4.1.1 类定义

```python
class BaseTrainer(ABC):
    """
    所有 Trainer 的抽象基类
    
    职责:
    - 随机种子管理
    - 输出目录创建
    - 日志初始化 (file + console)
    - 优化器构建 (YOLOv8 风格参数分组)
    - 学习率调度 (warmup + cosine/linear)
    - EMA / AMP 支持
    - 检查点保存/加载
    """
```

#### 4.1.2 抽象方法

| 方法 | 描述 |
|------|------|
| `train()` | 主训练入口，子类必须实现 |
| `_training_step()` | 单步训练逻辑 |
| `_build_state()` | 构建 Agent 输入状态 |

#### 4.1.3 主要公共方法

```python
def _setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
    """构建优化器 (YOLOv8 风格参数分组)
    
    参数分组策略:
    - 组0 (bn/bias): weight_decay=0
    - 组1 (conv weights): 正常 weight_decay
    - 组2 (other): 正常 weight_decay
    """

def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
    """构建学习率调度器
    
    支持:
    - constant: 恒定学习率
    - linear: 线性衰减至 lr * lr_final
    - cosine: 余弦退火至 lr * lr_final
    """

def _save_checkpoint(self, tag: str = 'latest') -> None:
    """保存检查点
    
    内容: model_state_dict, optimizer_state_dict, scheduler_state_dict,
          epoch, step, loss_history, config
    """

def _load_checkpoint(self, path: Path) -> None:
    """加载检查点并恢复训练状态"""
```

#### 4.1.4 YOLOv8 优化器参数分组

```python
# 参数分组逻辑
groups = [
    {'params': bn_params, 'weight_decay': 0.0},           # BatchNorm
    {'params': bias_params, 'weight_decay': 0.0},         # Bias
    {'params': other_params, 'weight_decay': weight_decay} # Other
]
```

**支持的优化器类型**（YOLOv8 风格）：
- `adam`: 标准 Adam
- `adamw`: AdamW（解耦权重衰减）
- `sgd`: SGD with Nesterov momentum
- `nadam`: NAdam（Nesterov-accelerated Adam）
- `radam`: RAdam（Rectified Adam）
- `rmsprop`: RMSprop
- `auto`: 自动选择（默认 AdamW）

#### 4.1.5 EMA 模型平均 (YOLOv8 风格)

**文件位置**: `base_trainer.py` 中的 `ModelEMA` 类

EMA（Exponential Moving Average）模型平均是一种训练增强技术，维护模型权重的指数移动平均版本，通常用于：
- **推理稳定性**：EMA 模型通常比训练中的瞬时模型更稳定
- **泛化能力**：平均后的权重往往泛化更好
- **验证评估**：可使用 EMA 模型进行验证，作为"最佳模型"候选

**YOLOv8 风格实现**：

```python
class ModelEMA:
    """指数移动平均模型 (YOLOv8 风格)
    
    特点:
    - 动态衰减: decay(x) = ema_decay * (1 - exp(-x / tau))
    - 自适应调整: 初期快速跟随，后期稳定平滑
    """
    
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - exp(-x / tau))
        
    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * model.state_dict()[k].detach()
```

**配置参数**：
```yaml
training:
  use_ema: true         # 启用 EMA 模型平均
  ema_decay: 0.9999     # 最大 EMA 衰减系数
  ema_tau: 2000         # 衰减时间常数 (YOLOv8 默认值)
```

**动态衰减公式**：
$$d(x) = \alpha \cdot (1 - e^{-x/\tau})$$

其中 $\alpha$ 为 `ema_decay`，$\tau$ 为 `ema_tau`，$x$ 为更新次数。

**验证阶段双模型评估**：
启用 EMA 后，验证阶段会同时评估原模型和 EMA 模型：
- 原模型输出保存到 `validation/`
- EMA 模型输出保存到 `validation_ema/`
- 返回 `(model_loss, ema_loss)` 元组

**注意**：此 EMA 是**模型权重级别**的平均，与正则化部分的"策略稳定性 EMA"（控制信号级别）是不同的概念。

#### 4.1.6 验证阶段 (Validation)

验证阶段的流程设计与训练阶段**完全一致**，确保 loss 值可比。详细机制见 [5.5 训练/验证一致性机制](#55-训练验证一致性机制)。

**核心特点**：
- **流程对齐**：预热、Loss 计算、Buffer 更新等流程与训练完全一致
- **不含正则化项**：验证 Loss = Primary Loss（无 reg_loss）
- **验证步数**：`validation_steps` 不包含 warmup 步数

**验证步数设置**：
- 建议至少覆盖 3-5 个涡脱周期
- 涡脱周期 $T = D / (St \times U)$
- 详细参数参考见 [附录D. 涡脱周期](#附录d-涡脱周期)

**验证阶段输出文件** (`validation/` 目录)：

| 文件 | 内容 |
|------|------|
| `validation_curves.npz` | 曲线数据 (NumPy 格式，含元数据) |
| `validation_curves.csv` | 曲线数据 (表格格式) |
| `validation_curves.png` | 3×2 可视化图 |
| `raw/fields.h5` | 流场数据 |
| `raw/force_coeffs.csv` | 力系数（含 warmup） |

#### 验证曲线统一格式

**NPZ 和 CSV 使用完全相同的键名**，便于跨工具分析：

| 字段 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `env_step` | int | ✅ | 环境步索引 (0-based) |
| `amplitude` | float | ✅ | 射流幅值 |
| `cd` | float | ✅ | 阻力系数 |
| `cl` | float | ✅ | 升力系数 |
| `loss` | float | ⬜ | Primary Loss (Supervision 模式) |
| `velocity_u` | float | ⬜ | 探针速度 u 分量 |
| `velocity_v` | float | ⬜ | 探针速度 v 分量 |
| `pressure` | float | ⬜ | 探针压力 |
| `reward` | float | ⬜ | 累积奖励 (DRL 模式) |

**NPZ 额外包含元数据**：
- `target_velocity`: 目标速度 [u, v]

**加载示例**：
```python
import numpy as np
import pandas as pd

# NPZ (含元数据)
data = np.load('validation_curves.npz')
print(data['cd'], data['cl'])
print(data['target_velocity'])  # 元数据

# CSV (纯时序)
df = pd.read_csv('validation_curves.csv')
print(df[['env_step', 'amplitude', 'cd', 'cl']])
```

> 📌 **跨 Trainer 一致性**：JetTrainer、DRLTrainer、AFCValidator 的验证输出均使用此统一格式

> 💡 **工具**: 使用 `scripts/calc_vortex_shedding_period.py --demo` 快速查看涡脱周期参考

---

#### 4.1.7 检查点保存与加载 (断点续训) (YOLOv8 风格)

**检查点格式**：
```python
ckpt_data = {
    'epoch': self.current_epoch,
    'step': self.current_step,
    'update_count': self.update_count,
    'best_fitness': self.best_fitness,
    'model': agent.network.state_dict(),
    'ema': ema.state_dict() if ema else None,
    'optimizer': optimizer.state_dict(),
    'scaler': grad_scaler.state_dict() if amp else None,
    'date': datetime.now().isoformat(),
    'version': __version__,
}
```

**保存策略** (YOLOv8 风格)：
- `last.pt`: 每个 epoch 保存
- `best.pt`: 当 loss 达到新低时保存
- `epoch_N.pt`: 按 `checkpoint_epoch_interval` 间隔保存

**层次化设计**：
- `BaseTrainer._save_checkpoint()`: 通用逻辑
- `BaseTrainer._load_checkpoint()`: 通用逻辑
- `JetTrainer._get_model_state_dict()`: 个性化模型序列化
- `JetTrainer._load_model_state_dict()`: 个性化模型加载

**断点续训 (checkpoint + resume)**

**实现位置**：`BaseTrainer`（通用功能，所有 Trainer 可用）

**设计**：统一的 `checkpoint` 参数 + `resume` 开关，简化配置。

**参数说明**（定义在 `TrainingBaseConfig`）：

| 参数 | 类型 | 说明 |
|------|------|------|
| `checkpoint` | `str / null` | 检查点路径 |
| `resume` | `bool` | 控制训练模式 (true=严格断点续训, false=若有checkpoint则仅加载权重) |

**使用场景对比**：

| 场景 | checkpoint | resume | 行为 |
|------|------------|--------|------|
| 从头训练 | `null` | `false` | 随机初始化，新建目录 |
| 迁移学习 | `"path/to/ckpt.pt"` | `false` | 仅加载权重，新建目录 |
| 断点续训 | `"path/to/ckpt.pt"` | `true` | 恢复完整状态，**复用目录** |

**Checkpoint epoch 语义**：

```
checkpoint 中 epoch 字段表示 "已完成轮数"（1-based）:
- epoch=0: 未开始训练
- epoch=1: 已完成第 1 轮
- epoch=N: 已完成前 N 轮

Resume 后训练循环: for epoch in range(current_epoch, total_epochs)
→ 若 current_epoch >= total_epochs，跳过训练直接进入 validation
```

**BaseTrainer 提供的方法**：

```python
# 路径解析
_resolve_checkpoint_path() -> Optional[Path]

# 断点续训 (恢复全部状态)
_resume_training()

# 仅加载权重 (不恢复训练状态)
_load_checkpoint_weights()
```

**Resume 恢复内容**：

```python
def _resume_training(self):  # BaseTrainer 方法
    ckpt = torch.load(checkpoint_path)
    self._load_model_state_dict(ckpt['model'])    # 模型权重
    optimizer.load_state_dict(ckpt['optimizer'])  # 优化器状态
    ema.load_state_dict(ckpt['ema'])              # EMA 状态
    grad_scaler.load_state_dict(ckpt['scaler'])   # AMP scaler
    current_epoch = ckpt['epoch']                 # 训练进度 (已完成轮数)
```

**配置示例**：

```yaml
training:
  checkpoint: "runs/afc_train/exp/checkpoints/last.pt"
  resume: true  # true=断点续训, false=若有checkpoint则仅加载权重
```

**支持的 checkpoint 格式**：

| 格式 | 结构 | 来源 |
|------|------|------|
| AFCAgent 格式 | `{network_state_dict: ...}` | `agent.save_weights()` |
| Trainer 格式 | `{model: state_dict, ...}` | `_save_checkpoint()` |
| 纯 state_dict | `{mlp_extractor..., action_net...}` | `torch.save(net.state_dict())` |

### 4.2 JetTrainer 核心实现

**文件**: `afc_supervision/engines/trainers/jet_trainer.py`  
**代码量**: ~2500 行

JetTrainer 是射流主动流动控制的核心训练器，继承 BaseTrainer 并实现完整的可微分训练循环。

#### 4.2.1 类定义

```python
class JetTrainer(BaseTrainer):
    """
    基于可微分模拟的射流 AFC 训练器
    
    核心功能:
    - 单环境/多环境训练 (ParallelSolverManager)
    - 探针 MSE/MAE Loss
    - 力系数 Loss
    - 多种正则化 (L2, 时序平滑, 策略稳定性, 熵, KL)
    - PPO 式重要性采样
    - sinusoidal 入口边界
    - envstep_state 历史堆叠
    - relative_action 相对动作模式
    - 断点续训 (Resume)
    """
```

#### 4.2.2 初始化流程

```python
def __init__(self, config: Union[str, Path, Dict, JetTrainerConfig]):
    # 1. 解析配置
    self.cfg = self._parse_config(config)
    
    # 2. 初始化基类 (随机种子、输出目录、日志)
    super().__init__(self.cfg.training, self.cfg.output)
    
    # 3. 初始化训练状态
    self._prev_amplitude = torch.tensor(0.0)
    self._step_losses: List[torch.Tensor] = []
    self._state_history_buffer: List[Dict] = []
    
    # 4. 初始化组件
    self._setup_env()           # Solver 初始化
    self._setup_probe_points()  # 探针点加载
    self._setup_agent()         # AFCAgent 初始化
    self._setup_optimizer_for_jet()  # 优化器
    self._setup_tensorboard()   # TensorBoard
```

#### 4.2.3 核心方法

| 方法 | 描述 |
|------|------|
| `train()` | 主训练入口，返回训练结果字典 |
| `_setup_env()` | 初始化 Solver (单环境/多环境) |
| `_setup_agent()` | 初始化 AFCAgent |
| `_setup_probe_points()` | 加载探针位置 (用于 `main_input_type='probes'`) |
| `_build_state()` | 构建 Agent 输入状态字典 |
| `_compute_loss()` | 计算单步 Loss (主 Loss + 正则化) |
| `_reset_env()` | 重置环境状态 (每 epoch 开始) |
| `_detach_graph()` | 分离计算图 (每次 backward 后) |
| `_run_validation()` | 验证阶段 |

#### 4.2.4 状态构建 (_build_state)

```python
def _build_state(self) -> Dict:
    """构建 Agent 输入状态
    
    返回字典结构:
    {
        'time': float,           # 当前物理时间
        'step': int,             # 当前步数
        'probes': {
            'target': {'u': float, 'v': float},
            'probe_1': {...},    # 仅 main_input_type='probes' 时
            ...
        },
        'force_coeffs': {        # 仅 include_force_coeffs=True 时
            'Cd': float,
            'Cl': float,
        },
    }
    """
```

### 4.3 AFCAgent 策略网络

**文件**: `afc_supervision/agents/afc_agent.py`  
**代码量**: ~380 行

AFCAgent 继承 MLPPolicy，专为 AFC 任务设计的策略网络。

#### 4.3.1 类定义

```python
class AFCAgent(MLPPolicy):
    """
    AFC 专用策略网络
    
    核心功能:
    - 历史状态缓冲管理 (_state_history)
    - 历史幅值管理 (_amplitude_history)
    - 权重保存/加载
    - 与 JetObstacle 无缝对接
    
    输入特征支持:
    - target_velocity: 目标速度 [u, v]
    - envstep_state: 历史状态堆叠 (与 DRLinFluids 一致)
    - prev_amplitudes: 历史幅值序列
    - probes: 探针数据
    - force_coeffs: 力系数 [Cd, Cl]
    """
```

#### 4.3.2 核心方法

| 方法 | 描述 |
|------|------|
| `compute_action(state)` | 计算控制信号，返回 (amplitude, PolicyNetOutput) |
| `update_history(state)` | 更新历史状态缓冲区 (envstep_state 用) |
| `update_amplitude_history(amp)` | 更新历史幅值 (prev_amplitude 用) |
| `set_target_velocity(vel)` | 设置目标速度 (由 Trainer 调用) |
| `reset()` | 重置 Agent 状态 (每 epoch 调用) |
| `_build_input(state)` | 构建神经网络输入向量 |
| `_extract_probe_features(state)` | 提取探针/力系数特征 (历史堆叠用) |

#### 4.3.3 输入向量构建

`_build_input()` 方法与 SimpleTrainer 的 CustomMLPPolicy 保持一致：

```python
def _build_input(self, state: Dict) -> List[float]:
    """构建输入向量（按固定顺序拼接）
    
    顺序:
    1. target_velocity [u, v]      # if include_target_velocity
    2. prev_amplitudes [a1, a2...] # if include_prev_amplitude_num > 0
    3. 探针+力系数特征 (支持历史堆叠)
       - envstep_state=1: 仅当前 state
       - envstep_state>1: 堆叠历史+当前 (最旧在前)
    """
```

**历史堆叠示意** (`envstep_state=3`):

```
输入向量 = [target_vel] + [prev_amps] + [hist_t-2] + [hist_t-1] + [current_t]
                                          ↑           ↑           ↑
                                     _state_history 缓冲 →       当前 state
```

#### 4.3.4 使用示例

```python
# 初始化
agent = AFCAgent(config=agent_cfg, num_nozzles=2, device='cuda')
agent.set_target_velocity([0.0, 0.233])

# 训练循环
for step in range(steps):
    state = build_state()  # 由 Trainer 构建
    amplitude, policy_output = agent.compute_action(state)
    
    # 执行控制...
    
    agent.update_history(state)
    agent.update_amplitude_history(amplitude.item())

# 每个 epoch 开始时
agent.reset()
    """清空历史缓冲区 (每 epoch 开始时调用)"""

def _prepare_state(self, state: Dict) -> torch.Tensor:
    """准备网络输入张量 (包含历史堆叠)"""
```

### 4.4 Loss 函数模块

**文件**: `afc_supervision/losses/`

#### 4.4.1 探针 Loss

```python
def probe_mse_loss(current: Tensor, target: Tensor) -> Tensor:
    """探针速度 MSE Loss
    
    L = mean((u - u_target)² + (v - v_target)²)
    """

def probe_mae_loss(current: Tensor, target: Tensor) -> Tensor:
    """探针速度 MAE Loss
    
    L = mean(|u - u_target| + |v - v_target|)
    """
```

#### 4.4.2 力系数 Loss

```python
def force_coeffs_loss(Cd: Tensor, Cl: Tensor, 
                      Cd_target: float = 0.0, 
                      Cl_target: float = 0.0) -> Tensor:
    """力系数 Loss
    
    L = (Cd - Cd_target)² + (Cl - Cl_target)²
    """
```

#### 4.4.3 流场 Loss

流场 Loss 用于直接优化流场特性，支持区域指定功能：

**区域指定 (Region Specification)**:
```python
from myafc.afc_supervision.losses import RegionConfig, field_rms_loss

# 方式 1: 使用 RegionConfig
region = RegionConfig(x_start=0.3, y_start=0.2, x_end=0.8, y_end=0.8)
loss = field_rms_loss(velocity_field, region=region)

# 方式 2: 使用列表 [x_start, y_start, x_end, y_end]
loss = field_rms_loss(velocity_field, region=[0.3, 0.2, 0.8, 0.8])
```

| 流场 Loss | 功能 | 说明 |
|-----------|------|------|
| `field_rms_loss` | 区域均方根 | 最小化指定区域的速度波动 |
| `fft_mode_loss` | FFT 模态 | 最小化非零 FFT 模态幅值 |
| `fluctuation_sparse_loss` | 波动稀疏 | 自动定位并惩罚波动最大点 |
| `region_velocity_loss` | 区域速度 | 最小化指定区域的速度波动 |

#### 4.4.4 模态分解 Loss (POD/DMD)

模态分解 Loss 用于分析和优化时序流场数据：

```python
from myafc.afc_supervision.losses import pod_mode_loss, dmd_mode_loss

# POD 模态 Loss - 最小化非稳态模态能量
snapshots = torch.stack([v_t1, v_t2, v_t3, ...])  # 多个时间步快照
loss = pod_mode_loss(snapshots, n_modes=5, exclude_first_k=1)

# DMD 模态 Loss - 最小化非稳态模态幅值
loss = dmd_mode_loss(snapshots, n_modes=5, dt=0.005)

# 稳态残差 Loss - 原流场与稳态模态的差异
loss = steady_state_residual_loss(snapshots, n_modes=5)
```

底层可微分操作已封装到 `myafc.utils.differentiable_ops`:
```python
from myafc.utils import differentiable_pod, differentiable_dmd

# POD 分解
modes, coeffs, singular_values = differentiable_pod(snapshots, n_modes=5)

# DMD 分解
modes, eigenvalues, amplitudes = differentiable_dmd(snapshots, n_modes=5, dt=0.005)
```

#### 4.4.5 力系数 DRLinFluids 风格 Loss

参考 DRLinFluids 论文的奖励函数设计：

```python
from myafc.afc_supervision.losses import drl_reward_loss

# DRLinFluids reward: R = cd_baseline - Cd - alpha_lift × |Cl|
# loss = -reward = Cd - cd_baseline + alpha_lift × |Cl| (最大化 reward = 最小化 loss)
loss = drl_reward_loss(
    cd=current_cd,
    cl=current_cl,
    cd_baseline=1.549,    # 无控制 Cd (必须正确设置!)
    alpha_lift=0.1,       # 升力惩罚系数 (论文默认 0.1)
)
```

> **案例**: `cd_baseline` 可设为无控制时的 Cd 基线值！
> - 方柱 Re=100: `cd_baseline ≈ 1.549`
> - 圆柱 Re=100: `cd_baseline ≈ 3.205`

**DRLinFluids 参数配置**:
```yaml
loss:
  force_drl_weight: 1.0      # >0 启用
  cd_baseline: 1.549         # 无控制基线 Cd (**必须正确设置**)
  alpha_lift: 0.1            # 升力惩罚系数 (论文默认 0.1)
  
  # 时间平均配置 (类似 DRLinFluids 论文)
  force_drl_avg_window: action_duration  # 可选: null, 1, 正整数, 'action_duration', 'control_interval'
  force_drl_cl_abs: true                  # true: 平均 |Cl|, false: 平均 Cl
```

**时间平均说明**:

DRLinFluids 论文中使用时间平均的 Cd/Cl 计算 reward，而非瞬时值。

| 参数 | 值 | 说明 |
|------|------|------|
| `force_drl_avg_window` | `null`/`1` | 不平均，使用瞬时值 (默认) |
| | 正整数 `N` | 最近 N 个 env-step 滑动平均 |
| | `'action_duration'` | 自动设为 action_duration |
| | `'control_interval'` | 自动设为 control_interval |
| `force_drl_cl_abs` | `true` | 平均 \|Cl\| (抑制升力振荡) |
| | `false` | 平均 Cl (正负可能抵消) (默认) |

设置 `force_drl_avg_window: action_duration` 和 `force_drl_cl_abs: true` 时，升力项在动作持续窗口内使用升力系数的绝对值。

#### 4.4.6 涡脱抑制 Loss

基于 Cl 历史的 FFT 分析抑制涡脱频率：

```python
from myafc.afc_supervision.losses import vortex_shedding_loss, cl_oscillation_loss

# 涡脱抑制 - 基于 FFT 分析
cl_history = torch.stack([cl_t1, cl_t2, ...])  # Cl 时间序列
loss = vortex_shedding_loss(cl_history, target_frequency=0.0, weight=1.0)

# 简化版 - Cl 振荡幅值
loss = cl_oscillation_loss(cl_history, weight=1.0)
```

**配置参数**:
```yaml
loss:
  force_vortex_weight: 1.0     # >0 启用
  vortex_cl_history_len: 50    # FFT 分析窗口长度
```

#### 4.4.7 Loss 类型汇总表

| 类别 | Loss 名称 | 配置权重参数 | 功能说明 |
|------|----------|-------------|----------|
| **探针** | `probe_mse_loss` | `probe_mse_weight` | 速度追踪 MSE |
| | `probe_mae_loss` | `probe_mae_weight` | 速度追踪 MAE（对异常值鲁棒） |
| **力系数** | `force_coeffs_loss` | `force_weight` | Cd/Cl 目标值 MSE |
| | `drl_reward_loss` | `force_drl_weight` | DRLinFluids 风格 reward |
| | `vortex_shedding_loss` | `force_vortex_weight` | 涡脱频率抑制 |
| **流场** | `field_rms_loss` | `field_rms_weight` | 区域 RMS |
| | `fft_mode_loss` | `field_fft_weight` | FFT 模态幅值 |
| | `fluctuation_sparse_loss` | `field_fluctuation_weight` | 局部波动抑制 |
| **模态** | `pod_mode_loss` | `modal_pod_weight` | 非稳态 POD 模态能量 |
| | `dmd_mode_loss` | `modal_dmd_weight` | 不稳定 DMD 模态幅值 |
| | `steady_state_residual_loss` | `modal_residual_weight` | 稳态残差 |

**流场 Loss 区域配置 (`field_loss_region`)**

`field_loss_region` 用于指定**所有基于流场计算的 Loss** 的计算区域（相对坐标 0~1）。

**适用范围**：
- **直接流场 Loss**: `field_rms_loss`, `fft_mode_loss`, `fluctuation_sparse_loss`
- **模态 Loss**: `pod_mode_loss`, `dmd_mode_loss`, `steady_state_residual_loss`

> 模态 Loss 使用的快照会在存入缓冲区前应用区域裁剪，确保模态分析只在指定区域进行。

**配置格式**：
```yaml
loss:
  # 显式指定区域: [x_start, y_start, x_end, y_end]
  field_loss_region: [0.35, 0.2, 0.9, 0.8]  # 障碍物后方矩形区域
  
  # 或使用自动计算的下游默认区域
  field_loss_region: null  # 默认值，自动计算障碍物下游区域
```

**自动计算逻辑** (当 `field_loss_region: null`)：

根据障碍物配置自动计算下游区域：
- **方形障碍物**: `x_start = center_x + size_x / (2 * Lx) + gap`
- **圆形障碍物**: `x_start = center_x + radius / Lx + gap`
- 默认 `gap = 0.02`，避免边界效应

```python
# 自动计算示例 (方柱, center=[0.25, 0.5], size=[0.05, 0.05], Lx=2.0)
x_right_rel = 0.25 + 0.05 / (2 * 2.0) = 0.2625
x_start = 0.2625 + 0.02 = 0.2825
# 最终 region = [0.2825, 0.0, 1.0, 1.0]
```

**应用场景**：
- 聚焦尾流区域，忽略上游影响
- 避免障碍物边界附近的数值误差
- 减少计算量（仅在感兴趣区域计算）

#### 4.4.8 自定义 Loss

支持通过配置文件加载自定义 Loss 函数：

```yaml
# 配置文件
loss:
  custom_loss_path: "path/to/my_loss.py"
  custom_loss_func: "my_custom_loss"
```

```python
# 自定义函数签名
def my_custom_loss(state, amplitude, target=None, **kwargs):
    # 自定义计算逻辑
    ...
    return loss
```

### 4.5 正则化模块

正则化项用于提升训练稳定性和策略泛化能力。总损失函数：

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \lambda_1 \mathcal{L}_{L2} + \lambda_2 \mathcal{L}_{smooth} + \lambda_3 \mathcal{L}_{stab} - \lambda_{ent} H + \lambda_{kl} D_{KL}$$

#### 4.5.1 正则化项汇总

| 正则化项 | 公式 | 配置参数 | 功能 |
|---------|------|---------|------|
| 幅值 L2 | $\mathcal{L} = a^2$ | `amplitude_l2_weight` | 惩罚过大幅值，鼓励节能 |
| 时序平滑 | $\mathcal{L} = (a_t - a_{t-1})^2$ | `temporal_smoothness_weight` | 抑制控制震荡 |
| 策略稳定性 | $\mathcal{L} = (a_t - \bar{a}_{EMA})^2$ | `policy_stability_weight` | 防止策略过冲 |
| 熵正则化 | $\mathcal{L} = -H(\pi)$ | `entropy_weight` | 保持探索性 |
| KL 散度 | $\mathcal{L} = D_{KL}(\pi_{new} \| \pi_{old})$ | `kl_weight` | 限制策略更新幅度 |

#### 4.5.2 正则化项生效机制

**核心原则**：每个环境步都施加正则化项，但正则化项**数值仅在动作更新步计算，其他步复用缓存张量**。

```python
# _compute_loss 中的正则化缓存逻辑
if self.cfg.loss.entropy_weight > 0:
    if is_net_forward_step:
        # 动作更新步：计算并缓存
        self._cached_reg_entropy = self._compute_entropy_loss()
    # 复用缓存值
    if self._cached_reg_entropy is not None:
        loss = loss - self.cfg.loss.entropy_weight * self._cached_reg_entropy
```

**设计理由**：
1. **梯度链连通**：缓存张量保持计算图引用，梯度可正确反传到 PolicyNet
2. **避免重复计算**：非动作更新步复用缓存，提升效率
3. **显存管理**：每次 `backward()` 后清空缓存

#### 4.5.3 各正则化项详解

##### 幅值 L2 正则化

$$\mathcal{L}_{L2} = a^2$$

- **问题**：策略网络可能输出过大的控制幅值，导致流场剧烈扰动、数值不稳定、控制能耗过高
- **目标**：惩罚过大控制幅值，防止数值不稳定和能耗过高
- **配置**：`amplitude_l2_weight: 0.001`
- **注意**：幅值从 0 开始时该项极小（~1e-10），日志可能显示 0.0000

##### 时序平滑正则化

**标准形式**：
$$\mathcal{L}_{smooth} = (a_t - a_{t-1})^2$$

**带软约束** (`action_rate_limit = r > 0`)：
$$\mathcal{L}_{smooth} = \max(0, |a_t - a_{t-1}| - r)^2$$

- **问题**：相邻时间步的控制信号剧烈变化会导致：1) 流场非物理震荡；2) 训练 loss 剧烈波动难以收敛；3) 实际应用中执行器无法跟随
- **目标**：抑制控制信号剧烈变化，提升训练稳定性和控制可行性
- **配置**：`temporal_smoothness_weight: 0.01`
- **软约束作用**：仅惩罚超过阈值的变化量

##### 策略稳定性正则化 (EMA)

**EMA 更新**：
$$\bar{a}_{EMA} = \alpha \cdot a_t + (1-\alpha) \cdot \bar{a}_{EMA,prev}$$

**Loss**：
$$\mathcal{L}_{stab} = (a_t - \bar{a}_{EMA})^2$$

- **问题**：策略更新过快导致输出短时剧烈波动（过冲），表现为 loss 持续上升后缓慢回落
- **目标**：防止策略"过冲"（loss 先升后降）
- **配置**：`policy_stability_weight: 0.01`, `policy_stability_momentum: 0.1`
- **机制**：EMA 使用 detach 后的值更新，不影响梯度；当前输出保持梯度

##### 熵正则化 (高斯策略)

**高斯分布的熵**：
$$H(\pi) = \sum_{i=1}^{d} \frac{1}{2}\log(2\pi e \sigma_i^2)$$

**Loss** (负熵)：
$$\mathcal{L}_{entropy} = -H(\pi)$$

- **问题**：策略网络过早收敛到确定性极值（标准差趋近于零），丧失探索能力，陷入局部最优
- **目标**：防止策略过早收敛到确定性极值，保持探索能力
- **配置**：`entropy_weight: 0.01`
- **原理**：最小化 Loss = 最大化熵 = 鼓励探索

##### KL 散度正则化 (高斯策略)

$$D_{KL}(\pi_{new} \| \pi_{old}) = \sum_{i=1}^{d} \left[ \log\frac{\sigma_{old,i}}{\sigma_{new,i}} + \frac{\sigma_{new,i}^2 + (\mu_{new,i} - \mu_{old,i})^2}{2\sigma_{old,i}^2} - \frac{1}{2} \right]$$

- **问题**：单次参数更新过大导致策略剧变，引发训练崩溃（loss 爆炸、策略退化、环境/数据分布随 agent 剧烈变化导致训练不稳定）
- **目标**：约束新旧策略分布差异，实现稳定更新
- **配置**：`kl_weight: 0.01`
- **实现**：维护 `policy_net_old` 副本，每次 `optimizer.step()` 前同步权重

**实现要点**：

1. 在动作更新步，缓存 state 副本（`_current_state_for_kl`），确保新旧策略比较时使用相同输入
2. 每次 `optimizer.step()` **前**，将当前网络权重同步到 `policy_net_old`
3. 使用 `policy_net_old` 和缓存的 state 计算旧分布（无梯度），与当前分布计算 KL 散度

**与 PPO 的区别**：
- PPO 是 RL 算法，需要在多个 epoch 中使用旧策略采样的 trajectory
- 本框架是监督学习，直接比较当前网络和上一次更新前网络的输出分布

#### 4.5.4 问题诊断与配置建议

| 问题现象 | 推荐配置 |
|:---|:---|
| Loss 大起大落 | `temporal_smoothness_weight: 0.01` |
| 策略收敛过快/陷入局部最优 | `entropy_weight: 0.01` |
| Loss 持续上升后缓慢回落（过冲） | `policy_stability_weight: 0.01` |
| 训练不稳定/更新过大 | `kl_weight: 0.01` |
| 控制幅值过大 | `amplitude_l2_weight: 0.001` |

**综合配置示例**：
```yaml
loss:
  amplitude_l2_weight: 0.001
  temporal_smoothness_weight: 0.01
  entropy_weight: 0.005
  policy_stability_weight: 0.01
  policy_stability_momentum: 0.1
  kl_weight: 0.01
  action_rate_limit: 0.5        # 可选：超过此值才惩罚
  disable_last_epochs: 5        # 可选：最后 N epoch 禁用正则化
```

**相关参考文献**：

1. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017
2. **TRPO**: Schulman et al., "Trust Region Policy Optimization", ICML 2015
3. **SAC**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", ICML 2018
4. **DDPG**: Lillicrap et al., "Continuous control with deep reinforcement learning", ICLR 2016
5. **Natural Policy Gradient**: Kakade, "A Natural Policy Gradient", NIPS 2001
6. **DAgger**: Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011

---

## 5. 训练流程详解

### 5.1 整体训练流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          JetTrainer.train()                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │  FOR epoch = 1 to num_epochs:                                │     │
│   │  ├─ _reset_env()  # 重置流场、清空历史                         │     │
│   │  │                                                           │     │
│   │  ├─ [Warmup 阶段] env_warmup_envsteps 步                      │     │
│   │  │   FOR step in range(warmup):                              │     │
│   │  │   ├─ solver.step(update_controls=False, apply_jets=False) │     │
│   │  │   └─ agent.update_history(state)                          │     │
│   │  │                                                           │     │
│   │  ├─ _detach_graph()  # 分离计算图                             │     │
│   │  │                                                           │     │
│   │  ├─ [Training 阶段] num_interaction_per_mini_episode 次       │     │
│   │  │   FOR interaction in range(num_interactions):             │     │
│   │  │   ├─ FOR local_step in range(control_interval):           │     │
│   │  │   │   ├─ IF local_step == 0: 更新控制信号                   │     │
│   │  │   │   ├─ apply_jets = (local_step < action_duration)      │     │
│   │  │   │   ├─ solver.step()                                    │     │
│   │  │   │   └─ _compute_loss() → step_losses.append()           │     │
│   │  │   │                                                       │     │
│   │  │   └─ IF accumulated >= envstep_per_value:                 │     │
│   │  │       ├─ weighted_loss = Σ(w_t * L_t)                     │     │
│   │  │       ├─ optimizer.zero_grad()                            │     │
│   │  │       ├─ weighted_loss.backward()                         │     │
│   │  │       ├─ grad_clip (if enabled)                           │     │
│   │  │       ├─ optimizer.step()                                 │     │
│   │  │       ├─ scheduler.step()                                 │     │
│   │  │       └─ _detach_graph()                                  │     │
│   │  │                                                           │     │
│   │  └─ [Epoch 结束] 日志记录、检查点保存                          │     │
│   └──────────────────────────────────────────────────────────────┘     │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  [验证阶段] _run_validation(validation_steps)                │      │
│   └─────────────────────────────────────────────────────────────┘      │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  [保存结果] 训练曲线、最终检查点、结果字典                      │      │
│   └─────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Mini-Episode 结构

```
Mini-Episode (1 Epoch):
┌─────────────────────────────────────────────────────────────────┐
│ [Warmup] → [Interaction 1] → [Interaction 2] → ... → [Int. N]   │
│  (填充历史)   control_interval 步                                │
└─────────────────────────────────────────────────────────────────┘

单个 Interaction 内部:
┌────────────────────────────────────────────────────────────────┐
│ Step 0: 更新控制 → [施加 action_duration 步] → [无控制]          │
└────────────────────────────────────────────────────────────────┘
```

**关键参数关系**:
- `envstep_per_value = num_of_interaction_per_agent_update × control_interval`
- 总环境步数 = `env_warmup_envsteps + num_interaction × control_interval`

#### 5.2.1 ⚠️ Reset 机制（关键）

**每个 epoch 和验证阶段开始时必须调用 `_reset_env()`**，确保所有状态正确重置：

```
_reset_env() 重置内容:
┌─────────────────────────────────────────────────────────────────┐
│ ✓ solver.reset()          # 流场重置到初始状态                   │
│ ✓ agent.reset()           # Agent 历史清空                      │
│ ✓ _state_history_buffer   # 历史状态缓冲清空（envstep_state 用）  │
│ ✓ _policy_output_ema      # 策略稳定性 EMA 清空                  │
│ ✓ _cached_reg_*           # 正则化缓存清空                       │
│ ✓ _prev_amplitude         # 上一步幅值归零                       │
│ ✓ _detach_graph()         # 计算图分离                          │
└─────────────────────────────────────────────────────────────────┘
```

> **⚠️ 重要**: 训练和验证必须使用相同的 reset 逻辑，否则会导致：
> - 验证阶段继承训练阶段的历史状态
> - 控制信号在 epoch 开始时出现异常震荡
> - smoothing_alpha 表现不一致

### 5.3 分段反向传播策略

为避免长 rollout 导致的显存溢出，采用**分段更新 (Truncated BPTT)**：

```
Timeline:
  step:  0   1   2   ... 29  30  31  ...
         │   │   │       │   │   │
  graph: ├───┴───┴───...─┴───┤   │
         │ 计算图累积 30 步   │   │
         └───────────────────┘   │
                     ▼           │
                backward()       │
                optimizer.step() │
                detach(state)    │
                                 │
                                 ├───────────...
                                 │ 新计算图开始
                                 └───────────...

优点:
  1. 显存占用可控（仅保留 envstep_per_value 步的计算图）
  2. 梯度仍能在分段内传播，保留时序依赖信息
  3. 灵活权衡：更长的 interval → 更好的梯度 vs 更多显存
```

从整体时间线看，每个 Mini-Episode 内含多个 interaction，每 `envstep_per_value` 步进行一次神经网络更新：

```
┌───────────────────────────────────────────────────────────────────────┐
│                     一个 Mini-Episode (1 Epoch)                       │
├────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┤
│  Int 0 │  Int 1 │  Int 2 │  ...   │ Int N-1│  ...   │  ...   │  ...   │
├────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┤
│<---- envstep_per_reward (T) ----> │<---- envstep_per_reward (T) ----> │
│           NN Update 1             │           NN Update 2             │
└───────────────────────────────────────────────────────────────────────┘
```

### 5.4 Loss 累积与时间权重

本框架采用**多步累积 Loss → 统一反向传播**的训练模式。每次神经网络更新前，先累积 $T$ 步的 Loss，再进行一次 `backward()`。

#### 5.4.1 核心公式

**总 Loss** (加权累积):
$$\mathcal{L}_{total} = \sum_{t=0}^{T-1} \bar{w}_t \cdot \mathcal{L}_t$$

其中 $\bar{w}_t$ 为第 $t$ 步的时间权重，$\mathcal{L}_t$ 为第 $t$ 步的 Loss（含主 Loss + 正则化）。

**关键参数**:
| 符号 | 配置项 | 含义 |
|:---:|:---|:---|
| $T$ | `envstep_per_value` | 神经网络更新周期（环境步数） |
| $K$ | `control_interval` | 单个 interaction 的环境步数 |
| $D$ | `action_duration` | 射流施加的环境步数 |
| $N$ | $\lceil T/K \rceil$ | interaction 数量 |
| $\gamma$ | `loss_gamma` | 衰减率 |

#### 5.4.2 权重模式

**等权重 (`loss_weights='equal'`)**:
$$\bar{w}_t = \frac{1}{T}$$

---

**指数权重 (`loss_weights='exponential'`)** — 基于 Interaction 结构的分段指数权重:

对于第 $n$ 个 interaction ($n = 0, 1, \ldots, N-1$) 在环境步 $t$ 的权重：

$$W_{n,t} = \begin{cases} 
0, & t < nK \quad \text{(该 interaction 尚未开始)} \\
1, & nK \leq t < nK + D \quad \text{(射流施加中)} \\
\gamma^{t - nK - D + 1}, & t \geq nK + D \quad \text{(持续衰减至周期结束)}
\end{cases}$$

> ⚠️ **关键特点**：每个 interaction 的衰减**持续到 $t=T-1$**，而非仅到该 interaction 结束 $(n+1)K$。

> ⚠️ **D = K 特例**：当 `action_duration = control_interval` 时，每个 interaction 整个周期都处于射流施加阶段（权重=1），衰减仅在该 interaction 结束后开始。由于求 interaction 维度平均，越晚的步有越多 interaction 贡献权重，**权重呈递增趋势**（而非递减），这是正常趋势，是符合预期的。

**最终权重** (在 interaction 维度求平均)：
$$\bar{w}_t = \frac{1}{N} \sum_{n=0}^{N-1} W_{n,t}$$

#### 5.4.3 权重矩阵示例

**符号示例** ($T=30, K=10, D=5, \gamma, N=3$):

```
                 t=0  1  2  3  4  5  6  7  8  9 10 11 ... 19 20 21 ... 29
               ┌─────────────────────────────────────────────────────────
 Interaction 0 │ 1  1  1  1  1  γ¹ γ² γ³ γ⁴ γ⁵ γ⁶ γ⁷ ... γ¹⁵ γ¹⁶ γ¹⁷ ... γ²⁵
 Interaction 1 │ 0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  γ¹ γ² ... γ¹⁵
 Interaction 2 │ 0  0  0  0  0  0  0  0  0  0  0  0  ... 0  1  1  1  1  1  γ¹ γ² γ³ γ⁴ γ⁵
               └─────────────────────────────────────────────────────────
 平均后 w̄_t    │ ←─── 逐步叠加，在 interaction 维度平均 ───→
```

**具体数值示例** ($\gamma=0.9$):

```
        t: 0   1   2   3   4   5   6   7   8   9  10  11  ... 19  20  21  ... 29
      ┌────────────────────────────────────────────────────────────────────────
Int 0 │ 1   1   1   1   1  .90 .81 .73 .66 .59 .53 .48 ... .21 .19 .17 ... .07
Int 1 │ 0   0   0   0   0   0   0   0   0   0   1   1  ... .59 .53 .48 ... .21
Int 2 │ 0   0   0   0   0   0   0   0   0   0   0   0  ...  0   1   1  ... .59
      └────────────────────────────────────────────────────────────────────────
 Avg  │.33 .33 .33 .33 .33 .30 .27 .24 .22 .20 .51 .49 ... .27 .57 .55 ... .29
```

#### 5.4.4 伪代码

```python
def compute_weights(T, K, D, gamma):
    """计算 Interaction-based 时间权重"""
    N = ceil(T / K)
    W = zeros(N, T)
    for n in range(N):
        start = n * K
        for t in range(start, T):  # 持续到 T，非 (n+1)*K
            local_t = t - start
            W[n, t] = 1.0 if local_t < D else gamma ** (local_t - D + 1)
    return mean(W, axis=0)  # interaction 维度平均

# 累积 Loss
losses = []
for t in range(T):
    L_t = compute_loss(state_t)  # 主 Loss + 正则化
    losses.append(L_t)

# 加权求和
weights = compute_weights(T, K, D, gamma)
total_loss = sum(w * L for w, L in zip(weights, losses))

# 反向传播
total_loss.backward()
optimizer.step()
```

#### 5.4.5 不满数据处理

当 episode 结束时累积步数 $T' < T$ 时：
1. 实际 interaction 数：$N' = \lceil T'/K \rceil$
2. 重新计算权重矩阵 $W' \in \mathbb{R}^{N' \times T'}$
3. 应用相同的衰减公式，仅范围缩小到 $[0, T'-1]$

### 5.5 训练/验证一致性机制

训练和验证阶段的行为一致性是正确评估模型性能的关键。本节详解两阶段的异同及实现机制。

#### 5.5.1 核心设计：共享方法 + 参数区分

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                      训练/验证一致性架构                                       │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   共享方法层（训练/验证行为一致）                                               │
│   ┌─────────────────────────────────────────────────────────────────────────┐ │
│   │  _env_step()           # 控制时机判断、状态构建                           │ │
│   │  _do_solver_step()     # 仿真推进、历史更新                               │ │
│   │  _update_force_buffers()  # 统一力系数 buffer 更新                        │ │
│   │  _build_state()        # 状态字典构建                                    │ │
│   │  _compute_primary_loss()  # 主 Loss 计算（无正则化）                      │ │
│   └─────────────────────────────────────────────────────────────────────────┘ │
│                                  ↓                                            │
│   差异化参数层（reason='epoch' vs 'validation'）                              │
│   ┌─────────────────────────────────────────────────────────────────────────┐ │
│   │  _reset_env(reason)    # 根据 reason 区分初始化策略                       │ │
│   │  _compute_loss()       # 训练专用：primary + reg                          │ │
│   │  backward/optimizer    # 训练专用：参数更新                               │ │
│   └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

#### 5.5.2 关键方法调用对比

| 执行阶段 | 训练 (`_train_single_epoch`) | 验证 (`_run_single_validation`) |
|----------|------------------------------|--------------------------------|
| 环境重置 | `_reset_env('epoch')` | `_reset_env('validation')` |
| Warmup | ✅ 使用 `_env_step(is_warmup=True)` | ✅ 相同 |
| Buffer 更新 | ✅ 调用 `_update_force_buffers()` | ✅ 相同 |
| 控制时机 | ✅ `_env_step()` 判断 | ✅ 相同 |
| Loss 计算 | `_compute_loss()` (含正则化) | `_compute_primary_loss()` (仅主 Loss) |
| Backward | ✅ 每 `envstep_per_value` 步 | ❌ `torch.no_grad()` |
| 计算图 | 每次 backward 后 `_detach_graph()` | 无需分离 |
| Runner 力系数 | ❌ 不使用 | ✅ `runner._record_force_coeffs()` 每步记录 |
| Runner 探针采样 | ❌ 不使用 | ✅ `runner._record_probes()` 每步采样所有探针 |
| Runner 可视化 | ✅ `_record_training_visual()` 按 interval | ✅ `runner._save_visual()` 按 interval |

#### 5.5.3 `_update_force_buffers()` 统一更新机制

此方法确保 DRL 时间平均和涡脱 FFT 在训练/验证阶段行为完全一致：

```python
def _update_force_buffers(self, cd=None, cl=None):
    """统一更新所有力系数相关的 history buffer
    
    调用时机: 每个 env-step 执行 solver.step() 后
    作用范围:
      - DRL 时间平均 buffer (_drl_cd/cl_history_buffer)
      - 涡脱 FFT buffer (_cl_history_buffer)
    """
    loss_cfg = self.cfg.loss
    need_update = (loss_cfg.force_drl_weight > 0 or loss_cfg.force_vortex_weight > 0)
    
    if not need_update:
        return
    
    # 按需计算力系数（使用缓存方法）
    if cd is None or cl is None:
        cd, cl = self._get_force_coeffs()
    
    # 更新 DRL buffer
    if loss_cfg.force_drl_weight > 0:
        self._update_drl_force_history(cd, cl)
    
    # 更新涡脱 FFT buffer
    if loss_cfg.force_vortex_weight > 0:
        self._update_cl_history_buffer(cl)
```

**调用位置**:
- 训练阶段: 每步 `_do_solver_step()` 后
- 验证阶段: 预热阶段 + 正式验证阶段（每步都调用）

#### 5.5.4 `_get_force_coeffs()` 统一力系数获取

此方法为训练和验证阶段提供**统一的力系数获取接口**，带缓存以避免重复计算：

```python
def _get_force_coeffs(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取当前力系数（带缓存）
    
    在同一 env-step 内多次调用时复用缓存。
    缓存在 _do_solver_step() 后自动清空。
    
    Returns:
        (cd, cl): 阻力系数和升力系数
    """
    if self._cached_force_coeffs is None:
        cd, cl, _, _, _, _ = self.solver.compute_force_coeffs()
        self._cached_force_coeffs = (cd, cl)
    return self._cached_force_coeffs
```

**缓存机制**:
```
┌─────────────────────────────────────────────────────────────────┐
│ 每个 env-step 内:                                                │
│   1. _do_solver_step() 执行后清空 _cached_force_coeffs          │
│   2. 首次调用 _get_force_coeffs() → 计算并缓存                   │
│   3. 后续调用 → 复用缓存 (避免重复计算)                          │
└─────────────────────────────────────────────────────────────────┘
```

**训练/验证一致性**: 训练和验证阶段都使用此方法获取 Cd/Cl，确保数据一致：
- 训练: `_step_cd_record.append(cd)` 
- 验证: `val_cd.append(cd)`

#### 5.5.5 Loss 计算的区别

**训练阶段**: 使用 `_compute_loss()` 返回三元组

```python
# 返回: (total_loss, primary_loss, reg_loss)
total_loss, primary_loss, reg_loss = self._compute_loss(
    current_vel, amplitude, policy_output, is_update_step
)
# total_loss = primary_loss + reg_loss
```

**验证阶段**: 使用 `_compute_primary_loss()` 仅计算主 Loss

```python
loss = self._compute_primary_loss(current_velocity=(u, v))
# 无正则化项，确保验证 loss 与训练主 loss 可比
```

**设计理由**：
- 正则化项用于**训练稳定性**，不应影响**模型性能评估**
- 验证 loss 应反映策略的**实际控制效果**
- 两者使用相同的 `_compute_primary_loss()` 保证主 Loss 计算逻辑一致

#### 5.5.6 初始化差异 (`_reset_env`)

| 特性 | 训练 (`reason='epoch'`) | 验证 (`reason='validation'`) |
|------|------------------------|------------------------------|
| 流场初始化 | 支持随机帧 (`solver_initial_resume_step`) | 固定使用 `resume_step` |
| 噪声增强 | 由 `noise_envs` 控制 | 由 `validation_noise_enabled` 控制 (默认 `false`) |
| 缓存清理 | ✅ 清空所有 buffer | ✅ 相同 |

```python
def _reset_env(self, reason: str = 'epoch'):
    # 判断是否使用随机帧初始化（仅训练阶段）
    use_random_resume = (
        reason == 'epoch' and 
        cfg.solver_initial_resume_step is not None and
        self._has_valid_resume_from()
    )
    
    # 噪声增强：根据配置决定是否施加
    apply_noise = (
        cfg.noise_std > 0 and (
            (reason == 'epoch' and cfg.noise_envs) or
            (reason == 'validation' and cfg.validation_noise_enabled)
        )
    )
```

#### 5.5.7 时序流程对比图

```
训练阶段 (_train_single_epoch)
═══════════════════════════════════════════════════════════════════════════
│ _reset_env('epoch')                                                      │
│   ├─ 随机帧初始化 (若配置)                                                │
│   └─ 噪声增强 (若 noise_envs=true)                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ [Warmup]                                                                 │
│   for step in range(env_warmup_envsteps):                                │
│       _env_step(is_warmup=True)                                          │
│       _do_solver_step()                                                  │
│       _update_force_buffers()  ← 统一方法                                 │
├──────────────────────────────────────────────────────────────────────────┤
│ _detach_graph()                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ [Training]                                                               │
│   for interaction in range(num_interaction):                             │
│       for local_step in range(control_interval):                         │
│           _env_step(is_warmup=False)                                     │
│           agent.compute_action() if is_update                            │
│           _do_solver_step()                                              │
│           _update_force_buffers()  ← 统一方法                             │
│           _compute_loss()  ← 训练专用: primary + reg                     │
│       if accumulated >= envstep_per_value:                               │
│           backward + optimizer.step                                      │
│           _detach_graph()                                                │
═══════════════════════════════════════════════════════════════════════════

验证阶段 (_run_single_validation)
═══════════════════════════════════════════════════════════════════════════
│ _reset_env('validation')                                                 │
│   ├─ 固定帧初始化 (resume_step)                                          │
│   └─ 噪声增强 (若 validation_noise_enabled=true)                          │
├──────────────────────────────────────────────────────────────────────────┤
│ [Warmup] - 与训练完全相同                                                 │
│   for step in range(env_warmup_envsteps):                                │
│       _env_step(is_warmup=True)                                          │
│       _do_solver_step()                                                  │
│       _update_force_buffers()  ← 统一方法                                 │
├──────────────────────────────────────────────────────────────────────────┤
│ _detach_graph()                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ [Validation] with torch.no_grad():                                       │
│   for local_step in range(validation_steps - warmup):                    │
│       _env_step(is_warmup=False)                                         │
│       agent.compute_action() if is_update                                │
│       _do_solver_step()                                                  │
│       _update_force_buffers()  ← 统一方法                                 │
│       _compute_primary_loss()  ← 仅主 Loss，无正则化                      │
═══════════════════════════════════════════════════════════════════════════
```

#### 5.5.8 训练/验证一致性的关键保障

1. **Buffer 更新统一**: `_update_force_buffers()` 方法在训练和验证的每个 env-step 都调用
2. **Warmup 行为一致**: 两阶段都使用相同的 warmup 逻辑填充 history buffer
3. **主 Loss 共享**: `_compute_primary_loss()` 被 `_compute_loss()` 内部调用，确保计算逻辑一致
4. **控制时机相同**: 两阶段都使用 `_env_step()` 判断控制信号更新时机
5. **重复计算消除**: 验证阶段 runner 记录在前、缓存复用在后，每步仅调用一次 `solver.compute_force_coeffs()`、目标探针仅插值一次

### 5.6 梯度流与可微分机制

#### 5.6.1 梯度传递路径

```
                                 ┌─────────────────────────────────┐
                                 │        Loss Function            │
                                 │   L = MSE(u_probe, u_target)    │
                                 └───────────────┬─────────────────┘
                                                 │ ∂L/∂u_probe
                                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         Probe Sampling                                  │
│   u_probe = field.sample(velocity_field, probe_position)               │
│   (PhiFlow 可微分采样)                                                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ ∂L/∂velocity_field
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         PhiFlow Solver Step                             │
│   velocity_{t+1} = advect.semi_lagrangian(velocity_t, ...)             │
│   velocity_{t+1}, pressure = fluid.make_incompressible(...)            │
│   (可微分 PDE 求解器)                                                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ ∂L/∂velocity_t
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         Control Application                             │
│   velocity_t = apply_jet_bc(velocity_{t-1}, amplitude)                 │
│   (可微分边界条件施加)                                                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ ∂L/∂amplitude
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         AFCAgent (MLP)                                  │
│   amplitude = policy(state)                                            │
│   (标准 PyTorch 网络)                                                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ ∂L/∂θ (Agent 参数梯度)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         Optimizer Update                                │
│   θ_{new} = θ_{old} - lr * ∂L/∂θ                                       │
└────────────────────────────────────────────────────────────────────────┘
```

#### 5.6.2 可微分性保证要点

```python
# ❌ 错误: 断开梯度链
obs = sample_probes(state).detach()  # detach 会阻断梯度

# ✓ 正确: 保持梯度流
obs = sample_probes(state)  # 不要 detach

# ❌ 错误: 使用非可微分操作
action_np = action.numpy()  # 转 numpy 会断开梯度

# ✓ 正确: 全程使用 PyTorch 张量
velocity = apply_control_torch(velocity, action)

# ❌ 错误: 原地修改张量
state.velocity.values = new_values  # 可能断开梯度

# ✓ 正确: 创建新张量
velocity_new = velocity + delta  # 保持梯度
```

#### 5.6.3 计算图管理

```python
def _detach_graph(self):
    """分离计算图，释放显存
    
    每次 backward 后调用，截断梯度链
    """
    from phi.math import stop_gradient
    
    self.solver.velocity = stop_gradient(self.solver.velocity)
    self.solver.pressure = stop_gradient(self.solver.pressure)
    
    # 清空 step_losses 列表
    self._step_losses.clear()
```

---

## 6. 配置参数详解

### 6.1 配置文件结构

```yaml
env_config_path: "cfgs4env/xxx.yaml"  # 环境配置路径

training: { ... }       # 训练参数
agent: { ... }          # 网络参数
input_features: { ... } # 输入特征配置
inlet: { ... }          # 入口边界配置
loss: { ... }           # Loss 配置
output: { ... }         # 输出配置
```

### 6.2 训练参数 (training)

#### 6.2.1 训练循环

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `epochs` | int | 100 | 训练轮数 |
| `env_warmup_envsteps` | int | 50 | 预热步数 (≥ envstep_state - 1) |
| `action_duration` | int | 26 | 射流持续步数 |
| `control_interval` | int/str | 26 | 控制更新间隔，支持 `'equal2action_duration'` |
| `num_of_interaction_per_agent_update` | int | 2 | 每次更新累积的 interaction 数 |
| `num_interaction_per_mini_episode` | int | 50 | 每 epoch 的 interaction 数 |

#### 6.2.2 优化器

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `optimizer` | str | 'adam' | adam / adamw / sgd |
| `learning_rate` | float | 0.001 | 初始学习率 |
| `optimizer_momentum` | float | 0.937 | SGD 动量 (YOLOv8 默认) |
| `optimizer_weight_decay` | float | 0.0005 | 权重衰减 |
| `grad_clip` | float | 1.0 | 梯度裁剪 (0=不裁剪) |

#### 6.2.3 学习率调度

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `lr_schedule` | str | 'cosine' | constant / linear / cosine |
| `lr_final` | float | 0.01 | 最终 LR 比例 |
| `warmup_updates` | int | 10 | LR 预热次数 |
| `warmup_bias_lr` | float | 0.1 | 预热初始 LR 比例 |
| `warmup_momentum` | float | 0.8 | 预热初始动量 |

#### 6.2.4 多环境并行训练

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `num_envs` | int | 1 | 并行环境数 |
| `env_variants` | list | null | 各环境变体的待覆盖配置 |

> **⚠️ 多环境并行训练详解**
>
> 多环境并行（Multi-Environment Parallel Training）通过 `ParallelSolverManager` 实现，可同时运行多个具有不同配置的 CFD 仿真环境，提升训练鲁棒性和泛化能力。
>
> **核心语义**：
> - `num_envs=1`: 仅使用基础环境（来自 `env_config_path`），不启用并行
> - `num_envs=n` (n>1): 1 个基础环境 + (n-1) 个派生环境 = n 个并行环境
> - `env_variants`: 长度必须为 `n_envs-1`，为每个派生环境提供配置覆盖
>
> **配置示例**：
> ```yaml
> training:
>   num_envs: 4
>   env_variants:
>     # Env 1: 不同入口速度
>     - solver:
>         boundary:
>           velocity:
>             x-: { type: inflow, value: [2.1, 0.0] }
>     # Env 2: 不同黏度
>     - solver:
>         fluid:
>           viscosity: 0.002
>     # Env 3: 不同障碍物位置
>     - solver:
>         obstacle:
>           center: [0.30, 0.50]
> ```
>
> **可覆盖的配置项**（任意 solver YAML 中的参数均可覆盖）：
> | 类别 | 示例参数 |
> |------|---------|
> | 边界条件 | `boundary.velocity.x_minus.value`, `boundary.velocity.x_plus.type` |
> | 物理参数 | `fluid.viscosity`, `fluid.density` |
> | 障碍物几何 | `obstacle.center`, `obstacle.size`, `obstacle.radius` |
> | 射流参数 | `jets.nozzles`, `jets.smoothing_alpha` |
> | 初始条件 | `initial.resume_from`, `initial.resume_step` |
>
> **Loss 聚合机制**：
> ```
> ┌─────────────────────────────────────────────────────────────────┐
> │  total_loss = 0                                                 │
> │  for env in [Env 0, Env 1, ..., Env n-1]:                       │
> │      env_loss = accumulate_loss_over_envsteps(env)              │
> │      total_loss += env_loss                                     │
> │  total_loss /= num_envs  # 跨环境平均                            │
> │  total_loss.backward()                                          │
> └─────────────────────────────────────────────────────────────────┘
> ```
>
> **Reset 逻辑**：
>
> Solver 的 `reset()` 方法统一了所有重置场景，通过 `use_h5` 参数控制初始化来源：
>
> ```python
> # 从 h5 文件重置（默认）
> solver.reset(use_h5=True, step=-1)   # 加载指定帧
> 
> # 从配置重置（零初始化 + 边界修正）
> solver.reset(use_h5=False)           # 不从 h5 加载
> ```
>
> - 单环境 (`num_envs=1`): `self.solver.reset(use_h5, step)` 直接调用 `KarmanTorchSolver.reset()`
> - 多环境 (`num_envs>1`): `self.solver` 是 `ParallelSolverManager`，其 `reset(use_h5, steps)` 会遍历所有子环境执行重置
>
> **注意事项**：
> - 单环节和多环境的 reset 方法的参数名存在差异：`step` vs `steps`，请注意区分
> - PhiFlow 存在全局状态（JIT 缓存），多线程并行可能不安全，建议使用 `parallel=False` 顺序执行
> - 各环境的梯度独立计算，最后在 loss 层面平均后统一 backward
> - 显存消耗随 `num_envs` 线性增加，请根据 GPU 容量调整

#### 6.2.5 噪声增强（数据增强）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `noise_envs` | bool | false | 是否启用初始流场噪声增强 |
| `noise_std` | float | 0.01 | 噪声标准差（相对于参考速度） |
| `noise_observation` | bool | false | 是否向观测探针添加噪声 |
| `validation_noise_enabled` | bool | false | 是否在验证阶段也施加噪声增强 |

> **⚠️ 噪声增强详解 (对多环境并行训练模式同样提供支持)**
>
> 噪声增强（Noise Augmentation / Data Augmentation）通过向初始流场和/或观测数据添加随机噪声，提升策略网络对扰动的鲁棒性。
>
> **1. 初始流场噪声 (`noise_envs`)**
>
> 在每个 epoch 开始时的 `_reset_env()` 中，向初始速度场添加高斯噪声：
> ```python
> def _apply_noise_to_velocity_field(self):
>     noise_std = self.training_cfg.noise_std
>     ref_velocity = max(abs(inflow_velocity_x), 0.1)  # 参考速度
>     
>     # 向 u, v 分量添加高斯噪声
>     vel_x += randn_like(vel_x) * noise_std * ref_velocity
>     vel_y += randn_like(vel_y) * noise_std * ref_velocity
> ```
>
> **默认行为**：噪声仅在 `reason='epoch'` 时施加（训练阶段），验证阶段 (`reason='validation'`) 默认不施加噪声。
>
> **验证阶段噪声 (`validation_noise_enabled`)**：
> 当设置 `validation_noise_enabled: true` 时，验证阶段也会施加与训练阶段相同的噪声增强。这在以下场景中有用：
> - 评估策略在扰动环境下的鲁棒性
> - 需要验证和训练保持一致的噪声条件
> - 测试策略对噪声的敏感度
>
> **2. 观测噪声 (`noise_observation`)**
>
> 在 `_build_state()` 中向采样的探针速度添加噪声：
> ```python
> if self.cfg.training.noise_observation:
>     u = u + torch.randn_like(u) * noise_std
>     v = v + torch.randn_like(v) * noise_std
> ```
>
> **配置示例**：
> ```yaml
> training:
>   noise_envs: true             # 启用初始流场噪声
>   noise_std: 0.05              # 噪声标准差 (5% 参考速度)
>   noise_observation: true      # 启用观测噪声（可选）
>   validation_noise_enabled: false  # 是否在验证阶段也施加噪声（默认禁用）
> ```
>
> **使用场景**：
> | 场景 | 推荐配置 |
> |------|---------|
> | 提升泛化能力 | `noise_envs: true, noise_std: 0.01~0.05` |
> | 模拟传感器噪声 | `noise_observation: true, noise_std: 0.01` |
> | 组合使用 | 两者可同时启用，进一步增强鲁棒性 |
>
> **注意事项**：
> - `noise_std` 过大可能导致仿真不稳定或策略难以收敛
> - 建议从小值开始（0.01），逐步增加
> - 与 `num_envs` 配合使用效果更佳（每个环境有不同的噪声实例）

#### 6.2.6 随机初始帧（训练多样性增强）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `solver_initial_resume_step` | int/list | null | 初始帧索引或范围 `[min, max]` |
| `sync_resume_step_across_envs` | bool | true | 多环境是否统一使用同一帧 |

> **⚠️ 随机初始帧详解**
>
> 随机初始帧（Random Initial Frame）功能允许在每个 epoch 开始时从预设范围内随机选择 h5 文件中的帧作为初始流场，增加训练数据多样性。
>
> **1. 参数说明**
>
> | 值 | 效果 |
> |---|------|
> | `null` | 禁用，使用 solver 配置的 `resume_step` （若为 `null` 则自动取最后一帧） |
> | `int` | 固定使用指定帧（如 `100` 或 `-1`） |
> | `[min, max]` | 从范围内随机选择（如 `[-2000, -1]`） |
>
> **负索引支持**：`-1` 表示最后一个物理时间步，`-100` 表示从最后物理时间步向前倒数第 100 个时间步。

> **语义细节（关键）**：
> - 负端点先转换为“物理时间步边界”。
> - 若 h5 为跳间隔保存（如仅保存 `10, 20, 30, ...`），边界可落在未保存步上，最终在物理边界区间内筛选已保存帧。
>
> **混合/逆序端点支持**：`[100, -100]` 这类配置会先将负端点解析为真实步号，再自动归一化为 `[low, high]` 后筛选可用步。
>
> **2. 边界处理**
>
> 当随机区间边界内不存在任何已保存帧时：
> ```
> 示例：h5 保存步为 `10,20,...,2000`，配置区间为 `[11, 19]`
> → 该物理区间内无已保存帧
> → 警告并回退使用首个可用物理步
> ```
>
> **3. 多环境行为**
>
> | `sync_resume_step_across_envs` | 行为 |
> |------|------|
> | `true` (默认) | 所有子环境使用相同的随机帧 |
> | `false` | 每个子环境独立随机选择不同帧 |
>
> **4. 执行顺序**
>
> ```
> _reset_env('epoch') 执行顺序:
> ┌──────────────────────────────────────────────────────────────────┐
> │ 1. 随机初始帧选择 (solver_initial_resume_step)                   │
> │    ↓                                                             │
> │ 2. 噪声增强 (noise_envs) - 向选定帧的流场添加噪声                 │
> │    ↓                                                             │
> │ 3. Agent 状态重置                                                │
> │    ↓                                                             │
> │ 4. 缓存清理                                                      │
> └──────────────────────────────────────────────────────────────────┘
> ```
>
> **5. 配置示例**
>
> ```yaml
> training:
>   # 从最后 2000 帧中随机选择
>   solver_initial_resume_step: [-2000, -1]
>   sync_resume_step_across_envs: true
>   
>   # 与噪声增强配合使用，进一步增加多样性
>   noise_envs: true
>   noise_std: 0.02
> ```
>
> **6. 前置条件**
>
> - 必须在 solver 配置中设置 `initial.resume_from` 指向有效的 h5 文件
> - 若 `resume_from` 未设置或为 `null`，此功能静默跳过
>
> **7. 训练/验证模式差异**
>
> | 模式 | 流场初始化来源 |
> |------|----------------|
> | 训练 (`reason='epoch'`) | 若配置了 `solver_initial_resume_step`，支持从该范围随机选帧；否则使用 `env_config` 的 `resume_step` |
> | 验证 (`reason='validation'`) | `env_config` 中的 `initial.resume_step`，若为 null 则默认使用 -1 |
>
> ```
> 训练模式:
>   [INFO] [JetTrainer] 随机帧初始化 | 帧: 1330
>   [INFO] [Solver] 从 h5 恢复 | 路径: .../fields.h5 | 时间步: 1330
>
> 验证模式:
>   [INFO] [JetTrainer] 开始验证 (20 步)...
>   [INFO] [Solver] 从 h5 恢复 | 路径: .../fields.h5 | 时间步: 2000  ← 使用 resume_step=-1
> ```
>
> **注意**：默认情况下验证阶段不会施加噪声增强（`noise_envs`），确保验证结果可重复。若需要在验证阶段也施加噪声（如测试鲁棒性），可设置 `validation_noise_enabled: true`。

#### 6.2.7 全局参数（覆盖 env_config）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `smoothing_alpha` | float | null | 控制信号时间平滑系数，覆盖 solver 配置 |

> **⚠️ 全局参数说明**:
> - `smoothing_alpha`: 设置后**同时作用于训练和验证阶段**
> - 若为 `null`，则使用 `env_config_path` 中 `solver.jets.smoothing_alpha` 的值
> - 值为 `1.0` 表示不平滑（原始输出直接施加）
> - 值为 `0.4` 表示 60% 历史 + 40% 当前（较强平滑）

### 6.3 Loss 配置 (loss)

#### 6.3.1 主 Loss

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `probe_mse_weight` | float | 1.0 | 探针 MSE 权重 |
| `probe_mae_weight` | float | 0.0 | 探针 MAE 权重 |
| `force_weight` | float | 0.0 | 力系数权重 |
| `target_probe_position` | [x, y] | [0.2616, 0.5236] | 目标探针位置 (相对坐标) |
| `target_velocity` | [u, v] | [0.0, 0.233] | 目标速度 |
| `cd_target` | float | 0.0 | Cd 目标值 |
| `cl_target` | float | 0.0 | Cl 目标值 |

#### 6.3.2 时间权重

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `loss_weights` | str | 'exponential' | equal / exponential |
| `loss_gamma` | float | 0.99 | 指数衰减因子 γ |

#### 6.3.3 正则化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `amplitude_l2_weight` | float | 0.0 | 幅值 L2 正则化 |
| `temporal_smoothness_weight` | float | 0.01 | 时序平滑正则化 |
| `action_rate_limit` | float | null | 动作变化率软约束 |
| `policy_stability_weight` | float | 0.0 | EMA 策略稳定性 |
| `policy_stability_momentum` | float | 0.1 | EMA 动量 |
| `entropy_weight` | float | 0.01 | 熵正则化 |
| `kl_weight` | float | 0.0 | KL 散度正则化 |
| `disable_regularization_last_epochs` | int / str | 0 | 最后 N 个 epoch 禁用所有正则化项 |

> **⚠️ `disable_regularization_last_epochs` 说明**:
> - 在最后 N 个 epoch 禁用所有正则化项，让策略充分收敛
> - 取值: `0` = 不禁用（默认），`n` = 最后 n 个 epoch 禁用，`'all'` = 所有 epoch 禁用
> - 当 n ≥ epochs 时等效于 `'all'`

### 6.4 Agent 配置 (agent)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `hidden_dims` | list | [512, 512] | 隐藏层维度 |
| `amplitude_max` | float | 4.0 | 输出幅值上限 |
| `angle_deg_fixed` | float | 90.0 | 固定角度 (null=网络输出) |
| `action_scale` | str | 'softsign' | tanh / clamp / softsign |
| `relative_action` | bool | false | 相对动作模式 |
| `deterministic` | bool | false | 确定性策略 |
| `pretrained_path` | str | null | 预训练权重路径 (见 §4.1.7) |

> **`pretrained_path` 说明**:
> - 支持相对路径（相对于 `env_config_path` 目录）或绝对路径
> - 网络结构必须与预训练模型匹配
> - 仅加载权重，不恢复训练状态（epoch、optimizer 等）
> - 详见 §4.1.7 预训练权重加载

### 6.5 输入特征配置 (input_features)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `main_input_type` | str | 'target_probe' | null / target_probe / probes |
| `probe_fields` | list | ['u', 'v', 'p'] | 探针字段 |
| `include_target_velocity` | bool | true | 包含目标速度 |
| `include_prev_amplitude_num` | int | 1 | 历史幅值数量 |
| `include_force_coeffs` | bool | false | 包含力系数 |
| `envstep_state` | int | 1 | 历史状态堆叠步数 |

> **⚠️ `envstep_state` 参数作用范围说明**:
> 
> `envstep_state` 控制历史状态堆叠的步数，其**同时作用于**：
> 1. **main_input_type** 指定的主输入类型（target_probe / probes）
> 2. **include_force_coeffs=true** 时的力系数 `[Cd, Cl]`
> 
> 例如 - 堆叠后的输入向量结构：
> ```
> [target_velocity] + [prev_amplitudes] + [probe_t-2, force_t-2] + [probe_t-1, force_t-1] + [probe_t, force_t]
>                                         └─────────────────────────────────────────────────────────────────┘
>                                                        envstep_state=3 时的历史堆叠部分
> ```
> 
> **注意**：预热步数 `env_warmup_envsteps` 应 ≥ `envstep_state - 1`，以确保历史缓冲区填充完毕。

### 6.6 入口边界配置 (inlet)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `mode` | str/null | `null` | 边界模式 (见下表) |
| `fixed_velocity` | [u, v] | [1.0, 0.0] | fixed 模式速度 |
| `amplitude` | float | 0.5 | sinusoidal 振幅 |
| `frequency` | float | 0.1 | sinusoidal 频率 (Hz) |
| `phase_deg` | float | 0.0 | sinusoidal 相位 (度) |
| `offset` | [u, v] | [1.0, 0.0] | sinusoidal 偏移 |

**`mode` 可选值**:

| 值 | 说明 |
|-----|------|
| `null` / `~` | 不覆盖，使用 `env_config_path` 中的边界配置 (**默认**) |
| `'fixed'` | 固定速度入口 |
| `'zero_gradient'` | 零梯度边界 |
| `'sinusoidal'` | 正弦变化速度入口 |

---

## 7. API 参考

### 7.1 JetTrainer

```python
class JetTrainer(BaseTrainer):
    """射流 AFC 训练器"""
    
    def __init__(self, config: Union[str, Path, Dict, JetTrainerConfig]) -> None:
        """
        初始化训练器
        
        Args:
            config: 配置来源 (YAML 路径 / Dict / JetTrainerConfig)
        """
    
    def train(self) -> Dict[str, Any]:
        """
        执行训练
        
        Returns:
            {
                'final_loss': float,       # 最终 Loss
                'loss_history': List[float],
                'exp_dir': Path,           # 实验目录
                'best_loss': float,
                'total_steps': int,
            }
        """
    
    def close(self) -> None:
        """清理资源 (TensorBoard writer 等)"""
```

### 7.2 AFCAgent

```python
class AFCAgent(MLPPolicy):
    """AFC 策略网络"""
    
    def __init__(
        self,
        input_features_cfg: InputFeaturesConfig,
        hidden_dims: List[int],
        amplitude_max: float = 4.0,
        angle_deg_fixed: Optional[float] = 90.0,
        action_scale: str = 'softsign',
        relative_action: bool = False,
        deterministic: bool = False,
    ) -> None:
        """初始化"""
    
    def forward(self, state: Dict) -> PolicyNetOutput:
        """
        前向传播
        
        Args:
            state: 状态字典 (来自 _build_state())
        
        Returns:
            PolicyNetOutput(amplitude, angle_deg, mean, log_std)
        """
    
    def update_history(self, state: Dict) -> None:
        """更新历史状态缓冲区"""
    
    def clear_history(self) -> None:
        """清空历史缓冲区"""
```

### 7.3 PolicyNetOutput

```python
@dataclass
class PolicyNetOutput:
    """策略网络输出"""
    amplitude: torch.Tensor   # 控制幅值
    angle_deg: torch.Tensor   # 角度 (度)
    mean: torch.Tensor        # 原始均值
    log_std: torch.Tensor     # 对数标准差
```

---

## 附录A. JetTrainer 功能清单

JetTrainer 提供完整的可微分 AFC 训练功能，涵盖以下特性：

### A.1 训练循环
- ✅ 多 epoch 训练
- ✅ env_warmup_envsteps (环境预热步数)
- ✅ `action_duration` 控制持续步数
- ✅ `control_interval` 控制信号更新周期 (包含语法糖 `'equal2action_duration'`)
- ✅ `num_of_interaction_per_agent_update` 累积更新神经网络
- ✅ `num_interaction_per_mini_episode` 每 epoch interaction 数

### A.2 多环境并行
- ✅ ParallelSolverManager 支持
- ✅ env_variants 配置变体

### A.3 损失函数
- ✅ probe_mse / probe_mae
- ✅ force_coeffs (Cd, Cl)
- ✅ 时间权重 (equal / exponential / interaction-based)

### A.4 正则化
- ✅ amplitude_l2 (幅值 L2)
- ✅ temporal_smoothness (时序平滑)
- ✅ action_rate_limit (动作变化率软约束)
- ✅ policy_stability (EMA 策略稳定性)
- ✅ entropy (熵正则化)
- ✅ kl_weight (KL 散度)
- ✅ disable_last_epochs (最后 N epoch 禁用正则化)

### A.5 优化器
- ✅ YOLOv8 风格参数分组 (bn/bias 无 weight_decay)
- ✅ 学习率调度 (constant / linear / cosine)
- ✅ warmup 支持 (warmup_updates, warmup_bias_lr, warmup_momentum)

### A.6 高级特性
- ✅ EMA 模型 (指数移动平均)
- ✅ AMP (混合精度训练)
- ✅ grad_clip (梯度裁剪)
- ✅ importance_sampling (重要性采样 - PPO 风格)
- ✅ sinusoidal 入口边界 (配置覆盖)
- ✅ smoothing_alpha (配置覆盖 - 控制信号平滑)
- ✅ deterministic_policy (配置覆盖 - 确定性策略)
- ✅ random_seed (配置覆盖 - 随机种子)

### A.7 输入特性
- ✅ envstep_state (历史状态堆叠)
- ✅ relative_action (相对动作模式)
- ✅ include_prev_amplitude_num (历史幅值输入)
- ✅ include_target_velocity (目标速度输入)
- ✅ include_force_coeffs (力系数输入)
- ✅ main_input_type='probes' (多探针输入)
- ✅ normalize_input (输入特征自动归一化)

### A.8 输出与监控
- ✅ TensorBoard 日志
- ✅ 检查点保存/加载
- ✅ 训练曲线自动绘制
- ✅ 验证阶段

---

## 附录B. 符号表与公式推导

### B.1 符号定义

| 符号 | 含义 | 单位/范围 | 说明 |
|------|------|----------|------|
| $t$ | 时间步索引 | $t \in \{1, 2, ..., T\}$ | 从 1 开始的环境步编号 |
| $T$ | 总步数 | `envstep_per_value` | 一次神经网络参数更新所累积的环境步数 |
| $\mathcal{L}_t$ | 第 $t$ 步 Loss | $\mathbb{R}^+$ | 包含主 Loss（探针/力系数）+ 正则化 Loss |
| $w_t$ | 第 $t$ 步时间权重 | $\sum_t w_t = 1$ | 用于加权累积多步 Loss，强调不同时间步的贡献 |
| $\gamma$ | 指数衰减因子 | $(0, 1]$ | 控制时序权重衰减速度，越小衰减越快（配置项: `loss_gamma`） |
| $a_t$ | 第 $t$ 步控制幅值 | $[-A_{max}, A_{max}]$ | 神经网络输出的控制信号幅值 |
| $\bar{a}_{EMA}$ | EMA 幅值 | $\mathbb{R}$ | 历史控制幅值的指数移动平均，用于策略稳定性正则化 |
| $\alpha$ | EMA 动量系数 | $(0, 1]$ | 控制 EMA 更新速度，越大则新值权重越高（配置项: `policy_stability_momentum`） |
| $\pi$ | 策略分布 | 高斯分布 | 神经网络输出的动作分布 $\pi(a|s) = \mathcal{N}(\mu, \sigma^2)$ |
| $\mu, \sigma$ | 策略均值、标准差 | $\mathbb{R}, \mathbb{R}^+$ | 高斯策略的两个参数，由 PolicyNet 输出 |
| $H(\pi)$ | 策略熵 | $\mathbb{R}$ | 衡量策略的随机性/探索性，熵越大探索性越强 |

### B.2 Loss 累积公式

> 详细内容请查阅 [5.4 Loss 累积与时间权重](#54-loss-累积与时间权重)。

**核心公式**：$\mathcal{L}_{total} = \sum_{t=0}^{T-1} \bar{w}_t \cdot \mathcal{L}_t$

### B.3 正则化公式

> 详细内容请查阅 [4.5 正则化模块](#45-正则化模块)。

**总正则化 Loss**:
$$\mathcal{L}_{reg} = \lambda_1 \mathcal{L}_{L2} + \lambda_2 \mathcal{L}_{smooth} + \lambda_3 \mathcal{L}_{stability} + \lambda_4 \mathcal{L}_{entropy} + \lambda_5 \mathcal{L}_{KL}$$

---

## 附录C. 常见问题与解决方案

### C.1 训练问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss 不下降/下降有限 | 学习率/训练轮次长度与数量/神经网络容量 不合适 | 增大 `learning_rate` 或 增加 `num_interaction_per_mini_episode` / `epochs` 或 增大 `hidden_dims` |
| Loss 震荡 / 控制信号越过目标值(过冲) | 学习率及其调度不合适 | 减小 `learning_rate` 或使用递减的 `lr_schedule` |
| 控制信号收敛位置不在目标位置(指主 Loss 定义的位置) | 正则化不合适或与主 Loss 的相对大小不合适 | 增大主 Loss 的权重或减小正则化权重或增大 `disable_regularization_last_epochs` |
| 控制信号震荡 | 缺少平滑正则化 | 增加 `temporal_smoothness_weight` |
| | 动作变化过快 | 启用 `action_rate_limit` |
| 显存不足 | 计算图过长 | 减少 `num_of_interaction_per_agent_update` |
| 梯度爆炸 | 裁剪阈值过大 | 降低 `grad_clip` |
| | 幅值过大 | 降低 `amplitude_max` |
| 正则化 = 0 | 幅值太小 | 预期行为：幅值从 0 开始时 L2 正则化极小 |
| | 变化率 < 阈值 | 预期行为：启用 `action_rate_limit` 时小变化不惩罚 |
| | 控制信号未生效 | 检查 `jet_obstacle` 配置和 `amplitude_max` |
| 验证 loss 与训练 loss 趋势不一致 | 检查 loss 配置 | 确认训练和验证使用相同 loss 类型 |

### C.2 DRL Loss 配置问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 验证图中 loss 与力系数趋势不匹配 | loss 类型不一致 | 检查验证和训练使用的 loss 配置 |
| 时间平均后 backward 报错 | 历史缓冲区未分离 | 检查 `_detach_graph()` 是否分离 DRL 缓冲区 |

### C.3 Solver 问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| Reset 时压力求解发散 | JetObstacle 状态未正确重置 | 确保 `jet_obstacle.reset()` 在 `_init_velocity_pressure()` 之前调用 |
| "Direct solution does not satisfy tolerance" | 初始速度场与边界不兼容 | 检查 `inlet.fixed_velocity` 是否为 `[Ux, Uy]` 格式 |
| "Possible rank deficiency" 警告 | 压力矩阵可能奇异 | 确保有至少一个 Dirichlet 压力边界 |
| 入口模式 fixed 无效 | fixed_velocity 格式错误 | 使用列表格式 `[2.0, 0.0]` 而非标量 |

### C.4 梯度问题

| 问题 | 检查点 |
|------|--------|
| 梯度为 None | 1. 检查是否在 `torch.no_grad()` 上下文中 |
| | 2. 检查是否有 `.detach()` 或 `.numpy()` |
| 梯度全零 | 1. 检查 PhiFlow `enable_grad=True` |
| | 2. 检查探针采样是否保持梯度 |

### C.5 模型格式转换工具

提供 可微分优化AFC框架(如 JetTrainer)下的 checkpoint (`.pt`) 与 SB3 (`.zip`) 模型之间的双向转换。

**工具路径**: `myafc/src/myafc/scripts/convert_model_format.py`

#### 使用场景

| 方向 | 场景 | 说明 |
|------|------|------|
| JetTrainer → SB3 | 迁移学习 | 将监督学习预训练的策略用于 DRL 微调 |
| SB3 → JetTrainer | 迁移学习 / 可微分优化 | 将 DRL 策略用于可微分模拟优化 |

#### 命令示例

```bash
# JetTrainer → SB3
python convert_model_format.py \
    --source checkpoints/best.pt \
    --target output/model.zip \
    --direction sup2sb3 \
    --trainer-config configs/jet_trainer.yaml

# SB3 → JetTrainer
python convert_model_format.py \
    --source final_model.zip \
    --target output/converted.pt \
    --direction sb32sup \
    --trainer-config configs/jet_trainer.yaml

# 分析兼容性
python convert_model_format.py \
    --analyze \
    --sup-path checkpoints/best.pt \
    --sb3-path final_model.zip \
    --env-config cfgs4env/karman_torch_drl_square.yaml
```

#### 注意事项

- 网络架构差异：JetTrainer 使用 `MLPPolicy`，SB3 使用 `SACPolicy`
- 转换仅迁移**隐藏层权重**，输出层可能需要重新训练
- 输入/输出维度必须匹配才能成功转换
- 建议将转换后的模型作为**预训练起点**，而非最终模型

---

## 附录D. 涡脱周期

本附录提供涡脱周期相关参数的参考数据，用于设置 `validation_steps` 等关键参数。

### D.1 Strouhal 数说明

**涡脱周期**与 **Strouhal 数** 的关系：

$$
T_{vortex} = \frac{D}{St \cdot U}
$$

其中：
- $T_{vortex}$：涡脱周期 (s)
- $D$：特征长度（障碍物直径/边长）(m)
- $St$：Strouhal 数（无量纲）
- $U$：来流速度 (m/s)

对应的环境步数：

$$
n_{steps} = \frac{T_{vortex}}{dt} = \frac{D}{St \cdot U \cdot dt}
$$

### D.2 DRLinFluids 案例参数

以下参数来自 DRLinFluids 开源库和论文：

#### DRLinFluids 开源库案例

| 参数 | 圆柱案例 | 方柱案例 |
|------|---------|----------|
| **几何参数** | | |
| 特征长度 D | 0.1 m (直径) | 0.1 m (边长) |
| 计算域 | 2.2×0.41 m | 2.032×1.016 m |
| 分辨率 | 220×64 | 258×127 |
| 障碍物中心 | (0.2, 0.2) | (0.254, 0.508) |
| **流体参数** | | |
| 来流速度 U | 1.0 m/s | 1.5 m/s |
| 运动粘度 ν | 0.001 m²/s | 0.001 m²/s |
| 雷诺数 Re | 100 | 150 |
| **时间参数** | | |
| 时间步长 dt | 0.004 s | 0.004 s |
| **涡脱参数** | | |
| Strouhal 数 St | ~0.167 | ~0.137 |
| 涡脱周期 | ~0.6 s | ~0.486 s |
| 周期步数 | ~150 steps | ~122 steps |

#### DRLinFluids 论文案例 (不同 Re)

| 几何 | Re | St | 涡脱周期 (s) | 周期步数 |
|------|----|----|-------------|----------|
| 方柱 | 100 | 0.139 | 0.719 s | ~180 steps |
| 方柱 | 400 | 0.140 | 0.714 s | ~179 steps |
| 方柱 | 1000 | 0.117 | 0.855 s | ~214 steps |

> **注**：论文案例参数为 D=0.1m, U=1.0m/s, dt=0.004s

### D.3 典型 Strouhal 数参考

圆柱绕流 Strouhal 数经验公式 (Roshko, 1954)：

$$
St = 0.212 \left(1 - \frac{21.2}{Re}\right) \quad (50 < Re < 10^4)
$$

方柱绕流 Strouhal 数经验公式 (Okajima, 1982)：

$$
St = 0.135 \left(1 - \frac{5.0}{Re}\right) \quad (Re > 100)
$$

### D.4 validation_steps 设置建议

- **建议覆盖 3-5 个涡脱周期**
- 对于 DRLinFluids 方柱案例 (Re=150)：建议 `validation_steps ≈ 400-600`
- 对于 DRLinFluids 圆柱案例 (Re=100)：建议 `validation_steps ≈ 450-750`

### D.5 三种周期

在经典绕流案例中，存在三种典型的周期，分别是流场中的涡脱周期、阻力系数 Cd 震荡周期和升力系数 Cl 震荡周期，三者关系如下：

$$
T_{Cl} = T_{shedding} \quad , \quad T_{Cd} = \frac{T_{shedding}}{2}
$$

### D.6 工具使用

可使用 `myafc.utils.estimate_strouhal()` 函数估算涡脱周期：

```python
from myafc.utils import estimate_strouhal

St, T_vortex, n_steps = estimate_strouhal(
    Re=150, 
    shape='square', 
    D=0.1, 
    U=1.5, 
    dt=0.004
)
print(f"St={St:.3f}, T={T_vortex:.3f}s, steps={n_steps}")
```

---
