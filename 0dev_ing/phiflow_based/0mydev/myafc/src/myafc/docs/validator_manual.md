# AFC Validator 技术手册

> **位置**: `myafc/src/myafc/engines/validators/`

---

## 1. 概述

`AFCValidator` 使用训练配置和模型检查点运行独立验证。JetTrainer 的 `.pt` 文件包含 AFCAgent 权重。验证 SB3 的 `.zip` 模型还依赖 Stable-Baselines3 和单独的 `myafc.afc_drl_sb3` 环境模块。

--------|----------|------|----------|
| **JetTrainer** | `.pt` | PyTorch 检查点，含 AFCAgent 网络权重 | 独立验证循环 |
| **DRLTrainer** | `.zip` | Stable-Baselines3 格式 (SAC/PPO/TD3) | ⭐ 内部调用 AFCValidator |

### 设计理念

- **统一验证接口**：无论模型来源，使用相同的 API
- **配置驱动**：直接解析 Trainer 配置文件，无需额外 Validator 配置
- **对齐 JetTrainer**：验证流程与 `JetTrainer._run_single_validation()` 保持一致
- **Trainer 内部集成**：DRLTrainer 的 `run_final_validation()` 现在内部调用 AFCValidator，确保训练-验证一致性

---

## 2. 快速开始

### 2.1 安装

```bash
pip install -e myafc/
```

### 2.2 基本用法

**方式 1: 直接传入参数**

```python
from myafc.engines.validators import AFCValidator

# 验证 JetTrainer 模型
validator = AFCValidator(
    trainer_config="configs/jet_trainer_benchmark.yaml",
    model_path="runs/exp1/checkpoints/best.pt",
    warmup_steps=100,
    validation_steps=500,
)
results = validator.run()
print(f"mean_Cd: {results['mean_cd']:.4f}")
```

**方式 2: 使用配置文件**

```python
validator = AFCValidator(config="configs/val/val_benchmark.yaml")
results = validator.run()
```

---

## 3. 验证流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      Validation Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌────────────────┐    ┌───────────────────┐   │
│  │  Warmup  │ → │   Validation    │ → │   Save Results    │   │
│  │  Phase   │    │     Phase      │    │   & Visualize     │   │
│  └──────────┘    └────────────────┘    └───────────────────┘   │
│                                                                   │
│  warmup_steps     validation_steps     npz/csv/png/mp4          │
│  (不计入统计)      (记录详细数据)       (完整输出)               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 步数语义 (CFD Step)

> ⭐ **关键**: `warmup_steps` 和 `validation_steps` 都是 **CFD steps** (env-steps)

**与 JetTrainer 对齐**：所有时间步相关操作都以 CFD step 为单位。

**DRL 模式特殊说明**：

```
1 DRL step = interaction_period × CFD steps

示例 (interaction_period=26, warmup=100, validation=500):
  - 总 CFD 步数 = 100 + 500 = 600 (不乘以 interaction_period!)
  - 每 26 个 CFD step 获取一次新的 agent action
```

### 3.2 Warmup 阶段

- 执行 `warmup_steps` 步环境推进 (CFD steps)
- 填充历史 buffer（用于 `envstep_state` 特征）
- **不记录到验证统计**

### 3.3 验证阶段

- 执行 `validation_steps` 步 (CFD steps)
- 每步记录：Cd、Cl、控制幅值、奖励（DRL 模式）
- 按 `visuals.interval` 保存可视化帧

---

## 4. 配置详解

### 4.1 ValidatorConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `trainer_config_path` | str | 必填 | Trainer 配置文件路径 |
| `model_path` | str | 必填 | 模型文件路径 (.pt/.zip) |
| `model_type` | str | 'auto' | 模型类型 ('auto'/'jet'/'drl') |
| `warmup_steps` | int | 100 | 预热步数 |
| `validation_steps` | int | 500 | 验证步数 |
| `deterministic` | bool | True | 是否使用确定性策略 |
| `use_ema` | bool | False | 是否使用 EMA 权重 (JetTrainer) |
| `output_dir` | str | None | 输出目录 (None=自动创建) |
| `device` | str | 'auto' | 计算设备 |

### 4.2 可视化配置 (visuals)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | True | 是否启用可视化 |
| `interval` | int | 50 | 帧保存间隔 |
| `quantities` | list | ['velocity_magnitude', 'pressure'] | 物理量列表 |
| `save_endpoints` | bool | True | 保存初始/最终帧 |
| `save_fields` | bool | False | 保存原始流场 (npz) |
| `generate_video` | bool | True | 生成 mp4 视频 |
| `video_fps` | int | 10 | 视频帧率 |

**支持的物理量**:
- `velocity_magnitude`: 速度大小
- `velocity_x`: x 方向速度
- `velocity_y`: y 方向速度
- `pressure`: 压力场
- `vorticity`: 涡量场

---

## 5. 输出目录结构

```
validation_<timestamp>/
├── validation_curves.npz    # 曲线数据 (NumPy 格式)
├── validation_curves.csv    # 曲线数据 (表格格式)
├── validation_curves.png    # 曲线可视化图
├── raw/                     # 原始流场数据 (可选)
│   └── fields.npz           # 多时刻流场 (velocity_x, velocity_y, pressure)
└── frames/                  # 可视化图片
    ├── velocity_magnitude/
    │   ├── step000001.png
    │   ├── step000050.png
    │   └── ...
    ├── pressure/
    ├── vorticity/
    ├── velocity_magnitude.mp4   # 合成视频
    └── pressure.mp4
```

### 5.1 曲线数据格式 (统一格式 v2.0)

> 📌 **跨 Trainer 一致性**：JetTrainer、DRLTrainer、AFCValidator 的验证输出均使用此统一格式

**NPZ 和 CSV 使用完全相同的键名**：

| 字段 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `env_step` | int | ✅ | 环境步索引 (0-based) |
| `amplitude` | float | ✅ | 射流幅值 |
| `cd` | float | ✅ | 阻力系数 |
| `cl` | float | ✅ | 升力系数 |
| `loss` | float | ⬜ | Primary Loss (Supervision 模式) |
| `velocity_u` | float | ⬜ | 探针速度 u 分量 |
| `velocity_v` | float | ⬜ | 探针速度 v 分量 |
| `reward` | float | ⬜ | 累积奖励 (DRL 模式) |

**validation_curves.npz** (含元数据):
```python
import numpy as np
data = np.load('validation_curves.npz')
print(data['env_step'])    # 步索引
print(data['cd'])          # Cd 数组
print(data['cl'])          # Cl 数组
print(data['amplitude'])   # 控制幅值 (注意：单数形式)
print(data['reward'])      # 奖励 (DRL 模式，可选)
print(data['target_velocity'])  # 元数据 (仅 NPZ)
```

**validation_curves.csv** (纯时序):
| env_step | amplitude | cd | cl | loss | reward |
|----------|-----------|-----|-----|------|--------|
| 0 | 0.5 | 1.234 | 0.567 | 0.0123 | -0.123 |
| 1 | 0.48 | 1.231 | 0.554 | 0.0121 | -0.121 |

---

## 6. 模型加载机制

### 6.1 JetTrainer 模型 (.pt)

检查点结构：
```python
{
    'epoch': int,
    'step': int,
    'model': OrderedDict,      # AFCAgent 权重
    'ema': {                   # EMA 权重 (可选)
        'ema_model_state_dict': OrderedDict
    },
    'optimizer': OrderedDict,
    'best_fitness': float,
}
```

加载流程：
1. 解析 Trainer 配置 → 获取 `agent` 和 `input_features` 配置
2. 从环境配置计算输入维度
3. 创建 `AFCAgent` 实例
4. 加载权重 (`model` 或 `ema`)

### 6.2 DRLTrainer 模型 (.zip)

使用 Stable-Baselines3 标准加载：
```python
from stable_baselines3 import SAC
model = SAC.load("final_model.zip", env=env)
```

---

## 7. 高级用法

### 7.1 自定义验证步数

```python
validator = AFCValidator(
    trainer_config="configs/jet_trainer.yaml",
    model_path="checkpoints/best.pt",
    warmup_steps=50,        # 自定义预热
    validation_steps=1000,  # 长时验证
)
```

### 7.2 使用 EMA 模型

```python
validator = AFCValidator(
    trainer_config="configs/jet_trainer.yaml",
    model_path="checkpoints/best.pt",
    use_ema=True,  # 使用 EMA 权重
)
```

### 7.3 保存原始流场用于后分析

```yaml
# val_benchmark.yaml
visuals:
  save_fields: true
  interval: 10  # 更密集的保存
```

```python
# 后续分析
import numpy as np
fields = np.load('validation/raw/fields.npz')
print(fields['velocity_x'].shape)  # (n_steps, nx, ny)
```

### 7.4 批量验证

```python
import glob
from myafc.engines.validators import AFCValidator

checkpoints = glob.glob("runs/exp*/checkpoints/best.pt")
results = []

for ckpt in checkpoints:
    validator = AFCValidator(
        trainer_config="configs/jet_trainer.yaml",
        model_path=ckpt,
    )
    result = validator.run()
    results.append({
        'checkpoint': ckpt,
        **result
    })

# 汇总结果
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('batch_validation_results.csv')
```

---

## 8. 常见问题

### Q1: 如何判断应该用哪个 Trainer 配置？

Validator 会自动检测配置类型：
- 含 `agent` 字段 → JetTrainer
- 含 `algorithm` 字段 → DRLTrainer

### Q2: 模型和配置不匹配怎么办？

确保 `model_path` 和 `trainer_config_path` 来自同一次训练。Validator 会发出警告但仍尝试加载。

### Q3: 视频生成失败

需要安装 ffmpeg：
```bash
# Windows (Chocolatey)
choco install ffmpeg

# Linux
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Q4: CUDA 内存不足

使用 CPU 验证：
```python
validator = AFCValidator(..., device='cpu')
```

---

## 9. API 参考

### AFCValidator

```python
class AFCValidator:
    def __init__(
        self,
        config: Optional[Union[str, Path, Dict, ValidatorConfig]] = None,
        trainer_config: Optional[Union[str, Path]] = None,
        model_path: Optional[str] = None,
        warmup_steps: Optional[int] = None,
        validation_steps: Optional[int] = None,
        output_dir: Optional[str] = None,
        device: str = 'auto',
    ): ...
    
    def run(self) -> Dict[str, Any]:
        """执行验证
        
        Returns:
            dict: {
                'warmup_steps': int,
                'validation_steps': int,
                'mean_cd': float,
                'mean_cl': float,
                'mean_amplitude': float,
                'std_cd': float,
                'std_cl': float,
                'output_dir': str,
                'mean_reward': float,  # DRL 模式
                'total_reward': float, # DRL 模式
            }
        """
```

### ValidatorConfig

```python
@dataclass
class ValidatorConfig:
    trainer_config_path: str = ''
    model_path: str = ''
    model_type: Literal['auto', 'jet', 'drl'] = 'auto'
    warmup_steps: int = 100
    validation_steps: int = 500
    deterministic: bool = True
    use_ema: bool = False
    output_dir: Optional[str] = None
    device: str = 'auto'
    visuals: ValidationVisualsConfig = field(...)
```

---

## 10. 相关文档

- [JetTrainer 技术手册](architecture-afc_supervision-JetTrainer_技术手册.md)
- [方柱算例环境配置](../configs/cfgs4env/karman_torch_drl_square.yaml)
