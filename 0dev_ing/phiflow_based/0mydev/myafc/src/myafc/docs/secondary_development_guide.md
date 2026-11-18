# 二次开发指南：实现新的 AFC 案例

> **适用范围**: myafc 框架下的可微分监督学习式 AFC 策略训练

---

## 目录

1. [概述](#1-概述)
2. [框架架构总览](#2-框架架构总览)
3. [核心组件接口规范](#3-核心组件接口规范)
   - 3.1 [Solver 接口规范](#31-solver-接口规范)
   - 3.2 [Runner 接口规范](#32-runner-接口规范)
   - 3.3 [JetObstacle 接口规范](#33-jetobstacle-接口规范)
   - 3.4 [Policy 接口规范](#34-policy-接口规范)
4. [配置文件编写规范](#4-配置文件编写规范)
   - 4.1 [环境配置 (cfgs4env)](#41-环境配置-cfgs4env)
   - 4.2 [训练器配置 (Trainer Config)](#42-训练器配置-trainer-config)
5. [二次开发流程](#5-二次开发流程)
   - 5.1 [实现新的 Solver](#51-实现新的-solver)
   - 5.2 [实现新的 Runner](#52-实现新的-runner)
   - 5.3 [实现新的控制策略](#53-实现新的控制策略)
6. [完整示例：实现一个新的 AFC 案例](#6-完整示例实现一个新的-afc-案例)
7. [常见问题与调试技巧](#7-常见问题与调试技巧)
8. [附录](#8-附录)

---

## 1. 概述

### 1.1 框架定位

本框架是一个 **基于 PhiFlow 可微分模拟的监督学习式 AFC (Active Flow Control) 策略训练框架**，核心特点：

- **端到端可微分**：控制信号 → CFD 求解 → 观测数据，全流程保持梯度链路
- **模块化设计**：Solver / Runner / Trainer / Policy 分层解耦
- **YAML 驱动**：所有仿真与训练参数通过配置文件管理，无需修改代码

### 1.2 二次开发场景

典型的二次开发需求包括：

| 场景 | 需要开发的组件 |
|------|----------------|
| 新的几何形状（如翼型、多障碍物） | Solver + 可能需要新的 JetObstacle |
| 新的边界条件类型 | Solver（边界解析逻辑） |
| 新的控制策略（如 RNN、Transformer） | Policy |
| 新的观测/奖励类型 | Runner（数据采集逻辑） |
| 新的训练算法 | Trainer |

### 1.3 前置要求

```python
# 必须使用 PhiFlow Torch 后端以保证可微分性
from phi.torch.flow import (
    CenteredGrid, StaggeredGrid, 
    advect, diffuse, fluid,
    Solve, Box, Sphere, vec,
)
```

---

## 2. 框架架构总览

```
myafc/src/myafc/
├── env_kernels/                    # CFD 环境核心层
│   ├── solvers/                    # 求解器实现
│   │   ├── karman_torch_solver.py  # 参考实现: Kármán 绕流
│   │   ├── parallel_solver_manager.py  # 多环境并行管理器
│   │   └── fluid_control/          # 流动控制组件
│   │       ├── jet_obstacle.py     # 射流障碍物
│   │       └── control_policy/     # 控制策略
│   └── solver_wrappers/            # 求解器包装层
│       └── solver_runner.py        # Runner: 数据采集、可视化
│
├── afc_supervision/                # 监督学习训练层
│   ├── engines/
│   │   └── trainers/               # 训练器实现
│   │       ├── base_trainer.py     # 训练器基类
│   │       └── jet_trainer.py      # AFC 训练器
│   └── losses/                     # 损失函数
│
├── configs/                        # 配置文件
│   ├── cfgs4env/                   # 环境配置 (Solver + Runner)
│   └── *.yaml                      # 训练器配置
│
└── utils/                          # 工具函数
    ├── __init__.py                 # get_logger() 等
    └── phiflow_io.py               # PhiFlow H5 I/O
```

### 2.1 数据流

```
┌─────────────┐    step()    ┌─────────────┐   record_*()  ┌─────────────┐
│   Solver    │ ──────────>  │   Runner    │ ──────────>   │   Trainer   │
│ (CFD 求解)  │              │ (数据采集)   │              │ (梯度优化)   │
└─────────────┘              └─────────────┘               └─────────────┘
       ▲                            │                            │
       │                            │ probes, force_coeffs       │ loss.backward()
       │                            ▼                            ▼
       │                     ┌─────────────┐              ┌─────────────┐
       │                     │   State     │   ───────>   │   Policy    │
       └─────────────────────│  (观测)     │              │ (控制信号)  │
                             └─────────────┘              └─────────────┘
```

---

## 3. 核心组件接口规范

### 3.1 Solver 接口规范

Solver 是 CFD 求解的核心，负责流场的时间推进。新 Solver 必须实现以下接口：

#### 3.1.1 必须实现的属性

```python
class MySolver:
    """自定义 Solver 必须提供的属性"""
    
    # === 流场状态 (必需) ===
    velocity: StaggeredGrid     # 速度场 (交错网格)
    pressure: CenteredGrid      # 压力场 (中心网格)
    
    # === 时间状态 (必需) ===
    current_step: int           # 当前时间步编号
    current_time: float         # 当前物理时间 (秒)
    
    # === 配置 (必需) ===
    cfg: SolverConfig           # 配置对象，需包含 time.dt 等
    
    # === 障碍物 (如适用) ===
    obstacles: List[Obstacle]   # PhiFlow 障碍物列表
    jet_obstacle: JetObstacle   # 射流障碍物 (可选，若启用射流)
    
    # === 可微分控制 (必需) ===
    differentiable: bool        # 是否保持梯度
```

#### 3.1.2 必须实现的方法

```python
def __init__(self, cfg_path: str, log_file: Optional[str] = None):
    """
    初始化求解器
    
    Args:
        cfg_path: YAML 配置文件路径
        log_file: 日志文件路径 (可选)
    
    初始化流程:
        1. 解析配置 → parse_config(cfg_path)
        2. 设置后端 → _setup_backend()
        3. 初始化场 → _init_fields()
    """
    pass

def step(
    self,
    state: Optional[Dict[str, Any]] = None,
    update_controls: bool = True,
    apply_jets: bool = True,
) -> Tuple[StaggeredGrid, CenteredGrid]:
    """
    执行单步求解 (核心方法)
    
    Args:
        state: 状态字典，传递给控制策略
            常用字段:
                - 'time': float, 当前物理时间
                - 'probes': Dict, 探针数据
                - 'force_coeffs': Dict, 力系数
        update_controls: 是否更新控制信号 (step 开始时)
        apply_jets: 是否施加射流边界条件
    
    Returns:
        (velocity, pressure): 更新后的流场
    
    可微分要求:
        - differentiable=True 时，所有操作必须保持梯度
        - differentiable=False 时，应使用 torch.no_grad()
    
    求解流程 (参考实现):
        1. 更新射流控制信号
        2. 黏性扩散 (diffuse.explicit)
        3. 施加边界条件 (with_boundary + apply_jet_bc)
        4. 对流 (advect.semi_lagrangian)
        5. 外力项 (可选)
        6. 压力投影 (fluid.make_incompressible)
        7. 更新 current_step, current_time
    """
    pass

def reset(
    self,
    use_h5: bool = True,
    step: Optional[int] = None,
    resume_strict: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    重置求解器状态
    
    Args:
        use_h5: 是否从 H5 文件初始化流场
        step: H5 帧索引 (-1 表示最后一帧)
        resume_strict: 严格续算模式
            - False: 热启动，时间从 0 开始
            - True: 接续 H5 的时间步
    
    Returns:
        H5 加载信息 Dict (use_h5=True 时)
    
    重置内容:
        1. 流场 (velocity, pressure)
        2. 时间状态 (current_step, current_time)
        3. JetObstacle 控制状态 (如适用)
    """
    pass

def compute_force_coeffs(
    self,
    U_ref: Optional[float] = None,
    A_ref: Optional[float] = None,
    include_viscous: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    计算力系数 (可微分实现)
    
    Args:
        U_ref: 参考速度，None 时自动获取
        A_ref: 参考面积/长度，None 时自动计算
        include_viscous: 是否包含粘性力
    
    Returns:
        (Cd, Cl, Fx, Fy, U_ref, A_ref)
        - Cd, Cl, Fx, Fy: torch.Tensor，支持梯度反传
        - U_ref, A_ref: float，实际使用的参考量
    
    公式:
        Cd = Fx / (0.5 * ρ * U² * A)
        Cl = Fy / (0.5 * ρ * U² * A)
    """
    pass

def set_differentiable(self, enabled: bool):
    """动态切换可微分模式"""
    self.differentiable = enabled
```

#### 3.1.3 可选方法

```python
def set_jet_policy(self, policy):
    """设置/替换射流控制策略 (用于 RL 训练)"""
    if self.jet_obstacle is not None:
        self.jet_obstacle.policy = policy

def run(self, num_steps: Optional[int] = None) -> Dict[str, Any]:
    """执行完整仿真 (无进度条，建议使用 Runner)"""
    pass
```

#### 3.1.4 配置数据类

```python
from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional, Dict

@dataclass
class DomainConfig:
    """求解域配置"""
    size: List[float] = dataclass_field(default_factory=lambda: [200.0, 100.0])  # [Lx, Ly]
    resolution: List[int] = dataclass_field(default_factory=lambda: [128, 64])   # [Nx, Ny]

@dataclass
class ObstacleConfig:
    """障碍物配置"""
    enabled: bool = True
    shape: str = 'sphere'           # 'sphere' / 'box' / 自定义
    center: List[float] = dataclass_field(default_factory=lambda: [0.25, 0.5])
    radius: float = 10.0            # sphere 时使用
    size: List[float] = dataclass_field(default_factory=lambda: [10.0, 10.0])  # box 时使用
    rotation: float = 0.0           # 旋转角度 (弧度)

@dataclass
class FluidConfig:
    """流体物性配置"""
    density: float = 1.0
    viscosity: float = 0.001

@dataclass
class TimeConfig:
    """时间步进配置"""
    dt: float = 1.0
    num_steps: int = 100

@dataclass
class RuntimeConfig:
    """运行时配置"""
    differentiable: bool = True
    device: str = "cuda"
    random_seed: Optional[int] = 42
    verbose: str = 'INFO'

@dataclass
class SolverConfig:
    """完整配置"""
    domain: DomainConfig = dataclass_field(default_factory=DomainConfig)
    obstacle: ObstacleConfig = dataclass_field(default_factory=ObstacleConfig)
    fluid: FluidConfig = dataclass_field(default_factory=FluidConfig)
    time: TimeConfig = dataclass_field(default_factory=TimeConfig)
    runtime: RuntimeConfig = dataclass_field(default_factory=RuntimeConfig)
    # ... 其他配置
```

---

### 3.2 Runner 接口规范

Runner 是 Solver 的高层包装，提供数据采集、可视化和进度管理功能。

#### 3.2.1 必须实现的属性

```python
class MyRunner:
    """Runner 必须提供的属性"""
    
    solver: MySolver                    # 被包装的 Solver
    cfg: RunnerConfig                   # Runner 配置
    
    # === 数据缓存 ===
    force_coeffs: List[Dict]            # 力系数记录
    probes: Dict[str, List]             # 探针数据记录
    probes_jet: List[Dict]              # 射流探针记录
    fields: List[Dict]                  # 场数据缓存
    
    # === 输出 ===
    output_dir: Optional[Path]          # 输出目录
```

#### 3.2.2 必须实现的方法

```python
def __init__(self, solver, config):
    """
    初始化 Runner
    
    Args:
        solver: Solver 实例
        config: YAML 路径或配置字典
    """
    pass

def run(self, num_steps: int = None, show_progress: bool = True) -> Dict[str, Any]:
    """
    执行仿真并采集数据
    
    Args:
        num_steps: 仿真步数，None 时从配置读取
        show_progress: 是否显示进度条
    
    Returns:
        {
            'force_coeffs': List[Dict],  # 力系数记录
            'probes': Dict[str, List],   # 探针数据
            'fields': List[Dict],        # 场数据缓存
        }
    
    执行流程:
        for each step:
            1. solver.step()
            2. _record_force_coeffs()   # 按 interval
            3. _record_probes()         # 按 interval
            4. _record_jet_probes()     # 按 interval
            5. _save_fields_h5()        # 按 interval
            6. _save_visual()           # 按 interval
        
        仿真结束后:
            - 保存 CSV 文件
            - 生成视频
            - 绘制统计图
    """
    pass

def reset_caches(self, reset_solver: bool = True):
    """
    重置数据缓存
    
    Args:
        reset_solver: 是否同时重置 Solver
    
    清空内容:
        - force_coeffs, probes, probes_jet, fields
        - H5 文件句柄
        - 时间偏移量
    """
    pass
```

#### 3.2.3 数据采集方法 (内部实现)

```python
def _record_force_coeffs(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    记录力系数 (可微分)
    
    Returns:
        (Cd, Cl): torch.Tensor，保持梯度
    
    存储格式:
        {
            'step': int,
            'time': float,
            'Cd': float,
            'Cl': float,
            'Fx': float,
            'Fy': float,
        }
    """
    pass

def _record_probes(self) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """
    记录探针数据 (可微分)
    
    探针位置从 cfg.probes.locations 读取
    
    Returns:
        {probe_name: (u, v, p), ...}
        每个值为 torch.Tensor，保持梯度
    """
    pass

def _record_jet_probes(self) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """
    记录射流探针数据 (可微分)
    
    探针位置自动从 JetObstacle 获取
    
    Returns:
        {nozzle_name: (u, v, p), ...}
    """
    pass
```

#### 3.2.4 配置数据类

```python
@dataclass
class OutputConfig:
    enabled: bool = True
    root: str = './outputs'
    subdir: str = 'run'

@dataclass
class ForceCoeffsConfig:
    enabled: bool = False
    interval: int = 1
    filename: str = 'force_coeffs.csv'
    reference_velocity: Optional[float] = None
    reference_length: Optional[float] = None

@dataclass
class ProbesConfig:
    enabled: bool = False
    interval: int = 1
    filename: str = 'probes.csv'
    locations: List[Dict] = dataclass_field(default_factory=list)
    # 位置格式:
    #   { name: 'probe1', fraction: [0.5, 0.5] }  # 相对坐标
    #   { name: 'probe2', units: [1.5, 0.5] }     # 绝对坐标

@dataclass
class VisualsConfig:
    enabled: bool = False
    interval: int = 10
    quantities: List[str] = dataclass_field(default_factory=lambda: [
        'velocity_magnitude', 'pressure', 'vorticity'
    ])

@dataclass
class RunnerConfig:
    output: OutputConfig = dataclass_field(default_factory=OutputConfig)
    force_coeffs: ForceCoeffsConfig = dataclass_field(default_factory=ForceCoeffsConfig)
    probes: ProbesConfig = dataclass_field(default_factory=ProbesConfig)
    visuals: VisualsConfig = dataclass_field(default_factory=VisualsConfig)
    verbose: str = 'INFO'
```

---

### 3.3 JetObstacle 接口规范

JetObstacle 是实现射流边界条件的核心组件，继承自 `phi.physics.fluid.Obstacle`。

#### 3.3.1 必须实现的属性

```python
class MyJetObstacle(Obstacle):
    """射流障碍物"""
    
    # === 配置 ===
    jet_config: JetConfig           # 射流配置
    
    # === 状态 ===
    _current_controls: List[Tuple[torch.Tensor, torch.Tensor]]  # 当前控制信号
    _policy: Optional[Policy]       # 控制策略
    
    # === 缓存 ===
    _nozzle_cache: Optional[NozzleCache]  # 喷嘴几何缓存
    _velocity_template: StaggeredGrid     # 速度场模板
```

#### 3.3.2 必须实现的方法

```python
def __init__(
    self,
    geometry: Geometry,
    jet_config: JetConfig,
    policy: Optional[Policy] = None,
    velocity_template: Optional[StaggeredGrid] = None,
):
    """
    初始化射流障碍物
    
    Args:
        geometry: 障碍物几何 (Sphere/Box)
        jet_config: 射流配置
        policy: 控制策略
        velocity_template: 速度场模板 (用于初始化喷嘴缓存)
    """
    pass

def update_controls(self, state: Dict[str, Any]):
    """
    更新控制信号 (step 开始时调用)
    
    Args:
        state: 状态字典，传递给 Policy
            常用字段: 'time', 'probes', 'force_coeffs'
    
    流程:
        1. Policy 输出原始信号
        2. 存储到 _current_controls
    """
    pass

def apply_controls(self):
    """
    控制信号后处理
    
    流程:
        1. 饱和裁剪: |A| <= saturation
        2. 缩放: A *= amplitude_scale
        3. 时间平滑: A = (1-α)*A_prev + α*A
        4. 空间分布: uniform / cosine
        5. 重建射流速度场
    """
    pass

def apply_jet_bc(self, velocity: StaggeredGrid) -> StaggeredGrid:
    """
    在速度场上施加射流边界条件
    
    Args:
        velocity: 输入速度场
    
    Returns:
        修改后的速度场 (喷嘴区域设为射流速度)
    
    Note:
        此方法在求解流程中多次调用:
        - 扩散后
        - 对流前
        - 压力投影后
    """
    pass

def reset(self):
    """
    重置控制状态
    
    清空:
        - _current_controls
        - _prev_applied_amplitudes
        - Policy 内部状态 (如有)
    """
    pass

def set_velocity_template(self, velocity: StaggeredGrid):
    """
    设置速度场模板 (初始化喷嘴缓存)
    
    Args:
        velocity: 速度场模板
    
    Note:
        必须在 apply_jet_bc 之前调用
    """
    pass

def get_nozzle_probe_locations(self) -> List[Dict]:
    """
    获取喷嘴探针位置
    
    Returns:
        [
            {
                'fluid': (x, y),    # 流体域采样点
                'nozzle': (x, y),   # 喷嘴核心点
            },
            ...
        ]
    """
    pass
```

#### 3.3.3 JetConfig 配置

```python
@dataclass
class JetNozzle:
    """单个喷嘴配置"""
    arc_deg: Tuple[float, float]  # 角度范围 [start, end]

@dataclass
class JetConfig:
    """射流配置"""
    nozzles: List[JetNozzle]      # 喷嘴列表
    saturation: float = 10.0      # 饱和裁剪
    amplitude_scale: float = 1.0  # 缩放系数
    smoothing_alpha: float = 1.0  # 平滑系数 (α=1 无平滑)
    profile: str = 'uniform'      # 空间分布: 'uniform' / 'cosine'
```

**射流角度约定 (物理绝对坐标系)**:
```
                90° (y+, ↑)
                    │
                    │
180° (x-, ←) ───────┼─────── 0° (x+, →)
                    │
                    │
               270° (y-, ↓)
```

---

### 3.4 Policy 接口规范

Policy 是控制策略的抽象，输出控制信号。

#### 3.4.1 必须实现的方法

```python
class MyPolicy:
    """控制策略基类"""
    
    def __call__(
        self, 
        state: Dict[str, Any],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算控制信号
        
        Args:
            state: 状态字典
                常用字段:
                    - 'probes': Dict[str, (u, v, p)]
                    - 'force_coeffs': Dict (Cd, Cl, ...)
                    - 'time': float
            deterministic: 是否使用确定性策略
        
        Returns:
            (amplitude, angle): 每个喷嘴的控制信号
            - amplitude: shape (num_nozzles,) 或 (1,)
            - angle: shape (num_nozzles,) 或 (1,)
        """
        pass
    
    def reset(self):
        """重置策略内部状态 (如 RNN 隐藏状态)"""
        pass
    
    def parameters(self):
        """返回可训练参数 (用于优化器)"""
        pass
    
    @classmethod
    def from_config(cls, config: Dict) -> 'MyPolicy':
        """从配置字典创建策略"""
        pass
```

#### 3.4.2 内置策略类型

| 类型 | 说明 | 配置示例 |
|------|------|----------|
| `ConstantPolicy` | 常量输出 | `amplitude: 4.0, angle_deg: 90.0` |
| `SinusoidalPolicy` | 正弦波输出 | `amplitude: 4.0, frequency: 0.5` |
| `MLPPolicy` | 多层感知机 | `hidden_dims: [512, 512]` |

---

## 4. 配置文件编写规范

### 4.1 环境配置 (cfgs4env)

环境配置文件定义 Solver 和 Runner 的参数，存放于 `configs/cfgs4env/` 目录。

#### 4.1.1 完整模板

```yaml
# =============================================================================
# 环境配置文件模板
# =============================================================================

meta:
  description: |
    配置文件描述
  notes: "备注信息"

solver:
  # --- 求解域 ---
  domain:
    size: [2.032, 1.016]          # 物理尺寸 [Lx, Ly]
    resolution: [258, 127]        # 网格数 [Nx, Ny]

  # --- 障碍物 ---
  obstacle:
    enabled: true
    shape: box                    # sphere / box
    center: [0.25, 0.5]           # 相对坐标 [0~1]
    size: [0.0508, 0.0508]        # 绝对尺寸 (box)
    # radius: 0.025               # 绝对尺寸 (sphere)
    rotation: 0.0                 # 旋转角度 (弧度)

  # --- 流体物性 ---
  fluid:
    density: 1.0                  # ρ
    viscosity: 0.001              # μ (Re = ρUL/μ)

  # --- 初始条件 ---
  initial:
    velocity: null                # [Ux, Uy]，null=从 inflow 获取
    resume_from: null             # H5 文件路径
    resume_step: -1               # 恢复帧，-1=最后一帧
    resume_strict: false          # 严格续算模式

  # --- 边界条件 ---
  boundary:
    velocity:
      x-: { type: inflow, value: [2.0, 0.0] }   # 入口
      x+: { type: zero-gradient }               # 出口
      y-: { type: wall }                        # 无滑移壁面
      y+: { type: wall }
    pressure:
      x-: { type: zero-gradient }
      x+: { type: dirichlet, value: 0.0 }       # 参考压力
      y-: { type: zero-gradient }
      y+: { type: zero-gradient }

  # --- 射流控制 ---
  jets:
    enabled: true
    nozzles:
      - { arc_deg: [45.0, 48.0] }    # 角度范围
      - { arc_deg: [312.0, 315.0] }
    saturation: 4.0                   # 饱和裁剪
    amplitude_scale: 1.0              # 缩放系数
    smoothing_alpha: 0.4              # 平滑系数
    profile: uniform                  # uniform / cosine
    policy:
      type: mlp                       # constant / sinusoidal / mlp
      hidden_dims: [64, 32]
      amplitude_max: 4.0
      angle_deg_fixed: 90.0           # 固定角度

  # --- 时间步进 ---
  time:
    dt: 5e-4                          # 时间步长
    num_steps: 3000                   # 默认步数

  # --- 运行时 ---
  runtime:
    differentiable: true              # 可微分模式
    device: cuda                      # cuda / cpu
    random_seed: 42
    verbose: 'INFO'

# =============================================================================
# Runner 配置
# =============================================================================
runner:
  verbose: 'INFO'
  
  output:
    enabled: true
    root: './outputs'
    subdir: 'my_case'

  force_coeffs:
    enabled: true
    interval: 1
    filename: 'force_coeffs.csv'
    reference_velocity: null          # null=自动
    reference_length: null            # null=自动

  probes:
    enabled: true
    interval: 1
    filename: 'probes.csv'
    locations:
      - { name: probe1, fraction: [0.5, 0.5] }     # 相对坐标
      - { name: probe2, units: [1.0, 0.5] }        # 绝对坐标

  jet_probes:
    enabled: true
    interval: 1
    filename: 'jet_probes.csv'

  visuals:
    enabled: true
    interval: 10
    quantities:
      - velocity_magnitude
      - pressure
      - vorticity
    save_endpoints: true
    folder: 'frames'
    video_fps: 10

  fields:
    save:
      enabled: true
      interval: 10
      filename: 'fields.h5'
      quantities:
        - velocity
        - pressure
    cache:
      enabled: false
      interval: 50
```

#### 4.1.2 边界条件类型

| 类型 | 用途 | 速度/压力 |
|------|------|-----------|
| `inflow` | 入口（Dirichlet） | 速度 |
| `wall` / `no-slip` | 无滑移壁面 | 速度 |
| `zero-gradient` / `neumann` / `outflow` | 零梯度边界 | 速度/压力 |
| `dirichlet` | 固定值 | 压力 |
| `periodic` | 周期边界 | 速度/压力 |

#### 4.1.3 喷嘴位置示例

```yaml
# 方柱四角喷嘴
nozzles:
  - { arc_deg: [45.0, 48.0] }     # 右上角
  - { arc_deg: [132.0, 135.0] }   # 左上角
  - { arc_deg: [225.0, 228.0] }   # 左下角
  - { arc_deg: [312.0, 315.0] }   # 右下角

# 圆柱顶部/底部喷嘴
nozzles:
  - { arc_deg: [85, 95] }         # 顶部 (90° 附近)
  - { arc_deg: [265, 275] }       # 底部 (270° 附近)
```

---

### 4.2 训练器配置 (Trainer Config)

训练器配置定义训练过程的参数，独立于环境配置。

#### 4.2.1 完整模板

```yaml
# =============================================================================
# SimpleTrainer 配置文件
# =============================================================================

# 基础环境配置文件路径（相对或绝对）
base_config_path: "cfgs4env/karman_torch_drl_square.yaml"

# =============================================================================
# 多环境配置
# =============================================================================
num_envs: 1                        # 并行环境数量

# 环境变体（覆盖 base_config_path 中的参数）
env_variants: null
# 示例:
# env_variants:
#   - { solver: { fluid: { viscosity: 0.002 }}}
#   - { solver: { fluid: { viscosity: 0.001 }}}

# =============================================================================
# MLP 网络配置
# =============================================================================
mlp:
  hidden_dims: [512, 512]          # 隐藏层维度
  amplitude_max: 1.5               # 输出幅值上限
  angle_deg_fixed: 90.0            # 固定角度，null=学习
  shared_output: true              # 喷嘴共享输出
  normalize_input: true            # 输入归一化
  action_scale: 'softsign'         # tanh / clamp / softsign
  relative_action: false           # 相对动作模式
  deterministic_policy: false      # 确定性策略
  agent_model_path: null           # 预训练权重路径

# =============================================================================
# 输入特征配置
# =============================================================================
input_features:
  main_input_type: 'target_probe'  # null / 'target_probe' / 'probes'
  probe_fields: ['u', 'v', 'p']    # 探针数据字段
  include_target_velocity: true    # 包含目标速度
  include_prev_amplitude_num: 1    # 历史幅值步数
  include_force_coeffs: true       # 包含力系数
  envstep_state: 1                 # 历史状态堆叠步数

# =============================================================================
# 损失函数配置
# =============================================================================
loss:
  loss_type: 'probe_mse'           # probe_mse / probe_mae / force_coeffs
  target_probe_position: [0.26, 0.52]  # 目标探针位置
  target_velocity: [0.0, 0.233]    # 目标速度

  # 损失权重
  loss_weights:
    probe: 1.0
    Cd: 0.0
    Cl: 0.0
    energy: 0.0

# =============================================================================
# 训练配置
# =============================================================================
training:
  # --- 训练循环 ---
  epochs: 100                      # mini-episode 数量
  control_interval: 26             # 控制间隔 (env-steps)
  action_duration: 26              # 控制施加时长 (env-steps)
  num_of_interaction_per_agent_update: 2  # 每次更新的 interaction 数

  # --- 优化器 ---
  learning_rate: 1e-3
  lr_final: 0.01                   # 最终 LR 比例
  lr_schedule: 'cosine'            # constant / linear / cosine
  optimizer: 'adam'                # adam / adamw / sgd
  weight_decay: 5e-4
  grad_clip: 1.0                   # 梯度裁剪

  # --- EMA & AMP ---
  use_ema: false
  ema_decay: 0.9999
  use_amp: false

  # --- 其他 ---
  random_seed: 42
  checkpoint_interval: 50
  checkpoint: null                 # 断点续训路径
  resume: false                    # 断点续训模式
  verbose: 'INFO'

# =============================================================================
# 环境配置
# =============================================================================
env:
  warmup_envsteps: 100             # 预热步数
  h5_file: null                    # 热启动文件

# =============================================================================
# 输出配置
# =============================================================================
output:
  project: 'runs/afc_train'
  name: 'exp'
```

---

## 5. 二次开发流程

### 5.1 实现新的 Solver

#### 步骤 1: 创建配置数据类

```python
# my_solver.py
from dataclasses import dataclass, field as dataclass_field

@dataclass
class MyDomainConfig:
    # 自定义域配置
    pass

@dataclass
class MySolverConfig:
    domain: MyDomainConfig = dataclass_field(default_factory=MyDomainConfig)
    # ... 其他配置
```

#### 步骤 2: 实现 Solver 类

```python
from phi.torch.flow import (
    CenteredGrid, StaggeredGrid, Box,
    advect, diffuse, fluid, Solve,
)
from phi.physics.fluid import Obstacle
from myafc.utils import get_logger

LOGGER = get_logger()

class MySolver:
    def __init__(self, cfg_path: str, log_file=None):
        self.cfg = self._parse_config(cfg_path)
        self._setup_backend()
        self._init_fields()
        
        self.differentiable = self.cfg.runtime.differentiable
        self.current_step = 0
        self.current_time = 0.0
    
    def step(self, state=None, update_controls=True, apply_jets=True):
        """核心求解逻辑"""
        dt = self.cfg.time.dt
        
        # 1. 扩散
        velocity = diffuse.explicit(self.velocity, self.kinematic_viscosity, dt)
        
        # 2. 对流
        velocity = advect.semi_lagrangian(velocity, velocity, dt)
        
        # 3. 压力投影
        velocity, pressure = fluid.make_incompressible(
            velocity, 
            obstacles=self.obstacles,
            solve=Solve('auto', x0=self.pressure)
        )
        
        # 4. 更新状态
        self.velocity = velocity
        self.pressure = pressure
        self.current_step += 1
        self.current_time += dt
        
        return velocity, pressure
    
    def reset(self, use_h5=True, step=None, resume_strict=None):
        # 重置逻辑
        pass
    
    def compute_force_coeffs(self, U_ref=None, A_ref=None, include_viscous=True):
        # 力系数计算
        pass
```

#### 步骤 3: 注册到配置解析

确保新 Solver 可以从配置文件创建实例。

### 5.2 实现新的 Runner

Runner 的实现相对简单，主要是包装 Solver 并添加数据采集功能：

```python
class MyRunner:
    def __init__(self, solver, config):
        self.solver = solver
        self.cfg = self._parse_config(config)
        
        # 初始化缓存
        self.force_coeffs = []
        self.probes = {}
        
    def run(self, num_steps=None, show_progress=True):
        from tqdm import tqdm
        
        iterator = tqdm(range(num_steps)) if show_progress else range(num_steps)
        
        for i in iterator:
            # 1. 求解
            self.solver.step()
            
            # 2. 采集数据
            if self.cfg.force_coeffs.enabled:
                self._record_force_coeffs()
            if self.cfg.probes.enabled:
                self._record_probes()
        
        return {
            'force_coeffs': self.force_coeffs,
            'probes': self.probes,
        }
```

### 5.3 实现新的控制策略

#### 示例：实现 LSTM 策略

```python
import torch
import torch.nn as nn

class LSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_nozzles, amplitude_max):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2 * num_nozzles)  # amp + angle
        self.amplitude_max = amplitude_max
        self.num_nozzles = num_nozzles
        
        self._hidden = None
    
    def __call__(self, state, deterministic=False):
        # 构建输入
        x = self._build_input(state)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        
        # LSTM 前向
        out, self._hidden = self.lstm(x, self._hidden)
        out = self.fc(out.squeeze(0).squeeze(0))
        
        # 分离 amplitude 和 angle
        amp = torch.tanh(out[:self.num_nozzles]) * self.amplitude_max
        angle = torch.sigmoid(out[self.num_nozzles:]) * 2 * 3.14159
        
        return amp, angle
    
    def reset(self):
        self._hidden = None
    
    def _build_input(self, state):
        # 从 state 构建输入特征
        features = []
        if 'probes' in state:
            for name, (u, v, p) in state['probes'].items():
                features.extend([u.item(), v.item(), p.item()])
        return torch.tensor(features, dtype=torch.float32)
    
    @classmethod
    def from_config(cls, config):
        return cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_layers', 2),
            num_nozzles=config['num_nozzles'],
            amplitude_max=config.get('amplitude_max', 4.0),
        )
```

---

## 6. 完整示例：实现一个新的 AFC 案例

以 **翼型绕流** 为例，展示完整的二次开发流程。

### 6.1 创建配置文件

```yaml
# configs/cfgs4env/airfoil_afc.yaml

solver:
  domain:
    size: [3.0, 1.5]
    resolution: [300, 150]

  obstacle:
    enabled: true
    shape: airfoil                # 自定义形状
    center: [0.3, 0.5]
    chord: 0.3                    # 翼弦长度
    aoa_deg: 10.0                 # 攻角

  fluid:
    density: 1.0
    viscosity: 0.0001             # Re = 30000

  jets:
    enabled: true
    nozzles:
      - { location: 'leading_edge', width: 0.02 }
      - { location: 'trailing_edge', width: 0.02 }
    # ...

  boundary:
    velocity:
      x-: { type: inflow, value: [1.0, 0.0] }
      x+: { type: zero-gradient }
      y-: { type: zero-gradient }
      y+: { type: zero-gradient }
    pressure:
      x-: { type: zero-gradient }
      x+: { type: dirichlet, value: 0.0 }
      y-: { type: zero-gradient }
      y+: { type: zero-gradient }

  time:
    dt: 1e-4
    num_steps: 10000

runner:
  # ...
```

### 6.2 实现翼型 Solver

```python
# env_kernels/solvers/airfoil_solver.py

from myafc.env_kernels.solvers.karman_torch_solver import KarmanTorchSolver

class AirfoilSolver(KarmanTorchSolver):
    """翼型绕流求解器"""
    
    def _init_obstacles(self):
        """重写障碍物初始化，支持翼型"""
        cfg = self.cfg
        
        if cfg.obstacle.shape == 'airfoil':
            # 创建翼型几何
            geom = self._create_airfoil_geometry(
                center=cfg.obstacle.center,
                chord=cfg.obstacle.chord,
                aoa_deg=cfg.obstacle.aoa_deg,
            )
        else:
            # 回退到父类实现
            return super()._init_obstacles()
        
        # 初始化射流 (如果启用)
        if cfg.jets.enabled:
            self._init_jet_obstacle(geom)
        else:
            from phi.physics.fluid import Obstacle
            self.obstacles = [Obstacle(geom)]
    
    def _create_airfoil_geometry(self, center, chord, aoa_deg):
        """创建 NACA 翼型几何"""
        from phi.geom import Polygon
        import numpy as np
        
        # NACA 0012 坐标 (简化示例)
        t = np.linspace(0, 2*np.pi, 100)
        # ... 计算翼型坐标
        
        # 创建 Polygon 几何
        return Polygon(vertices=...)
```

### 6.3 创建训练配置

```yaml
# configs/airfoil_trainer.yaml

base_config_path: "cfgs4env/airfoil_afc.yaml"

num_envs: 4
env_variants:
  - { solver: { fluid: { viscosity: 0.0002 }}}  # Re = 15000
  - { solver: { fluid: { viscosity: 0.0001 }}}  # Re = 30000
  - { solver: { fluid: { viscosity: 0.00005 }}} # Re = 60000
  - { solver: { fluid: { viscosity: 0.00003 }}} # Re = 100000

mlp:
  hidden_dims: [256, 256, 128]
  amplitude_max: 2.0

loss:
  loss_type: 'force_coeffs'        # 优化升阻比
  loss_weights:
    Cd: 1.0
    Cl: -0.5                       # 最大化升力

training:
  epochs: 200
  learning_rate: 5e-4
```

### 6.4 运行训练

```python
# examples/train_airfoil.py

from myafc.afc_supervision.engines.trainers import JetTrainer

trainer = JetTrainer.from_config('configs/airfoil_trainer.yaml')
results = trainer.train()
```

---

## 7. 常见问题与调试技巧

### 7.1 梯度断开问题

**症状**: Loss 不下降，或 `grad` 为 None

**排查**:
```python
# 检查梯度链路
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient: {name}")
```

**常见原因**:
1. 使用了 `torch.no_grad()` 或 `.detach()`
2. 使用了非可微分操作（如 `np.array()` 转换）
3. PhiFlow 操作中使用了 `phi.flow` 而非 `phi.torch.flow`

### 7.2 压力求解器发散

**症状**: `pressure` 出现 NaN 或极大值

**排查**:
```python
# 检查速度场散度
div = field.divergence(velocity)
print(f"Max divergence: {math.max(abs(div.values))}")
```

**解决方案**:
1. 减小时间步长 `dt`
2. 调整求解器参数：`Solve('CG', rel_tol=1e-5, max_iterations=5000)`
3. 检查边界条件是否正确

### 7.3 内存不足

**症状**: CUDA out of memory

**解决方案**:
1. 减少 `num_of_interaction_per_agent_update`
2. 降低网格分辨率
3. 使用 `torch.cuda.empty_cache()` 释放缓存
4. 启用梯度检查点 (gradient checkpointing)

### 7.4 控制信号无效

**症状**: 射流不影响流场

**排查**:
1. 检查 `jets.enabled: true`
2. 确认喷嘴角度正确
3. 验证 `apply_jets=True` 传递正确
4. 检查 `saturation` 和 `amplitude_scale` 设置

---

## 8. 附录

### 8.1 PhiFlow 关键 API

```python
# 网格创建
velocity = StaggeredGrid(values, boundary, bounds, x=Nx, y=Ny)
pressure = CenteredGrid(values, boundary, bounds, x=Nx, y=Ny)

# 求解操作
velocity = diffuse.explicit(velocity, diffusivity, dt)
velocity = advect.semi_lagrangian(velocity, velocity, dt)
velocity, pressure = fluid.make_incompressible(velocity, obstacles, solve=Solve(...))

# 采样
point = Point(vec(x=x, y=y))
value = field.at(point)

# 边界条件
velocity = velocity.with_boundary(boundary_dict)
```

### 8.2 日志使用

```python
from myafc.utils import get_logger, set_log_level

LOGGER = get_logger()

set_log_level('DEBUG')  # 详细日志
set_log_level('INFO')   # 正常日志
set_log_level('WARNING')  # 仅警告

LOGGER.info("信息消息")
LOGGER.debug("调试消息")
LOGGER.warning("警告消息")
```

### 8.3 参考文件

| 文件 | 说明 |
|------|------|
| `karman_torch_solver.py` | 参考 Solver 实现 |
| `solver_runner.py` | 参考 Runner 实现 |
| `jet_obstacle.py` | JetObstacle 完整实现 |
| `base_trainer.py` | Trainer 基类 |
| `karman_torch_drl_square.yaml` | 环境配置示例 |
| `simple_trainer_benchmark.yaml` | 训练器配置示例 |

---
