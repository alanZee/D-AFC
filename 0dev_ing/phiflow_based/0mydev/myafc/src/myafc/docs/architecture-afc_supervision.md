# AFC 监督学习系统架构

## 核心模块抽象层级

系统采用分层架构，自顶向下各层职责明确：

```
Trainer (SimpleTrainer)                              ← 训练引擎：损失计算、梯度更新、正则化
    │
    ├── Runner (SolverRunner)  ← 包装 Solver ：后处理、可视化等
    │       │
    └──-----└── Solver (KarmanTorchSolver)           ← 流场求解：可微分 NS 方程
                    └── JetObstacle  ← 射流边界条件管理（继承并替代 PhiFlow 原生 Obstacle ，对接 Solver 与 Policy）
                            │
                            └── Policy (MLPPolicy / CustomMLPPolicy)  ← 统一的标准化接口，对接 JetObstacle 与 策略神经网络，并实现输入输出解析
                                    │
                                    └── PolicyNet                     ← 神经网络：MLP + 高斯采样 (state→(amp,angle))
```

**设计理念**：各层解耦，Trainer 只管训练逻辑，Solver 只管物理求解，Policy 只管控制策略。

---

## 模块职责总览

| 模块 | 核心职责 | 关键接口 |
|:---|:---|:---|
| **SimpleTrainer** | 训练循环、损失计算、正则化、参数优化 | `train()`, `_compute_loss()` |
| **KarmanTorchSolver** | 可微分 NS 方程求解、边界条件管理 | `step(update_controls, apply_jets)` |
| **SolverRunner** | 仿真后处理、评估验证、可视化 | `run_eval()`, `plot_flow()` |
| **JetObstacle** | 障碍物几何 + 喷嘴配置 + 射流速度场 | `apply_jet_bc()`, `velocity` 属性 |
| **MLPPolicy** | 从 state 构建输入、解析输出为控制信号 | `__call__(state) → (amps, angles)` |
| **PolicyNet** | MLP 前向传播、EMA 归一化、动作缩放 | `forward(obs) → PolicyOutput` |

---

## 训练循环流程

```python
# SimpleTrainer.train() 核心逻辑
for env_step in range(total_steps):
    
    # 1. 判断是否更新控制（每 control_interval 步更新一次）
    should_update = (env_step % control_interval == 0)
    
    # 2. 推进仿真（内部按需调用 policy 获取动作）
    solver.step(update_controls=should_update, apply_jets=True)
    
    # 3. 计算 loss（每步都施加正则化项，但正则化项的值仅在 should_update 时更新）
    loss = self._compute_loss(is_net_forward_step=should_update)
    step_losses.append(loss)
    
    # 4. 累积 envstep_per_reward 步后，加权求和并反向传播
    if len(step_losses) >= envstep_per_reward:
        weighted_loss = sum(w * L for w, L in zip(weights, step_losses))
        weighted_loss.backward()   # 梯度通过 solver → policy → PolicyNet 传递
        optimizer.step()
        step_losses.clear()
```

---

## Solver.step() 调用流程

```python
def step(self, update_controls=True, apply_jets=True):
    
    # 1. 可选：更新控制动作（仅在 update_controls=True 时调用 policy）
    if update_controls and self.jet_obstacle:
        state = self.get_state()                        # 收集探针/力系数
        amps, angles = self.jet_obstacle.policy(state)  # Policy 前向推理
        self.jet_obstacle.set_controls(amps, angles)
    
    # 2. 扩散 → 施加射流BC
    velocity = diffuse.explicit(velocity, viscosity, dt)
    velocity = self.jet_obstacle.apply_jet_bc(velocity) if apply_jets else velocity
    
    # 3. 对流 → 施加射流BC
    velocity = advect.semi_lagrangian(velocity, velocity, dt)
    velocity = self.jet_obstacle.apply_jet_bc(velocity) if apply_jets else velocity
    
    # 4. 压力投影（JetObstacle.velocity 作为障碍物边界条件）
    velocity, pressure = fluid.make_incompressible(velocity, obstacles=[self.jet_obstacle])
    
    # 5. 最终施加射流BC
    velocity = self.jet_obstacle.apply_jet_bc(velocity) if apply_jets else velocity
```

---

## 数据流示意

```
    SimpleTrainer                  KarmanTorchSolver              MLPPolicy / PolicyNet
         │                               │                                │
         ├── solver.step() ─────────────►│                                │
         │                               ├── policy(state) ──────────────►│
         │                               │                                ├── PolicyNet.forward()
         │                               │◄── (amps, angles) ─────────────┤
         │                               ├── apply_jet_bc()               │
         │◄── probe_velocity / force ────┤                                │
         │                               │                                │
         ├── _compute_loss()             │                                │
         │                               │                                │
         ├── loss.backward() ────────────┼───► 梯度反传至 PolicyNet ──────►│
         │                               │                                │
         └── optimizer.step() ───────────┴───────────────────────────────►│
```

---

## 正则化机制

每个环境步都施加正则化项到 loss 中，但正则化项的**数值仅在动作更新步刷新**（复用上一次的值直到下一次 policy forward），以保持梯度链连通。

| 正则化项 | 公式 | 作用 |
|:---|:---|:---|
| 幅值 L2 | $L = a^2$ | 抑制极端动作 |
| 时序平滑 | $L = (a_t - a_{t-1})^2$ | 防止控制信号剧变 |
| EMA 稳定性 | $L = (a_t - \bar{a}_{EMA})^2$ | 防止策略过冲 |
| 熵正则 | $H = \sum \frac{1}{2}\log(2\pi e\sigma^2)$ | 鼓励探索 |
| KL 散度 | $D_{KL}(\pi_{new} \| \pi_{old})$ | 稳定策略更新 |

详见 [JetTrainer 手册的正则化说明](architecture-afc_supervision-JetTrainer_技术手册.md#45-正则化模块)。

---

## 配置层次

```yaml
# solver_config.yaml（基础配置）
domain: { size: [2.0, 1.0], resolution: [256, 128] }
obstacle: { shape: sphere, center: [0.25, 0.5], radius: 0.05 }
jets:
  enabled: true
  nozzles: [{ arc_deg: [80, 100] }, { arc_deg: [260, 280] }]
  policy:
    type: custom_mlp
    hidden_dims: [64, 64]
    max_amplitude: 1.0

# train_config.yaml（训练配置，继承并覆盖基础配置）
base_config_path: solver_config.yaml

training:
  total_interactions: 100
  control_interval: 10
  learning_rate: 0.001

loss:
  loss_type: probe_mse
  amplitude_l2_weight: 0.001
  entropy_weight: 0.01
```

**继承规则**：train_config 中的参数会覆盖 base_config 中的同名参数。
