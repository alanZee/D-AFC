"""
流体控制模块

提供带射流边界条件的障碍物类和控制策略接口。

核心组件:
    - JetObstacle: 继承自 PhiFlow 的 Obstacle，可直接用于 make_incompressible()
    - JetConfig: 射流配置（喷嘴、饱和度、平滑系数等）
    - JetNozzle: 单个喷嘴配置（角度范围）
    - BasePolicy: 策略基类
    - ConstantPolicy: 常量策略
    - SinusoidalPolicy: 正弦策略
    - MLPPolicy: MLP 策略（用于 RL）

使用方式:
    1. 创建 JetObstacle 时传入 geometry, jet_config, policy
    2. 设置速度场模板: jet_obstacle.set_velocity_template(velocity)
    3. 在每步 step 开始时更新控制: jet_obstacle.update_controls(state)
    4. 使用原生 make_incompressible: fluid.make_incompressible(velocity, obstacles=[jet_obstacle])

多环境并行:
    使用 ParallelSolverManager 创建多个独立 Solver 实例，
    通过多线程实现并行执行。详见 myafc.env_kernels.solvers.parallel_solver_manager
"""
from .jet_obstacle import JetObstacle, JetConfig, JetNozzle
from .control_policy import BasePolicy, ConstantPolicy, SinusoidalPolicy, MLPPolicy

__all__ = [
    'JetObstacle', 'JetConfig', 'JetNozzle',
    'BasePolicy', 'ConstantPolicy', 'SinusoidalPolicy', 'MLPPolicy',
]
