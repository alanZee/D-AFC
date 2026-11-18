"""
流体求解器模块

核心组件:
    - KarmanTorchSolver: 单环境 Karman 涡街求解器
    - ParallelSolverManager: 多环境并行求解器管理器

子模块:
    - fluid_control: 射流控制组件 (JetObstacle, 策略等)
    - forcing: 外力场组件

多环境并行:
    使用 ParallelSolverManager 可创建多个独立的 KarmanTorchSolver 实例，
    每个实例可拥有完全独立的配置（包括边界条件、流体属性等），
    通过多线程实现并行执行。

示例:
    ```python
    from myafc.env_kernels.solvers import KarmanTorchSolver, ParallelSolverManager
    
    # 单环境
    solver = KarmanTorchSolver(cfg_path)
    
    # 多环境并行
    manager = ParallelSolverManager(
        base_cfg_path=cfg_path,
        n_envs=4,
        env_configs=[
            {'solver': {'boundary': {'velocity': {'x-': {'value': [1.0, 0.0]}}}}},
            {'solver': {'boundary': {'velocity': {'x-': {'value': [2.0, 0.0]}}}}},
            ...
        ],
    )
    results = manager.step_all(parallel=True)
    ```
"""
from .karman_torch_solver import KarmanTorchSolver
from .parallel_solver_manager import ParallelSolverManager, EnvState

__all__ = [
    'KarmanTorchSolver',
    'ParallelSolverManager',
    'EnvState',
]
