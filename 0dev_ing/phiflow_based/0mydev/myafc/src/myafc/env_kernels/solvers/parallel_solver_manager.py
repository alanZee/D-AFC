"""
ParallelSolverManager - 多实例 CFD Solver 管理器

为每个环境创建独立的 KarmanSolver 实例，每个实例可以有完全独立的配置
（包括边界条件、黏度、障碍物等）。

**重要说明**:
    PhiFlow 内部使用全局状态（如 JIT 缓存），不是线程安全的。
    因此 step_all(parallel=True) 可能导致竞态条件错误。
    建议使用 parallel=False 顺序执行，或使用多进程方式。

使用方法:
    manager = ParallelSolverManager(
        base_cfg_path="config.yaml",
        n_envs=4,
        env_variants=[
            {"solver": {"boundary": {"velocity": {"x-": {"value": [1.0, 0.0]}}}}},
            {"solver": {"boundary": {"velocity": {"x-": {"value": [2.0, 0.0]}}}}},
            ...
        ]
    )
    
    # 顺序执行一步（推荐，避免 PhiFlow 线程安全问题）
    results = manager.step_all(parallel=False)
    
    # 获取所有环境状态
    states = manager.get_all_states()
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from copy import deepcopy
import tempfile
import yaml
import time
from typing import Union

from myafc.utils import get_logger

LOGGER = get_logger()


@dataclass
class EnvState:
    """单个环境的状态数据"""
    env_idx: int
    velocity: np.ndarray       # (Nx+1, Ny+1, 2)
    pressure: np.ndarray       # (Nx, Ny)
    force_coeffs: Optional[Tuple[float, float]] = None  # (Cd, Cl)
    current_step: int = 0
    current_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'env_idx': self.env_idx,
            'velocity': self.velocity,
            'pressure': self.pressure,
            'force_coeffs': self.force_coeffs,
            'current_step': self.current_step,
            'current_time': self.current_time,
        }


def _create_solver_process(cfg_path: str, env_config: Dict, device: str = 'cuda'):
    """
    在独立进程中创建 Solver
    
    注意: 此函数会在子进程中执行
    """
    import sys
    # 确保导入路径正确
    from myafc.env_kernels.solvers.karman_torch_solver import KarmanSolver
    from myafc.env_kernels.configs.solver_config import SolverConfig
    
    # 合并配置
    if env_config:
        # 创建临时配置文件
        with open(cfg_path, 'r', encoding='utf-8') as f:
            base_cfg = yaml.safe_load(f)
        
        # 深度合并配置
        merged_cfg = _deep_merge(base_cfg, env_config)
        
        temp_dir = tempfile.gettempdir()
        temp_cfg_path = Path(temp_dir) / f'env_cfg_{mp.current_process().pid}.yaml'
        with open(temp_cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(merged_cfg, f, sort_keys=False)
        cfg_path = str(temp_cfg_path)
    
    # 创建 solver
    solver = KarmanSolver(cfg_path=cfg_path)
    return solver


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典"""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _solver_step_worker(args):
    """
    Solver step worker function (用于 multiprocessing)
    
    Args:
        args: (solver_state_dict, control, state_dict, env_idx)
    
    Returns:
        (env_idx, velocity, pressure, force_coeffs, step, time)
    """
    solver_pickle, control, state_dict, env_idx = args
    
    # 多进程演示占位：不推进求解器，场数据和力系数均为空。
    
    return (env_idx, None, None, None, 0, 0.0)


class ParallelSolverManager:
    """
    多环境并行 Solver 管理器
    
    使用 ThreadPoolExecutor 实现多线程并行（因为 PyTorch/CUDA 在多进程中有限制）
    每个环境独立的 KarmanSolver 实例，支持完全不同的配置。
    
    多环境语义:
    - n_envs=1: 仅使用基础环境，无派生
    - n_envs=n (n>1): 1个基础环境 + (n-1)个派生环境 = n个环境
    - env_variants: 长度为 n_envs-1，为每个派生环境提供差异化配置
    
    示例:
        n_envs=3, env_variants=[{...}, {...}]
        → Env 0: 基础配置
        → Env 1: 基础配置 + env_variants[0] 覆盖
        → Env 2: 基础配置 + env_variants[1] 覆盖
    
    注意:
    - 使用多线程而非多进程，因为 CUDA 张量不能跨进程共享
    - 如需真正的多进程并行，需要使用 CPU 模式或共享内存
    """
    
    def __init__(
        self,
        base_cfg_path: str,
        n_envs: int = 1,
        env_variants: Optional[List[Dict]] = None,
        max_workers: Optional[int] = None,
        use_multiprocessing: bool = False,  # 是否使用多进程（CPU模式）
        device: str = 'cuda',
    ):
        """
        Args:
            base_cfg_path: 基础配置文件路径（Env 0 使用此配置）
            n_envs: 总环境数量（1=仅基础环境，n>1=1个基础环境 + n-1个派生环境）
            env_variants: 派生环境的配置覆盖列表，长度必须为 n_envs-1
            max_workers: 最大并行工作线程/进程数，None=使用 n_envs
            use_multiprocessing: 是否使用多进程（仅 CPU 模式有效）
            device: 设备 ('cuda' 或 'cpu')
        """
        self.base_cfg_path = base_cfg_path
        self.n_envs = n_envs
        self.env_variants = env_variants or []
        self.max_workers = max_workers or n_envs
        self.use_multiprocessing = use_multiprocessing and (device == 'cpu')
        self.device = device
        
        # 验证 env_variants 长度
        expected_variants = n_envs - 1
        if len(self.env_variants) != expected_variants:
            raise ValueError(
                f"env_variants 长度 ({len(self.env_variants)}) 必须等于 n_envs-1 ({expected_variants})。"
                f"n_envs={n_envs} 表示 1个基础环境 + {expected_variants}个派生环境。"
            )
        
        # 创建所有 solver 实例
        self.solvers: List = []
        self._create_solvers()
        
        LOGGER.info(f"[ParallelSolverManager] 初始化完成 | n_envs={n_envs} (1 base + {n_envs-1} derived) | workers={self.max_workers}")
    
    def _create_solvers(self):
        """创建所有 solver 实例
        
        Env 0: 使用基础配置
        Env 1~n-1: 使用基础配置 + env_variants[env_idx-1] 覆盖
        """
        from myafc.env_kernels.solvers.karman_torch_solver import KarmanTorchSolver
        
        LOGGER.info(f"[ParallelSolverManager] 创建 {self.n_envs} 个 Solver 实例 (1 base + {self.n_envs-1} derived)...")
        
        for env_idx in range(self.n_envs):
            if env_idx == 0:
                # Env 0: 使用基础配置
                cfg_path = self.base_cfg_path
                env_type = "base"
            else:
                # Env 1~n-1: 使用 env_variants[env_idx-1] 覆盖基础配置
                env_config = self.env_variants[env_idx - 1]
                
                with open(self.base_cfg_path, 'r', encoding='utf-8') as f:
                    base_cfg = yaml.safe_load(f)
                
                merged_cfg = _deep_merge(base_cfg, env_config)
                
                temp_dir = tempfile.gettempdir()
                temp_cfg_path = Path(temp_dir) / f'parallel_env_{env_idx}.yaml'
                with open(temp_cfg_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(merged_cfg, f, sort_keys=False)
                cfg_path = str(temp_cfg_path)
                env_type = "derived"
            
            solver = KarmanTorchSolver(cfg_path=cfg_path)
            self.solvers.append(solver)
            
            LOGGER.info(f"  Env {env_idx} ({env_type}): 创建成功 | 入口流速: {self._get_inflow_velocity(solver)}")
    
    def _get_inflow_velocity(self, solver) -> List[float]:
        """获取 solver 的入口流速"""
        try:
            bc = solver.cfg.boundary.velocity
            if hasattr(bc, 'x_minus') and bc.x_minus:
                return bc.x_minus.get('value', [None, None])
            return [None, None]
        except:
            return [None, None]
    
    def step_all(
        self,
        controls: Optional[List[Any]] = None,
        states: Optional[List[Dict]] = None,
        parallel: bool = True,
    ) -> List[EnvState]:
        """
        并行执行所有环境的一步仿真
        
        Args:
            controls: 每个环境的控制信号列表
            states: 每个环境的状态字典列表
            parallel: 是否并行执行
        
        Returns:
            所有环境的状态列表
        """
        if controls is None:
            controls = [None] * self.n_envs
        if states is None:
            states = [{}] * self.n_envs
        
        if parallel and self.n_envs > 1:
            return self._step_parallel(controls, states)
        else:
            return self._step_sequential(controls, states)
    
    def _step_sequential(
        self,
        controls: List[Any],
        states: List[Dict],
    ) -> List[EnvState]:
        """顺序执行所有环境"""
        results = []
        for env_idx in range(self.n_envs):
            result = self._step_single(env_idx, controls[env_idx], states[env_idx])
            results.append(result)
        return results
    
    def _step_parallel(
        self,
        controls: List[Any],
        states: List[Dict],
    ) -> List[EnvState]:
        """并行执行所有环境"""
        results = [None] * self.n_envs
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for env_idx in range(self.n_envs):
                future = executor.submit(
                    self._step_single, env_idx, controls[env_idx], states[env_idx]
                )
                futures[future] = env_idx
            
            for future in as_completed(futures):
                env_idx = futures[future]
                try:
                    results[env_idx] = future.result()
                except Exception as e:
                    LOGGER.error(f"[ParallelSolverManager] Env {env_idx} step 失败: {e}")
                    raise
        
        return results
    
    def _step_single(
        self,
        env_idx: int,
        control: Optional[Any],
        state: Dict,
    ) -> EnvState:
        """执行单个环境的一步"""
        solver = self.solvers[env_idx]
        
        # 更新控制（如果有 jet_obstacle）
        if control is not None and solver.jet_obstacle is not None:
            # 设置控制策略
            class TempPolicy:
                def __init__(self, ctrl):
                    self.ctrl = ctrl
                def __call__(self, state):
                    return self.ctrl
            
            solver.jet_obstacle._policy = TempPolicy(control)
            solver.jet_obstacle.update_controls(state)
        
        # 执行一步
        velocity, pressure = solver.step(
            state=state,
            update_controls=(control is not None),
            apply_jets=(control is not None and solver.jet_obstacle is not None),
        )
        
        # 提取数据
        vel_tensor = velocity.staggered_tensor()
        vel_native = vel_tensor.native(vel_tensor.shape).cpu().numpy()
        
        p_tensor = pressure.values
        p_native = p_tensor.native(p_tensor.shape).cpu().numpy()
        
        # 计算力系数（如果有方法）
        force_coeffs = None
        if hasattr(solver, 'compute_force_coeffs'):
            try:
                cd, cl = solver.compute_force_coeffs()
                force_coeffs = (float(cd), float(cl))
            except:
                pass
        
        return EnvState(
            env_idx=env_idx,
            velocity=vel_native,
            pressure=p_native,
            force_coeffs=force_coeffs,
            current_step=solver.current_step,
            current_time=solver.current_time,
        )
    
    def reset_all(self, env_indices: Optional[List[int]] = None):
        """
        重置指定环境（或全部）
        
        Args:
            env_indices: 要重置的环境索引列表，None=全部
        """
        if env_indices is None:
            env_indices = list(range(self.n_envs))
        
        for env_idx in env_indices:
            self.solvers[env_idx].reset(use_h5=True)
        
        LOGGER.info(f"[ParallelSolverManager] 重置环境: {env_indices}")
    
    def reset(
        self,
        use_h5: bool = True,
        steps: Optional[Union[int, List[int]]] = None,
        resume_strict: Optional[bool] = None,
    ) -> Optional[List[Dict]]:
        """重置所有环境（兼容单 Solver 接口）
        
        统一的重置接口，与 KarmanTorchSolver.reset() 语义一致。
        
        Args:
            use_h5: 是否从 h5 文件初始化流场
                - True（默认）: 从 h5 文件加载
                - False: 从配置参数初始化
            steps: h5 帧索引（仅 use_h5=True 时有效）
                - None: 各环境使用自身配置中的 resume_step
                - int: 所有环境使用同一帧
                - List[int]: 每个环境使用不同帧，长度必须等于 n_envs
            resume_strict: 严格续算模式（仅 use_h5=True 时有效）
                - None: 各环境使用自身配置中的 initial.resume_strict
                - False: 热启动模式，时间步从 0 开始
                - True: 严格续算模式，时间步和物理时间接续 h5 帧
        
        Returns:
            use_h5=True 时返回各环境的加载信息列表，否则返回 None
        """
        if not use_h5:
            # 从配置参数初始化
            for solver in self.solvers:
                solver.reset(use_h5=False)
            LOGGER.info(f"[ParallelSolverManager] 从配置参数重置 {self.n_envs} 个环境")
            return None
        
        # 从 h5 文件初始化
        if steps is None:
            steps = [self.solvers[i].cfg.initial.resume_step for i in range(self.n_envs)]
        
        if isinstance(steps, int):
            steps = [steps] * self.n_envs
        
        if len(steps) != self.n_envs:
            raise ValueError(f"[ParallelSolverManager] steps 长度 ({len(steps)}) 必须等于 n_envs ({self.n_envs})")
        
        resume_infos = []
        for env_idx, step in enumerate(steps):
            info = self.solvers[env_idx].reset(use_h5=True, step=step, resume_strict=resume_strict)
            resume_infos.append(info)
        
        loaded_steps = [info['step'] for info in resume_infos]
        LOGGER.info(f"[ParallelSolverManager] 从 h5 重置 | 帧: {loaded_steps}")
        return resume_infos
    
    def get_all_states(self) -> List[EnvState]:
        """获取所有环境的当前状态"""
        states = []
        for env_idx in range(self.n_envs):
            solver = self.solvers[env_idx]
            
            vel_tensor = solver.velocity.staggered_tensor()
            vel_native = vel_tensor.native(vel_tensor.shape).cpu().numpy()
            
            p_tensor = solver.pressure.values
            p_native = p_tensor.native(p_tensor.shape).cpu().numpy()
            
            states.append(EnvState(
                env_idx=env_idx,
                velocity=vel_native,
                pressure=p_native,
                force_coeffs=None,
                current_step=solver.current_step,
                current_time=solver.current_time,
            ))
        
        return states
    
    def get_solver(self, env_idx: int):
        """获取指定环境的 solver"""
        return self.solvers[env_idx]
    
    @property
    def dt(self) -> float:
        """返回第一个环境的时间步长"""
        return self.solvers[0].cfg.time.dt if self.solvers else 0.0
    
    @property
    def cfg(self):
        """返回第一个环境的配置（用于与单 solver 接口兼容）"""
        return self.solvers[0].cfg if self.solvers else None
    
    @property
    def resolution(self) -> List[int]:
        """返回分辨率"""
        return self.solvers[0].cfg.domain.resolution if self.solvers else [0, 0]
    
    @property
    def domain_size(self) -> List[float]:
        """返回域大小"""
        return self.solvers[0].cfg.domain.size if self.solvers else [0.0, 0.0]
    
    def __getattr__(self, name: str):
        """代理未定义的属性到第一个 solver（用于与单 solver 接口兼容）
        
        这样 ParallelSolverManager 可以作为 KarmanTorchSolver 的透明替代
        """
        # 避免无限递归：solvers 必须通过 __dict__ 访问
        if name == 'solvers':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'solvers'")
        
        solvers = self.__dict__.get('solvers', [])
        if solvers:
            return getattr(solvers[0], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value):
        """设置属性时，如果是 solver 的属性，则同步设置到所有 solver"""
        # 这些是 ParallelSolverManager 自身的属性
        own_attrs = {'base_cfg_path', 'n_envs', 'env_variants', 'max_workers', 
                     'use_multiprocessing', 'device', 'solvers'}
        if name in own_attrs:
            super().__setattr__(name, value)
        elif 'solvers' in self.__dict__ and self.__dict__['solvers']:
            # 同步设置到所有 solver
            for solver in self.__dict__['solvers']:
                setattr(solver, name, value)
        else:
            super().__setattr__(name, value)
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'solvers'):
            self.solvers.clear()


class AsyncParallelSolverManager(ParallelSolverManager):
    """
    异步并行 Solver 管理器
    
    使用异步执行模式，允许在等待结果时执行其他操作。
    适合与神经网络训练配合使用。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._pending_futures = {}
    
    def step_all_async(
        self,
        controls: Optional[List[Any]] = None,
        states: Optional[List[Dict]] = None,
    ) -> List:
        """
        异步执行所有环境的一步
        
        Returns:
            futures 列表
        """
        if controls is None:
            controls = [None] * self.n_envs
        if states is None:
            states = [{}] * self.n_envs
        
        futures = []
        for env_idx in range(self.n_envs):
            future = self._executor.submit(
                self._step_single, env_idx, controls[env_idx], states[env_idx]
            )
            futures.append(future)
            self._pending_futures[future] = env_idx
        
        return futures
    
    def wait_for_results(self, futures: List) -> List[EnvState]:
        """等待异步执行完成并返回结果"""
        results = [None] * len(futures)
        
        for future in as_completed(futures):
            env_idx = self._pending_futures.pop(future, None)
            if env_idx is not None:
                results[env_idx] = future.result()
        
        return results
    
    def __del__(self):
        self._executor.shutdown(wait=False)
        super().__del__()
