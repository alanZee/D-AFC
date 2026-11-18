"""
Kármán 圆柱绕流求解器（基于 PhiFlow Torch 后端）

功能特性:
- 支持长方形求解域，长宽与分辨率可独立配置
- 支持初始流场速度自定义设置（或自动从 inflow 边界获取）
- 支持边界条件独立配置（参考 PhiFlow extrapolation）
- 支持流体物性（黏度、密度）配置
- 支持多种障碍物类型（圆柱、方柱等）及旋转
- 支持外力场（forcing）配置
- 支持结构物表面射流边界条件（JetObstacle）
- 支持可微/不可微模式切换
- 支持 GPU/CPU 设备选择

参考:
- Solver-in-the-Loop (karman-2d-phi2/karman.py)
- PhiFlow 官方文档与示例

使用方法:
    from myafc.env_kernels.solvers.karman_torch_solver import KarmanTorchSolver
    solver = KarmanTorchSolver("path/to/config.yaml")
    results = solver.run()  # 或逐步: solver.step()
"""
from __future__ import annotations
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field as dataclass_field

import yaml
import torch
import numpy as np

# ============================================================
# 日志配置（使用 myafc.utils.logging 统一模块）
# ============================================================
from myafc.utils import get_logger, setup_file_logging, set_log_level
# 使用现代化的 PhiFlow I/O 工具
from myafc.utils.phiflow_io import load_fields_h5, centered_to_staggered

LOGGER = get_logger()

# ============================================================
# PhiFlow 导入 (Torch 后端)
# ============================================================
from phi.torch.flow import (
    CenteredGrid, StaggeredGrid,
    Box, Sphere,
    math, extrapolation, channel, spatial, batch,
    advect, diffuse, fluid,
    Solve, vec, Noise,
    jit_compile,
)
from phi.physics.fluid import Obstacle

# JetObstacle 导入（延迟导入）
JetObstacle = None


# 导入 utils 模块会自动执行 monkey-patch（模块级副作用）
from myafc.env_kernels.solvers.utils import _InletZeroGradient


def _lazy_import_jet_obstacle():
    """延迟导入 JetObstacle"""
    global JetObstacle
    if JetObstacle is None:
        try:
            from myafc.env_kernels.solvers.fluid_control import JetObstacle as _JetObstacle
            JetObstacle = _JetObstacle
        except ImportError:
            pass
    return JetObstacle


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class DomainConfig:
    """求解域配置"""
    size: List[float] = dataclass_field(default_factory=lambda: [200.0, 100.0])
    resolution: List[int] = dataclass_field(default_factory=lambda: [128, 64])


@dataclass
class ObstacleConfig:
    """障碍物配置"""
    enabled: bool = True
    shape: str = 'sphere'  # 'sphere' / 'box'
    center: List[float] = dataclass_field(default_factory=lambda: [0.25, 0.5])  # 相对坐标 [0~1]
    radius: float = 10.0   # 绝对尺寸（用于 sphere）
    size: List[float] = dataclass_field(default_factory=lambda: [10.0, 10.0])  # 绝对尺寸（用于 box）
    rotation: float = 0.0  # 旋转角度（弧度）


@dataclass
class FluidConfig:
    """流体物性配置"""
    density: float = 1.0
    viscosity: float = 0.001


@dataclass 
class InitialConditionConfig:
    """初始条件配置
    
    resume_strict 模式说明:
        False (默认): 热启动模式
            - 仅使用场文件初始化流场（速度、压力）
            - 输出保存到新目录，仿真时间从零开始
            - 适合：基于已有流场开始新的独立仿真
        
        True: 严格续算模式
            - 接着场文件的指定时间步继续仿真
            - 输出保持与场文件所属目录一致
            - 删除指定时间步之后的所有数据并重新生成
            - 仿真时间、图片编号等都接续
            - 适合：仿真中断后继续运行
    """
    velocity: Optional[List[float]] = None  # None = 自动从 inflow 获取
    resume_from: Optional[str] = None  # h5 文件路径
    resume_step: int = -1  # 从哪一步恢复，-1 表示最后一帧
    resume_strict: bool = False  # 是否严格续算模式


@dataclass
class BoundaryConfig:
    """边界条件配置
    
    inlet_profile 说明:
        'uniform'   — 均匀入口 U(y) = value (常数)
        'parabolic' — 抛物线入口 U(y) = 1.5*value * 4y(H-y)/H²
    
    value 约定:
        inflow.value 始终代表截面平均速度 U_mean，无论 profile 类型。
        Re = U_mean * D / ν，Cd = Fx / (0.5ρ U_mean² A)
        parabolic 时内部计算 U_max = 1.5 × U_mean (峰值速度)
        
        DRLinFluids 参考: OpenFOAM 的 maxValue 参数 = U_max (峰值)，
        而 magUInf (力系数参考速度) = U_mean；两种参数化等价。
    
    ⚠️ NOTE 入口边界条件实现 (uniform / parabolic):
        通过 _init_inflow_cache() 将入口边界统一改为 _InletZeroGradient()。
        避免 ConstantExtrapolation(vec) 的 vector 维度在 mac_cormack 中导致
        closest_values pad 断言失败。
        入口面值存储在 StaggeredGrid 数据中，可被 _enforce_inflow() 修改。
    """
    velocity: Dict[str, Dict] = dataclass_field(default_factory=dict)
    pressure: Dict[str, Dict] = dataclass_field(default_factory=dict)
    inlet_profile: str = 'parabolic'  # 'uniform' / 'parabolic'


@dataclass
class ForcingConfig:
    """外力配置"""
    enabled: bool = False
    type: str = 'constant'  # 'constant' / 'kolmogorov'
    # constant forcing 参数
    force: List[float] = dataclass_field(default_factory=lambda: [0.0, 0.0])
    # kolmogorov forcing 参数
    amplitude: float = 1.0
    wavenumber: int = 4
    noise_scale: float = 0.0


@dataclass
class JetsConfig:
    """
    射流控制配置
    
    射流边界条件通过 JetObstacle 实现:
        - JetObstacle 继承自 PhiFlow 的 Obstacle 类
        - JetObstacle.velocity 属性返回空间变化的射流速度场
        - 在 make_incompressible() 中自动应用射流 BC
        
    工作流程:
        1. step 开始时调用 jet_obstacle.update_controls(state) 计算控制信号
        2. make_incompressible(velocity, obstacles=[jet_obstacle]) 进行压力投影
           - PhiFlow 内部调用 apply_boundary_conditions
           - 自动将障碍物区域的速度设置为 jet_obstacle.velocity
    """
    enabled: bool = False
    nozzles: List[Dict] = dataclass_field(default_factory=list)
    smoothing_alpha: float = 1.0  # 平滑系数
    saturation: float = 10.0  # 幅值上限（作用于缩放之前）
    amplitude_scale: float = 1.0
    profile: str = 'uniform'
    policy: Dict = dataclass_field(default_factory=dict)


@dataclass
class TimeConfig:
    """时间步进配置
    
    diffusion_method 说明:
        'explicit'      — diffuse.explicit()，条件稳定 (dt < dx²/(2ν))，速度快
        'implicit'      — diffuse.implicit()，无条件稳定，使用 CG 线性求解器
                          ⚠️ NOTE 已知限制: PhiFlow 当前版本对 StaggeredGrid + 非周期 BC
                          支持不完善，运行时可能触发 MixedShape 错误，此时自动回退
                          到 auto_substep。推荐直接使用 auto_substep。
        'auto_substep'  — diffuse.explicit(substeps=N)，自动计算子步数保证稳定性（推荐）
    
    advection_method 说明:
        'semi_lagrangian' — advect.semi_lagrangian()，一阶精度，数值扩散较大（默认）
        'mac_cormack'     — advect.mac_cormack()，二阶精度，使用前向+后向追溯
                            减少数值扩散，有助于更好地捕捉涡脱落振荡
                            计算量约为 semi_lagrangian 的 2 倍
    """
    dt: float = 1.0
    num_steps: int = 100
    diffusion_method: str = 'auto_substep'  # 'explicit' / 'implicit' / 'auto_substep'
    advection_method: str = 'semi_lagrangian'  # 'semi_lagrangian' / 'mac_cormack'


@dataclass
class RuntimeConfig:
    """运行时配置"""
    differentiable: bool = True
    device: str = "cuda"
    random_seed: Optional[int] = 42
    # 日志详细级别: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    verbose: str = 'INFO'


@dataclass
class SolverConfig:
    """求解器完整配置"""
    domain: DomainConfig = dataclass_field(default_factory=DomainConfig)
    obstacle: ObstacleConfig = dataclass_field(default_factory=ObstacleConfig)
    fluid: FluidConfig = dataclass_field(default_factory=FluidConfig)
    initial: InitialConditionConfig = dataclass_field(default_factory=InitialConditionConfig)
    boundary: BoundaryConfig = dataclass_field(default_factory=BoundaryConfig)
    forcing: ForcingConfig = dataclass_field(default_factory=ForcingConfig)
    jets: JetsConfig = dataclass_field(default_factory=JetsConfig)
    time: TimeConfig = dataclass_field(default_factory=TimeConfig)
    runtime: RuntimeConfig = dataclass_field(default_factory=RuntimeConfig)


# ============================================================
# 配置解析工具
# ============================================================
def parse_config(cfg_path: str) -> SolverConfig:
    """从 YAML 文件解析配置"""
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    
    cfg_dict = raw.get('solver', raw)
    config = SolverConfig()
    
    if 'domain' in cfg_dict:
        config.domain = DomainConfig(**cfg_dict['domain'])
    if 'obstacle' in cfg_dict:
        config.obstacle = ObstacleConfig(**cfg_dict['obstacle'])
    if 'fluid' in cfg_dict:
        config.fluid = FluidConfig(**cfg_dict['fluid'])
    if 'initial' in cfg_dict:
        config.initial = InitialConditionConfig(**cfg_dict['initial'])
    if 'boundary' in cfg_dict:
        config.boundary = BoundaryConfig(**cfg_dict['boundary'])
    if 'forcing' in cfg_dict:
        config.forcing = ForcingConfig(**cfg_dict['forcing'])
    if 'jets' in cfg_dict:
        config.jets = JetsConfig(**cfg_dict['jets'])
    if 'time' in cfg_dict:
        time_dict = cfg_dict['time'].copy()
        # 确保 dt 是 float（YAML 可能把科学计数法解析为字符串）
        if 'dt' in time_dict:
            time_dict['dt'] = float(time_dict['dt'])
        config.time = TimeConfig(**time_dict)
    if 'runtime' in cfg_dict:
        config.runtime = RuntimeConfig(**cfg_dict['runtime'])
    
    return config


def _parse_single_boundary_value(bc_spec: Dict, is_velocity: bool = True) -> Any:
    """解析单个边界条件"""
    from phi.torch.flow import vec, ZERO_GRADIENT, PERIODIC
    
    bc_type = bc_spec.get('type', 'zero-gradient').lower()
    value = bc_spec.get('value', None)
    
    if bc_type in ('dirichlet', 'inflow'):
        if value is None:
            return 0
        if isinstance(value, dict):
            return vec(x=value.get('x', 0.0), y=value.get('y', 0.0))
        if isinstance(value, list):
            return vec(x=value[0], y=value[1]) if is_velocity else value[0]
        return value
    
    elif bc_type in ('wall', 'no-slip'):
        return 0
    
    elif bc_type in ('zero-gradient', 'neumann', 'outflow'):
        return ZERO_GRADIENT
    
    elif bc_type == 'periodic':
        return PERIODIC
    
    elif bc_type in ('symmetry', 'slip', 'mirror'):
        LOGGER.warning(f"[Solver] 边界类型 '{bc_type}' 尚未准确实现，暂用零梯度近似")
        return ZERO_GRADIENT
    
    else:
        LOGGER.warning(f"[Solver] 未知边界类型 '{bc_type}'，使用零梯度")
        return ZERO_GRADIENT


def _find_inflow_velocity(boundary_cfg: Dict[str, Dict]) -> Tuple[Optional[List[float]], int]:
    """
    从边界配置中查找 inflow 类型边界的速度值
    
    Returns:
        (velocity, count): inflow 速度值和 inflow 边界数量
    """
    inflow_velocity = None
    inflow_count = 0
    
    for side, bc_spec in boundary_cfg.items():
        if isinstance(bc_spec, dict):
            bc_type = bc_spec.get('type', '').lower()
            if bc_type == 'inflow':
                inflow_count += 1
                value = bc_spec.get('value')
                if value is not None:
                    if isinstance(value, list):
                        inflow_velocity = value
                    elif isinstance(value, dict):
                        inflow_velocity = [value.get('x', 0.0), value.get('y', 0.0)]
    
    return inflow_velocity, inflow_count


def _find_inflow_key(boundary_cfg: Dict[str, Dict]) -> Optional[str]:
    """查找 inflow 边界的 side key（如 'x-'）"""
    for side, bc_spec in boundary_cfg.items():
        if isinstance(bc_spec, dict) and bc_spec.get('type', '').lower() == 'inflow':
            return side
    return None


def build_boundary_dict(boundary_cfg: Dict[str, Dict], is_velocity: bool = True) -> Dict:
    """根据边界配置构建 PhiFlow 边界字典"""
    from phi.torch.flow import ZERO_GRADIENT, vec
    
    if not boundary_cfg:
        if is_velocity:
            return {'x-': vec(x=1, y=0), 'x+': ZERO_GRADIENT, 'y': 0}
        else:
            return {'x': ZERO_GRADIENT, 'y': ZERO_GRADIENT}
    
    result = {}
    for side in ['x-', 'x+', 'y-', 'y+']:
        bc_spec = boundary_cfg.get(side, {'type': 'zero-gradient'})
        result[side] = _parse_single_boundary_value(bc_spec, is_velocity)
    
    return result


# ============================================================
# Kármán 圆柱绕流求解器
# ============================================================
class KarmanTorchSolver:
    """
    基于 PhiFlow Torch 后端的 Kármán 圆柱绕流求解器
    
    核心求解流程:
    1. 黏性扩散 (diffuse)
    2. 边界条件施加 (apply BCs)
    3. 对流 (advect)
    4. 外力项 (forcing) [可选]
    5. 压力投影 (make_incompressible)
    """

    # 连续回退阈值: 超过此次数后直接使用 CG-relaxed
    _FALLBACK_STREAK_THRESHOLD = 10
    
    def __init__(self, cfg_path: str, log_file: Optional[str] = None):
        """
        初始化求解器
        
        Args:
            cfg_path: YAML 配置文件路径
            log_file: 日志文件路径（可选，会自动创建目录）
        """
        # 设置日志文件
        if log_file:
            setup_file_logging(log_file)
        
        # 解析配置
        self.cfg = parse_config(cfg_path)
        
        # 设置日志级别（配置解析后，确保后续日志按配置输出）
        set_log_level(self.cfg.runtime.verbose)
        
        # 现在才输出配置信息（受 verbose 设置控制）
        LOGGER.info(f"[Solver] 加载配置文件: {cfg_path}")
        
        self._validate_config()
        self._setup_backend()
        self._init_fields()
        
        # 可微分控制
        self.differentiable = self.cfg.runtime.differentiable

        # JIT 编译 _step_impl（用于 no_grad 模式）
        try:
            self._step_impl_jit = math.jit_compile(self._step_impl)
            LOGGER.debug("[Solver] JIT 编译 _step_impl 成功")
        except Exception as e:
            LOGGER.warning(f"[Solver] JIT 编译失败，回退到普通调用: {e}")
            self._step_impl_jit = self._step_impl

        self.current_step = 0
        self.current_time = 0.0
        LOGGER.info(f"[Solver] 求解器初始化完成 | 域: {self.cfg.domain.size} | "
                    f"分辨率: {self.cfg.domain.resolution} | dt: {self.cfg.time.dt}"
                    f"{'' if self.differentiable else ' | 梯度禁用'}")
    
    def _validate_config(self):
        """验证配置合法性（含 CFL / 扩散稳定性检查）"""
        cfg = self.cfg
        
        # 检查 inflow 边界数量（最多一个）
        inflow_vel, inflow_count = _find_inflow_velocity(cfg.boundary.velocity)
        if inflow_count > 1:
            raise ValueError(f"inflow 类型的边界条件最多只能有 1 个，当前有 {inflow_count} 个")
        
        # 验证 diffusion_method
        valid_methods = ('explicit', 'implicit', 'auto_substep')
        if cfg.time.diffusion_method not in valid_methods:
            raise ValueError(
                f"[Solver] ❌ 未知的 diffusion_method: '{cfg.time.diffusion_method}'. "
                f"可选: {valid_methods}"
            )
        
        # 验证 inlet_profile
        valid_profiles = ('uniform', 'parabolic')
        if cfg.boundary.inlet_profile not in valid_profiles:
            raise ValueError(
                f"[Solver] ❌ 未知的 inlet_profile: '{cfg.boundary.inlet_profile}'. "
                f"可选: {valid_profiles}"
            )
        
        # ===== 数值稳定性检查 =====
        # 显式扩散稳定性: dt < dx_min² / (2ν)
        # 对流 CFL (semi-Lagrangian 不严格要求但影响精度): CFL = U*dt/dx
        dt = cfg.time.dt
        Lx, Ly = cfg.domain.size
        Nx, Ny = cfg.domain.resolution
        dx, dy = Lx / Nx, Ly / Ny
        dx_min = min(dx, dy)
        nu = cfg.fluid.viscosity
        diffusion_method = cfg.time.diffusion_method
        
        if nu > 0:
            dt_diffusion_max = dx_min**2 / (2 * nu)
            if dt > dt_diffusion_max:
                if diffusion_method == 'explicit':
                    # 显式方法: dt 超限直接报错
                    raise ValueError(
                        f"[Solver] ❌ 显式扩散不稳定! "
                        f"dt={dt:.2e} > dt_max={dt_diffusion_max:.2e} "
                        f"(dx_min={dx_min:.4f}, ν={nu:.2e}). "
                        f"请减小 dt 至 {dt_diffusion_max:.2e} 以下，"
                        f"或切换 diffusion_method 为 'implicit' / 'auto_substep'。"
                        f"DRLinFluids 参考: dt=5e-4"
                    )
                elif diffusion_method == 'auto_substep':
                    # auto_substep: 自动计算子步数，仅警告
                    import math as _math
                    substeps = _math.ceil(nu * dt / (0.4 * dx_min**2))
                    LOGGER.warning(
                        f"[Solver] ⚠️ dt={dt:.2e} 超过显式扩散稳定上限 dt_max={dt_diffusion_max:.2e}，"
                        f"auto_substep 将自动使用 {substeps} 个子步以保证稳定性"
                    )
                else:
                    # implicit: 无条件稳定，仅提示
                    LOGGER.info(
                        f"[Solver] dt={dt:.2e} 超过显式扩散稳定上限 dt_max={dt_diffusion_max:.2e}，"
                        f"使用隐式扩散 (diffuse.implicit)，无条件稳定。"
                        f"⚠️ 注意: PhiFlow 对 StaggeredGrid+非周期BC 支持有限，"
                        f"运行时若出错将自动回退到 auto_substep"
                    )
            # 接近上限时警告 (余量 < 50%)，仅对显式方法有意义
            elif dt > dt_diffusion_max * 0.5 and diffusion_method == 'explicit':
                LOGGER.warning(
                    f"[Solver] ⚠️ 显式扩散稳定性余量不足! "
                    f"dt={dt:.2e}, dt_max={dt_diffusion_max:.2e} "
                    f"(余量 {(1 - dt/dt_diffusion_max)*100:.0f}%). "
                    f"建议减小 dt 或切换 diffusion_method"
                )
        
        advection_method = cfg.time.advection_method
        LOGGER.info(f"[Solver] 扩散方法: {diffusion_method} | 对流方案: {advection_method} | 入口速度分布: {cfg.boundary.inlet_profile}")
        
        if inflow_vel is not None:
            U_mag = (inflow_vel[0]**2 + inflow_vel[1]**2)**0.5
            cfl = U_mag * dt / dx_min
            if cfl > 2.0:
                LOGGER.warning(
                    f"[Solver] ⚠️ CFL = {cfl:.2f} > 2.0 (U={U_mag:.1f}, dt={dt:.2e}, "
                    f"dx_min={dx_min:.4f}). 精度可能较差，推荐 CFL < 1.0"
                )
            elif cfl > 1.0:
                LOGGER.info(f"[Solver] CFL = {cfl:.2f} (可接受，推荐 < 1.0)")
        
        LOGGER.info(f"[Solver] 配置验证通过 | inflow 边界数: {inflow_count}")
    
    def _setup_backend(self):
        """配置后端与设备"""
        device = self.cfg.runtime.device.lower()
        
        if device == 'cuda' and not torch.cuda.is_available():
            LOGGER.warning("[Solver] CUDA 不可用，回退到 CPU")
            device = 'cpu'
        
        # 保存最终设备名称（供后续组件使用）
        self._device = device
        
        # 设置 PhiFlow 后端默认设备
        # PhiFlow 使用 'GPU' / 'CPU' 表示设备
        from phiml.backend.torch import TORCH
        phiflow_device = 'GPU' if device == 'cuda' else 'CPU'
        if TORCH.set_default_device(phiflow_device):
            LOGGER.info(f"[Solver] PhiFlow 后端设备: {phiflow_device}")
        else:
            LOGGER.warning(f"[Solver] PhiFlow 设置设备 {phiflow_device} 失败，使用默认设备")
        
        if self.cfg.runtime.random_seed is not None:
            torch.manual_seed(self.cfg.runtime.random_seed)
            math.seed(self.cfg.runtime.random_seed)
            LOGGER.info(f"[Solver] 随机种子: {self.cfg.runtime.random_seed}")
    
    def _init_fields(self):
        """初始化场变量"""
        cfg = self.cfg
        Lx, Ly = cfg.domain.size
        Nx, Ny = cfg.domain.resolution
        
        self.bounds = Box(x=Lx, y=Ly)
        
        # 构建边界条件
        self.vel_boundary = build_boundary_dict(cfg.boundary.velocity, is_velocity=True)
        self.pressure_boundary = build_boundary_dict(cfg.boundary.pressure, is_velocity=False)
        
        # 初始化障碍物
        self._init_obstacles()
        
        # 初始化 forcing
        self._init_forcing()
        
        # 黏性系数
        self.kinematic_viscosity = cfg.fluid.viscosity / cfg.fluid.density
        
        # resume 信息（供 runner 使用）
        self.resume_info: Optional[Dict[str, Any]] = None
        
        # 入口 profile 缓存（由 _init_inflow_cache 初始化）
        self._inflow_stag_profile = None
        self._inflow_enabled = False
        
        # 压力求解自适应回退：按策略独立计数
        # 记录“连续使用某个回退策略成功”的次数
        self._pressure_fallback_streaks = {}
        # 自适应首选策略：连续触发阈值后，后续优先直接尝试该策略
        self._pressure_preferred_fallback = None
        
        # 检查是否从 h5 文件恢复（优先于 initial.velocity）
        if cfg.initial.resume_from and os.path.exists(cfg.initial.resume_from):
            self.resume_info = self._load_from_h5(cfg.initial.resume_from, cfg.initial.resume_step)
            self.resume_info['strict'] = cfg.initial.resume_strict
            # 恢复场景也需要初始化入口缓存
            self._init_inflow_cache()
            return
        
        # 初始化速度场和压力场（公共逻辑）
        self._init_velocity_pressure()
        
        # 初始化入口边界条件（设置 tensor-valued extrapolation）
        self._init_inflow_cache()
    
    def _build_parabolic_inflow_bc(self):
        """构建抛物线入口速度分布（仅供初始场使用）
        
        抛物线速度分布: U(y) = U_max * 4*y*(H-y) / H²
        其中 U_max = 1.5 * U_mean，使得截面平均速度 = U_mean
        
        运行时的入口边界条件通过 _init_inflow_cache() 设置
        tensor-valued extrapolation 实现，无需手动修改 staggered tensor。
        
        注意: Ny 使用 cfg.domain.resolution（stagger 点数），因初始场
        的 values 需要与 StaggeredGrid(x=Nx, y=Ny) 参数匹配。
        
        Returns:
            (U_max_x, y_centers, profile_x) or None if no inflow
        """
        cfg = self.cfg
        Ly = cfg.domain.size[1]
        Ny = cfg.domain.resolution[1]
        
        inflow_vel, _ = _find_inflow_velocity(cfg.boundary.velocity)
        if inflow_vel is None:
            LOGGER.warning("[Solver] inlet_profile='parabolic' 但未找到 inflow 边界，跳过")
            return None
        
        U_mean_x = inflow_vel[0]
        
        # 计算 cell center y 坐标
        dy = Ly / Ny
        y_centers = torch.linspace(dy / 2, Ly - dy / 2, Ny, dtype=torch.float32)
        
        # 抛物线分布: U(y) = U_max * 4*y*(H-y)/H²
        # U_max = 1.5 * U_mean 保证截面平均 = U_mean
        U_max_x = 1.5 * U_mean_x
        profile_x = U_max_x * 4.0 * y_centers * (Ly - y_centers) / (Ly ** 2)
        
        LOGGER.debug(f"[Solver] 抛物线入口: U_max={U_max_x:.4f}, U_mean={U_mean_x:.4f}, Ny={Ny}")
        return U_max_x, y_centers, profile_x
    
    def _init_inflow_cache(self):
        """初始化入口 profile 运行时强制施加 (uniform / parabolic)
        
        策略:
            1. 将入口 (x-) 速度边界改为 _InletZeroGradient()
               → 行为与 ZERO_GRADIENT 完全相同 (valid_outer_faces=True)
               → 避免 ConstantExtrapolation 带 vector 维度导致
                 mac_cormack closest_values pad 断言失败
            2. 通过 _enforce_inflow() 在 _step_impl 的每个子步骤
               直接修改 staggered_tensor 入口面值
               → uniform: 常数 profile
               → parabolic: 抛物线 profile
        """
        cfg = self.cfg
        inflow_vel, _ = _find_inflow_velocity(cfg.boundary.velocity)
        inflow_key = _find_inflow_key(cfg.boundary.velocity)
        
        if inflow_vel is None or inflow_key is None:
            if cfg.boundary.inlet_profile == 'parabolic':
                LOGGER.warning("[Solver] inlet_profile='parabolic' 但未找到 inflow 边界，跳过")
            self._inflow_enabled = False
            self._inflow_stag_profile = None
            return
        
        U_mean_x = inflow_vel[0]
        
        # 将入口速度边界改为 _InletZeroGradient
        # → valid_outer_faces=True: boundary face 被存储在 grid 数据中
        # → 避免 vec(x=.., y=..) 的 ConstantExtrapolation 在 mac_cormack 中触发维度断言
        self.vel_boundary[inflow_key] = _InletZeroGradient()
        
        # 预计算入口 profile
        # PhiFlow StaggeredGrid(x=N) → N-1 cells; u_x 在入口面有 Ny_cells 个值
        Ny_stag = cfg.domain.resolution[1]
        Ny_cells = Ny_stag - 1
        Ly = cfg.domain.size[1]
        
        if cfg.boundary.inlet_profile == 'parabolic':
            dy = Ly / Ny_cells
            y_centers = torch.linspace(dy / 2, Ly - dy / 2, Ny_cells, dtype=torch.float32)
            U_max_x = 1.5 * U_mean_x
            profile = U_max_x * 4.0 * y_centers * (Ly - y_centers) / (Ly ** 2)
            LOGGER.info(f"[Solver] 入口 profile=parabolic 运行时强制施加: 已启用 "
                        f"(Ny_cells={Ny_cells}, U_max={U_max_x:.4f}, U_mean={U_mean_x:.4f})")
        else:
            # uniform: 所有面上均为 U_mean_x
            profile = torch.full((Ny_cells,), U_mean_x, dtype=torch.float32)
            LOGGER.info(f"[Solver] 入口 profile=uniform 运行时强制施加: 已启用 "
                        f"(Ny_cells={Ny_cells}, U_mean={U_mean_x:.4f})")
        
        self._inflow_stag_profile = profile
        self._inflow_enabled = True
        self._inflow_U_mean_x = U_mean_x
    
    def _enforce_inflow(self, velocity):
        """在 StaggeredGrid 的入口面强制施加入口速度分布 (uniform / parabolic)
        
        配合 _InletZeroGradient 入口边界使用。
        修改 staggered_tensor[0, :Ny_cells, 0] = profile 后重建 StaggeredGrid。
        必须在 _step_impl 的每个子步骤（扩散/对流/投影前后）调用。
        
        Args:
            velocity: 当前 StaggeredGrid 速度场
        
        Returns:
            修改后的 StaggeredGrid
        """
        if not self._inflow_enabled:
            return velocity
        
        vel_st = velocity.staggered_tensor()
        vel_native = vel_st.native(vel_st.shape)
        
        profile = self._inflow_stag_profile
        Ny_cells = len(profile)
        
        # 确保设备一致
        if profile.device != vel_native.device:
            profile = profile.to(vel_native.device)
            self._inflow_stag_profile = profile
        if profile.dtype != vel_native.dtype:
            profile = profile.to(vel_native.dtype)
            self._inflow_stag_profile = profile
        
        # 修改入口面: staggered_tensor[0, :Ny_cells, 0] = profile (uniform 或 parabolic)
        # 同时设入口面 u_y = 0: staggered_tensor[0, :, 1] = 0.0
        modified = vel_native.clone()
        modified[0, :Ny_cells, 0] = profile
        modified[0, :, 1] = 0.0  # 入口 u_y = 0
        
        result_phi = math.tensor(modified, spatial('x', 'y'), math.channel('vector'))
        return StaggeredGrid(
            values=result_phi,
            boundary=velocity.extrapolation,
            bounds=velocity.bounds,
        )
    
    def _init_velocity_pressure(self, verbose: bool = True):
        """初始化速度场和压力场（公共逻辑）
        
        被 _init_fields() 和 reset() 共同调用。
        
        Args:
            verbose: 是否输出详细日志（初始化时 True，reset 时 False）
        """
        cfg = self.cfg
        Nx, Ny = cfg.domain.resolution
        
        # 确定初始速度
        if cfg.initial.velocity is not None:
            Ux, Uy = cfg.initial.velocity
            if verbose:
                LOGGER.info(f"[Solver] 使用配置的初始速度: [{Ux}, {Uy}]")
        else:
            # 自动从 inflow 边界获取
            inflow_vel, _ = _find_inflow_velocity(cfg.boundary.velocity)
            if inflow_vel is not None:
                Ux, Uy = inflow_vel
                if verbose:
                    LOGGER.info(f"[Solver] 从 inflow 边界获取初始速度: [{Ux}, {Uy}]")
            else:
                Ux, Uy = 0.0, 0.0
                if verbose:
                    LOGGER.info("[Solver] 无 inflow 边界，初始速度设为 [0, 0]")
        
        # 初始速度场（支持抛物线入口分布）
        if cfg.boundary.inlet_profile == 'parabolic' and Ux != 0.0:
            # 使用 _build_parabolic_inflow_bc 获取抛物线分布
            parabolic_data = self._build_parabolic_inflow_bc()
            if parabolic_data is not None:
                U_max_x, _, profile_x = parabolic_data
                # 广播到 (Nx, Ny)
                profile_2d = profile_x.unsqueeze(0).expand(Nx, Ny)
                init_values = vec(
                    x=math.tensor(profile_2d.numpy(), spatial(x=Nx, y=Ny)),
                    y=math.tensor(float(Uy) * np.ones((Nx, Ny)), spatial(x=Nx, y=Ny))
                )
                self.velocity = StaggeredGrid(
                    values=init_values,
                    boundary=self.vel_boundary,
                    bounds=self.bounds,
                    x=Nx, y=Ny
                )
                if verbose:
                    LOGGER.info(f"[Solver] 抛物线初始速度场: U_max={U_max_x:.4f}, U_mean={Ux:.4f}")
            else:
                # 无 inflow 边界，回退到均匀
                self.velocity = StaggeredGrid(
                    values=(Ux, Uy),
                    boundary=self.vel_boundary,
                    bounds=self.bounds,
                    x=Nx, y=Ny
                )
        else:
            self.velocity = StaggeredGrid(
                values=(Ux, Uy),
                boundary=self.vel_boundary,
                bounds=self.bounds,
                x=Nx, y=Ny
            )
        
        # 压力投影 (初始化: 无 x0, differentiable=False)
        self.velocity, self.pressure = self._solve_pressure(
            self.velocity, self.obstacles)
        
        # 转换为物理压力: p_physical = p_phiflow × ρ / dt
        rho = cfg.fluid.density
        dt = cfg.time.dt
        self.pressure = self.pressure * (rho / dt)
        
        # JetObstacle 设置速度场模板（必须在速度场初始化之后）
        if self.jet_obstacle is not None:
            self.jet_obstacle.set_velocity_template(self.velocity)

    def _reset_pressure_fallback_tracking(self):
        """清空压力回退计数与首选策略。"""
        for key in list(self._pressure_fallback_streaks.keys()):
            self._pressure_fallback_streaks[key] = 0
        self._pressure_preferred_fallback = None

    def _mark_pressure_fallback_success(self, strategy_name: str, used_fallback: bool) -> int:
        """更新“按策略独立连续回退计数”并触发首选策略。

        Args:
            strategy_name: 本次成功策略名
            used_fallback: 是否属于回退成功（即非首策略）

        Returns:
            当前策略的连续回退计数（非回退成功时返回 0）
        """
        if not used_fallback:
            self._reset_pressure_fallback_tracking()
            return 0

        # 仅保留当前策略的连续计数，其余清零
        for key in list(self._pressure_fallback_streaks.keys()):
            if key != strategy_name:
                self._pressure_fallback_streaks[key] = 0

        current = self._pressure_fallback_streaks.get(strategy_name, 0) + 1
        self._pressure_fallback_streaks[strategy_name] = current

        # 首次达到阈值时记录首选策略。
        if current == self._FALLBACK_STREAK_THRESHOLD:
            self._pressure_preferred_fallback = strategy_name
            LOGGER.info(
                f"[Solver] 压力投影: {strategy_name} 连续回退{current}次, "
                f"后续优先使用该策略")

        return current
    
    def _solve_pressure(self, velocity, obstacles, x0=None, differentiable=False):
        """统一压力投影 — 多级回退策略 (含按策略自适应快速路径)
        
        根据 x0 和 differentiable 动态构建回退链:
        - x0 可用时: 优先迭代法 (warm start 收敛快)
        - x0 为 None 时: 优先直接法 (无初始猜测时最可靠)
        - differentiable=True: 仅使用可微求解器 (避免 scipy-direct 中断梯度链)
        
        最终以 CG 宽松容差 (1e-3) 兜底, 确保 GPU float32 下也能收敛。

        自适应机制: 对“每个回退策略”独立计数, 连续达到阈值后,
        后续步骤优先快速尝试该策略。
        
        回退策略:
            有 x0:  solver-in-the-loop → auto(x0) → CG(x0, 1e-5) → CG(x0, 1e-3) → [scipy-direct*]
            无 x0:  [scipy-direct*] → solver-in-the-loop → auto → CG(1e-5) → CG(1e-3)
            * scipy-direct 仅在 differentiable=False 时启用
        
        Args:
            velocity: 速度场 (StaggeredGrid)
            obstacles: 障碍物列表
            x0: 压力初始猜测 (时间步进时为上一步 pressure, 初始化时为 None)
            differentiable: True 时仅用可微求解器 (CG/auto)
            
        Returns:
            (velocity, pressure) 投影后的场
        """
        # --- 构建 CG-relaxed 兜底 Solve ---
        relaxed_solve = Solve('CG', 1e-3, 1e-3,
                              max_iterations=10000,
                              rank_deficiency=0,
                              **(dict(x0=x0) if x0 is not None else {}))
        
        # --- 构建策略链 ---
        strategies = []
        if x0 is not None:
            # 有初始猜测 → 迭代法优先 (warm start 收敛快)
            # like solver-in-the-loop: v, p = fluid.make_incompressible(velocity, obstacles=obstacles)
            strategies.append(('solver-in-the-loop', Solve())) # Solve() 为 fluid.make_incompressible() 中 solve 参数的默认值
            strategies.append(('auto+x0', Solve('auto', x0=x0)))
            strategies.append(('CG+x0', Solve('CG', 1e-5, 1e-5,
                                               max_iterations=5000,
                                               rank_deficiency=0, x0=x0)))
            strategies.append(('CG-relaxed+x0', relaxed_solve))
            if not differentiable:
                strategies.append(('scipy-direct',
                                   Solve('scipy-direct', rank_deficiency=0)))
        else:
            # 无初始猜测 → 直接法优先 (确定性, 不需迭代)
            if not differentiable:
                strategies.append(('scipy-direct',
                                   Solve('scipy-direct', rank_deficiency=0)))
            # like solver-in-the-loop: v, p = fluid.make_incompressible(velocity, obstacles=obstacles)
            strategies.append(('solver-in-the-loop', Solve())) # Solve() 为 fluid.make_incompressible() 中 solve 参数的默认值
            strategies.append(('auto', Solve('auto', rank_deficiency=0)))
            strategies.append(('CG', Solve('CG', 1e-5, 1e-5,
                                            max_iterations=10000,
                                            rank_deficiency=0)))
            strategies.append(('CG-relaxed', relaxed_solve))

        # --- 自适应: 若存在首选回退策略，先尝试一次快速路径 ---
        preferred_name = getattr(self, '_pressure_preferred_fallback', None)
        if preferred_name is not None:
            preferred_solve = None
            for name, solve in strategies:
                if name == preferred_name:
                    preferred_solve = solve
                    break

            if preferred_solve is not None:
                try:
                    v, p = fluid.make_incompressible(
                        velocity, obstacles=obstacles, solve=preferred_solve)
                    self._mark_pressure_fallback_success(
                        preferred_name, used_fallback=True)
                    return v, p
                except Exception as e:
                    LOGGER.warning(
                        f"[Solver] 压力投影: 快速路径 {preferred_name} 失败 "
                        f"({type(e).__name__}), 回退到完整策略链")
                    self._pressure_preferred_fallback = None
                    if preferred_name in self._pressure_fallback_streaks:
                        self._pressure_fallback_streaks[preferred_name] = 0
        
        for i, (name, solve) in enumerate(strategies):
            try:
                v, p = fluid.make_incompressible(
                    velocity, obstacles=obstacles, solve=solve)
                # 成功且非最后策略 → 重置连续回退计数
                if i < len(strategies) - 1:
                    if i > 0:
                        streak = self._mark_pressure_fallback_success(
                            name, used_fallback=True)
                        LOGGER.warning(
                            f"[Solver] 压力投影: 回退成功, 最终使用 {name} "
                            f"(第{i+1}策略, 连续{streak}次)")
                    else:
                        self._mark_pressure_fallback_success(name, used_fallback=False)
                else:
                    # 成功但用了最后策略 → 累计连续回退
                    streak = self._mark_pressure_fallback_success(
                        name, used_fallback=True)
                    if i > 0:
                        LOGGER.warning(
                            f"[Solver] 压力投影: 回退成功, 最终使用 {name} "
                            f"(第{i+1}策略, 连续{streak}次)")
                return v, p
            except Exception as e:
                is_last = (i == len(strategies) - 1)
                LOGGER.warning(
                    f"[Solver] 压力投影: {name} 失败 ({type(e).__name__})"
                    + ("" if is_last else ", 尝试下一策略"))
                if is_last:
                    raise RuntimeError(
                        f"压力投影: 所有求解策略均失败。"
                        f"可能原因: GPU float32 精度不足 + 高分辨率。"
                        f"建议: 使用 resume_from 热启动, 或降低分辨率。"
                    ) from e
    
    def _load_from_h5(self, h5_path: str, step: int = -1) -> Dict[str, Any]:
        """从 h5 文件加载场数据（使用现代化 PhiFlow I/O）
        
        使用 myafc.utils.phiflow_io 的 load_fields_h5 函数。
        
        Returns:
            Dict 包含恢复的元信息:
                - 'step': 恢复的时间步
                - 'time': 恢复的物理时间
                - 'h5_path': h5 文件路径
                - 'has_pressure': 是否加载了压力场
        """
        cfg = self.cfg
        Nx, Ny = cfg.domain.resolution
        
        # 使用现代化的 load_fields_h5 函数
        fields, info = load_fields_h5(
            h5_path=h5_path,
            step=step,
            bounds=self.bounds,
            velocity_boundary=self.vel_boundary,
            pressure_boundary=self.pressure_boundary,
        )
        
        # ── 分辨率一致性校验（不匹配时自动插值）──
        need_interpolation = False
        h5_res_info = None
        
        if 'resolution' in info:
            h5_res = list(info['resolution'])
            cfg_res = list(cfg.domain.resolution)
            if h5_res != cfg_res:
                need_interpolation = True
                h5_res_info = h5_res
                LOGGER.warning(
                    f"[Solver] ⚠️ H5 文件分辨率 {h5_res} 与当前配置分辨率 {cfg_res} 不匹配，"
                    f"将自动插值到目标分辨率。\n"
                    f"  H5 文件: {h5_path}"
                )
        else:
            # H5 文件没有存储分辨率元数据，检查加载后的场形状
            if 'velocity' in fields:
                vel_shape = fields['velocity'].shape
                vel_x = vel_shape.get_size('x') if 'x' in vel_shape else None
                vel_y = vel_shape.get_size('y') if 'y' in vel_shape else None
                if vel_x is not None and vel_y is not None:
                    if int(vel_x) != Nx or int(vel_y) != Ny:
                        need_interpolation = True
                        h5_res_info = [int(vel_x), int(vel_y)]
                        LOGGER.warning(
                            f"[Solver] ⚠️ H5 速度场形状 ({int(vel_x)}, {int(vel_y)}) "
                            f"与配置分辨率 ({Nx}, {Ny}) 不匹配，将自动插值到目标分辨率。\n"
                            f"  H5 文件: {h5_path}"
                        )
        
        # 如果需要插值，创建目标网格并 resample
        if need_interpolation:
            target_velocity_grid = StaggeredGrid(
                values=0, boundary=self.vel_boundary,
                bounds=self.bounds, x=Nx, y=Ny
            )
            target_pressure_grid = CenteredGrid(
                values=0, boundary=self.pressure_boundary,
                bounds=self.bounds, x=Nx, y=Ny
            )
            if 'velocity' in fields:
                fields['velocity'] = fields['velocity'].at(target_velocity_grid)
                LOGGER.info(
                    f"[Solver] 速度场已从 {h5_res_info} 插值到 [{Nx}, {Ny}]"
                )
            if 'pressure' in fields:
                fields['pressure'] = fields['pressure'].at(target_pressure_grid)
                LOGGER.info(
                    f"[Solver] 压力场已从 {h5_res_info} 插值到 [{Nx}, {Ny}]"
                )
        
        resume_info = {
            'h5_path': h5_path,
            'step': info['step'],
            'time': info.get('time', info['step'] * cfg.time.dt),
            'has_pressure': 'pressure' in fields,
        }
        
        # 设置速度场
        if 'velocity' in fields:
            self.velocity = fields['velocity']
            LOGGER.info(f"[Solver] 从 h5 恢复 | 路径: {h5_path} | 时间步: {resume_info['step']} | "
                       f"时间: {resume_info['time']:.4f}")
        else:
            raise ValueError(f"h5 文件中缺少 velocity 数据: {h5_path}")
        
        # 设置压力场
        if resume_info['has_pressure']:
            self.pressure = fields['pressure']
            LOGGER.info("[Solver] 从 h5 恢复 | 含压力场")
        else:
            # 无压力数据时，重新求解压力
            self.velocity, self.pressure = self._solve_pressure(
                self.velocity, self.obstacles)
            # 转换为物理压力
            rho = cfg.fluid.density
            dt = cfg.time.dt
            self.pressure = self.pressure * (rho / dt)
            LOGGER.info("[Solver] 从 h5 恢复 | 压力重新求解")
        
        # JetObstacle 设置速度场模板（必须在速度场初始化之后）
        if self.jet_obstacle is not None:
            self.jet_obstacle.set_velocity_template(self.velocity)
        
        return resume_info
        
    def _init_obstacles(self):
        """初始化障碍物"""
        cfg = self.cfg
        
        # 初始化 JetObstacle（如果配置了 jets）
        self.jet_obstacle = None
        
        if not cfg.obstacle.enabled:
            self.obstacles = []
            LOGGER.info("[Solver] 障碍物: 禁用")
            return
        
        Lx, Ly = cfg.domain.size
        # 相对坐标转换为物理坐标
        cx = cfg.obstacle.center[0] * Lx
        cy = cfg.obstacle.center[1] * Ly
        
        shape_type = cfg.obstacle.shape.lower()
        
        if shape_type == 'sphere':
            r = cfg.obstacle.radius  # 绝对尺寸
            # FixedSphere 使用较小的距离下限计算小半径圆柱的 SDF。
            from myafc.env_kernels.solvers.utils import FixedSphere
            geom = FixedSphere(center=vec(x=cx, y=cy), radius=r)
            LOGGER.info(f"[Solver] 障碍物: 圆柱 | 中心: ({cx:.4f}, {cy:.4f}) | 半径: {r:.4f}")
        
        elif shape_type == 'box':
            sx, sy = cfg.obstacle.size  # 绝对尺寸
            geom = Box(x=(cx - sx/2, cx + sx/2), y=(cy - sy/2, cy + sy/2))
            LOGGER.info(f"[Solver] 障碍物: 方柱 | 中心: ({cx:.2f}, {cy:.2f}) | 尺寸: ({sx:.2f}, {sy:.2f})")
        
        else:
            LOGGER.warning(f"[Solver] 未知障碍物类型 '{shape_type}'，使用圆柱")
            r = cfg.obstacle.radius
            from myafc.env_kernels.solvers.utils import FixedSphere
            geom = FixedSphere(center=vec(x=cx, y=cy), radius=r)
        
        # 应用旋转
        if cfg.obstacle.rotation != 0:
            geom = geom.rotated(cfg.obstacle.rotation)
            LOGGER.info(f"[Solver] 障碍物旋转: {np.degrees(cfg.obstacle.rotation):.1f}°")
        
        # 检查是否启用射流控制
        if cfg.jets.enabled and cfg.jets.nozzles:
            self._init_jet_obstacle(geom)
        else:
            self.obstacles = [Obstacle(geom)]
    
    def _init_jet_obstacle(self, geom):
        """初始化射流障碍物"""
        _lazy_import_jet_obstacle()
        if JetObstacle is None:
            LOGGER.warning("[Solver] ⚠️ JetObstacle 模块不可用，使用普通 Obstacle")
            self.obstacles = [Obstacle(geom)]
            return
        
        from myafc.env_kernels.solvers.fluid_control import JetConfig
        
        cfg = self.cfg.jets
        
        # 创建 JetConfig
        jet_config = JetConfig.from_dict({
            'nozzles': cfg.nozzles,
            'smoothing_alpha': cfg.smoothing_alpha,
            'saturation': cfg.saturation,
            'amplitude_scale': cfg.amplitude_scale,
            'profile': cfg.profile,
        })
        
        # 创建策略（传入喷嘴数量）
        num_nozzles = len(cfg.nozzles)
        policy = self._create_jet_policy(cfg.policy, num_nozzles)
        
        # 创建 JetObstacle
        self.jet_obstacle = JetObstacle(
            geometry=geom,
            jet_config=jet_config,
            policy=policy,
        )
        
        # 设置设备
        self.jet_obstacle.set_device(self._device)
        
        # 从配置读取 deterministic_policy 设置
        deterministic_policy = cfg.policy.get('deterministic_policy', False)
        self.jet_obstacle.deterministic = deterministic_policy
        
        # 保存不带射流的普通障碍物引用（用于 apply_jets=False 时的压力投影）
        self.base_obstacle = Obstacle(geom)
        
        # 需要在第一次 step 之前设置速度场模板
        self.obstacles = [self.jet_obstacle]
        
        # 独立运行时使用配置中的策略。训练器和验证器可从外部设置控制信号：
        #     - JetTrainer/AFCValidator: 通过 set_controls_external() 设置
        #     - DRLTrainer (SB3): 通过 PhiFlowEnv.step() → set_controls_external() 设置
        policy_type = cfg.policy.get('type', 'none')
        LOGGER.info(f"[Solver] Jets: 启用 | 喷嘴数: {num_nozzles} | 内置策略: {policy_type} (训练与验证时可能被外部策略覆盖)")
    
    def _create_jet_policy(self, policy_cfg: Dict, num_nozzles: int = 1):
        """创建射流控制策略
        
        Args:
            policy_cfg: 策略配置字典
            num_nozzles: 喷嘴数量
        
        注意: device 参数由 runtime.device 统一控制
        """
        if not policy_cfg:
            return None
        
        policy_type = policy_cfg.get('type', '').lower()
        
        # 自动设置 device 和 num_nozzles
        policy_cfg = dict(policy_cfg)  # 避免修改原配置
        policy_cfg['device'] = self._device
        policy_cfg['num_nozzles'] = num_nozzles
        
        if policy_type == 'constant':
            from myafc.env_kernels.solvers.fluid_control.control_policy import ConstantPolicy
            return ConstantPolicy.from_config(policy_cfg)
        
        elif policy_type == 'sinusoidal':
            from myafc.env_kernels.solvers.fluid_control.control_policy import SinusoidalPolicy
            return SinusoidalPolicy.from_config(policy_cfg)
        
        elif policy_type == 'mlp':
            from myafc.env_kernels.solvers.fluid_control.control_policy import MLPPolicy
            return MLPPolicy.from_config(policy_cfg)
        
        else:
            LOGGER.warning(f"[Solver] ⚠️ 未知策略类型 '{policy_type}'，不使用策略")
            return None
    
    def set_jet_policy(self, policy):
        """设置/替换射流控制策略（用于 RL 训练）"""
        if self.jet_obstacle is not None:
            self.jet_obstacle.policy = policy
            LOGGER.info(f"[Solver] 设置射流策略: {type(policy).__name__}")
        else:
            LOGGER.warning("[Solver] ⚠️ 未启用射流控制，无法设置策略")
    
    def _init_forcing(self):
        """初始化外力场"""
        cfg = self.cfg
        
        if not cfg.forcing.enabled:
            self.forcing_generator = None
            LOGGER.info("[Solver] Forcing: 禁用")
            return
        
        Lx, Ly = cfg.domain.size
        Nx, Ny = cfg.domain.resolution
        
        forcing_type = cfg.forcing.type.lower()
        
        if forcing_type == 'constant':
            from myafc.env_kernels.solvers.forcing import ConstantForcing
            self.forcing_generator = ConstantForcing(
                force=cfg.forcing.force,
                bounds=self.bounds,
                x=Nx, y=Ny
            )
            LOGGER.info(f"[Solver] Forcing: 定常 | 力: {cfg.forcing.force}")
        
        elif forcing_type == 'kolmogorov':
            from myafc.env_kernels.solvers.forcing import KolmogorovForcing
            self.forcing_generator = KolmogorovForcing(
                amplitude=cfg.forcing.amplitude,
                wavenumber=cfg.forcing.wavenumber,
                noise_scale=cfg.forcing.noise_scale,
                bounds=self.bounds,
                x=Nx, y=Ny
            )
            LOGGER.info(f"[Solver] Forcing: Kolmogorov | A={cfg.forcing.amplitude} | "
                        f"k={cfg.forcing.wavenumber} | noise={cfg.forcing.noise_scale}")
        
        else:
            LOGGER.warning(f"[Solver] 未知 forcing 类型 '{forcing_type}'，禁用 forcing")
            self.forcing_generator = None
    
    def step(
        self,
        state: Optional[Dict[str, Any]] = None,  # 用于 jet poliycy
        update_controls: bool = True,
        apply_jets: bool = True,
    ) -> Tuple[StaggeredGrid, CenteredGrid]:
        """
        执行单步求解
        
        控制信号流程（核心设计）:
        1. step 开始时调用 update_controls(state) 计算控制信号
           - state 包含历史流场信息（probes, force_coeffs 等）和当前步时间
           - 控制信号存储在 jet_obstacle._current_controls 中
        2. step 过程中调用 apply_jet_bc() 使用已存储的控制信号
        
        求解流程:
        1. 更新射流控制信号（基于历史信息和当前时间）
        2. 黏性扩散 (显式欧拉)
        3. 域边界条件施加
        4. 辅助射流边界条件 (对流前，与域 BC 同等地位)
        5. 对流 (semi-Lagrangian)
        6. 外力项 (forcing)
        7. 压力投影 (make_incompressible，射流参与散度计算)
        8. 辅助射流边界条件 (压力投影后，确保边界值准确)
        
        可微分控制:
        - differentiable=True: 保持梯度，用于训练
        - differentiable=False: 禁用梯度，节省内存，用于推断/数据收集
        
        压力约定:
        - step() 之外: 压力为物理压力 p_physical (Pa)
        - step() 之内: 压力为 PhiFlow 投影压力 p_phiflow = p_physical × dt / ρ
        - 转换在 step() / _step_impl() 开头和结尾自动完成
        
        Args:
            state: 状态字典，传递给射流策略（如 probes, force_coeffs, time）
            update_controls: 是否更新控制信号（默认 True）
            apply_jets: 是否施加射流 BC（默认 True）
        
        Returns:
            (velocity, pressure) - 注意: 返回的 pressure 是物理压力
        """
        if self.differentiable:
            return self._step_impl(state, update_controls, apply_jets)
        else:
            with torch.no_grad():
                return self._step_impl_jit(state, update_controls, apply_jets)
    
    def _step_impl(
        self,
        state: Optional[Dict[str, Any]] = None,
        update_controls: bool = True,
        apply_jets: bool = True,
    ) -> Tuple[StaggeredGrid, CenteredGrid]:
        """step() 的核心实现（分离出来以便 no_grad 包装）"""
        cfg = self.cfg
        dt = cfg.time.dt
        rho = cfg.fluid.density
        
        velocity = self.velocity
        pressure = self.pressure
        
        # ===== 选择压力投影使用的障碍物 =====
        # 压力投影使用静止障碍物维持无滑移条件。
        # apply_jet_bc 单独施加喷口速度，避免将速度赋给整个障碍物。
        if hasattr(self, 'base_obstacle') and self.base_obstacle is not None:
            obstacles_for_projection = [self.base_obstacle]
        else:
            obstacles_for_projection = self.obstacles
        
        # ===== 0. 更新射流控制信号（step 开始时）=====
        # update_controls: 从策略获取原始控制信号
        # apply_controls: 控制信号后处理, 为后续的 apply_jet_bc 做准备; 裁剪→缩放→平滑→空间分布→重建速度场
        # apply_jet_bc: 在流场中施加射流 BC
        if self.jet_obstacle is not None:
            if update_controls:
                jet_state = state.copy() if state else {}
                jet_state['time'] = self.current_time  # 当前步开始时间
                self.jet_obstacle.update_controls(jet_state)
            if apply_jets:
                # 每个施加射流的时间步都执行完整处理流程
                self.jet_obstacle.apply_controls()
        
        # ===== 物理压力 → PhiFlow 投影压力 =====
        pressure = pressure * (dt / rho)
        
        # --- 1. 黏性扩散 ---
        if self.kinematic_viscosity > 0:
            # --- 1a. 施加域边界 BC + 入口 profile ---
            velocity = velocity.with_boundary(self.vel_boundary)
            velocity = self._enforce_inflow(velocity)
            # --- 1b. 扩散前施加射流 BC ---
            if self.jet_obstacle is not None and apply_jets:
                velocity = self.jet_obstacle.apply_jet_bc(velocity)
            
            diffusion_method = cfg.time.diffusion_method
            if diffusion_method == 'implicit':
                # 隐式扩散: 无条件稳定，使用 CG 线性求解器
                # 注意: PhiFlow 当前版本对 StaggeredGrid + 非周期 BC 支持有限，
                #       可能触发 MixedShape/bake_extrapolation 错误，此时自动回退到 auto_substep
                try:
                    velocity = diffuse.implicit(
                        velocity, diffusivity=self.kinematic_viscosity, dt=dt,
                        solve=Solve('CG', rel_tol=1e-5, abs_tol=1e-5)
                    )
                except (AttributeError, NotImplementedError, Exception) as e:
                    if not hasattr(self, '_implicit_fallback_warned'):
                        LOGGER.warning(
                            f"[Solver] ⚠️ diffuse.implicit() 失败 ({type(e).__name__}: {e}), "
                            f"自动回退到 auto_substep 显式扩散。"
                            f"这通常是 PhiFlow 对 StaggeredGrid + 非周期边界的已知限制。"
                        )
                        self._implicit_fallback_warned = True
                    # 回退到 auto_substep
                    import math as _math
                    dx_min = min(cfg.domain.size[0] / cfg.domain.resolution[0],
                                 cfg.domain.size[1] / cfg.domain.resolution[1])
                    substeps = max(1, _math.ceil(self.kinematic_viscosity * dt / (0.4 * dx_min**2)))
                    velocity = diffuse.explicit(
                        velocity, diffusivity=self.kinematic_viscosity, dt=dt,
                        substeps=substeps
                    )
            elif diffusion_method == 'auto_substep':
                # 自动子步显式扩散: 计算所需子步数以满足 CFL < 0.4
                import math as _math
                dx_min = min(cfg.domain.size[0] / cfg.domain.resolution[0],
                             cfg.domain.size[1] / cfg.domain.resolution[1])
                substeps = max(1, _math.ceil(self.kinematic_viscosity * dt / (0.4 * dx_min**2)))
                velocity = diffuse.explicit(
                    velocity, diffusivity=self.kinematic_viscosity, dt=dt,
                    substeps=substeps
                )
            else:
                # 显式扩散: 条件稳定，dt < dx²/(2ν)
                velocity = diffuse.explicit(velocity, diffusivity=self.kinematic_viscosity, dt=dt)
        
        # --- 2. 对流 ---
        # --- 2a. 施加域边界 BC + 入口 profile ---
        velocity = velocity.with_boundary(self.vel_boundary)
        velocity = self._enforce_inflow(velocity)
        # --- 2b. 对流前施加射流 BC ---
        if self.jet_obstacle is not None and apply_jets:
            velocity = self.jet_obstacle.apply_jet_bc(velocity)
        if cfg.time.advection_method == 'mac_cormack':
            velocity = advect.mac_cormack(velocity, velocity, dt=dt)
        else:
            velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        
        # --- 3. 外力项 ---
        if self.forcing_generator is not None:
            # --- 3a. 施加域边界 BC + 入口 profile ---
            velocity = velocity.with_boundary(self.vel_boundary)
            velocity = self._enforce_inflow(velocity)
            # --- 3b. 外力前施加射流 BC ---
            if self.jet_obstacle is not None and apply_jets:
                velocity = self.jet_obstacle.apply_jet_bc(velocity)
            forcing_field = self.forcing_generator.get_field(template=velocity)
            velocity = velocity + forcing_field * dt
        
        # --- 4. 压力投影 ---
        # JetObstacle 继承自 Obstacle，velocity 属性返回射流速度场
        # PhiFlow 原生的 make_incompressible 内部通过 apply_boundary_conditions 自动应用射流 BC
        
        # --- 4a. 施加域边界 BC + 入口 profile ---
        velocity = velocity.with_boundary(self.vel_boundary)
        velocity = self._enforce_inflow(velocity)

        velocity, pressure = self._solve_pressure(
            velocity, obstacles_for_projection,
            x0=pressure, differentiable=self.differentiable)
        
        # --- 5. 求解流程结束后施加射流 BC ---
        # 确保最终速度场在喷嘴处正确（PhiFlow 的 apply_boundary_conditions 只处理固体内部）
        velocity = self._enforce_inflow(velocity)
        if self.jet_obstacle is not None and apply_jets:
            velocity = self.jet_obstacle.apply_jet_bc(velocity)
        
        # ===== PhiFlow 投影压力 → 物理压力 =====
        pressure = pressure * (rho / dt)
        
        # 更新状态
        self.velocity = velocity
        self.pressure = pressure
        self.current_step += 1
        self.current_time += dt
        
        return self.velocity, self.pressure

    def set_differentiable(self, enabled: bool):
        """
        动态切换可微分模式
        
        Args:
            enabled: True=保持梯度（训练），False=禁用梯度（推断）
        """
        self.differentiable = enabled
    
    def run(self, num_steps: Optional[int] = None) -> Dict[str, Any]:
        """执行完整仿真（无进度显示，建议使用 SolverRunner）"""
        steps = num_steps if num_steps is not None else self.cfg.time.num_steps
        
        for _ in range(steps):
            self.step()
        
        stats = self._compute_statistics()
        LOGGER.info(f"[Solver] 仿真完成 | 总步数: {self.current_step} | 总时间: {self.current_time:.4f}")
        
        return {
            'velocity': self.velocity,
            'pressure': self.pressure,
            'stats': stats,
            'steps_completed': self.current_step
        }
    
    def _compute_statistics(self) -> Dict[str, float]:
        """计算场统计量
        
        注意：StaggeredGrid 的 x 和 y 分量采样在不同位置（face-centered），
        需要分别通过 velocity['x'] 和 velocity['y'] 访问各分量。
        """
        vel_x = self.velocity['x'].values
        vel_y = self.velocity['y'].values
        p = self.pressure.values
        
        return {
            'max_vel_x': float(math.max(vel_x)),
            'min_vel_x': float(math.min(vel_x)),
            'mean_vel_x': float(math.mean(vel_x)),
            'max_vel_y': float(math.max(vel_y)),
            'min_vel_y': float(math.min(vel_y)),
            'mean_vel_y': float(math.mean(vel_y)),
            'max_pressure': float(math.max(p)),
            'min_pressure': float(math.min(p)),
            'mean_pressure': float(math.mean(p)),
        }
    
    def reset(
        self, 
        use_h5: bool = True,
        step: Optional[int] = None,
        resume_strict: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """重置求解器（流场和控制状态）
        
        统一的重置接口，支持两种初始化模式：
        1. 从 h5 文件加载流场（默认，use_h5=True）
        2. 从配置参数初始化流场（use_h5=False）
        
        Args:
            use_h5: 是否从 h5 文件初始化流场
                - True（默认）: 从 h5 文件加载，需配置 resume_from
                - False: 从配置参数初始化（均匀入口速度场）
            step: h5 帧索引（仅 use_h5=True 时有效）
                - None: 使用配置中的 resume_step（默认 -1，即最后一帧）
                - int: 指定帧索引（负数表示从末尾倒数）
            resume_strict: 严格续算模式（仅 use_h5=True 时有效）
                - None: 使用配置中的 initial.resume_strict（默认 False）
                - False: 热启动模式，时间步从 0 开始
                - True: 严格续算模式，时间步和物理时间接续 h5 帧
        
        Returns:
            use_h5=True 时返回 h5 加载信息 Dict，否则返回 None
            
        Raises:
            ValueError: use_h5=True 但未配置有效的 h5 文件路径
            
        Note:
            不会重建障碍物和 JetObstacle，Trainer 覆盖的配置
            （如 smoothing_alpha、deterministic_policy）会保持不变。
        """
        resume_info = None
        
        # 1. 重置 jet obstacle 控制状态
        if self.jet_obstacle is not None:
            self.jet_obstacle.reset()
        
        # 2. 初始化流场
        if use_h5:
            # 从 h5 文件加载
            cfg = self.cfg
            h5_path = cfg.initial.resume_from
            step = (cfg.initial.resume_step or -1) if step is None else step
            
            if not h5_path or not os.path.exists(h5_path):
                raise ValueError(f"[Solver] reset(use_h5=True) 需要有效的 resume_from 路径，当前: {h5_path}")
            
            resume_info = self._load_from_h5(h5_path, step)
            LOGGER.info(f"[Solver] 从 h5 重置 | 帧: {resume_info['step']} | 时间: {resume_info['time']:.4f}")
            
            # JetObstacle 设置速度场模板
            if self.jet_obstacle is not None:
                self.jet_obstacle.set_velocity_template(self.velocity)
        else:
            # 从配置参数初始化 (verbose=False 避免重复日志)
            self._init_velocity_pressure(verbose=False)
            LOGGER.info("[Solver] 从配置参数重置")
        
        # 3. 设置时间状态
        # 确定是否使用严格续算模式
        if resume_strict is None:
            resume_strict = getattr(self.cfg.initial, 'resume_strict', False)
        
        if use_h5 and resume_strict and resume_info is not None:
            # 严格续算模式：接续 h5 帧的时间步和物理时间
            self.current_step = resume_info['step']
            self.current_time = resume_info['time']
            LOGGER.info(f"[Solver] 严格续算模式 | 接续步: {self.current_step} | 时间: {self.current_time:.4f}")
        else:
            # 热启动模式：时间步从 0 开始
            self.current_step = 0
            self.current_time = 0.0
        
        return resume_info
    
    # ============================================================
    # 力与力系数计算
    # ============================================================
    
    def compute_force_coeffs(
        self,
        U_ref: Optional[float] = None,
        A_ref: Optional[float] = None,
        include_viscous: bool = True,
    ) -> Tuple:
        """
        计算障碍物受力和力系数（可微分实现，通用几何形状）
        
        使用 SDF 梯度法进行表面积分，适用于任意几何形状（圆柱、方柱、旋转方柱等）。
        
        原理:
            总力 = 压力力 + 粘性力
            F_pressure = -∫_∂Ω p * n * dA （压力垂直于壁面）
            F_viscous  = ∫_∂Ω τ * t * dA （粘性剪切力切向于壁面）
            
            其中:
            - n: 从固体指向流体的单位法向量 (∇SDF/|∇SDF|)
            - t: 切向量 (垂直于n)
            - τ = μ * ∂u_t/∂n: 壁面剪切应力
        
        力系数公式:
            Cd = Fx / (0.5 * ρ * U² * A)
            Cl = Fy / (0.5 * ρ * U² * A)
        
        Args:
            U_ref: 参考速度 (= U_mean)，None 时自动从 inflow.value 获取
                   注: inflow.value 始终代表 U_mean (截面平均速度)，
                   无论 inlet_profile 为 uniform 或 parabolic，与 DRLinFluids magUInf 一致
            A_ref: 参考面积（2D 为特征长度），None 时自动计算
            include_viscous: 是否包含粘性剪切应力贡献（默认True）
        
        Returns:
            (Cd, Cl, Fx, Fy, U_ref, A_ref): 力系数、力和参考量
            - Cd, Cl, Fx, Fy: torch.Tensor，支持梯度
            - U_ref, A_ref: float，实际使用的参考量
        """
        import torch
        
        if not self.obstacles:
            zero = torch.tensor(0.0)
            return zero, zero, zero, zero, 0.0, 0.0
        
        cfg = self.cfg
        rho = cfg.fluid.density
        
        # 自动获取参考速度
        if U_ref is None:
            inflow_vel, _ = _find_inflow_velocity(cfg.boundary.velocity)
            if inflow_vel is not None:
                U_ref = (inflow_vel[0]**2 + inflow_vel[1]**2)**0.5
            else:
                U_ref = 1.0
        
        # 自动计算参考面积（特征长度）
        if A_ref is None:
            A_ref = self._compute_reference_area()
        
        # 计算力（根据形状选择最优方法）
        # 仅未旋转的 Box 使用简化边缘积分法，其他情况使用通用边界单元法
        is_axis_aligned_box = (cfg.obstacle.shape.lower() == 'box' and 
                               abs(cfg.obstacle.rotation) < 1e-6)
        if is_axis_aligned_box:
            Fx, Fy = self._forces_box_edges(include_viscous=include_viscous)
        else:
            Fx, Fy = self._forces_boundary_cells(include_viscous=include_viscous)
        
        # 计算力系数
        dyn_p = 0.5 * rho * U_ref**2
        if dyn_p > 0:
            Cd = Fx / (dyn_p * A_ref)
            Cl = Fy / (dyn_p * A_ref)
        else:
            Cd = Fx * 0
            Cl = Fy * 0
        
        return Cd, Cl, Fx, Fy, U_ref, A_ref
    
    def _get_actual_grid_dims(self) -> Tuple[int, int, float, float]:
        """获取实际网格单元格维度（从压力场推断）
        
        PhiFlow 的 StaggeredGrid(x=N) 创建 N-1 个流体单元格。
        config.domain.resolution 是 stagger 点数，实际单元格数 = resolution - 1。
        此方法返回实际单元格维度，供力计算和速度梯度等使用。
        
        Returns:
            (Nx_cells, Ny_cells, dx, dy): 实际单元格数和网格间距
        """
        p_native = self.pressure.values.native(('y', 'x'))
        Ny_cells, Nx_cells = p_native.shape
        Lx, Ly = self.cfg.domain.size
        dx = Lx / Nx_cells
        dy = Ly / Ny_cells
        return Nx_cells, Ny_cells, dx, dy
    
    def _compute_velocity_gradient(self) -> Dict:
        """
        计算速度场梯度（模块化，供力计算和力矩计算复用）
        
        Returns:
            Dict 包含:
            - du_dx, du_dy, dv_dx, dv_dy: 速度梯度分量
            - u, v: 中心网格上的速度分量
        """
        import torch
        
        pressure = self.pressure
        velocity = self.velocity
        
        # 使用实际网格维度（不用 config resolution，因为 PhiFlow StaggeredGrid(x=N) → N-1 个单元格）
        Nx, Ny, dx, dy = self._get_actual_grid_dims()
        
        # 获取速度场（插值到中心网格）
        vel_centered = velocity @ pressure
        u = vel_centered.values.vector['x'].native(('y', 'x'))
        v = vel_centered.values.vector['y'].native(('y', 'x'))
        
        # 计算速度梯度（使用可微分切片操作）
        # x 方向梯度
        du_dx = torch.empty_like(u)
        dv_dx = torch.empty_like(v)
        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx
        
        # y 方向梯度
        du_dy = torch.empty_like(u)
        dv_dy = torch.empty_like(v)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy
        dv_dy[0, :] = (v[1, :] - v[0, :]) / dy
        dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy
        
        return {
            'u': u,
            'v': v,
            'du_dx': du_dx,
            'du_dy': du_dy,
            'dv_dx': dv_dx,
            'dv_dy': dv_dy,
        }
    
    def _compute_viscous_stress(self, vel_grad: Dict) -> Tuple:
        """
        计算粘性应力张量分量
        
        应力张量 σ_ij = μ*(∂u_i/∂x_j + ∂u_j/∂x_i)
        
        Args:
            vel_grad: 速度梯度字典（来自 _compute_velocity_gradient）
        
        Returns:
            (sigma_xx, sigma_yy, sigma_xy): 应力张量分量
        """
        mu = self.cfg.fluid.viscosity  # 动力粘度
        
        du_dx = vel_grad['du_dx']
        du_dy = vel_grad['du_dy']
        dv_dx = vel_grad['dv_dx']
        dv_dy = vel_grad['dv_dy']
        
        sigma_xx = 2 * mu * du_dx
        sigma_yy = 2 * mu * dv_dy
        sigma_xy = mu * (du_dy + dv_dx)
        
        return sigma_xx, sigma_yy, sigma_xy
    
    def _forces_box_edges(self, include_viscous: bool = True) -> Tuple:
        """
        边缘积分法计算障碍物受力（仅适用于轴对齐的 Box，精度最高）
        
        原理:
            对于 Box 形状，直接在固体外侧第一层流体网格上进行积分：
            F = -∮_∂Ω p * n dL + ∮_∂Ω τ·n dL
            
            其中 n 是从固体指向流体的单位外法向量。
        
        Args:
            include_viscous: 是否包含粘性剪切应力贡献
        
        Returns:
            (Fx, Fy): x 和 y 方向的总力（torch.Tensor，支持梯度）
        """
        import torch
        
        cfg = self.cfg
        Lx, Ly = cfg.domain.size
        Nx, Ny, dx, dy = self._get_actual_grid_dims()
        
        # 获取压力场
        p = self.pressure.values.native(('y', 'x'))  # shape: (Ny, Nx)
        
        # 获取障碍物几何信息
        obs = cfg.obstacle
        cx = obs.center[0] * Lx  # 绝对坐标
        cy = obs.center[1] * Ly
        
        if obs.shape.lower() == 'box':
            sx, sy = obs.size
            half_sx, half_sy = sx / 2, sy / 2
            
            # 计算边界索引（固体内部边缘）
            i_left = int((cx - half_sx) / dx)
            i_right = int((cx + half_sx) / dx)
            j_bottom = int((cy - half_sy) / dy)
            j_top = int((cy + half_sy) / dy)
            
            # 确保索引在有效范围内
            i_left = max(1, i_left)
            i_right = min(Nx - 2, i_right)
            j_bottom = max(1, j_bottom)
            j_top = min(Ny - 2, j_top)
            
            # 压力力积分（使用固体外侧第一层流体网格）
            # 左边界（前缘）: n = (-1, 0), F = -p * n = +p
            Fx_left = p[j_bottom:j_top+1, i_left-1].sum() * dy
            # 右边界（后缘）: n = (+1, 0), F = -p * n = -p
            Fx_right = -p[j_bottom:j_top+1, i_right+1].sum() * dy
            # 下边界: n = (0, -1), F = -p * n = +p
            Fy_bottom = p[j_bottom-1, i_left:i_right+1].sum() * dx
            # 上边界: n = (0, +1), F = -p * n = -p
            Fy_top = -p[j_top+1, i_left:i_right+1].sum() * dx
            
            Fx_pressure = Fx_left + Fx_right
            Fy_pressure = Fy_bottom + Fy_top
            
            # 粘性力（可选）
            Fx_viscous = p.sum() * 0.0  # 保持梯度链
            Fy_viscous = p.sum() * 0.0
            
            if include_viscous and self.kinematic_viscosity > 0:
                vel_grad = self._compute_velocity_gradient()
                sigma_xx, sigma_yy, sigma_xy = self._compute_viscous_stress(vel_grad)
                
                # 左边界: n = (-1, 0), τ·n = (-σ_xx, -σ_xy)
                Fx_viscous += -sigma_xx[j_bottom:j_top+1, i_left-1].sum() * dy
                Fy_viscous += -sigma_xy[j_bottom:j_top+1, i_left-1].sum() * dy
                # 右边界: n = (+1, 0), τ·n = (σ_xx, σ_xy)
                Fx_viscous += sigma_xx[j_bottom:j_top+1, i_right+1].sum() * dy
                Fy_viscous += sigma_xy[j_bottom:j_top+1, i_right+1].sum() * dy
                # 下边界: n = (0, -1), τ·n = (-σ_xy, -σ_yy)
                Fx_viscous += -sigma_xy[j_bottom-1, i_left:i_right+1].sum() * dx
                Fy_viscous += -sigma_yy[j_bottom-1, i_left:i_right+1].sum() * dx
                # 上边界: n = (0, +1), τ·n = (σ_xy, σ_yy)
                Fx_viscous += sigma_xy[j_top+1, i_left:i_right+1].sum() * dx
                Fy_viscous += sigma_yy[j_top+1, i_left:i_right+1].sum() * dx
            
            return Fx_pressure + Fx_viscous, Fy_pressure + Fy_viscous
        
        else:
            # 对于非 Box 形状（或旋转Box、圆形等），使用通用边界单元积分法
            return self._forces_boundary_cells(include_viscous=include_viscous)
    
    def _get_boundary_cell_info(self) -> Dict[str, Any]:
        """
        获取边界单元信息（内部方法，供力和力矩计算复用）
        
        使用 SDF 加权的边界积分方法。
        
        核心思路：
        1. 只选取"紧贴边界"的流体单元（|SDF| < h 且 SDF > 0）
        2. 使用 SDF 梯度计算法向量
        3. 每个单元的积分长度 dL_i 根据其 SDF 值进行加权
        
        对于圆柱这样的光滑边界：
        - 使用"切过单元"的边界长度近似
        - dL_i ≈ h / |cos(θ)| 其中 θ 是法向量与网格轴的夹角
        - 简化为 dL_i ≈ h * |∇φ|（因为 |∇φ| ≈ 1 在边界附近）
        
        Returns:
            dict: 包含以下键
                - boundary_mask: 边界单元掩码 (Ny, Nx)
                - boundary_weight: 每个边界单元的积分权重 (Ny, Nx)
                - nx, ny: 单位法向量场 (Ny, Nx)
                - p: 压力场 (Ny, Nx)
                - dL: 基础积分长度元
                - X, Y: 网格坐标 (Ny, Nx)
                - device: torch device
                - debug_info: 调试信息字典
        
        注意: 
            PhiFlow 的 approximate_signed_distance 在几何中心附近使用 eps 参数
            导致 SDF 值不准确（总是 >= sqrt(eps)），因此自己计算准确的 SDF。
        """
        import torch
        
        cfg = self.cfg
        Lx, Ly = cfg.domain.size
        Nx, Ny, dx, dy = self._get_actual_grid_dims()
        h = (dx + dy) / 2  # 平均网格尺寸
        
        p = self.pressure.values.native(('y', 'x'))
        device = p.device
        
        # 网格坐标 (提前计算，用于 SDF 和返回)
        x_coords = torch.linspace(dx/2, Lx - dx/2, Nx, device=device)
        y_coords = torch.linspace(dy/2, Ly - dy/2, Ny, device=device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (Ny, Nx)
        
        # 使用统一的 SDF 工具（避免 PhiFlow 的 eps 问题）
        from myafc.env_kernels.solvers.utils import compute_sdf_and_gradient
        
        geom = self.obstacles[0].geometry
        shape_type = cfg.obstacle.shape.lower()
        
        # 计算 SDF 和梯度（使用统一接口）
        sdf_info = compute_sdf_and_gradient(
            geom, X, Y, dx, dy, device=str(device), use_phiflow_fallback=True
        )
        sdf = sdf_info['sdf']
        nx = sdf_info['nx']
        ny = sdf_info['ny']
        
        # 计算理论周长（用于校验）
        if shape_type == 'sphere':
            r = float(geom.radius)
            theoretical_perimeter = 2 * np.pi * r
        elif shape_type == 'box':
            sx, sy = cfg.obstacle.size
            theoretical_perimeter = 2 * (sx + sy)
        else:
            theoretical_perimeter = None
        
        # ========== 边界单元选取 ==========
        # 方法：选取 0 < SDF < h 的流体单元（紧贴边界的第一层）
        boundary_mask = ((sdf > 0) & (sdf < h)).float()
        
        # ========== 积分长度加权 ==========
        # 对于每个边界单元，其贡献的边界长度与该单元被边界"切过"的长度有关
        # 使用简化公式：dL_i = h / max(|nx|, |ny|)
        # 这考虑了边界切过单元时的角度校正
        # 例如：当法向量沿 x 轴时（nx=1, ny=0），dL = h（水平切过）
        #       当法向量沿对角线时（nx=ny=0.707），dL = h/0.707 ≈ 1.41h
        
        with torch.no_grad():
            # 计算角度校正因子
            max_n_component = torch.maximum(torch.abs(nx), torch.abs(ny)).clamp(min=0.5)
            angle_correction = 1.0 / max_n_component
            
            # 限制校正因子范围（避免极端情况）
            angle_correction = angle_correction.clamp(min=1.0, max=2.0)
        
        # 最终的边界权重（需要保持可微分）
        boundary_weight = boundary_mask * angle_correction
        
        # ========== 周长归一化校正（可选，提高精度）==========
        # 如果知道理论周长，可以进行全局校正
        if theoretical_perimeter is not None:
            with torch.no_grad():
                # 计算当前积分长度
                current_integration_length = (boundary_weight * h).sum().item()
                if current_integration_length > 1e-8:
                    # 计算校正因子
                    perimeter_correction = theoretical_perimeter / current_integration_length
                    # 只有当偏差超过 10% 时才校正
                    if abs(perimeter_correction - 1.0) > 0.1:
                        boundary_weight = boundary_weight * perimeter_correction
        
        # 调试信息
        debug_info = {
            'num_boundary_cells': int(boundary_mask.sum().item()),
            'theoretical_perimeter': theoretical_perimeter,
            'actual_integration_length': float((boundary_weight * h).sum().item()),
            'h': h,
        }
        
        return {
            'boundary_mask': boundary_mask, 
            'boundary_weight': boundary_weight,
            'nx': nx, 'ny': ny,
            'p': p, 'dL': h, 'X': X, 'Y': Y, 'device': device,
            'debug_info': debug_info,
        }
    
    def _forces_boundary_cells(self, include_viscous: bool = True) -> Tuple:
        """
        边界单元法计算障碍物受力（适用于任意形状）
        
        使用加权边界积分方法：
        - boundary_weight 包含了角度校正和周长归一化
        - 这确保了总积分长度等于理论周长
        
        Args:
            include_viscous: 是否包含粘性剪切应力贡献
        
        Returns:
            (Fx, Fy): x 和 y 方向的总力（torch.Tensor，支持梯度）
        """
        import torch
        
        info = self._get_boundary_cell_info()
        boundary_mask = info['boundary_mask']
        boundary_weight = info['boundary_weight']  # 包含角度校正和周长归一化
        nx, ny = info['nx'], info['ny']
        p, dL, device = info['p'], info['dL'], info['device']
        
        # 压力力: F_p = -Σ p * n * (weight * dL)
        # boundary_weight 已包含校正因子
        Fx_pressure = -(p * nx * boundary_weight * dL).sum()
        Fy_pressure = -(p * ny * boundary_weight * dL).sum()
        
        # 粘性力
        Fx_viscous = p.sum() * 0.0
        Fy_viscous = p.sum() * 0.0
        
        if include_viscous and self.kinematic_viscosity > 0:
            vel_grad = self._compute_velocity_gradient()
            sigma_xx, sigma_yy, sigma_xy = self._compute_viscous_stress(vel_grad)
            Fx_viscous = ((sigma_xx * nx + sigma_xy * ny) * boundary_weight * dL).sum()
            Fy_viscous = ((sigma_xy * nx + sigma_yy * ny) * boundary_weight * dL).sum()
        
        return Fx_pressure + Fx_viscous, Fy_pressure + Fy_viscous
    
    def _compute_reference_area(self) -> float:
        """
        计算参考面积（2D 为特征长度）
        
        对于圆柱：直径 2R
        对于方柱：垂直于来流方向的投影宽度
        """
        cfg = self.cfg
        obs = cfg.obstacle
        
        if obs.shape.lower() == 'sphere':
            return 2.0 * obs.radius
        elif obs.shape.lower() == 'box':
            # 考虑旋转的情况
            inflow_vel, _ = _find_inflow_velocity(cfg.boundary.velocity)
            if inflow_vel is None:
                inflow_vel = [1.0, 0.0]
            
            U_mag = np.sqrt(inflow_vel[0]**2 + inflow_vel[1]**2)
            if U_mag > 0:
                perp_x, perp_y = -inflow_vel[1] / U_mag, inflow_vel[0] / U_mag
            else:
                perp_x, perp_y = 0.0, 1.0
            
            sx, sy = obs.size
            theta = obs.rotation
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            half_sx, half_sy = sx / 2, sy / 2
            corners = [
                (half_sx, half_sy), (-half_sx, half_sy),
                (-half_sx, -half_sy), (half_sx, -half_sy),
            ]
            
            projections = []
            for lx, ly in corners:
                rx = lx * cos_t - ly * sin_t
                ry = lx * sin_t + ly * cos_t
                proj = rx * perp_x + ry * perp_y
                projections.append(proj)
            
            return max(projections) - min(projections)
        else:
            if hasattr(obs, 'radius') and obs.radius:
                return 2.0 * obs.radius
            elif hasattr(obs, 'size') and obs.size:
                return max(obs.size)
            return 1.0
    
    def compute_torque(
        self,
        pivot: Optional[Tuple[float, float]] = None,
        include_viscous: bool = True,
    ) -> Tuple:
        """
        计算障碍物受到的合力矩（复用 _get_boundary_cell_info）
        
        原理: M_z = ∫(rx * Fy - ry * Fx) dL
        
        Args:
            pivot: 力矩参考点，None 时使用障碍物中心
            include_viscous: 是否包含粘性贡献
        
        Returns:
            (Mz, Mz_pressure, Mz_viscous): 总力矩及分解
        """
        import torch
        
        if not self.obstacles:
            zero = torch.tensor(0.0)
            return zero, zero, zero
        
        # 复用边界单元信息
        info = self._get_boundary_cell_info()
        boundary_mask = info['boundary_mask']
        boundary_weight = info['boundary_weight']  # 包含角度校正和周长归一化
        nx, ny, p, dL = info['nx'], info['ny'], info['p'], info['dL']
        X, Y, device = info['X'], info['Y'], info['device']
        
        # 力矩参考点
        if pivot is None:
            geom = self.obstacles[0].geometry
            x_pivot = float(geom.center.vector['x'])
            y_pivot = float(geom.center.vector['y'])
        else:
            x_pivot, y_pivot = pivot
        
        rx, ry = X - x_pivot, Y - y_pivot
        
        # 压力力矩: M_z = Σ(rx * Fy_p - ry * Fx_p)
        # 使用 boundary_weight 替代 boundary_mask
        Fx_p = -p * nx * boundary_weight * dL
        Fy_p = -p * ny * boundary_weight * dL
        Mz_pressure = (rx * Fy_p - ry * Fx_p).sum()
        
        # 粘性力矩
        Mz_viscous = torch.tensor(0.0, device=device)
        if include_viscous and self.kinematic_viscosity > 0:
            vel_grad = self._compute_velocity_gradient()
            sigma_xx, sigma_yy, sigma_xy = self._compute_viscous_stress(vel_grad)
            Fx_v = (sigma_xx * nx + sigma_xy * ny) * boundary_weight * dL
            Fy_v = (sigma_xy * nx + sigma_yy * ny) * boundary_weight * dL
            Mz_viscous = (rx * Fy_v - ry * Fx_v).sum()
        
        return Mz_pressure + Mz_viscous, Mz_pressure, Mz_viscous
    
    # ============================================================
    # 其他自定义接口
    # ============================================================
    
    def add_boundary_condition(self, region: Box, bc_type: str, value: Any = None):
        """添加精细边界条件（接口定义，暂未实现）"""
        LOGGER.warning("[Solver] add_boundary_condition 功能暂未实现")
        raise NotImplementedError("精细边界条件功能待实现")


# ============================================================
# 命令行入口
# ============================================================
def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kármán 圆柱绕流求解器 (PhiFlow Torch)")
    parser.add_argument('--cfg', '-c', type=str, required=True, help='YAML 配置文件路径')
    parser.add_argument('--steps', '-n', type=int, default=None, help='仿真步数')
    parser.add_argument('--log', '-l', type=str, default=None, help='日志文件路径')
    args = parser.parse_args()
    
    solver = KarmanTorchSolver(args.cfg, log_file=args.log)
    results = solver.run(num_steps=args.steps)
    
    print("\n========== 最终统计量 ==========")
    for k, v in results['stats'].items():
        print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()
