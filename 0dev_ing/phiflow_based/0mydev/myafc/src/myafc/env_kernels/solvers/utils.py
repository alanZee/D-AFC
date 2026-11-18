"""

此模块包含求解器使用的辅助函数。

包含:
    - _InletZeroGradient: 入口专用 ZERO_GRADIENT (解决 mac_cormack + ConstantExtrapolation 兼容问题)
    - FixedSphere: 修复的 Sphere 几何体类（解决 PhiFlow eps 问题）
    - make_incompressible_with_jets: 带射流边界条件的压力投影（已废弃）
    - SDF 计算工具: compute_sdf, compute_sdf_gradient, compute_sdf_and_gradient

注意: make_incompressible_with_jets 已被废弃。
      JetObstacle 继承自 PhiFlow 的 Obstacle，可直接用于原生 make_incompressible()。
      射流边界条件通过 JetObstacle.velocity 属性实现，PhiFlow 会在
      apply_boundary_conditions() 中自动将障碍物区域的速度设置为射流速度场。

FixedSphere 使用 eps=1e-9 计算距离。PhiFlow Sphere 使用 eps=1e-3 时，
半径小于 sqrt(1e-3) 的圆在圆心处可能得到正的 SDF。
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from phi.field import StaggeredGrid, CenteredGrid
    from phi.geom import Geometry
    from myafc.env_kernels.solvers.fluid_control.jet_obstacle import JetObstacle

# 使用 myafc.utils.logging 统一日志模块
from myafc.utils import get_logger

LOGGER = get_logger()


# ============================================================================
# _InletZeroGradient: 入口专用外推 + PhiFlow monkey-patch
# ============================================================================
# 目的: 将入口速度边界从 ConstantExtrapolation(vec(x=.., y=..)) 替换为
#   _InletZeroGradient (标量外推)，避免 mac_cormack 内部 closest_values
#   对分量级 staggered tensor 做 pad 时因 vector 维度不匹配而断言失败。
#
# 同时 patch _pressure_extrapolation / _accessible_extrapolation，
#   使入口在 make_incompressible 中获得正确的压力 BC (Neumann)。
# ============================================================================

from phi.torch.flow import ZERO_GRADIENT as _ZERO_GRADIENT, extrapolation
from phi.physics import fluid as _phi_fluid

_ZeroGradientClass = type(_ZERO_GRADIENT)


class _InletZeroGradient(_ZeroGradientClass):
    """入流专用的 ZERO_GRADIENT，行为完全相同，仅类型不同。
    
    用于:
    1. 避免 ConstantExtrapolation(vec) 在 mac_cormack 中触发 vector 维度断言
    2. 在 _pressure_extrapolation 中映射到 Neumann 压力 BC
    """
    def __init__(self):
        super().__init__(2)  # pad_rank=2


# Monkey-patch PhiFlow 的 BC 映射函数
_orig_pressure_ext = _phi_fluid._pressure_extrapolation
_orig_accessible_ext = _phi_fluid._accessible_extrapolation


def _patched_pressure_ext(vext):
    """_InletZeroGradient → BOUNDARY (Neumann 压力)"""
    if isinstance(vext, _InletZeroGradient):
        return extrapolation.BOUNDARY
    return _orig_pressure_ext(vext)


def _patched_accessible_ext(vext):
    """_InletZeroGradient → ZERO (不可通行)"""
    if isinstance(vext, _InletZeroGradient):
        return extrapolation.ZERO
    return _orig_accessible_ext(vext)


_phi_fluid._pressure_extrapolation = _patched_pressure_ext
_phi_fluid._accessible_extrapolation = _patched_accessible_ext


# ============================================================================
# FixedSphere: 修复的 Sphere 几何体
# ============================================================================

from phi.geom import Geometry

class FixedSphere(Geometry):
    """
    采用较小距离下限的球形几何体。
    
    继承自 PhiFlow 的 Geometry 基类，包装标准 Sphere 并重写 approximate_signed_distance 方法
    以使用更精确的 eps 值。
    
    标准 PhiFlow Sphere 使用 eps=1e-3 时，半径 0.0254 的圆在圆心处
    得到的带符号距离约为 +0.006。FixedSphere 使用 eps=1e-9，使该值为负。
    
    用法:
        >>> from myafc.env_kernels.solvers.utils import FixedSphere
        >>> sphere = FixedSphere(center=vec(x=0.1, y=0.1), radius=0.025)
        >>> obstacle = Obstacle(sphere)
    """
    
    def __init__(self, *args, **kwargs):
        """
        创建 FixedSphere 实例。
        
        Args:
            与 phi.geom.Sphere 相同的参数
        """
        from phi.geom import Sphere
        self._sphere = Sphere(*args, **kwargs)
    
    def approximate_signed_distance(self, location):
        """修复的 SDF 计算，使用更小的 eps 值"""
        from phi.geom._functions import norm
        from phi import math
        distance = norm(location - self._sphere.pos, 'vector', eps=1e-9)
        return math.min(distance - self._sphere.radius, self._sphere.shape.instance)
    
    def approximate_fraction_inside(self, other_geometry, balance=0.5):
        """使用修复的 SDF 计算 fraction inside"""
        from phi import math
        radius = other_geometry.bounding_radius()
        location = other_geometry.center
        distance = self.approximate_signed_distance(location)
        inside_fraction = balance - distance / radius
        inside_fraction = math.clip(inside_fraction, 0, 1)
        return inside_fraction
    
    # ========== Geometry 接口实现 ==========
    
    @property
    def center(self):
        return self._sphere.center
    
    @property
    def shape(self):
        return self._sphere.shape
    
    @property
    def volume(self) -> 'Tensor':
        return self._sphere.volume
    
    @property
    def faces(self):
        return self._sphere.faces
    
    @property
    def face_centers(self):
        return self._sphere.face_centers
    
    @property
    def face_areas(self):
        return self._sphere.face_areas
    
    @property
    def face_normals(self):
        return self._sphere.face_normals
    
    @property
    def boundary_elements(self):
        return self._sphere.boundary_elements
    
    @property
    def boundary_faces(self):
        return self._sphere.boundary_faces
    
    @property
    def face_shape(self):
        return self._sphere.face_shape
    
    @property
    def corners(self):
        return self._sphere.corners
    
    @property
    def spatial_rank(self) -> int:
        return self._sphere.spatial_rank
    
    def lies_inside(self, location):
        return self._sphere.lies_inside(location)
    
    def approximate_closest_surface(self, location):
        return self._sphere.approximate_closest_surface(location)
    
    def bounding_radius(self):
        return self._sphere.bounding_radius()
    
    def bounding_half_extent(self):
        return self._sphere.bounding_half_extent()
    
    def at(self, center):
        new_sphere = self._sphere.at(center)
        result = FixedSphere.__new__(FixedSphere)
        result._sphere = new_sphere
        return result
    
    def rotated(self, angle):
        return self._sphere.rotated(angle)
    
    def scaled(self, factor):
        new_sphere = self._sphere.scaled(factor)
        result = FixedSphere.__new__(FixedSphere)
        result._sphere = new_sphere
        return result
    
    def push(self, positions, outward=True, shift_amount=0):
        return self._sphere.push(positions, outward, shift_amount)
    
    def sample_uniform(self, *shape):
        return self._sphere.sample_uniform(*shape)
    
    def __invert__(self):
        """支持 ~geom 操作（用于 accessible mask 计算）"""
        from phi.geom._geom_ops import InvertedGeometry
        return InvertedGeometry(self)
    
    def __repr__(self):
        return f"FixedSphere(center={self._sphere.center}, radius={self._sphere.radius})"
    
    # 代理一些常用的 Sphere 特有属性
    @property
    def pos(self):
        return self._sphere.pos
    
    @property
    def radius(self):
        return self._sphere.radius
    
    # 代理所有其他属性到内部 Sphere
    def __getattr__(self, name):
        """代理所有未定义的属性访问到内部 Sphere"""
        # 避免无限递归
        if name == '_sphere':
            raise AttributeError(name)
        return getattr(self._sphere, name)


# ============================================================================
# SDF 计算工具
# ============================================================================

def compute_sdf(
    geometry: 'Geometry',
    X: 'np.ndarray | torch.Tensor',
    Y: 'np.ndarray | torch.Tensor',
    device: Optional[str] = None,
    use_phiflow_fallback: bool = True,
) -> 'np.ndarray | torch.Tensor':
    """
    计算几何体的有符号距离函数（SDF）
    
    对于简单几何形状（Sphere, Box），使用精确的解析公式计算，
    避免 PhiFlow approximate_signed_distance 中的 eps 精度问题。
    
    PhiFlow 问题说明：
        PhiFlow 的 Sphere.approximate_signed_distance 内部使用 
        vec_length(..., eps=1e-3)，导致在球心附近 SDF 值被限制为 
        sqrt(eps) - r ≈ 0.032 - r，无法正确识别球内部区域。
    
    Args:
        geometry: PhiFlow 几何体 (Sphere, Box, 或其他)
        X, Y: 网格坐标，shape 相同
        device: torch 设备（仅当输入为 torch.Tensor 时需要）
        use_phiflow_fallback: 对于未知几何是否回退到 PhiFlow
    
    Returns:
        SDF 值，shape 与 X, Y 相同
        - SDF < 0: 内部
        - SDF = 0: 边界
        - SDF > 0: 外部
    
    Raises:
        ValueError: 当几何类型不支持且 use_phiflow_fallback=False 时
    
    Example:
        >>> from phi.geom import Sphere
        >>> import torch
        >>> sphere = Sphere(center=(0.5, 0.5), radius=0.1)
        >>> X, Y = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
        >>> sdf = compute_sdf(sphere, X, Y, device='cpu')
    """
    from phi.geom import Box, Sphere
    
    # 检测输入类型
    is_torch = hasattr(X, 'device')
    
    # 辅助函数：检查是否为球形几何（包括 FixedSphere）
    def _is_sphere_like(geom):
        """检查几何体是否为球形（Sphere 或 FixedSphere）"""
        if isinstance(geom, Sphere):
            return True
        # FixedSphere 包装了 _sphere 属性
        if hasattr(geom, '_sphere') and isinstance(geom._sphere, Sphere):
            return True
        return False
    
    def _get_sphere_params(geom):
        """获取球形几何的中心和半径"""
        # FixedSphere 使用 _sphere 属性存储实际的 Sphere
        sphere = geom._sphere if hasattr(geom, '_sphere') else geom
        cx = float(sphere.center.vector['x'])
        cy = float(sphere.center.vector['y'])
        radius = float(sphere.radius)
        return cx, cy, radius
    
    if is_torch:
        import torch
        
        if _is_sphere_like(geometry):
            # 圆/球: SDF = |r - c| - R
            cx, cy, radius = _get_sphere_params(geometry)
            dist = torch.sqrt((X - cx)**2 + (Y - cy)**2)
            return dist - radius
        
        elif isinstance(geometry, Box):
            if geometry.is_axis_aligned:
                # 未旋转的 Box: 直接计算
                center = geometry.center
                half_size = geometry.half_size
                cx = float(center.vector['x'])
                cy = float(center.vector['y'])
                hx = float(half_size.vector['x'])
                hy = float(half_size.vector['y'])
                
                dx = torch.abs(X - cx) - hx
                dy = torch.abs(Y - cy) - hy
                
                outside = torch.sqrt(torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2)
                inside = torch.clamp(torch.maximum(dx, dy), max=0)
                return outside + inside
            else:
                # 旋转的 Box: 需要变换坐标或使用 PhiFlow
                if use_phiflow_fallback:
                    return _compute_sdf_via_phiflow(geometry, X, Y, device)
                raise ValueError(f"旋转 Box 的 SDF 计算需要启用 PhiFlow fallback")
        
        else:
            if use_phiflow_fallback:
                return _compute_sdf_via_phiflow(geometry, X, Y, device)
            raise ValueError(f"不支持的几何类型: {type(geometry)}")
    
    else:
        # NumPy 路径
        if _is_sphere_like(geometry):
            cx, cy, radius = _get_sphere_params(geometry)
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            return dist - radius
        
        elif isinstance(geometry, Box):
            if geometry.is_axis_aligned:
                center = geometry.center
                half_size = geometry.half_size
                cx = float(center.vector['x'])
                cy = float(center.vector['y'])
                hx = float(half_size.vector['x'])
                hy = float(half_size.vector['y'])
                
                dx = np.abs(X - cx) - hx
                dy = np.abs(Y - cy) - hy
                
                outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
                inside = np.minimum(np.maximum(dx, dy), 0)
                return outside + inside
            else:
                if use_phiflow_fallback:
                    return _compute_sdf_via_phiflow_numpy(geometry, X, Y)
                raise ValueError(f"旋转 Box 的 SDF 计算需要启用 PhiFlow fallback")
        
        else:
            if use_phiflow_fallback:
                return _compute_sdf_via_phiflow_numpy(geometry, X, Y)
            raise ValueError(f"不支持的几何类型: {type(geometry)}")


def _compute_sdf_via_phiflow(geometry, X, Y, device):
    """使用 PhiFlow API 计算 SDF（torch 版本）"""
    import torch
    from phi.torch.flow import CenteredGrid
    from phi.geom import Box
    
    shape = X.shape
    Nx, Ny = shape[1], shape[0]  # 假设 (Ny, Nx) 或 (Nx, Ny) 顺序
    
    # 获取边界
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())
    
    # 创建临时网格
    grid = CenteredGrid(
        values=0,
        boundary=0,
        bounds=Box(x=(x_min, x_max), y=(y_min, y_max)),
        x=Nx, y=Ny,
    )
    
    sdf_field = geometry.approximate_signed_distance(grid.points)
    sdf_native = sdf_field.native(('y', 'x'))
    
    if isinstance(sdf_native, torch.Tensor):
        return sdf_native.to(device)
    return torch.tensor(sdf_native, dtype=torch.float32, device=device)


def _compute_sdf_via_phiflow_numpy(geometry, X, Y):
    """使用 PhiFlow API 计算 SDF（numpy 版本）"""
    from phi.flow import CenteredGrid
    from phi.geom import Box
    
    shape = X.shape
    Nx, Ny = shape[1], shape[0]
    
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())
    
    grid = CenteredGrid(
        values=0,
        boundary=0,
        bounds=Box(x=(x_min, x_max), y=(y_min, y_max)),
        x=Nx, y=Ny,
    )
    
    sdf_field = geometry.approximate_signed_distance(grid.points)
    return np.array(sdf_field.native(('y', 'x')))


def compute_sdf_gradient(
    sdf: 'np.ndarray | torch.Tensor',
    dx: float,
    dy: float,
) -> Tuple['np.ndarray | torch.Tensor', 'np.ndarray | torch.Tensor']:
    """
    计算 SDF 的梯度（使用中心差分）
    
    Args:
        sdf: SDF 场，shape (Ny, Nx)
        dx, dy: 网格间距
    
    Returns:
        (grad_x, grad_y): SDF 梯度分量，shape 与 sdf 相同
    """
    is_torch = hasattr(sdf, 'device')
    
    if is_torch:
        import torch
        grad_x = torch.zeros_like(sdf)
        grad_y = torch.zeros_like(sdf)
        
        # 中心差分（内部）
        grad_x[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2 * dx)
        grad_y[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2 * dy)
        
        # 单边差分（边界）
        grad_x[:, 0] = (sdf[:, 1] - sdf[:, 0]) / dx
        grad_x[:, -1] = (sdf[:, -1] - sdf[:, -2]) / dx
        grad_y[0, :] = (sdf[1, :] - sdf[0, :]) / dy
        grad_y[-1, :] = (sdf[-1, :] - sdf[-2, :]) / dy
    else:
        grad_x = np.zeros_like(sdf)
        grad_y = np.zeros_like(sdf)
        
        grad_x[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2 * dx)
        grad_y[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2 * dy)
        
        grad_x[:, 0] = (sdf[:, 1] - sdf[:, 0]) / dx
        grad_x[:, -1] = (sdf[:, -1] - sdf[:, -2]) / dx
        grad_y[0, :] = (sdf[1, :] - sdf[0, :]) / dy
        grad_y[-1, :] = (sdf[-1, :] - sdf[-2, :]) / dy
    
    return grad_x, grad_y


def compute_sdf_normal(
    sdf: 'np.ndarray | torch.Tensor',
    dx: float,
    dy: float,
    eps: float = 1e-8,
) -> Tuple['np.ndarray | torch.Tensor', 'np.ndarray | torch.Tensor']:
    """
    计算 SDF 的单位法向量（∇SDF / |∇SDF|）
    
    法向量指向几何体外部（SDF 增大方向）
    
    Args:
        sdf: SDF 场，shape (Ny, Nx)
        dx, dy: 网格间距
        eps: 防止除零的小量
    
    Returns:
        (nx, ny): 单位法向量分量，shape 与 sdf 相同
    """
    grad_x, grad_y = compute_sdf_gradient(sdf, dx, dy)
    
    is_torch = hasattr(sdf, 'device')
    
    if is_torch:
        import torch
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2).clamp(min=eps)
    else:
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.maximum(grad_mag, eps)
    
    nx = grad_x / grad_mag
    ny = grad_y / grad_mag
    
    return nx, ny


def compute_sdf_and_gradient(
    geometry: 'Geometry',
    X: 'np.ndarray | torch.Tensor',
    Y: 'np.ndarray | torch.Tensor',
    dx: float,
    dy: float,
    device: Optional[str] = None,
    use_phiflow_fallback: bool = True,
) -> Dict[str, 'np.ndarray | torch.Tensor']:
    """
    一次性计算 SDF、梯度和法向量
    
    这是推荐的接口，避免重复计算。
    
    Args:
        geometry: PhiFlow 几何体
        X, Y: 网格坐标
        dx, dy: 网格间距
        device: torch 设备
        use_phiflow_fallback: 是否允许 PhiFlow fallback
    
    Returns:
        dict 包含:
            - 'sdf': 有符号距离场
            - 'grad_x', 'grad_y': SDF 梯度
            - 'nx', 'ny': 单位法向量
            - 'grad_mag': 梯度模长
    """
    sdf = compute_sdf(geometry, X, Y, device, use_phiflow_fallback)
    grad_x, grad_y = compute_sdf_gradient(sdf, dx, dy)
    
    is_torch = hasattr(sdf, 'device')
    eps = 1e-8
    
    if is_torch:
        import torch
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2).clamp(min=eps)
    else:
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.maximum(grad_mag, eps)
    
    nx = grad_x / grad_mag
    ny = grad_y / grad_mag
    
    return {
        'sdf': sdf,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'nx': nx,
        'ny': ny,
        'grad_mag': grad_mag,
    }


# 已废弃, 请使用 JetObstacle 直接用于 make_incompressible()
def make_incompressible_with_jets(
    velocity: 'StaggeredGrid',
    jet_obstacle: 'JetObstacle',
    solve=None,
):
    """
    带射流边界条件的压力投影（可微分实现）
    
    此函数实现了 PhiFlow 的 make_incompressible 逻辑，但正确处理射流BC：
    1. 将障碍物内部速度清零（壁面BC）
    2. 在射流区域注入射流速度（使用已存储的控制信号）（源/汇效应）
    3. 计算散度（射流参与）
    4. 压力求解
    5. 速度投影
    6. 再次强制射流BC
    
    射流BC的物理意义：
    - 射流作为 Dirichlet 边界，类似 inflow 边界
    - 在散度方程中，射流引入源/汇效应
    - 压力场会自动调整以满足连续性
   
    
    Args:
        velocity: 速度场（StaggeredGrid）
        jet_obstacle: 带射流配置的障碍物
        solve: Solve 配置
    
    Returns:
        (velocity, pressure): 无散度速度场和压力场
    """
    from phi.physics.fluid import _accessible_extrapolation, _pressure_extrapolation
    from phi.physics.fluid import masked_laplace, _balance_divergence
    from phi.field import divergence, Field, resample
    import phi.field as field
    from phiml.math import solve_linear, Solve, channel, NUMPY
    from phi.math import extrapolation, copy_with
    from phi import math
    
    if solve is None:
        solve = Solve('auto')
    
    input_velocity = velocity
    geometry = jet_obstacle.geometry
    
    # 1. 计算障碍物可达性 mask
    accessible_boundary = _accessible_extrapolation(input_velocity.extrapolation)
    with NUMPY:
        accessible = Field(velocity.geometry, ~geometry, accessible_boundary)
        hard_bcs = field.stagger(accessible, math.minimum, velocity.boundary, 
                                  at=velocity.sampled_at, dims=velocity.vector.item_names)
    active = accessible.with_extrapolation(extrapolation.NONE)
    
    # 2. 障碍物内部速度清零
    obs_mask = resample(geometry, velocity, soft=True, balance=1)
    velocity = field.safe_mul(1 - obs_mask, velocity)
    
    # 3. 注入射流速度（使用已存储的控制信号，仅内部区域）
    velocity = jet_obstacle.apply_jet_bc(velocity, include_outer=False)
    
    # 4. 计算散度
    div = divergence(velocity, order=2)
    div *= active
    
    # 5. 压力求解
    assert not channel(div), f"Divergence must not have any channel dimensions. v={input_velocity}"
    
    if not input_velocity.extrapolation.is_flexible:
        solve = solve.with_preprocessing(_balance_divergence, active)
        if solve.rank_deficiency is None:
            solve = copy_with(solve, rank_deficiency=1)
    
    if solve.x0 is None:
        pressure_extrapolation = _pressure_extrapolation(input_velocity.extrapolation)
        solve = copy_with(solve, x0=Field(div.geometry, 0, pressure_extrapolation))
    
    pressure = solve_linear(
        masked_laplace, div, solve,
        velocity.boundary, hard_bcs, active,
        wide_stencil=False, order=2, implicit=None, upwind=None, correct_skew=False
    )
    
    # 6. 速度投影
    grad_pressure = field.spatial_gradient(
        pressure, input_velocity.extrapolation, 
        at=velocity.sampled_at, order=2, scheme='green-gauss'
    )
    grad_pressure *= hard_bcs
    velocity = (velocity - grad_pressure).with_extrapolation(input_velocity.extrapolation)
    
    # 7. 再次强制射流BC（包含外侧）
    velocity = jet_obstacle.apply_jet_bc(velocity, include_outer=False)
    
    return velocity, pressure
