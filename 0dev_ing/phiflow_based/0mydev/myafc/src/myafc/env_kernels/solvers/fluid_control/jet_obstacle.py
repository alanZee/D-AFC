"""
JetObstacle - 带射流边界条件的障碍物

设计理念：
    JetObstacle 继承自 phi.physics.fluid.Obstacle
    重写 velocity 属性，返回根据当前控制信号计算的射流速度场 (PhiFlow 原生 make_incompressible 会使用该属性)
    提供 apply_jet_bc() 方法用于在求解流程中多次施加射流 BC
    此外，需要统一的喷口定位方法（基于 SDF + 角度）以便于支持任意几何形状（圆柱、方柱、旋转方柱等）

射流 BC 施加机制（参考 PhiFlow 求解域边界条件设计）:
    射流 BC 本质上是 Dirichlet BC（固定速度值），与求解域入口边界条件等效，需要在求解流程多个环节施加:
      1. 扩散后 (通过 solver 调用 apply_jet_bc)
      2. 对流前（通过 solver 调用 apply_jet_bc）
      3. 压力投影中（通过 JetObstacle.velocity 属性(PhiFlow 内部会调用 apply_boundary_conditions)）
      4. 求解 step 后 (确保最终速度场满足边界条件)
      补充说明：
        PhiFlow 求解域边界通过 StaggeredGrid.with_boundary() 在多个环节施加:
            - 扩散前 (diffuse.explicit 内部仅自动处理求解域边界)
            - 对流前 (通过 .with_boundary 或 extrapolation 机制)
            - 压力投影 (make_incompressible 中的 apply_boundary_conditions)
        PhiFlow 原生结构物边界仅在 压力投影 中施加 (make_incompressible 中的 apply_boundary_conditions)

喷嘴定位策略：
    1. 首先确定喷嘴网格点集合（基于角度范围 + 表面位置(使用 SDF )）
    2. 然后在这些网格点上应用空间分布（uniform/cosine）
    
    空间分布与缓存：
    - 喷嘴作用区域沿表面法向（SDF 梯度方向）延伸，而非径向
    - 对于静态障碍物，缓存喷嘴位置信息避免重复计算
    - cosine 分布：先确定网格点，再计算权重并归一化

射流角度约定（物理绝对坐标系）:
    - 0°   = x轴正方向 (→)
    - 90°  = y轴正方向 (↑)
    - 180° = x轴负方向 (←)
    - 270° = y轴负方向 (↓)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field as dataclass_field

import numpy as np
import torch

from phi.physics.fluid import Obstacle

# 使用 myafc.utils.logging 统一日志模块
from myafc.utils import get_logger

LOGGER = get_logger()

if TYPE_CHECKING:
    from phi.field import StaggeredGrid
    from phi.geom import Geometry


@dataclass
class JetNozzle:
    """单个喷嘴配置"""
    arc_deg: Tuple[float, float]  # 角度范围 [start, end]（度）
    
    def __post_init__(self):
        if isinstance(self.arc_deg, list):
            self.arc_deg = tuple(self.arc_deg)
    
    @property
    def arc_width_deg(self) -> float:
        """弧段宽度（度）"""
        start, end = self.arc_deg
        return (end - start) if end >= start else (360 - start + end)
    
    @property
    def arc_center_deg(self) -> float:
        """弧段中心角度（度）"""
        start, end = self.arc_deg
        if end >= start:
            return (start + end) / 2
        return ((start + end + 360) / 2) % 360


@dataclass
class JetConfig:
    """
    射流配置
    
    控制信号处理流程:
        1. Policy 输出原始控制信号 (amplitude, angle)
        2. 饱和裁剪: |A| <= saturation
        3. 缩放: A = A * amplitude_scale
        4. 平滑: A = (1-α)*A_prev + α*A_curr
        5. 空间分布: uniform / cosine
    
    平滑公式: A_smooth = (1 - alpha) * A_prev + alpha * A_current
        - alpha=0: 完全继承上帧（无响应）
        - alpha=1: 无平滑（完全跟随当前）
    """
    nozzles: List[JetNozzle] = dataclass_field(default_factory=list)
    saturation: float = 10.0       # 饱和裁剪: |A| <= saturation
    amplitude_scale: float = 1.0   # 缩放系数
    smoothing_alpha: float = 1.0   # 平滑系数
    profile: str = 'uniform'       # 空间分布: 'uniform' / 'cosine'
    
    @classmethod
    def from_dict(cls, cfg: Dict) -> 'JetConfig':
        nozzles = []
        for n in cfg.get('nozzles', []):
            if isinstance(n, dict):
                nozzles.append(JetNozzle(arc_deg=tuple(n.get('arc_deg', [0, 10]))))
            elif isinstance(n, JetNozzle):
                nozzles.append(n)
        return cls(
            nozzles=nozzles,
            saturation=cfg.get('saturation', 10.0),
            amplitude_scale=cfg.get('amplitude_scale', 1.0),
            smoothing_alpha=cfg.get('smoothing_alpha', 1.0),
            profile=cfg.get('profile', 'uniform'),
        )


@dataclass
class NozzleCache:
    """
    喷嘴几何信息缓存（用于静态障碍物）
    
    缓存内容：
    - nozzle_masks: 每个喷嘴的布尔 mask (List of Tensor)，包含核心+延伸区域
    - nozzle_core_masks: 每个喷嘴的核心区域 mask（仅第一层，即射流出口）
    - nozzle_weights: 每个喷嘴的空间分布权重 (List of Tensor)
    - total_mask: 所有喷嘴的联合 mask (Tensor)
    - grid_shape: 网格形状 (Nx+1, Ny+1)
    """
    nozzle_masks: List[torch.Tensor]       # 每个喷嘴的完整 mask（核心+延伸）
    nozzle_core_masks: List[torch.Tensor]  # 每个喷嘴的核心区域 mask（仅第一层）
    nozzle_weights: List[torch.Tensor]     # 每个喷嘴的分布权重
    total_mask: torch.Tensor               # 联合 mask
    grid_shape: Tuple[int, int]            # 网格形状


class JetObstacle(Obstacle):
    """
    带射流边界条件的障碍物（继承自 PhiFlow Obstacle）
    
    此类通过重写 velocity 属性，使其返回一个空间变化的射流速度场，
    从而可以直接用于原生 make_incompressible() 函数。

    控制信号流程：
    1. update_controls(state): 在 step 开始时调用，计算控制信号
    2. apply_jet_bc(velocity): 在求解流程中多次调用，施加射流 BC
    3. 压力投影: 通过 JetObstacle.velocity 属性自动施加射流 BC
    
    关键属性:
        - velocity: 返回射流速度场（StaggeredGrid），用于压力投影
        - current_controls: 当前控制信号 [(amp, angle), ...]
    
    关键方法:
        - update_controls(state): 更新控制信号
        - apply_jet_bc(velocity): 施加射流 BC
        - set_velocity_template(velocity): 设置速度场模板（通常在初始化时已完成）
    
    喷嘴缓存:
        喷嘴的精确位置 mask 在实例化时就会计算（如果提供了 velocity_template），
        可通过 _nozzle_cache 属性访问。Runner 等组件应直接使用此缓存进行可视化。
    """
    
    def __init__(
        self,
        geometry: 'Geometry',
        jet_config: Optional[JetConfig] = None,
        policy: Optional[Any] = None,
        velocity_template: Optional['StaggeredGrid'] = None,
        # 障碍物的平移速度固定为零。
        angular_velocity: float = 0,
    ):
        """
        Args:
            geometry: 障碍物几何形状 (Sphere/Box)
            jet_config: 射流配置
            policy: 控制策略
            velocity_template: 速度场模板（StaggeredGrid），用于初始化喷嘴缓存。
                              如果不提供，需要后续调用 set_velocity_template()。
                              推荐在创建时就提供，以便立即计算喷嘴缓存。
            angular_velocity: 障碍物角速度
        """
        # 先初始化射流相关属性
        self.jet_config = jet_config or JetConfig()
        self._policy = policy
        self._current_controls: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        # 存储神经网络原始输出（未处理），由 update_controls 写入
        self._raw_controls: Optional[List[Tuple[Any, Any]]] = None
        # 存储上一步最终实际施加在流场中的平滑后 amp（用于时间平滑）
        # 初始为 None，代表初始施加的是 0 速度
        self._prev_applied_amplitudes: Optional[np.ndarray] = None
        self._device = 'cpu'
        # 确定性策略模式：True 时 policy 直接输出均值，不进行随机采样
        self._deterministic = False
        self._velocity_template: Optional['StaggeredGrid'] = None
        self._jet_velocity_field: Optional['StaggeredGrid'] = None
        
        # 喷嘴几何缓存（静态障碍物只计算一次）
        self._nozzle_cache: Optional[NozzleCache] = None
        
        # 调用父类初始化
        super().__init__(geometry, velocity=0, angular_velocity=angular_velocity)
        
        # 如果提供了速度场模板，立即初始化喷嘴缓存
        if velocity_template is not None:
            self.set_velocity_template(velocity_template)
    
    # ============ 重写 Obstacle 属性 ============
    
    @property
    def velocity(self):
        """
        返回射流速度场（StaggeredGrid）
        
        这是关键的重写：PhiFlow 的 apply_boundary_conditions 会使用此属性
        来设置障碍物区域的速度。
        """
        if self._jet_velocity_field is not None:
            return self._jet_velocity_field
        return 0
    
    @velocity.setter
    def velocity(self, value):
        """忽略父类的 velocity 设置"""
        pass
    
    @property
    def is_stationary(self):
        """兼容父类API: 检查障碍物是否静止（此处"静止"的语义: 无射流且不旋转）"""
        if self.num_nozzles > 0 and self._jet_velocity_field is not None:
            return False
        return not self.is_rotating
    
    @property
    def is_moving(self):
        """兼容父类API: 检查障碍物是否移动（此处"移动"的语义: 有射流）"""
        return self.num_nozzles > 0 and self._jet_velocity_field is not None
    
    # ============ 射流相关属性 ============
    
    @property
    def policy(self):
        return self._policy
    
    @policy.setter
    def policy(self, p):
        self._policy = p
    
    @property
    def deterministic(self) -> bool:
        """确定性策略模式：True 时 policy 直接输出均值，不进行随机采样"""
        return self._deterministic
    
    @deterministic.setter
    def deterministic(self, value: bool):
        self._deterministic = value
    
    @property
    def num_nozzles(self) -> int:
        return len(self.jet_config.nozzles)
    
    @property
    def current_controls(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """当前控制信号"""
        if self._current_controls is None:
            return [(torch.tensor(0.0, device=self._device), 
                     torch.tensor(0.0, device=self._device))] * self.num_nozzles
        return self._current_controls
    
    def set_device(self, device: str):
        """设置计算设备"""
        self._device = device
        if self._policy is not None and hasattr(self._policy, 'device'):
            self._policy.device = device
    
    def set_velocity_template(self, velocity: 'StaggeredGrid'):
        """
        设置速度场模板并初始化喷嘴缓存
        
        Args:
            velocity: 速度场模板
        """
        self._velocity_template = velocity
        # 初始化/更新喷嘴缓存
        self._init_nozzle_cache()
        # 重建射流速度场
        self._rebuild_jet_velocity_field()
    
    # ============ 喷嘴缓存管理 ============
    
    def _init_nozzle_cache(self):
        """
        初始化喷嘴几何缓存（静态障碍物只需计算一次）
        
        喷嘴区域沿表面法向延伸：
        1. 计算 SDF 确定结构物内部区域
        2. 使用"到中心角度"确定喷嘴位于哪个表面
        3. 选取该表面的"内部最外层"网格点（紧贴表面的内部网格）
        4. 检查这些网格点的 8 邻域：如果邻居是"更内部"的网格（非流体、非最外层），
           则将其作为延伸网格加入喷嘴 mask
        5. 计算空间分布权重（cosine 分布会归一化）
        
        注意：当前使用"到中心角度"定位喷嘴，这适用于凸形障碍物。
        """
        if self._velocity_template is None or self.num_nozzles == 0:
            self._nozzle_cache = None
            return
        
        velocity = self._velocity_template
        device = self._get_device_from_field(velocity)
        
        # 获取域信息
        bounds = velocity.bounds
        res_shape = velocity.resolution
        Nx, Ny = res_shape.get_size('x'), res_shape.get_size('y')
        Lx, Ly = float(bounds.size['x']), float(bounds.size['y'])
        dx, dy = Lx / Nx, Ly / Ny
        
        # StaggeredGrid 网格坐标 (Nx+1, Ny+1)
        x_coords = torch.linspace(0, Lx, Nx+1, device=device)
        y_coords = torch.linspace(0, Ly, Ny+1, device=device)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # 计算 SDF（使用 PhiFlow 原生方法，支持任意几何）
        sdf = self._compute_sdf(X, Y, device)
        
        # 计算到障碍物中心的角度（用于喷嘴定位）
        center = self.geometry.center
        cx = float(center.vector['x'])
        cy = float(center.vector['y'])
        angles = torch.atan2(Y - cy, X - cx) * (180.0 / torch.pi) % 360
        
        # 结构物内部 mask: sdf < 0
        solid_interior_mask = sdf < 0
        
        # 第二层及更内部的网格：sdf < -threshold
        inner_layer_threshold = max(dx, dy) * 1.0
        inner_layers_mask = sdf < -inner_layer_threshold
        
        # ========== 为每个喷嘴计算 mask 和权重 ==========
        nozzle_masks = []
        nozzle_core_masks = []  # 仅核心区域（射流出口第一层）
        nozzle_weights = []
        total_mask = torch.zeros_like(X, dtype=torch.bool)
        
        for nozzle in self.jet_config.nozzles:
            # 计算喷嘴核心区域（基于表面和角度）
            nozzle_core_mask = self._compute_nozzle_first_layer(
                nozzle=nozzle,
                X=X, Y=Y, sdf=sdf, angles=angles,
                dx=dx, dy=dy,
                device=device,
            )
            nozzle_core_masks.append(nozzle_core_mask)  # 保存核心 mask
            
            # 法向延伸：检查核心区域每个点的 8 邻域，
            # 如果邻居在"更内部的层"（inner_layers），则加入延伸 mask
            extended_mask = self._extend_nozzle_by_neighbors(
                core_mask=nozzle_core_mask,
                inner_layers_mask=inner_layers_mask,
                shape=(Nx+1, Ny+1),
                device=device
            )
            
            # 最终喷嘴 mask = 核心 + 延伸
            nozzle_mask = nozzle_core_mask | extended_mask
            nozzle_masks.append(nozzle_mask)
            total_mask = total_mask | nozzle_mask
            
            # 计算空间分布权重
            if self.jet_config.profile == 'cosine':
                weight = self._compute_cosine_weight(angles, nozzle, nozzle_mask, device)
            else:
                weight = nozzle_mask.float()
            nozzle_weights.append(weight)
        
        self._nozzle_cache = NozzleCache(
            nozzle_masks=nozzle_masks,
            nozzle_core_masks=nozzle_core_masks,  # 保存核心 mask
            nozzle_weights=nozzle_weights,
            total_mask=total_mask,
            grid_shape=(Nx+1, Ny+1),
        )
    
    def _compute_nozzle_first_layer(
        self,
        nozzle: JetNozzle,
        X: torch.Tensor,
        Y: torch.Tensor,
        sdf: torch.Tensor,
        angles: torch.Tensor,
        dx: float,
        dy: float,
        device,
    ) -> torch.Tensor:
        """
        计算喷嘴在结构物内部最外层（第一层）的位置
        
        策略：
        1. 根据喷嘴角度确定所在表面
        2. 对于该表面，选取紧贴表面内侧的网格点
        3. 在角度范围内筛选
        
        这确保选取的是沿着表面分布的点，而非沿角度扇形分布的点。
        """
        from phi.geom import Box, Sphere
        
        arc_start, arc_end = nozzle.arc_deg
        arc_center = nozzle.arc_center_deg
        
        # 角度 mask（处理跨 0° 的情况）
        if arc_start <= arc_end:
            angle_mask = (angles >= arc_start) & (angles <= arc_end)
        else:
            angle_mask = (angles >= arc_start) | (angles <= arc_end)
        
        geom = self.geometry
        center = geom.center
        cx = float(center.vector['x'])
        cy = float(center.vector['y'])
        
        # ===== 对于未旋转的正放的 Box（方形）障碍物：基于表面位置选取 =====
        if isinstance(geom, Box) and geom.is_axis_aligned:
            half_size = geom.half_size
            hx = float(half_size.vector['x'])
            hy = float(half_size.vector['y'])
            
            # 结构物边界
            x_left = cx - hx
            x_right = cx + hx
            y_bottom = cy - hy
            y_top = cy + hy
            
            # 网格尺寸容差
            tol = max(dx, dy) * 1.5
            
            # 根据角度中心判断表面
            if 45 <= arc_center < 135:
                # 上表面：选取 y 接近 y_top 且 x 在范围内的内部点
                surface_mask = (
                    (Y >= y_top - tol) & (Y <= y_top) &  # y 紧贴上边界内侧
                    (X >= x_left) & (X <= x_right)       # x 在边界内
                )
            elif 135 <= arc_center < 225:
                # 左表面
                surface_mask = (
                    (X >= x_left) & (X <= x_left + tol) &  # x 紧贴左边界内侧
                    (Y >= y_bottom) & (Y <= y_top)         # y 在边界内
                )
            elif 225 <= arc_center < 315:
                # 下表面
                surface_mask = (
                    (Y >= y_bottom) & (Y <= y_bottom + tol) &  # y 紧贴下边界内侧
                    (X >= x_left) & (X <= x_right)             # x 在边界内
                )
            else:
                # 右表面 (arc_center >= 315 or arc_center < 45)
                surface_mask = (
                    (X >= x_right - tol) & (X <= x_right) &  # x 紧贴右边界内侧
                    (Y >= y_bottom) & (Y <= y_top)           # y 在边界内
                )
            
            # 结合角度和表面
            # 注意：这里 surface_mask 已经选取了表面上的点，
            # angle_mask 用于在表面上进一步筛选角度范围
            nozzle_core_mask = surface_mask & angle_mask & (sdf < 0)
        
        # ===== 圆形或其他几何形状：统一使用 SDF + 角度 =====
        # 对于圆形等非 Box 几何，最外层是 SDF 接近 0 的区域
        # 此处圆形和"其他"的处理逻辑完全相同，故合并
        else:
            outer_layer_threshold = max(dx, dy) * 1.0
            first_layer_mask = (sdf >= -outer_layer_threshold) & (sdf < 0)
            nozzle_core_mask = first_layer_mask & angle_mask
        
        return nozzle_core_mask
    
    def _extend_nozzle_by_neighbors(
        self,
        core_mask: torch.Tensor,
        inner_layers_mask: torch.Tensor,
        shape: Tuple[int, int],
        device,
    ) -> torch.Tensor:
        """
        通过 8 邻域检查将喷嘴核心区域向内部延伸
        
        对于核心区域（第一层/最外层）的每个点，检查其周围 8 个邻居：
        - 如果邻居在"更内部的层"（inner_layers_mask 为 True），
        - 则将该邻居加入延伸 mask
        
        这确保喷嘴区域沿表面法向延伸，而非径向延伸。
        
        Args:
            core_mask: 喷嘴核心区域 mask（第一层上的喷嘴位置）
            inner_layers_mask: 更内部层的 mask（第二层及更内）
            shape: 网格形状 (Nx+1, Ny+1)
            device: torch 设备
        
        Returns:
            extended_mask: 延伸区域的 mask
        """
        Nx1, Ny1 = shape
        extended_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        
        # 8 邻域偏移 (di, dj)
        # 包括：上下左右 + 四个对角
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]
        
        # 获取核心区域的索引
        core_indices = torch.nonzero(core_mask, as_tuple=False)  # shape: (N, 2)
        
        for idx in core_indices:
            i, j = int(idx[0]), int(idx[1])
            
            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                
                # 边界检查
                if 0 <= ni < Nx1 and 0 <= nj < Ny1:
                    # 如果邻居在更内部的层，加入延伸 mask
                    if inner_layers_mask[ni, nj]:
                        extended_mask[ni, nj] = True
        
        return extended_mask
    
    def _compute_cosine_weight(
        self, 
        angles: torch.Tensor, 
        nozzle: JetNozzle, 
        mask: torch.Tensor,
        device
    ) -> torch.Tensor:
        """
        计算喷嘴网格上的归一化余弦权重。
        
        计算步骤：
        1. 先确定喷嘴覆盖的网格点（mask）
        2. 在这些点上计算余弦权重
        3. 归一化使最大权重为 1.0
        
        这样无论喷嘴多窄、网格点是否在弧段正中心，
        喷嘴区域内的最大速度都能达到设定的振幅值。
        """
        if not mask.any():
            return torch.zeros_like(angles)
        
        arc_center = nozzle.arc_center_deg
        arc_width = nozzle.arc_width_deg
        
        # 计算角度偏移（处理跨 0° 的情况）
        angle_offset = angles - arc_center
        angle_offset = (angle_offset + 180) % 360 - 180
        
        # 余弦权重（在弧宽范围内）
        half_width = arc_width / 2
        raw_weight = torch.where(
            torch.abs(angle_offset) <= half_width,
            torch.cos(torch.tensor(torch.pi / arc_width, device=device) * angle_offset),
            torch.zeros_like(angle_offset)
        )
        
        # 应用 mask
        masked_weight = raw_weight * mask.float()
        
        # 归一化：确保最大权重为 1.0
        max_val = masked_weight.max()
        if max_val > 1e-8:
            masked_weight = masked_weight / max_val
        
        return masked_weight
    
    # ============ SDF 计算 ============
    
    def _compute_sdf(self, X: torch.Tensor, Y: torch.Tensor, device) -> torch.Tensor:
        """
        计算 SDF（使用统一的 SDF 工具，支持所有几何形状）
        
        Args:
            X, Y: 网格坐标 (任意 shape)
            device: torch 设备
        
        Returns:
            SDF tensor，shape 与 X, Y 相同
        """
        from myafc.env_kernels.solvers.utils import compute_sdf
        return compute_sdf(self.geometry, X, Y, device=str(device), use_phiflow_fallback=True)
    
    # ============ 控制信号更新 ============
    
    def update_controls(self, state: Optional[Dict[str, Any]] = None):
        """
        从策略获取原始控制信号（不做处理，只存储）
        
        处理流程会在 apply_controls() 中统一完成：
        裁剪 → 缩放 → 平滑 → 空间分布 → 重建速度场
        
        Args:
            state: 当前状态（如 time, probes, force_coeffs 等）
        """
        if self._policy is None or self.num_nozzles == 0:
            return
        
        raw_controls = self._policy(state, deterministic=self._deterministic)
        if raw_controls is None:
            return
        
        # 规范化格式并存储
        if isinstance(raw_controls, (list, tuple)):
            if len(raw_controls) == 0:
                return
            first = raw_controls[0]
            if isinstance(first, (int, float, torch.Tensor)) and not isinstance(first, tuple):
                raw_controls = [raw_controls] * self.num_nozzles
        
        # 存储原始控制信号（供 apply_controls 使用）
        self._raw_controls = raw_controls
    
    def set_controls_external(self, amplitudes, angles=None):
        """
        外部直接设置原始控制信号（注意：供 JetTrainer 等外部 Agent 使用，忽略有配置指定的原生内部策略）
        
        Args:
            amplitudes: 控制幅值（标量、列表或张量）
            angles: 控制角度（度），默认 90°
        """
        if self.num_nozzles == 0:
            return
        
        # 规范化幅值
        if isinstance(amplitudes, (int, float)):
            amplitudes = [amplitudes] * self.num_nozzles
        elif isinstance(amplitudes, torch.Tensor) and amplitudes.dim() == 0:
            amplitudes = [amplitudes] * self.num_nozzles
        elif isinstance(amplitudes, torch.Tensor):
            amplitudes = [amplitudes[i] if i < len(amplitudes) else amplitudes[-1] 
                         for i in range(self.num_nozzles)]
        
        # 规范化角度
        if angles is None:
            angles = [90.0] * self.num_nozzles
        elif isinstance(angles, (int, float)):
            angles = [angles] * self.num_nozzles
        
        # 存储原始控制信号
        self._raw_controls = [(amplitudes[i], angles[i]) for i in range(self.num_nozzles)]
    
    def apply_controls(self):
        """
        应用控制信号（每个施加射流的时间步调用）
        
        完整处理流程：
        1. 裁剪：饱和限制
        2. 缩放：amplitude_scale
        3. 平滑：时间平滑（与上一步实际施加值）（实际施加 = (1-α) * 上一步实际施加 + α * 神经网络原始输出）
        4. 空间分布：uniform/cosine（在 _rebuild_jet_velocity_field 中）
        """
        if self._raw_controls is None or self.num_nozzles == 0:
            return
        
        controls = []
        for i, ctrl in enumerate(self._raw_controls):
            if isinstance(ctrl, (list, tuple)) and len(ctrl) >= 2:
                amp, angle = ctrl[0], ctrl[1]
            else:
                amp = ctrl
                angle = torch.tensor(90.0, device=self._device)
            
            if not isinstance(amp, torch.Tensor):
                amp = torch.tensor(float(amp), device=self._device)
            if not isinstance(angle, torch.Tensor):
                angle = torch.tensor(float(angle), device=self._device)
            
            # 1. 裁剪（饱和限制）
            amp = torch.clamp(amp, -self.jet_config.saturation, self.jet_config.saturation)
            
            # 2. 缩放
            amp = amp * self.jet_config.amplitude_scale
            
            # 3. 时间平滑：实际施加 = (1-α) * 上一步实际施加 + α * 当前原始值
            alpha = self.jet_config.smoothing_alpha
            if self._prev_applied_amplitudes is not None and i < len(self._prev_applied_amplitudes):
                prev_applied = self._prev_applied_amplitudes[i]
            else:
                prev_applied = 0.0
            amp = (1 - alpha) * prev_applied + alpha * amp
            
            controls.append((amp, angle))
        
        self._current_controls = controls
        # 更新历史：存储本次最终施加的平滑后幅值
        self._prev_applied_amplitudes = np.array([float(c[0].detach().cpu()) for c in controls])
        
        # 4. 空间分布 + 重建射流速度场
        self._rebuild_jet_velocity_field()
    
    # ============ 射流速度场构建 ============
    
    def _rebuild_jet_velocity_field(self):
        """
        根据当前控制信号重建射流速度场
        
        使用缓存的喷嘴几何信息，仅重新计算速度值。
        """
        if self._velocity_template is None or self.num_nozzles == 0:
            self._jet_velocity_field = None
            return
        
        if self._nozzle_cache is None:
            self._init_nozzle_cache()
            if self._nozzle_cache is None:
                return
        
        from phi.torch.flow import math, StaggeredGrid, spatial
        
        velocity = self._velocity_template
        controls = self.current_controls
        device = self._get_device_from_field(velocity)
        
        Nx1, Ny1 = self._nozzle_cache.grid_shape
        
        # 累积射流速度
        total_vx = torch.zeros((Nx1, Ny1), device=device)
        total_vy = torch.zeros((Nx1, Ny1), device=device)
        
        for i, (amp, angle_deg) in enumerate(controls):
            if i >= self.num_nozzles:
                break
            
            weight = self._nozzle_cache.nozzle_weights[i]
            
            amp = amp.to(device)
            angle_deg = angle_deg.to(device)
            
            # 计算射流速度分量（使用绝对坐标系角度）
            angle_rad = angle_deg * (torch.pi / 180.0)
            jet_vx = amp * torch.cos(angle_rad)
            jet_vy = amp * torch.sin(angle_rad)
            
            # 累加（权重已经包含了空间分布）
            total_vx = total_vx + weight * jet_vx
            total_vy = total_vy + weight * jet_vy
        
        # 创建 StaggeredGrid
        staggered = torch.stack([total_vx, total_vy], dim=-1)
        staggered_phi = math.tensor(staggered, spatial('x', 'y'), math.channel('vector'))
        
        self._jet_velocity_field = StaggeredGrid(
            values=staggered_phi,
            boundary=velocity.extrapolation,
            bounds=velocity.bounds,
        )
    
    # ============ 射流 BC 施加 ============
    
    def apply_jet_bc(self, velocity: 'StaggeredGrid') -> 'StaggeredGrid':
        """
        将射流 BC 施加到速度场上
        
        在喷嘴作用区域替换速度值。
        此方法应在求解流程的多个环节调用。
        """
        if self._jet_velocity_field is None or self.num_nozzles == 0:
            return velocity
        
        if self._nozzle_cache is None:
            return velocity
        
        from phi.torch.flow import math, StaggeredGrid, spatial
        
        # 获取底层 tensor
        vel_st = velocity.staggered_tensor()
        jet_st = self._jet_velocity_field.staggered_tensor()
        
        vel_native = vel_st.native(vel_st.shape)
        jet_native = jet_st.native(jet_st.shape)
        
        # 使用缓存的 mask
        mask = self._nozzle_cache.total_mask.unsqueeze(-1).expand_as(vel_native).float()
        
        # 混合：喷嘴区域使用射流速度，其他区域保持原速度
        result_native = vel_native * (1 - mask) + jet_native * mask
        
        # 重建 StaggeredGrid
        result_phi = math.tensor(result_native, spatial('x', 'y'), math.channel('vector'))
        return StaggeredGrid(
            values=result_phi,
            boundary=velocity.extrapolation,
            bounds=velocity.bounds,
        )
    
    # ============ 辅助方法 ============
    
    def _get_device_from_field(self, velocity) -> str:
        """从速度场获取设备"""
        try:
            vel_x = velocity['x'].values
            native = vel_x.native(vel_x.shape)
            if isinstance(native, torch.Tensor):
                return native.device
        except:
            pass
        return self._device
    
    def reset(self):
        """重置状态（不重置几何缓存）"""
        self._current_controls = None
        self._raw_controls = None  # 重置原始控制信号
        self._prev_applied_amplitudes = None  # 重置后相当于初始施加 0 速度
        self._jet_velocity_field = None
        # 注意：不重置 _nozzle_cache，因为几何不变
        if self._policy is not None and hasattr(self._policy, 'reset'):
            self._policy.reset()
    
    def invalidate_cache(self):
        """强制使缓存失效（当几何发生变化时调用）"""
        self._nozzle_cache = None
    
    def get_nozzle_probe_locations(self) -> List[Dict[str, Tuple[float, float]]]:
        """
        获取每个喷嘴探针的采样位置（流体域网格 + 核心喷口网格）
        
        形状为 (Nx+1, Ny+1) 的 staggered_tensor 采用以下交错网格坐标：
            - staggered_tensor[i, j, 0] (vx) -> 物理位置 (i*dx, (j+0.5)*dy)
            - staggered_tensor[i, j, 1] (vy) -> 物理位置 ((i+0.5)*dx, j*dy)
            
            当通过 Point.at() 采样时，PhiFlow 会对 vx 和 vy 分别进行双线性插值。
            在对应速度分量的网格位置采样可取得该分量的网格值。
        
        策略：
            1. 找到所有 weight > 0 的网格点
            2. 对每个点计算 SDF，选择 SDF 最接近 0（最靠近边界）的点作为喷口
            3. 流体探针沿**射流方向**（而非几何法向）搜索第一个流体域网格
            4. 使用 vy 的物理位置映射：((i+0.5)*dx, j*dy)
        
        Returns:
            List of Dict: 每个喷嘴一个字典，包含：
            - 'fluid': (x, y)  流体域探针坐标（射流下游）
            - 'nozzle': (x, y) 核心喷口坐标（射流实际施加位置）
        """
        if self._nozzle_cache is None or self._velocity_template is None:
            return []
        
        bounds = self._velocity_template.bounds
        res_shape = self._velocity_template.resolution
        Nx, Ny = res_shape.get_size('x'), res_shape.get_size('y')
        Lx, Ly = float(bounds.size['x']), float(bounds.size['y'])
        dx, dy = Lx / Nx, Ly / Ny
        
        # 重新计算 SDF（用于判断流体域）
        device = self._get_device_from_field(self._velocity_template)
        x_coords = torch.linspace(0, Lx, Nx+1, device=device)
        y_coords = torch.linspace(0, Ly, Ny+1, device=device)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        sdf = self._compute_sdf(X, Y, device)
        
        # 获取当前射流控制方向（用于确定流体探针搜索方向）
        # 如果没有控制信号，默认使用 90° 向上
        jet_angle_deg = 90.0
        if self._current_controls and len(self._current_controls) > 0:
            # 取第一个喷嘴的角度作为参考（假设所有喷嘴同方向）
            _, angle = self._current_controls[0]
            if hasattr(angle, 'item'):
                jet_angle_deg = float(angle.item())
            elif hasattr(angle, 'cpu'):
                jet_angle_deg = float(angle.cpu().numpy())
            else:
                jet_angle_deg = float(angle)
        
        # 射流方向转换为网格步进 (di, dj)
        jet_rad = jet_angle_deg * (np.pi / 180.0)
        jet_dx, jet_dy = np.cos(jet_rad), np.sin(jet_rad)
        
        if abs(jet_dx) >= abs(jet_dy):
            jet_di = 1 if jet_dx > 0 else -1
            jet_dj = 0
        else:
            jet_di = 0
            jet_dj = 1 if jet_dy > 0 else -1
        
        locations = []
        for nozzle_idx, nozzle in enumerate(self.jet_config.nozzles):
            if nozzle_idx >= len(self._nozzle_cache.nozzle_weights):
                break
            
            # 使用 nozzle_weights 找到所有有权重的位置
            weight = self._nozzle_cache.nozzle_weights[nozzle_idx]
            
            if weight.max() < 1e-8:
                continue
            
            # 找到所有 weight > 0 的索引
            nonzero_indices = torch.nonzero(weight > 1e-8, as_tuple=False)  # (N, 2)
            
            if len(nonzero_indices) == 0:
                continue
            
            # 对每个点计算 SDF，找到最接近边界（SDF 接近 0）的点
            # 这个点就是射流出口位置
            best_idx = None
            best_sdf_abs = float('inf')
            
            for idx in nonzero_indices:
                i, j = int(idx[0]), int(idx[1])
                sdf_val = abs(float(sdf[i, j]))
                if sdf_val < best_sdf_abs:
                    best_sdf_abs = sdf_val
                    best_idx = (i, j)
            
            if best_idx is None:
                continue
            
            i_center, j_center = best_idx
            
            # 喷口物理坐标（使用 vy 分量的采样位置映射）
            # staggered_tensor[i, j, 1] (vy) -> ((i+0.5)*dx, j*dy)
            nozzle_x = (i_center + 0.5) * dx
            nozzle_y = j_center * dy
            
            # 沿射流方向搜索第一个流体网格点（SDF > 0）
            fluid_i, fluid_j = i_center, j_center
            found_fluid = False
            
            for step in range(1, max(Nx, Ny)):
                ni = i_center + step * jet_di
                nj = j_center + step * jet_dj
                
                # 边界检查
                if not (0 <= ni < Nx+1 and 0 <= nj < Ny+1):
                    break
                
                # 检查 SDF > 0（流体域）
                if sdf[ni, nj] > 0:
                    fluid_i, fluid_j = ni, nj
                    found_fluid = True
                    break
            
            if not found_fluid:
                # 备用：使用核心网格外一格（沿射流方向）
                fluid_i = min(max(i_center + jet_di, 0), Nx)
                fluid_j = min(max(j_center + jet_dj, 0), Ny)
            
            # 流体探针物理坐标（同样使用 vy 分量的采样位置映射）
            fluid_x = (fluid_i + 0.5) * dx
            fluid_y = fluid_j * dy
            
            locations.append({
                'fluid': (fluid_x, fluid_y),
                'nozzle': (nozzle_x, nozzle_y),
            })
        
        return locations
    
    # ============ 可视化辅助 ============
    
    def compute_sdf_numpy(self, velocity: 'StaggeredGrid') -> np.ndarray:
        """计算 SDF（用于可视化）"""
        device = self._get_device_from_field(velocity)
        bounds = velocity.bounds
        res_shape = velocity.resolution
        Nx, Ny = res_shape.get_size('x'), res_shape.get_size('y')
        Lx, Ly = float(bounds.size['x']), float(bounds.size['y'])
        
        x_coords = torch.linspace(0, Lx, Nx, device=device)
        y_coords = torch.linspace(0, Ly, Ny, device=device)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        sdf = self._compute_sdf(X, Y, device)
        return sdf.detach().cpu().numpy()


def create_jet_obstacle_from_config(cfg: Dict) -> JetObstacle:
    """从配置字典创建 JetObstacle"""
    from phi.torch.flow import Sphere, Box, vec
    
    geom_cfg = cfg.get('obstacle', cfg.get('geometry', {}))
    shape_type = geom_cfg.get('shape', 'sphere').lower()
    center = geom_cfg.get('center', [0.5, 0.5])
    domain_size = cfg.get('domain_size', cfg.get('domain', {}).get('size', [1, 1]))
    
    if all(0 <= c <= 1 for c in center):
        cx, cy = center[0] * domain_size[0], center[1] * domain_size[1]
    else:
        cx, cy = center
    
    if shape_type == 'sphere':
        radius = geom_cfg.get('radius', 10.0)
        geometry = Sphere(center=vec(x=cx, y=cy), radius=radius)
    else:
        size = geom_cfg.get('size', [10.0, 10.0])
        geometry = Box(x=(cx - size[0]/2, cx + size[0]/2), y=(cy - size[1]/2, cy + size[1]/2))
    
    jet_config = JetConfig.from_dict(cfg.get('jets', {}))
    return JetObstacle(geometry=geometry, jet_config=jet_config)
