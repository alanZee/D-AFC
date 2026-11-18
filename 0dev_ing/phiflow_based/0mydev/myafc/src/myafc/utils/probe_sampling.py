# -*- coding: utf-8 -*-
"""
probe_sampling — 统一的探针采样工具函数

所有 Trainer / Validator 的 _build_state 共用此函数，确保:
- probe_fields 过滤逻辑唯一且一致
- 采样 API 统一 (phi.torch.flow.field.sample)
- 浮点转换统一

调用方 (所有 _build_state 实现):
- JetTrainer._build_state
- AFCValidator._build_state
- SimpleTrainer._build_state
- PhiFlowEnv._build_state_for_history
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any


def sample_probe_fields(
    velocity_field,
    pressure_field,
    point,
    probe_fields: List[str],
    noise_std: float = 0.0,
) -> Dict[str, float]:
    """在指定点按 probe_fields 精确采样物理量

    所有组件 _build_state() 的唯一底层采样函数。

    Args:
        velocity_field: solver.velocity (PhiFlow 速度场, StaggeredGrid)
        pressure_field: solver.pressure (PhiFlow 压力场, CenteredGrid)
        point: 采样点 (PhiFlow Point/Geometry)
        probe_fields: 需采样的字段列表, ['u','v','p'] 的任意子集
        noise_std: 观测噪声标准差 (>0 时注入高斯噪声, 默认 0=无噪声)

    Returns:
        仅包含 probe_fields 中请求的字段, 如 {'u': 1.5, 'v': -0.2}
    """
    from phi.torch.flow import field

    need_velocity = ('u' in probe_fields or 'v' in probe_fields)
    need_pressure = 'p' in probe_fields
    data: Dict[str, float] = {}

    if need_velocity:
        vel = field.sample(velocity_field, point)
        u = vel.vector['x'].native().squeeze()
        v = vel.vector['y'].native().squeeze()
        if noise_std > 0: # 加噪
            import torch
            u = u + torch.randn_like(u) * noise_std
            v = v + torch.randn_like(v) * noise_std
        if 'u' in probe_fields:
            data['u'] = float(u)
        if 'v' in probe_fields:
            data['v'] = float(v)

    if need_pressure:
        p = field.sample(pressure_field, point).native().squeeze()
        if noise_std > 0: # 加噪
            import torch
            p = p + torch.randn_like(p) * noise_std
        data['p'] = float(p)

    return data


def build_probe_batch(named_points: Dict[str, Any]) -> Tuple[List[str], Any, List[Tuple[float, float]]]:
    """将具名探针点打包为可批量采样的 Point 几何体

    Args:
        named_points: {probe_name: Point} 或 {probe_name: point_like}

    Returns:
        (names, batch_point_geom, coords_xy)
        - names: 探针名称列表（与采样结果索引一一对应）
        - batch_point_geom: Point( instance('points') )，可一次性 field.sample
        - coords_xy: [(x, y), ...]，用于日志/CSV
    """
    from phi.torch.flow import math, instance, channel
    from phi.geom import Point

    names: List[str] = []
    coords: List[Tuple[float, float]] = []

    for name, point in named_points.items():
        center = point.center if hasattr(point, 'center') else point
        x = float(center.vector['x'].native().squeeze())
        y = float(center.vector['y'].native().squeeze())
        names.append(name)
        coords.append((x, y))

    points_tensor = math.wrap(coords, instance('points'), channel(vector='x,y'))
    batch_geom = Point(points_tensor)
    return names, batch_geom, coords


def sample_probe_fields_batch(
    velocity_field,
    pressure_field,
    probe_names: List[str],
    probe_geometry,
    probe_fields: List[str],
    noise_std: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """对多个探针点一次性采样并按 probe_fields 输出

    Args:
        velocity_field: solver.velocity
        pressure_field: solver.pressure
        probe_names: 探针名称列表，顺序需与 probe_geometry 对应
        probe_geometry: build_probe_batch 返回的 batch Point 几何
        probe_fields: ['u','v','p'] 的任意子集
        noise_std: 观测噪声标准差

    Returns:
        {name: {'u':..., 'v':..., 'p':...}, ...}
    """
    from phi.torch.flow import field

    need_velocity = ('u' in probe_fields or 'v' in probe_fields)
    need_pressure = 'p' in probe_fields

    n = len(probe_names)
    out: Dict[str, Dict[str, float]] = {name: {} for name in probe_names}

    if n == 0:
        return out

    u_all = v_all = p_all = None

    if need_velocity:
        vel = field.sample(velocity_field, probe_geometry)
        u_all = vel.vector['x'].native().reshape(-1)
        v_all = vel.vector['y'].native().reshape(-1)
        if noise_std > 0:
            import torch
            u_all = u_all + torch.randn_like(u_all) * noise_std
            v_all = v_all + torch.randn_like(v_all) * noise_std

    if need_pressure:
        p_all = field.sample(pressure_field, probe_geometry).native().reshape(-1)
        if noise_std > 0:
            import torch
            p_all = p_all + torch.randn_like(p_all) * noise_std

    for i, name in enumerate(probe_names):
        if 'u' in probe_fields:
            out[name]['u'] = float(u_all[i])
        if 'v' in probe_fields:
            out[name]['v'] = float(v_all[i])
        if 'p' in probe_fields:
            out[name]['p'] = float(p_all[i])

    return out
