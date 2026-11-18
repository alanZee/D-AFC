"""
phiflow_io.py - PhiFlow 现代化数据 I/O 工具模块

基于 PhiFlow 原生的 `phi.field.read()/write()` 和 `Scene` 机制。

主要功能：
1. 场数据读写（npz、h5 格式）
2. Scene 管理（兼容 PhiFlow 的数据管理范式）
3. 自动格式转换（CenteredGrid <-> StaggeredGrid）

参考实现: Solver-in-the-Loop/karman-2d-phi2/karman.py
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

from phi.torch.flow import (
    CenteredGrid, StaggeredGrid, Field,
    Box, Sphere,
    math, extrapolation, channel, spatial, batch,
    vec, field,
)

from myafc.utils import get_logger

LOGGER = get_logger()


# ============================================================
# 场数据读写
# ============================================================
def save_field_npz(
    fld: Field,
    path: Union[str, Path],
    name: str = 'field',
    compress: bool = True,
) -> Path:
    """保存单个场到 npz 文件
    
    Args:
        fld: PhiFlow Field 对象
        path: 保存目录
        name: 场名称（用于文件命名）
        compress: 是否压缩
    
    Returns:
        保存的文件路径
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    file_path = path / f'{name}.npz'
    
    # 使用 phi.field.write
    field.write(fld, file_path)
    
    LOGGER.debug(f"[IO] 保存场: {file_path}")
    return file_path


def load_field_npz(
    file_path: Union[str, Path],
    target_grid: Optional[Field] = None,
) -> Field:
    """从 npz 文件加载场
    
    Args:
        file_path: npz 文件路径
        target_grid: 目标网格（用于重采样）
    
    Returns:
        加载的 Field 对象
    """
    file_path = Path(file_path)
    
    # 使用 phi.field.read
    fld = field.read(file_path)
    
    # 如果指定了目标网格，重采样到该网格
    if target_grid is not None:
        fld = fld.at(target_grid)
    
    LOGGER.debug(f"[IO] 加载场: {file_path}")
    return fld


def save_fields_dict(
    fields_dict: Dict[str, Field],
    path: Union[str, Path],
    frame: int,
    compress: bool = True,
) -> Dict[str, Path]:
    """保存多个场到目录
    
    Args:
        fields_dict: {name: Field} 字典
        path: 保存目录
        frame: 帧编号
        compress: 是否压缩
    
    Returns:
        {name: file_path} 字典
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    for name, fld in fields_dict.items():
        file_path = path / f'{name}_{frame:06d}.npz'
        field.write(fld, file_path)
        saved_paths[name] = file_path
    
    return saved_paths


def load_fields_dict(
    path: Union[str, Path],
    frame: int,
    field_names: list,
    target_grids: Optional[Dict[str, Field]] = None,
) -> Dict[str, Field]:
    """从目录加载多个场
    
    Args:
        path: 目录路径
        frame: 帧编号
        field_names: 要加载的场名称列表
        target_grids: {name: target_grid} 字典（用于重采样）
    
    Returns:
        {name: Field} 字典
    """
    path = Path(path)
    
    loaded = {}
    for name in field_names:
        file_path = path / f'{name}_{frame:06d}.npz'
        if not file_path.exists():
            LOGGER.warning(f"[IO] 文件不存在: {file_path}")
            continue
        
        fld = field.read(file_path)
        
        # 重采样
        if target_grids and name in target_grids:
            fld = fld.at(target_grids[name])
        
        loaded[name] = fld
    
    return loaded


# ============================================================
# H5 格式支持
# ============================================================
def save_fields_h5(
    fields_dict: Dict[str, Field],
    h5_path: Union[str, Path],
    step: int,
    time: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
    append: bool = True,
) -> None:
    """保存场数据到 HDF5 文件
    
    Args:
        fields_dict: {name: Field} 字典
        h5_path: HDF5 文件路径
        step: 时间步
        time: 物理时间
        metadata: 元数据（domain_size, resolution 等）
        append: 是否追加模式
    """
    import h5py
    
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'a' if (append and h5_path.exists()) else 'w'
    
    with h5py.File(h5_path, mode) as f:
        # 写入元数据（仅新建文件时）
        if mode == 'w' and metadata:
            for key, val in metadata.items():
                f.attrs[key] = val
        
        # 创建时间步组
        grp_name = f'step_{step}'
        if grp_name in f:
            del f[grp_name]  # 覆盖已有数据
        
        grp = f.create_group(grp_name)
        grp.attrs['time'] = time
        grp.attrs['step'] = step
        
        # 保存各场
        for name, fld in fields_dict.items():
            # 转换为 numpy
            # 使用 is_staggered 属性区分 StaggeredGrid 和 CenteredGrid
            if hasattr(fld, 'is_staggered') and fld.is_staggered:
                # 对于 StaggeredGrid，需要创建一个参考 CenteredGrid 来插值
                # 获取边界和分辨率信息
                fld_bounds = fld.bounds
                resolution = fld.resolution
                Nx = resolution.get_size('x')
                Ny = resolution.get_size('y')
                
                # 创建参考 CenteredGrid
                ref_grid = CenteredGrid(
                    values=0.0,
                    boundary=extrapolation.ZERO,
                    bounds=fld_bounds,
                    x=Nx, y=Ny
                )
                # 插值到中心网格
                fld_centered = fld @ ref_grid
                data = fld_centered.values.numpy('x,y,vector')
            elif hasattr(fld, 'values'):
                # CenteredGrid 或其他 SampledField
                if 'vector' in fld.values.shape:
                    data = fld.values.numpy('x,y,vector')
                else:
                    data = fld.values.numpy('x,y')
            else:
                LOGGER.warning(f"[IO] 不支持的场类型: {type(fld)}")
                continue
            
            grp.create_dataset(name, data=data, compression='gzip')
    
    LOGGER.debug(f"[IO] 保存 H5: {h5_path} | step={step}")


def get_h5_steps(h5_path: Union[str, Path]) -> list[int]:
    """获取 HDF5 文件中存在的所有时间步
    
    Args:
        h5_path: HDF5 文件路径
        
    Returns:
        List[int]: 升序排列的时间步列表
    """
    import h5py
    from pathlib import Path
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 文件不存在: {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        steps = sorted([int(k.split('_')[1]) for k in f.keys() if k.startswith('step_')])
        return steps


def load_fields_h5(
    h5_path: Union[str, Path],
    step: int = -1,
    bounds: Optional[Box] = None,
    velocity_boundary: Any = None,
    pressure_boundary: Any = None,
) -> Tuple[Dict[str, Field], Dict[str, Any]]:
    """从 HDF5 文件加载场数据
    
    Args:
        h5_path: HDF5 文件路径
        step: 时间步（-1 表示最后一步）
        bounds: 计算域边界
        velocity_boundary: 速度场边界条件
        pressure_boundary: 压力场边界条件
    
    Returns:
        (fields_dict, info_dict):
            fields_dict: {name: Field} 字典
            info_dict: {'step': int, 'time': float, ...}
    """
    import h5py
    
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 文件不存在: {h5_path}")
    
    fields = {}
    info = {'h5_path': str(h5_path)}
    
    with h5py.File(h5_path, 'r') as f:
        # 获取时间步列表
        steps = sorted([int(k.split('_')[1]) for k in f.keys() if k.startswith('step_')])
        if not steps:
            raise ValueError(f"H5 文件中无有效数据: {h5_path}")
        
        # 处理帧索引
        # 负索引: 从末尾倒数，如 -1 表示最后一帧
        # 正索引: 直接作为 step 值
        if step < 0:
            # 负索引映射到实际帧
            list_idx = step  # Python 列表支持负索引
            if abs(step) > len(steps):
                # 索引越界，使用最后一帧
                LOGGER.warning(f"[phiflow_io] 负索引 {step} 超出范围 (共 {len(steps)} 帧)，使用最后一帧")
                target_step = steps[-1]
            else:
                target_step = steps[list_idx]
        else:
            target_step = step
            
        if target_step not in steps:
            available = steps[:5]
            raise ValueError(f"时间步 {target_step} 不在 H5 文件中，可用: {available}...")
        
        grp = f[f'step_{target_step}']
        
        info['step'] = target_step
        info['time'] = grp.attrs.get('time', 0.0)
        
        # 获取元数据
        if 'domain_size' in f.attrs:
            info['domain_size'] = list(f.attrs['domain_size'])
        if 'resolution' in f.attrs:
            info['resolution'] = list(f.attrs['resolution'])
        
        # 加载各场
        for name in grp.keys():
            data = grp[name][:]
            
            if name == 'velocity':
                # velocity 数据是 CenteredGrid 格式 (Nx, Ny, 2)
                # 转换为 StaggeredGrid
                if bounds is None:
                    raise ValueError("加载速度场需要提供 bounds")
                
                vel_centered = CenteredGrid(
                    values=math.tensor(data, spatial('x', 'y'), channel(vector='x,y')),
                    boundary=velocity_boundary or extrapolation.ZERO,
                    bounds=bounds,
                )
                
                Nx, Ny = data.shape[:2]
                fields['velocity'] = StaggeredGrid(
                    values=vel_centered,
                    boundary=velocity_boundary or extrapolation.ZERO,
                    bounds=bounds,
                    x=Nx, y=Ny
                )
                
            elif name == 'pressure':
                # pressure 是 CenteredGrid 格式 (Nx, Ny)
                if bounds is None:
                    raise ValueError("加载压力场需要提供 bounds")
                
                fields['pressure'] = CenteredGrid(
                    values=math.tensor(data, spatial('x', 'y')),
                    boundary=pressure_boundary or extrapolation.ZERO,
                    bounds=bounds,
                )
            
            else:
                # 通用场
                if len(data.shape) == 2:
                    fields[name] = CenteredGrid(
                        values=math.tensor(data, spatial('x', 'y')),
                        boundary=extrapolation.ZERO,
                        bounds=bounds,
                    )
                elif len(data.shape) == 3:
                    fields[name] = CenteredGrid(
                        values=math.tensor(data, spatial('x', 'y'), channel(vector='x,y')),
                        boundary=extrapolation.ZERO,
                        bounds=bounds,
                    )
    
    LOGGER.debug(f"[IO] 加载 H5: {h5_path} | step={target_step}")
    return fields, info


# ============================================================
# Scene 管理（兼容 PhiFlow 范式）
# ============================================================
class FieldScene:
    """场数据场景管理器
    
    兼容 PhiFlow 的 Scene 数据管理范式，但使用更灵活的实现。
    
    使用示例:
        scene = FieldScene(output_dir='runs/test')
        scene.write({'velocity': vel, 'pressure': p}, frame=0)
        vel, p = scene.read(['velocity', 'pressure'], frame=0)
    """
    
    def __init__(self, output_dir: Union[str, Path], format: str = 'npz'):
        """
        Args:
            output_dir: 输出目录
            format: 数据格式 ('npz' 或 'h5')
        """
        self.output_dir = Path(output_dir)
        self.format = format
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # H5 文件句柄（延迟初始化）
        self._h5_file = None
        self._h5_path = self.output_dir / 'fields.h5'
        
        # 元数据
        self.metadata = {}
    
    def write(
        self,
        data: Dict[str, Field],
        frame: int,
        time: Optional[float] = None,
    ) -> None:
        """写入场数据
        
        Args:
            data: {name: Field} 字典
            frame: 帧编号
            time: 物理时间（可选）
        """
        if self.format == 'npz':
            save_fields_dict(data, self.output_dir, frame)
        else:  # h5
            save_fields_h5(
                data, self._h5_path, frame,
                time=time or frame,
                metadata=self.metadata,
                append=True,
            )
    
    def read(
        self,
        field_names: list,
        frame: int,
        target_grids: Optional[Dict[str, Field]] = None,
    ) -> Dict[str, Field]:
        """读取场数据
        
        Args:
            field_names: 要读取的场名称列表
            frame: 帧编号
            target_grids: 目标网格字典（用于重采样）
        
        Returns:
            {name: Field} 字典
        """
        if self.format == 'npz':
            return load_fields_dict(self.output_dir, frame, field_names, target_grids)
        else:  # h5
            fields, _ = load_fields_h5(self._h5_path, frame)
            # 过滤并重采样
            result = {}
            for name in field_names:
                if name in fields:
                    fld = fields[name]
                    if target_grids and name in target_grids:
                        fld = fld.at(target_grids[name])
                    result[name] = fld
            return result
    
    def set_metadata(self, **kwargs):
        """设置元数据"""
        self.metadata.update(kwargs)
    
    def list_frames(self) -> list:
        """列出所有可用帧"""
        if self.format == 'npz':
            files = list(self.output_dir.glob('*_*.npz'))
            frames = set()
            for f in files:
                parts = f.stem.split('_')
                if len(parts) >= 2 and parts[-1].isdigit():
                    frames.add(int(parts[-1]))
            return sorted(frames)
        else:  # h5
            import h5py
            if not self._h5_path.exists():
                return []
            with h5py.File(self._h5_path, 'r') as f:
                steps = [int(k.split('_')[1]) for k in f.keys() if k.startswith('step_')]
            return sorted(steps)
    
    def close(self):
        """关闭资源"""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


# ============================================================
# 便捷函数
# ============================================================
def velocity_to_centered(velocity: StaggeredGrid, reference_grid: CenteredGrid) -> CenteredGrid:
    """将 StaggeredGrid 速度插值到 CenteredGrid
    
    Args:
        velocity: StaggeredGrid 速度场
        reference_grid: 参考的 CenteredGrid（如 pressure）
    
    Returns:
        CenteredGrid 速度场
    """
    return velocity @ reference_grid


def centered_to_staggered(
    velocity_centered: CenteredGrid,
    bounds: Box,
    boundary: Any = None,
    resolution: Optional[Tuple[int, int]] = None,
) -> StaggeredGrid:
    """将 CenteredGrid 速度转换为 StaggeredGrid
    
    Args:
        velocity_centered: CenteredGrid 速度场
        bounds: 计算域边界
        boundary: 边界条件
        resolution: 分辨率 (Nx, Ny)，如果不指定则从 velocity_centered 推断
    
    Returns:
        StaggeredGrid 速度场
    """
    if resolution is None:
        shape = velocity_centered.values.shape
        Nx = shape.get_size('x')
        Ny = shape.get_size('y')
    else:
        Nx, Ny = resolution
    
    return StaggeredGrid(
        values=velocity_centered,
        boundary=boundary or extrapolation.ZERO,
        bounds=bounds,
        x=Nx, y=Ny
    )
