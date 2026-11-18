"""
plot_utils - 曲线数据保存与绘图工具

为 JetTrainer、AFCValidator 和 DRLTrainer 提供统一的曲线绘图功能：
1. 验证曲线保存 (NPZ + CSV)
2. 标准 3×2 验证曲线绘图
3. 力系数/幅值独立绘图

使用示例:
    from myafc.utils import save_validation_curves, plot_validation_curves_3x2
    
    # 保存验证曲线数据
    data = {
        'time': np.array([...]),
        'amplitude': np.array([...]),
        'Cd': np.array([...]),
        'Cl': np.array([...]),
        'velocity_u': np.array([...]),
        'velocity_v': np.array([...]),
    }
    save_validation_curves(output_dir, data, metadata={'dt': 0.0005})
    
    # 绘制曲线
    plot_validation_curves_3x2(data, target_velocity=[0.0, -0.233], output_path='curves.png')
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

# 延迟导入 matplotlib 以避免启动时警告
_mpl_imported = False
def _ensure_matplotlib():
    global _mpl_imported
    if not _mpl_imported:
        import matplotlib
        matplotlib.use('Agg')
        _mpl_imported = True


def save_validation_curves(
    output_dir: Union[str, Path],
    data: Dict[str, np.ndarray],
    metadata: Optional[Dict] = None,
    prefix: str = 'validation',
) -> Tuple[Path, Path]:
    """保存验证曲线数据
    
    同时保存为 NPZ (含元数据) 和 CSV (便于查看) 格式。
    
    Args:
        output_dir: 输出目录
        data: 曲线数据字典，需包含以下键：
            - time: 时间序列
            - amplitude: 控制幅值
            - Cd: 阻力系数
            - Cl: 升力系数
            - velocity_u: 目标点 u 速度 (可选)
            - velocity_v: 目标点 v 速度 (可选)
        metadata: 元数据字典 (仅存于 NPZ)，如 {'dt': 0.0005, 'num_steps': 1000}
        prefix: 文件名前缀
    
    Returns:
        (npz_path, csv_path)
    """
    import pandas as pd
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 NPZ (含元数据)
    npz_path = output_dir / f'{prefix}_curves.npz'
    save_dict = {**data}
    if metadata:
        save_dict['metadata'] = np.array([metadata], dtype=object)
    np.savez(npz_path, **save_dict)
    
    # 保存 CSV (便于查看)
    csv_path = output_dir / f'{prefix}_curves.csv'
    # 只保存可以转为列的数据
    csv_data = {k: v for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 1}
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
    
    return npz_path, csv_path


def plot_validation_curves_3x2(
    data: Dict[str, np.ndarray],
    target_velocity: Optional[List[float]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = 'Validation Curves',
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """绘制标准 3×2 验证曲线图
    
    布局:
        [Amplitude]     [Cd]
        [Velocity U]    [Cl]
        [Velocity V]    [Combined Force]
    
    Args:
        data: 曲线数据字典
        target_velocity: 目标速度 [u, v]，用于绘制目标线
        output_path: 图像保存路径 (None = 不保存)
        title: 图像标题
        figsize: 图像尺寸
        dpi: 图像分辨率
        show: 是否显示图像
    
    Returns:
        matplotlib Figure 对象
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    
    time = data.get('time', np.arange(len(data.get('amplitude', []))))
    
    # Row 1: Amplitude, Cd
    ax = axes[0, 0]
    if 'amplitude' in data:
        ax.plot(time, data['amplitude'], 'b-', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Control Amplitude')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    if 'Cd' in data:
        ax.plot(time, data['Cd'], 'r-', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cd')
    ax.set_title('Drag Coefficient')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Velocity U, Cl
    ax = axes[1, 0]
    if 'velocity_u' in data:
        ax.plot(time, data['velocity_u'], 'g-', linewidth=1, label='u')
    if target_velocity:
        ax.axhline(y=target_velocity[0], color='k', linestyle='--', alpha=0.5, label=f'target={target_velocity[0]:.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Velocity U')
    ax.set_title('Target Probe Velocity U')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    if 'Cl' in data:
        ax.plot(time, data['Cl'], 'b-', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cl')
    ax.set_title('Lift Coefficient')
    ax.grid(True, alpha=0.3)
    
    # Row 3: Velocity V, Combined Force
    ax = axes[2, 0]
    if 'velocity_v' in data:
        ax.plot(time, data['velocity_v'], 'm-', linewidth=1, label='v')
    if target_velocity:
        ax.axhline(y=target_velocity[1], color='k', linestyle='--', alpha=0.5, label=f'target={target_velocity[1]:.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Velocity V')
    ax.set_title('Target Probe Velocity V')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    if 'Cd' in data and 'Cl' in data:
        ax.plot(time, data['Cd'], 'r-', linewidth=1, label='Cd')
        ax.plot(time, data['Cl'], 'b-', linewidth=1, label='Cl')
        ax.legend(loc='upper right')
    ax.set_xlabel('Time')
    ax.set_ylabel('Force Coeff.')
    ax.set_title('Combined Force Coefficients')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_validation_curves_3panels(
    data: Dict[str, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: str = 'Validation Curves',
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """绘制简化的 3 面板验证曲线图 (DRLTrainer 风格)
    
    布局:
        [Amplitude]
        [Cd]
        [Cl]
    
    Args:
        data: 曲线数据字典
        output_path: 图像保存路径
        title: 图像标题
        figsize: 图像尺寸
        dpi: 图像分辨率
        show: 是否显示
    
    Returns:
        matplotlib Figure 对象
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14)
    
    time = data.get('time', np.arange(len(data.get('amplitude', []))))
    
    # Panel 1: Amplitude
    ax = axes[0]
    if 'amplitude' in data:
        ax.plot(time, data['amplitude'], 'b-', linewidth=1)
    ax.set_ylabel('Amplitude')
    ax.set_title('Control Amplitude')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Cd
    ax = axes[1]
    if 'Cd' in data:
        ax.plot(time, data['Cd'], 'r-', linewidth=1)
    ax.set_ylabel('Cd')
    ax.set_title('Drag Coefficient')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Cl
    ax = axes[2]
    if 'Cl' in data:
        ax.plot(time, data['Cl'], 'g-', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cl')
    ax.set_title('Lift Coefficient')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_force_coeffs(
    cd_list: np.ndarray,
    cl_list: np.ndarray,
    time: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = 'Force Coefficients',
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """绘制力系数曲线
    
    Args:
        cd_list: 阻力系数序列
        cl_list: 升力系数序列
        time: 时间序列 (可选)
        output_path: 保存路径
        title: 标题
        figsize: 尺寸
        dpi: 分辨率
        show: 是否显示
    
    Returns:
        Figure 对象
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14)
    
    x = time if time is not None else np.arange(len(cd_list))
    
    axes[0].plot(x, cd_list, 'r-', linewidth=1)
    axes[0].set_ylabel('Cd')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, cl_list, 'b-', linewidth=1)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Cl')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_amplitude_history(
    amplitudes: np.ndarray,
    time: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = 'Control Amplitude History',
    figsize: Tuple[int, int] = (10, 4),
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """绘制控制幅值历史
    
    Args:
        amplitudes: 幅值序列
        time: 时间序列 (可选)
        output_path: 保存路径
        title: 标题
        figsize: 尺寸
        dpi: 分辨率
        show: 是否显示
    
    Returns:
        Figure 对象
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = time if time is not None else np.arange(len(amplitudes))
    ax.plot(x, amplitudes, 'b-', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def generate_video_from_frames(
    frames_dir: Union[str, Path],
    output_path: Union[str, Path],
    pattern: str = 'step*.png',
    fps: int = 10,
    codec: str = 'libx264',
    pix_fmt: str = 'yuv420p',
) -> Optional[Path]:
    """从帧序列生成视频
    
    使用 ffmpeg concat 模式生成 mp4 视频。
    
    Args:
        frames_dir: 帧图片目录
        output_path: 输出视频路径
        pattern: 帧文件 glob 模式
        fps: 帧率
        codec: 视频编码器
        pix_fmt: 像素格式
    
    Returns:
        输出视频路径，失败返回 None
    """
    import subprocess
    from glob import glob
    
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    
    # 获取帧文件列表
    frames = sorted(glob(str(frames_dir / pattern)))
    if not frames:
        return None
    
    # 创建临时文件列表
    filelist_path = frames_dir / 'filelist.txt'
    with open(filelist_path, 'w', encoding='utf-8') as f:
        for frame in frames:
            # 使用相对路径避免中文路径问题
            rel_path = Path(frame).name
            f.write(f"file '{rel_path}'\n")
            f.write(f"duration {1.0/fps}\n")
    
    # 调用 ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(filelist_path),
        '-c:v', codec,
        '-pix_fmt', pix_fmt,
        '-r', str(fps),
        str(output_path),
    ]
    
    try:
        subprocess.run(cmd, cwd=str(frames_dir), check=True, 
                      capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError:
        return None
    finally:
        # 清理临时文件
        if filelist_path.exists():
            filelist_path.unlink()
