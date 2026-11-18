"""
SolverRunner - 轻量级求解器运行包装类

提供以下功能:
- tqdm 进度条
- 力系数 (Cd/Cl) 计算与缓存/保存
- 探针数据采集与缓存/保存
- 射流探针自动放置与采集
- 流场数据缓存与 h5 保存（使用 PhiFlow 原生 I/O）
- 可视化图像保存
"""
from __future__ import annotations

# 设置 matplotlib 后端为 Agg (非交互式)，避免多线程 tkinter 问题
import matplotlib
matplotlib.use('Agg')

import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field as dataclass_field, asdict
from pathlib import Path

import yaml
import numpy as np
from datetime import datetime

# 使用 myafc.utils.logging 统一日志模块
from myafc.utils import get_logger, setup_file_logging, set_log_level
# 使用现代化的 PhiFlow I/O 工具
from myafc.utils.phiflow_io import save_fields_h5, load_fields_h5, FieldScene

LOGGER = get_logger()


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class OutputConfig:
    enabled: bool = True
    root: str = './outputs'
    subdir: str = 'run'
    skip_log_setup: bool = False  # 如果为 True，则不配置日志文件（由外部管理）
    skip_config_snapshot: bool = False  # 如果为 True，则不保存配置快照（由外部管理）


@dataclass
class FieldsSaveConfig:
    """场数据 H5 保存配置
    
    Attributes:
        enabled: 是否启用场数据保存
        interval: 保存间隔（每 N 步保存一次）
        filename: H5 文件名
        quantities: 要保存的场类型列表
            - 'velocity': 速度场（必选，始终保存）
            - 'pressure': 压力场（必选，始终保存）
            - 'vorticity': 涡量场（2D 为标量场）
            - 'velocity_magnitude': 速度大小场
            - 'sdf': Signed Distance Field（如果存在）
            - 'mask': 障碍物掩码（如果存在）
    """
    enabled: bool = True
    interval: int = 10
    filename: str = 'fields.h5'
    quantities: List[str] = dataclass_field(default_factory=lambda: ['velocity', 'pressure'])


@dataclass
class FieldsCacheConfig:
    enabled: bool = False
    interval: int = 10


@dataclass
class FieldsConfig:
    save: FieldsSaveConfig = dataclass_field(default_factory=FieldsSaveConfig)
    cache: FieldsCacheConfig = dataclass_field(default_factory=FieldsCacheConfig)


@dataclass
class ForceCoeffsConfig:
    enabled: bool = False
    interval: int = 1
    filename: str = 'force_coeffs.csv'
    reference_velocity: Optional[float] = None
    reference_length: Optional[float] = None
    include_torque: bool = False  # 是否计算并记录力矩


@dataclass
class JetProbesConfig:
    """射流探针配置
    
    自动在每个喷嘴核心区域中心放置探针，记录射流的实际速度和压强。
    记录保存在独立的 CSV 文件中，与常规探针分开。
    """
    enabled: bool = False              # true/false - 是否启用射流探针
    interval: int = 1                  # 采样间隔
    filename: str = 'jet_probes.csv'   # 保存文件名


@dataclass 
class ProbesConfig:
    enabled: bool = False
    interval: int = 1
    filename: str = 'probes.csv'
    locations: List[Dict] = dataclass_field(default_factory=list)


@dataclass
class VisualsConfig:
    enabled: bool = False
    interval: int = 10
    quantities: List[str] = dataclass_field(default_factory=lambda: ['velocity_magnitude', 'pressure'])
    save_endpoints: bool = True
    folder: str = 'frames'
    smooth: bool = True                          # 是否平滑插值（热力图）
    video_fps: int = 10                          # 合成视频帧率
    # 障碍物可视化选项
    obstacle_style: str = 'fill'  # 'outline'(仅轮廓) / 'fill'(填充) / 'both'(轮廓+填充) / 'none'(不显示)
    obstacle_fill_value: Optional[float] = None  # 填充值，None=使用 NaN


@dataclass
class DomainOverviewConfig:
    """域概览图配置"""
    enabled: bool = True
    filename: str = 'domain_overview.png'
    show_grid: bool = False             # 是否显示网格线
    show_nozzles: bool = True           # 是否显示射流喷嘴位置（标记点）
    show_nozzle_labels: bool = True     # 是否显示喷嘴角度文字标签
    show_probes: bool = True            # 是否显示探针位置（标记点）
    show_probes_labels: bool = True     # 是否显示探针名称文字标签
    probe_size: float = 6.0             # 探针标记大小
    nozzle_edge_only: bool = True       # 是否仅绘制喷口在结构物边缘的那一层网格（True=只显示表面网格，False=显示全部延伸区域）


@dataclass
class RunnerConfig:
    output: OutputConfig = dataclass_field(default_factory=OutputConfig)
    fields: FieldsConfig = dataclass_field(default_factory=FieldsConfig)
    force_coeffs: ForceCoeffsConfig = dataclass_field(default_factory=ForceCoeffsConfig)
    jet_probes: JetProbesConfig = dataclass_field(default_factory=JetProbesConfig)
    probes: ProbesConfig = dataclass_field(default_factory=ProbesConfig)
    visuals: VisualsConfig = dataclass_field(default_factory=VisualsConfig)
    domain_overview: DomainOverviewConfig = dataclass_field(default_factory=DomainOverviewConfig)
    # 日志详细级别: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    verbose: str = 'INFO'


def parse_runner_config(cfg_dict: Dict) -> RunnerConfig:
    """解析 runner 配置"""
    config = RunnerConfig()
    
    if 'output' in cfg_dict:
        config.output = OutputConfig(**cfg_dict['output'])
    
    if 'fields' in cfg_dict:
        fc = cfg_dict['fields']
        if 'save' in fc:
            config.fields.save = FieldsSaveConfig(**fc['save'])
        if 'cache' in fc:
            config.fields.cache = FieldsCacheConfig(**fc['cache'])
    
    if 'force_coeffs' in cfg_dict:
        config.force_coeffs = ForceCoeffsConfig(**cfg_dict['force_coeffs'])
    if 'jet_probes' in cfg_dict:
        config.jet_probes = JetProbesConfig(**cfg_dict['jet_probes'])
    if 'probes' in cfg_dict:
        config.probes = ProbesConfig(**cfg_dict['probes'])
    if 'visuals' in cfg_dict:
        config.visuals = VisualsConfig(**cfg_dict['visuals'])
    if 'domain_overview' in cfg_dict:
        config.domain_overview = DomainOverviewConfig(**cfg_dict['domain_overview'])
    if 'verbose' in cfg_dict:
        config.verbose = str(cfg_dict['verbose'])
    
    return config


# ============================================================
# SolverRunner
# ============================================================
class SolverRunner:
    """
    轻量级求解器运行包装类
    
    用法:
        solver = KarmanTorchSolver('config.yaml')
        runner = SolverRunner(solver, 'config.yaml')
        runner.run(100)  # 运行 100 步
        
        # 访问缓存
        print(runner.force_coeffs, runner.probes, runner.fields)
    """
    
    def __init__(self, solver, config):
        self.solver = solver
        
        # 解析配置
        if isinstance(config, str):
            with open(config, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
            cfg_dict = raw.get('runner', {})
            self._config_path = config
        else:
            # 如果是字典，检查是否包含 'runner' 键
            cfg_dict = config.get('runner', config) if isinstance(config, dict) else config
            self._config_path = None
        
        self.cfg = parse_runner_config(cfg_dict)
        
        # 缓存
        self.force_coeffs: List[Dict] = []
        self.probes: Dict[str, List] = {}
        self.probes_jet: List[Dict] = []  # 射流探针数据缓存
        self.fields: List[Dict] = []
        
        # 射流探针位置缓存（延迟初始化，在第一次记录时设置）
        self._jet_probe_locations: Optional[List[Tuple[float, float]]] = None
        
        # 常规探针批量采样缓存（减少每步 Point 构造与逐点 sample 开销）
        self._probe_batch_signature = None
        self._probe_batch_names: List[str] = []
        self._probe_batch_coords: List[Tuple[float, float]] = []
        self._probe_batch_geom = None
        
        # resume 状态
        self._resume_info = getattr(solver, 'resume_info', None)
        self._step_offset = 0  # 时间步偏移量
        self._time_offset = 0.0  # 物理时间偏移量
        self._frame_offset = 0  # 图片编号偏移量
        
        # 输出目录（根据 resume 模式决定）
        self._setup_output_dir()
        
        self._h5_file = None
        # 保存当前 runner 配置快照（仅在输出目录启用时）
        try:
            self._save_config_snapshot()
        except Exception as e:
            LOGGER.warning(f"[Runner] Failed to save runner config snapshot: {e}")
        
        # 设置日志级别
        # 注意: Runner 的 verbose 会覆盖 Solver 的 verbose 配置，
        # 确保日志级别由上层组件统一控制
        set_log_level(self.cfg.verbose)
        # 同步更新 solver 的 verbose 配置（如果 solver 支持）
        if hasattr(self.solver, 'cfg') and hasattr(self.solver.cfg, 'runtime'):
            self.solver.cfg.runtime.verbose = self.cfg.verbose
    
    def reset_caches(self, reset_solver: bool = True):
        """重置所有数据缓存（用于重复使用 runner 实例）
        
        清空 force_coeffs、probes、fields 等缓存，关闭并重新初始化 h5 文件句柄。
        调用此方法后，runner 可以重新开始记录数据。
        
        Args:
            reset_solver: 是否重置求解器状态，默认 True。
                          当上层已通过 reset() 等方法初始化流场时，设为 False。
        """
        # 重置求解器状态（可选）
        if reset_solver:
            self.solver.reset()

        # 关闭现有 h5 文件句柄
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None
        
        # 清空数据缓存
        self.force_coeffs.clear()
        self.probes.clear()
        self.probes_jet.clear()
        self.fields.clear()
        
        # 重置偏移量
        self._step_offset = 0
        self._time_offset = 0.0
        self._frame_offset = 0
        
        LOGGER.debug("[Runner] 数据缓存已重置")
    
    def _setup_output_dir(self):
        """设置输出目录（根据 resume 模式决定）"""
        if not self.cfg.output.enabled:
            self.output_dir = None
            return
        
        resume_info = self._resume_info
        
        if resume_info and resume_info.get('strict', False):
            # 严格续算模式：使用场文件所属目录
            h5_path = Path(resume_info['h5_path'])
            # h5 文件通常在 raw/ 子目录下，输出目录是其父目录的父目录
            if h5_path.parent.name == 'raw':
                self.output_dir = h5_path.parent.parent
            else:
                self.output_dir = h5_path.parent
            
            # 设置偏移量
            self._step_offset = resume_info['step']
            self._time_offset = resume_info['time']
            
            # 计算图片编号偏移（基于现有文件）
            self._frame_offset = self._compute_frame_offset()
            
            # 清理指定时间步之后的数据
            self._cleanup_after_step(resume_info['step'])
            
            LOGGER.info(f"[Runner] 严格续算模式 | 输出目录: {self.output_dir} | "
                       f"从步骤 {self._step_offset} (t={self._time_offset:.4f}) 继续")
        else:
            # 热启动模式（或非 resume）：使用配置的新目录
            self.output_dir = Path(self.cfg.output.root) / self.cfg.output.subdir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            if resume_info:
                LOGGER.info(f"[Runner] 热启动模式 | 从 {resume_info['h5_path']} 初始化 | "
                           f"新输出目录: {self.output_dir}")
        
        # 配置日志文件输出（可由外部跳过）
        if self.output_dir is not None and not self.cfg.output.skip_log_setup:
            log_path = self.output_dir / 'run.log'
            setup_file_logging(str(log_path))
            LOGGER.info(f"[Runner] 日志文件: {log_path}")
    
    def _compute_frame_offset(self) -> int:
        """计算严格续算模式下的图片编号偏移"""
        if self.output_dir is None:
            return 0
        
        frames_dir = self.output_dir / self.cfg.visuals.folder
        if not frames_dir.exists():
            return 0
        
        # 找到现有的最大帧编号
        max_frame = 0
        for qty in self.cfg.visuals.quantities:
            qty_dir = frames_dir / qty.lower()
            if qty_dir.exists():
                for f in qty_dir.glob('*.png'):
                    try:
                        # 文件名格式: step_00001.png 或类似
                        num = int(f.stem.split('_')[-1])
                        max_frame = max(max_frame, num)
                    except (ValueError, IndexError):
                        pass
        return max_frame
    
    def _cleanup_after_step(self, resume_step: int):
        """清理指定时间步之后的所有数据"""
        if self.output_dir is None:
            return
        
        LOGGER.info(f"[Runner] 清理步骤 {resume_step} 之后的数据...")
        
        # 1. 清理 h5 文件中的后续时间步
        raw_dir = self.output_dir / 'raw'
        if raw_dir.exists():
            h5_path = raw_dir / self.cfg.fields.save.filename
            if h5_path.exists():
                self._cleanup_h5_steps(h5_path, resume_step)
        
        # 2. 清理 CSV 文件中的后续记录
        self._cleanup_csv_after_step(raw_dir, resume_step)
        
        # 3. 清理图片（按步骤编号）
        self._cleanup_frames_after_step(resume_step)
    
    def _cleanup_h5_steps(self, h5_path: Path, resume_step: int):
        """清理 h5 文件中指定步骤之后的数据"""
        import h5py
        try:
            with h5py.File(h5_path, 'a') as f:
                steps_to_remove = []
                for key in f.keys():
                    if key.startswith('step_'):
                        step_num = int(key.split('_')[1])
                        if step_num > resume_step:
                            steps_to_remove.append(key)
                
                for key in steps_to_remove:
                    del f[key]
                
                if steps_to_remove:
                    LOGGER.info(f"[Runner]   h5: 删除 {len(steps_to_remove)} 个时间步")
        except Exception as e:
            LOGGER.warning(f"[Runner] 清理 h5 失败: {e}")
    
    def _cleanup_csv_after_step(self, raw_dir: Path, resume_step: int):
        """清理 CSV 文件中指定步骤之后的记录"""
        import csv
        
        # 包含射流探针 CSV
        csv_files = [
            self.cfg.force_coeffs.filename, 
            self.cfg.jet_probes.filename,
            self.cfg.probes.filename,
        ]
        
        for csv_name in csv_files:
            csv_path = raw_dir / csv_name
            if not csv_path.exists():
                continue
            
            try:
                # 读取所有行
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fieldnames = reader.fieldnames
                
                if not rows or not fieldnames:
                    continue
                
                # 找到 step 列并过滤
                if 'step' in fieldnames:
                    filtered = [r for r in rows if int(r['step']) <= resume_step]
                    removed_count = len(rows) - len(filtered)
                    
                    if removed_count > 0:
                        # 重写文件
                        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(filtered)
                        LOGGER.info(f"[Runner]   {csv_name}: 删除 {removed_count} 条记录")
            except Exception as e:
                LOGGER.warning(f"[Runner] 清理 {csv_name} 失败: {e}")
    
    def _cleanup_frames_after_step(self, resume_step: int):
        """清理指定步骤之后的图片（包括 final 帧）"""
        if self.output_dir is None:
            return
        
        frames_dir = self.output_dir / self.cfg.visuals.folder
        if not frames_dir.exists():
            return
        
        total_removed = 0
        for qty in self.cfg.visuals.quantities:
            qty_dir = frames_dir / qty.lower()
            if not qty_dir.exists():
                continue
            
            for f in qty_dir.glob('*.png'):
                try:
                    stem = f.stem  # e.g., "step000020" or "step000020_final"
                    # 处理 final 帧: step000020_final -> 删除
                    if '_final' in stem:
                        f.unlink()
                        total_removed += 1
                        continue
                    # 提取步骤编号: step000020 -> 20
                    if stem.startswith('step'):
                        num = int(stem[4:])  # 去掉 "step" 前缀
                        if num > resume_step:
                            f.unlink()
                            total_removed += 1
                except (ValueError, IndexError):
                    pass
        
        if total_removed > 0:
            LOGGER.info(f"[Runner]   frames: 删除 {total_removed} 张图片（含 final 帧）")

    
    def run(self, num_steps: int = None, show_progress: bool = True) -> Dict[str, Any]:
        """执行仿真
        
        Args:
            num_steps: 仿真步数。若为 None，则从配置文件 solver.time.num_steps 读取。
            show_progress: 是否显示进度条。
        
        Resume 模式说明:
            - resume_strict=False (热启动): 从零开始计步，输出到新目录
            - resume_strict=True (续算): 接续原来的步数，输出到原目录
        """
        # 从配置文件读取 num_steps（若未指定）
        if num_steps is None:
            num_steps = getattr(self.solver.cfg.time, 'num_steps', None)
            if num_steps is None:
                raise ValueError("num_steps 未指定且配置文件中未设置 solver.time.num_steps")
        
        # 严格续算模式下，设置 solver 的初始时间步和时间
        is_strict_resume = self._resume_info and self._resume_info.get('strict', False)
        if is_strict_resume:
            self.solver.current_step = self._step_offset
            self.solver.current_time = self._time_offset
            LOGGER.info(f"[Runner] 续算模式: 从步骤 {self._step_offset} (t={self._time_offset:.4f}) 继续 {num_steps} 步")
        
        # 域概览图（非续算模式或首次运行时生成）
        if self.cfg.domain_overview.enabled and self.output_dir:
            overview_path = self.output_dir / self.cfg.domain_overview.filename
            if not is_strict_resume or not overview_path.exists():
                self.plot_domain_overview(save_path=str(overview_path))
        
        # 进度条
        if show_progress:
            from tqdm import tqdm
            start_step = self._step_offset if is_strict_resume else 0
            iterator = tqdm(range(num_steps), desc=f'Simulating (from step {start_step})', unit='step')
        else:
            iterator = range(num_steps)
        
        # 保存初始帧（非续算模式时）
        if self.cfg.visuals.enabled and self.cfg.visuals.save_endpoints and not is_strict_resume:
            self._save_visual(0)
        
        # 主循环
        for i in iterator:
            self.solver.step()
            local_step = i + 1  # 本次运行的步数 (1, 2, 3, ...)
            global_step = self._step_offset + local_step if is_strict_resume else local_step  # 全局步数
            
            # 力系数（基于本次运行步数判断 interval）
            if self.cfg.force_coeffs.enabled and local_step % self.cfg.force_coeffs.interval == 0:
                self._record_force_coeffs()
            
            # 射流探针（基于本次运行步数判断 interval）
            if self.cfg.jet_probes.enabled and local_step % self.cfg.jet_probes.interval == 0:
                self._record_jet_probes()
            
            # 探针（基于本次运行步数判断 interval）
            if self.cfg.probes.enabled and local_step % self.cfg.probes.interval == 0:
                self._record_probes()
            
            # 场数据保存（基于本次运行步数判断 interval，保存时使用 solver.current_step）
            if self.cfg.fields.save.enabled and local_step % self.cfg.fields.save.interval == 0:
                self._save_fields_h5()
            
            # 场数据缓存（基于本次运行步数判断 interval）
            if self.cfg.fields.cache.enabled and local_step % self.cfg.fields.cache.interval == 0:
                self._cache_fields()
            
            # 可视化（基于本次运行步数判断 interval，文件名使用全局步数）
            if self.cfg.visuals.enabled and local_step % self.cfg.visuals.interval == 0:
                self._save_visual(global_step)
        
        # 最终帧（使用全局步数）
        final_step = self._step_offset + num_steps if is_strict_resume else num_steps
        if self.cfg.visuals.enabled and self.cfg.visuals.save_endpoints:
            self._save_visual(final_step, suffix='_final')
        
        # 关闭 h5
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        
        # 保存 CSV
        if self.cfg.output.enabled:
            self._save_csvs()
        
        # 绘制力系数图
        if self.cfg.force_coeffs.enabled and self.cfg.output.enabled and self.force_coeffs:
            self._plot_force_coeffs()
        
        # 生成视频
        if self.cfg.visuals.enabled and self.cfg.output.enabled:
            self._generate_videos()
        
        # 计算并打印统计量
        stats = self.solver._compute_statistics()
        print("\n========== 最终统计量 ==========")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}")
        print("================================")
        
        # 打印输出目录
        if self.output_dir:
            print(f"\n输出目录: {self.output_dir.resolve()}")

        return {'force_coeffs': self.force_coeffs, 'probes': self.probes, 'fields': self.fields}
    
    # ---- 存档配置文件 ----
    def _save_config_snapshot(self) -> None:
        """保存 runner 与 solver 的配置快照到 output_dir/config_snapshot.yaml
        
        仅保存一个配置文件，每次运行时覆盖。
        可通过 output.skip_config_snapshot=True 禁用此功能。
        """
        if self.output_dir is None:
            return
        if self.cfg.output.skip_config_snapshot:
            return

        # 转换 dataclass 到 dict
        try:
            runner_cfg = asdict(self.cfg)
        except Exception:
            runner_cfg = getattr(self.cfg, '__dict__', {})

        # 试图抓取 solver.cfg（若存在）
        solver_cfg = None
        try:
            if hasattr(self.solver, 'cfg'):
                try:
                    solver_cfg = asdict(self.solver.cfg)
                except Exception:
                    solver_cfg = getattr(self.solver.cfg, '__dict__', {})
        except Exception:
            solver_cfg = None

        snapshot = {'runner': runner_cfg}
        if solver_cfg is not None:
            snapshot['solver'] = solver_cfg

        # 直接保存到 output_dir，仅一个文件
        config_file = self.output_dir / 'config_snapshot.yaml'
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            LOGGER.warning(f"[Runner] Failed to write config snapshot: {e}")

    # ---- 力系数 ----
    def _record_force_coeffs(self):
        """计算力系数（可微分实现，可选力矩）
        
        返回 Cd, Cl 作为 tensor（支持梯度反传）。
        """
        solver = self.solver
        if not solver.obstacles:
            return None, None
        
        cfg = self.cfg.force_coeffs
        U_ref = cfg.reference_velocity
        A_ref = cfg.reference_length
        
        Cd, Cl, Fx, Fy, U_ref_used, A_ref_used = solver.compute_force_coeffs(U_ref=U_ref, A_ref=A_ref)
        
        record = {
            'step': solver.current_step, 'time': solver.current_time,
            'Cd': Cd.item(), 'Cl': Cl.item(),
            'Fx': Fx.item(), 'Fy': Fy.item(),
            'A_ref': A_ref_used, 'U_ref': U_ref_used,
        }
        
        # 可选：计算力矩
        if cfg.include_torque:
            Mz, Mz_p, Mz_v = solver.compute_torque()
            record['Mz'] = Mz.item()
            record['Mz_pressure'] = Mz_p.item()
            record['Mz_viscous'] = Mz_v.item()
        
        self.force_coeffs.append(record)
        return Cd, Cl
    
    # ---- 射流探针 ----
    def _record_jet_probes(self):
        """记录射流探针数据（可微分实现）
        
        在每个喷嘴出口的流体网格采样速度和压强，同时采集核心喷口的数据。
        探针位置通过网格索引直接定位，确保采样点在正确的流体域内。
        
        Returns:
            Dict[str, Tuple[u, v, p]] tensor 值，支持梯度反传
        """
        import torch
        from phi.torch.flow import math, vec
        from phi.geom import Point
        
        solver = self.solver
        
        # 检查是否有 JetObstacle
        if not hasattr(solver, 'jet_obstacle') or solver.jet_obstacle is None:
            return {}
        
        jet_obstacle = solver.jet_obstacle
        if jet_obstacle.num_nozzles == 0:
            return {}
        
        # 延迟初始化探针位置
        if self._jet_probe_locations is None:
            self._jet_probe_locations = jet_obstacle.get_nozzle_probe_locations()
            if self._jet_probe_locations:
                LOGGER.info(f"[Runner] 射流探针已初始化: {len(self._jet_probe_locations)} 个喷嘴")
                for i, loc in enumerate(self._jet_probe_locations):
                    fluid = loc['fluid']
                    nozzle = loc['nozzle']
                    LOGGER.debug(f"  nozzle_{i}: fluid=({fluid[0]:.4f}, {fluid[1]:.4f}), "
                                 f"nozzle=({nozzle[0]:.4f}, {nozzle[1]:.4f})")
        
        if not self._jet_probe_locations:
            return {}
        
        result = {}  # 返回可微分 tensor: {name: (u, v, p)}
        
        for i, loc in enumerate(self._jet_probe_locations):
            fluid_xy = loc['fluid']
            nozzle_xy = loc['nozzle']
            
            # ===== 流体域探针采样（主要数据，可微分）=====
            point = Point(vec(x=fluid_xy[0], y=fluid_xy[1]))
            vel = solver.velocity.at(point)
            p = solver.pressure.at(point)
            
            # 提取 tensor（保持计算图）
            u_t = math.mean(vel.values.vector['x'])
            v_t = math.mean(vel.values.vector['y'])
            p_t = math.mean(p.values)
            
            # 转换为 native tensor
            u_t = u_t.native() if hasattr(u_t, 'native') else u_t
            v_t = v_t.native() if hasattr(v_t, 'native') else v_t
            p_t = p_t.native() if hasattr(p_t, 'native') else p_t
            
            result[f'jet_nozzle_{i}'] = (u_t, v_t, p_t)
            
            # 计算标量值（用于日志）
            if isinstance(u_t, torch.Tensor):
                vel_mag = torch.sqrt(u_t**2 + v_t**2)
                u_val, v_val, p_val = u_t.item(), v_t.item(), p_t.item()
                vel_mag_val = vel_mag.item()
            else:
                vel_mag_val = np.sqrt(u_t**2 + v_t**2)
                u_val, v_val, p_val = float(u_t), float(v_t), float(p_t)
            
            # ===== 核心喷口采样（附加数据，用于验证射流施加效果）=====
            nozzle_point = Point(vec(x=nozzle_xy[0], y=nozzle_xy[1]))
            nozzle_vel = solver.velocity.at(nozzle_point)
            nozzle_p = solver.pressure.at(nozzle_point)
            
            nu_t = math.mean(nozzle_vel.values.vector['x'])
            nv_t = math.mean(nozzle_vel.values.vector['y'])
            np_t = math.mean(nozzle_p.values)
            
            nu_t = nu_t.native() if hasattr(nu_t, 'native') else nu_t
            nv_t = nv_t.native() if hasattr(nv_t, 'native') else nv_t
            np_t = np_t.native() if hasattr(np_t, 'native') else np_t
            
            if isinstance(nu_t, torch.Tensor):
                nozzle_vel_mag = torch.sqrt(nu_t**2 + nv_t**2)
                nu_val, nv_val, np_val = nu_t.item(), nv_t.item(), np_t.item()
                nozzle_vel_mag_val = nozzle_vel_mag.item()
            else:
                nozzle_vel_mag_val = np.sqrt(nu_t**2 + nv_t**2)
                nu_val, nv_val, np_val = float(nu_t), float(nv_t), float(np_t)
            
            # 记录到缓存（用于 CSV 输出）
            self.probes_jet.append({
                'nozzle_id': i,
                'step': solver.current_step,
                'time': solver.current_time,
                # 流体域探针
                'fluid_x': fluid_xy[0], 'fluid_y': fluid_xy[1],
                'fluid_u': u_val, 'fluid_v': v_val,
                'fluid_vel_mag': vel_mag_val, 'fluid_p': p_val,
                # 核心喷口探针
                'nozzle_x': nozzle_xy[0], 'nozzle_y': nozzle_xy[1],
                'nozzle_u': nu_val, 'nozzle_v': nv_val,
                'nozzle_vel_mag': nozzle_vel_mag_val, 'nozzle_p': np_val,
            })
        
        return result
        
        return result
    
    # ---- 探针 ----
    def _record_probes(self):
        """记录探针数据（可微分实现）
        
        返回 Dict[str, Tuple[u, v, p]] 的 tensor 值（支持梯度反传）。
        同时将结果记录到 self.probes 列表（转换为 float 用于日志）。
        """
        import torch
        from phi.torch.flow import math, vec, field, instance, channel
        from phi.geom import Point
        
        solver = self.solver
        Lx, Ly = solver.cfg.domain.size
        
        result = {}  # 返回值：{name: (u, v, p) tensors}

        locations = self.cfg.probes.locations
        signature = tuple(
            (
                loc.get('name', ''),
                tuple(loc.get('fraction', [])) if 'fraction' in loc else (),
                tuple(loc.get('units', [])) if 'units' in loc else (),
            )
            for loc in locations
        )

        # 缓存失效时重建批量采样几何
        if self._probe_batch_signature != signature:
            names: List[str] = []
            coords: List[Tuple[float, float]] = []

            for idx, loc in enumerate(locations):
                name = loc.get('name', f'probe_{idx}')
                if 'fraction' in loc:
                    x, y = loc['fraction'][0] * Lx, loc['fraction'][1] * Ly
                elif 'units' in loc:
                    x, y = loc['units']
                else:
                    continue
                names.append(name)
                coords.append((float(x), float(y)))

            self._probe_batch_names = names
            self._probe_batch_coords = coords
            if coords:
                points_tensor = math.wrap(coords, instance('points'), channel(vector='x,y'))
                self._probe_batch_geom = Point(points_tensor)
            else:
                self._probe_batch_geom = None
            self._probe_batch_signature = signature

        if not self._probe_batch_names or self._probe_batch_geom is None:
            return result

        vel = field.sample(solver.velocity, self._probe_batch_geom)
        p = field.sample(solver.pressure, self._probe_batch_geom)

        u_all = vel.vector['x'].native().reshape(-1)
        v_all = vel.vector['y'].native().reshape(-1)
        p_all = p.native().reshape(-1)

        for i, name in enumerate(self._probe_batch_names):
            x, y = self._probe_batch_coords[i]
            u_t = u_all[i]
            v_t = v_all[i]
            p_t = p_all[i]

            result[name] = (u_t, v_t, p_t)

            if name not in self.probes:
                self.probes[name] = []

            self.probes[name].append({
                'step': solver.current_step, 'time': solver.current_time,
                'x': x, 'y': y,
                'u': float(u_t.item() if isinstance(u_t, torch.Tensor) else u_t),
                'v': float(v_t.item() if isinstance(v_t, torch.Tensor) else v_t),
                'p': float(p_t.item() if isinstance(p_t, torch.Tensor) else p_t),
            })
        
        return result  # 返回 tensor dict 供 loss 使用
    
    # ---- 场数据 ----
    def _cache_fields(self):
        solver = self.solver
        # 转换为中心网格后再缓存
        vel_centered = solver.velocity @ solver.pressure
        self.fields.append({
            'step': solver.current_step, 'time': solver.current_time,
            'velocity': vel_centered.values.numpy('x,y,vector'),
            'pressure': solver.pressure.values.numpy('x,y'),
        })
    
    def _save_fields_h5(self):
        """使用 PhiFlow I/O 保存场数据到 H5
        
        根据配置中的 quantities 列表保存指定的场数据。
        支持的场类型:
            - 'velocity': 速度场（StaggeredGrid，会插值到中心网格后保存）
            - 'pressure': 压力场（CenteredGrid）
            - 'vorticity': 涡量场（由 velocity 计算得到）
            - 'velocity_magnitude': 速度大小场（标量）
            - 'sdf': Signed Distance Field（如果 solver 提供）
            - 'mask': 障碍物掩码（如果 solver 提供）
        """
        if not self.cfg.output.enabled:
            return
        
        from phi.torch.flow import field as phi_field
        
        solver = self.solver
        raw_dir = self.output_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        h5_path = raw_dir / self.cfg.fields.save.filename
        
        # 准备元数据（首次写入时）
        metadata = {
            'domain_size': solver.cfg.domain.size,
            'resolution': solver.cfg.domain.resolution,
            'dt': solver.cfg.time.dt,
            'array_dim': 'x,y',
        }
        
        # 获取配置中要保存的场列表（默认至少包含 velocity 和 pressure）
        quantities = self.cfg.fields.save.quantities
        # 确保 velocity 和 pressure 始终被保存
        required = {'velocity', 'pressure'}
        save_quantities = required.union(set(quantities))
        
        # 准备场数据字典
        fields_dict = {}
        
        # 参考网格（用于插值所有场到统一的中心网格）
        ref_grid = solver.pressure
        
        for qty in save_quantities:
            qty_lower = qty.lower()
            try:
                if qty_lower == 'velocity':
                    fields_dict['velocity'] = solver.velocity
                elif qty_lower == 'pressure':
                    fields_dict['pressure'] = solver.pressure
                elif qty_lower == 'vorticity':
                    # 计算涡量场（2D 情况下为标量场）
                    # curl 返回的是 dual grid，需要插值到中心网格以保持形状一致
                    vort = phi_field.curl(solver.velocity)
                    vort_centered = vort @ ref_grid  # 插值到中心网格
                    fields_dict['vorticity'] = vort_centered
                elif qty_lower == 'velocity_magnitude':
                    # 计算速度大小场
                    vel_mag = phi_field.vec_length(solver.velocity)
                    # vec_length 可能也返回非中心网格，确保插值
                    if hasattr(vel_mag, 'at'):
                        vel_mag = vel_mag @ ref_grid
                    fields_dict['velocity_magnitude'] = vel_mag
                elif qty_lower == 'sdf':
                    # SDF 场（如果 solver 提供）
                    if hasattr(solver, 'sdf') and solver.sdf is not None:
                        fields_dict['sdf'] = solver.sdf
                elif qty_lower == 'mask':
                    # 障碍物掩码（如果 solver 提供）
                    if hasattr(solver, 'obstacle_mask') and solver.obstacle_mask is not None:
                        fields_dict['mask'] = solver.obstacle_mask
                else:
                    LOGGER.warning(f"[Runner] 未知的场类型: {qty}，已跳过")
            except Exception as e:
                LOGGER.warning(f"[Runner] 计算场 {qty} 时出错: {e}")
        
        # 使用 save_fields_h5 函数
        save_fields_h5(
            fields_dict=fields_dict,
            h5_path=h5_path,
            step=solver.current_step,
            time=solver.current_time,
            metadata=metadata,
            append=True,
        )
    
    # ---- 可视化 ----
    def _save_visual(self, step: int, suffix: str = ''):
        if not self.cfg.output.enabled:
            return
        
        from phi.torch.flow import field
        import matplotlib.pyplot as plt
        
        solver = self.solver
        vis_cfg = self.cfg.visuals
        frames_dir = self.output_dir / vis_cfg.folder
        
        # 量映射（含流线图）
        qty_map = {
            'velocity_magnitude': (lambda: field.vec_length(solver.velocity), 'Vel Mag', 'heatmap'),
            'velocity_x': (lambda: solver.velocity.vector['x'], 'Vel X', 'heatmap'),
            'velocity_y': (lambda: solver.velocity.vector['y'], 'Vel Y', 'heatmap'),
            'pressure': (lambda: solver.pressure, 'Pressure', 'heatmap'),
            'vorticity': (lambda: field.curl(solver.velocity), 'Vorticity', 'heatmap'),
            'streamlines': (lambda: solver.velocity, 'Streamlines', 'streamlines'),
        }
        
        Lx, Ly = solver.cfg.domain.size
        
        for qty in vis_cfg.quantities:
            qty_lower = qty.lower()
            if qty_lower not in qty_map:
                continue
            
            # 创建该量的子目录
            qty_dir = frames_dir / qty_lower
            qty_dir.mkdir(parents=True, exist_ok=True)
            
            data_fn, title, plot_type = qty_map[qty_lower]
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                
                if plot_type == 'heatmap':
                    data_field = data_fn()
                    data = data_field.values.numpy('x,y')
                    
                    # 处理障碍物（平滑模式下平滑边缘）
                    if vis_cfg.obstacle_style in ('fill', 'both') and solver.obstacles:
                        data = self._mask_obstacle_region(data, Lx, Ly, smooth=vis_cfg.smooth)
                    
                    # 绘制热力图（interpolation控制平滑）
                    interp = 'bilinear' if vis_cfg.smooth else 'nearest'
                    im = ax.imshow(data.T, origin='lower', extent=[0, Lx, 0, Ly],
                                   aspect='equal', cmap='RdBu_r', interpolation=interp)
                    plt.colorbar(im, ax=ax, label=title)
                
                elif plot_type == 'streamlines':
                    vel = data_fn()
                    vel_centered = vel @ solver.pressure
                    u = vel_centered.values.numpy('x,y,vector')[..., 0].T
                    v = vel_centered.values.numpy('x,y,vector')[..., 1].T
                    Nx, Ny = u.shape[1], u.shape[0]
                    x = np.linspace(0, Lx, Nx)
                    y = np.linspace(0, Ly, Ny)
                    X, Y = np.meshgrid(x, y)
                    speed = np.sqrt(u**2 + v**2)
                    ax.streamplot(X, Y, u, v, color=speed, cmap='viridis', density=1.5, linewidth=0.8)
                
                # 绘制障碍物轮廓
                if vis_cfg.obstacle_style != 'none' and solver.obstacles:
                    self._draw_obstacle(ax, Lx, Ly, vis_cfg.obstacle_style)
                
                ax.set_xlim(0, Lx)
                ax.set_ylim(0, Ly)
                ax.set_aspect('equal')
                ax.set_title(f'{title} (t={solver.current_time:.4f})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                
                fig.savefig(qty_dir / f'step{step:06d}{suffix}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                LOGGER.warning(f"[Runner] 可视化 {qty} 失败: {e}")
    
    def _mask_obstacle_region(self, data: np.ndarray, Lx: float, Ly: float, smooth: bool = False) -> np.ndarray:
        """将障碍物区域的数据设为 NaN 或指定值
        
        Args:
            data: 原始场数据
            Lx, Ly: 域尺寸
            smooth: 是否平滑边缘（用于平滑可视化模式）
        """
        solver = self.solver
        obs = solver.cfg.obstacle
        vis_cfg = self.cfg.visuals
        
        Nx, Ny = data.shape
        dx, dy = Lx / Nx, Ly / Ny
        
        # 障碍物中心（物理坐标）
        cx = obs.center[0] * Lx
        cy = obs.center[1] * Ly
        
        # 创建网格坐标
        x = np.linspace(dx/2, Lx - dx/2, Nx)
        y = np.linspace(dy/2, Ly - dy/2, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # 计算有符号距离场（SDF）
        shape_type = obs.shape.lower()
        
        if shape_type == 'sphere':
            r = obs.radius
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2) - r  # 负值在内部
        
        elif shape_type == 'box':
            sx, sy = obs.size[0], obs.size[1]
            theta = obs.rotation
            
            # 将点转换到障碍物的局部坐标系
            dx_pts = X - cx
            dy_pts = Y - cy
            
            cos_t, sin_t = np.cos(-theta), np.sin(-theta)
            local_x = dx_pts * cos_t - dy_pts * sin_t
            local_y = dx_pts * sin_t + dy_pts * cos_t
            
            # 矩形 SDF
            qx = np.abs(local_x) - sx/2
            qy = np.abs(local_y) - sy/2
            dist = np.sqrt(np.maximum(qx, 0)**2 + np.maximum(qy, 0)**2) + np.minimum(np.maximum(qx, qy), 0)
        
        else:
            dist = np.ones_like(X)  # 无障碍物
        
        data = data.copy()
        fill_val = vis_cfg.obstacle_fill_value if vis_cfg.obstacle_fill_value is not None else np.nan
        
        if smooth:
            # 平滑模式：使用 SDF 创建平滑过渡
            # 过渡带宽度约为 2 个网格单元
            transition_width = 2.0 * max(dx, dy)
            # sigmoid 平滑过渡：内部=0，外部=1
            alpha = 1.0 / (1.0 + np.exp(-dist / (transition_width * 0.5)))
            # 混合原始数据和填充值
            if np.isnan(fill_val):
                # NaN 填充：将内部区域设为 NaN
                data[dist < -transition_width] = np.nan
                # 过渡区域保持原值（imshow 的插值会自然平滑）
            else:
                data = data * alpha + fill_val * (1 - alpha)
        else:
            # 非平滑模式：硬边界
            mask = dist <= 0
            data[mask] = fill_val
        
        return data
    
    def _draw_obstacle(self, ax, Lx: float, Ly: float, style: str):
        """在图上绘制障碍物"""
        import matplotlib.patches as mpatches
        from matplotlib.transforms import Affine2D
        
        solver = self.solver
        obs = solver.cfg.obstacle
        
        # 障碍物中心
        cx = obs.center[0] * Lx
        cy = obs.center[1] * Ly
        
        shape_type = obs.shape.lower()
        
        # 设置样式
        fill = style in ('fill', 'both')
        outline = style in ('outline', 'both')
        
        facecolor = 'white' if fill else 'none'
        edgecolor = 'black' if outline else 'none'
        linewidth = 1.5 if outline else 0
        
        if shape_type == 'sphere':
            r = obs.radius
            patch = mpatches.Circle(
                (cx, cy), r, 
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth
            )
            ax.add_patch(patch)
        
        elif shape_type == 'box':
            sx, sy = obs.size[0], obs.size[1]
            theta = obs.rotation
            
            # 创建矩形（左下角为锚点）
            rect = mpatches.Rectangle(
                (-sx/2, -sy/2), sx, sy,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth
            )
            
            # 应用变换：先旋转，再平移
            t = Affine2D().rotate(theta).translate(cx, cy) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
    
    # ---- CSV ----
    def _save_csvs(self):
        import csv
        raw_dir = self.output_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 判断是否为严格续算模式（追加 CSV）
        is_strict_resume = self._resume_info and self._resume_info.get('strict', False)
        
        if self.force_coeffs:
            fc_path = raw_dir / self.cfg.force_coeffs.filename
            mode = 'a' if (is_strict_resume and fc_path.exists()) else 'w'
            # 动态确定字段（根据是否含 torque）
            base_fields = ['step', 'time', 'Cd', 'Cl', 'Fx', 'Fy', 'A_ref', 'U_ref']
            if self.cfg.force_coeffs.include_torque:
                base_fields.extend(['Mz', 'Mz_pressure', 'Mz_viscous'])
            with open(fc_path, mode, newline='') as f:
                w = csv.DictWriter(f, fieldnames=base_fields)
                if mode == 'w':
                    w.writeheader()
                w.writerows(self.force_coeffs)
        
        # 射流探针 CSV（按 nozzle_id 分组排序）
        if self.probes_jet:
            jet_probes_path = raw_dir / self.cfg.jet_probes.filename
            mode = 'a' if (is_strict_resume and jet_probes_path.exists()) else 'w'
            sorted_jet_probes = sorted(self.probes_jet, key=lambda r: (r['nozzle_id'], r['step']))
            with open(jet_probes_path, mode, newline='') as f:
                fieldnames = [
                    'nozzle_id', 'step', 'time',
                    # 流体域探针
                    'fluid_x', 'fluid_y', 'fluid_u', 'fluid_v', 'fluid_vel_mag', 'fluid_p',
                    # 核心喷口探针
                    'nozzle_x', 'nozzle_y', 'nozzle_u', 'nozzle_v', 'nozzle_vel_mag', 'nozzle_p',
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == 'w':
                    w.writeheader()
                w.writerows(sorted_jet_probes)
        
        if self.probes:
            probes_path = raw_dir / self.cfg.probes.filename
            mode = 'a' if (is_strict_resume and probes_path.exists()) else 'w'
            with open(probes_path, mode, newline='') as f:
                rows = [{'probe': n, **r} for n, recs in self.probes.items() for r in recs]
                if rows:
                    w = csv.DictWriter(f, fieldnames=['probe', 'step', 'time', 'x', 'y', 'u', 'v', 'p'])
                    if mode == 'w':
                        w.writeheader()
                    w.writerows(rows)
    
    def _plot_force_coeffs(self):
        """绘制力系数和力的时间历程图
        
        保存在与 force_coeffs.csv 同目录下，文件名为 force_coeffs.png
        """
        import matplotlib.pyplot as plt
        
        if not self.force_coeffs:
            return
        
        raw_dir = self.output_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取数据
        steps = [r['step'] for r in self.force_coeffs]
        times = [r['time'] for r in self.force_coeffs]
        cd_history = [r['Cd'] for r in self.force_coeffs]
        cl_history = [r['Cl'] for r in self.force_coeffs]
        fx_history = [r['Fx'] for r in self.force_coeffs]
        fy_history = [r['Fy'] for r in self.force_coeffs]
        
        # 计算统计量
        n = len(cd_history)
        n_tail = max(1, n // 5)  # 最后 20% 用于稳态估计
        
        # 全量均值
        cd_mean, cd_std = np.mean(cd_history), np.std(cd_history)
        cl_mean, cl_std = np.mean(cl_history), np.std(cl_history)
        fx_mean, fy_mean = np.mean(fx_history), np.mean(fy_history)
        
        # 最后 20% 均值 (稳态估计)
        cd_tail = np.mean(cd_history[-n_tail:])
        cl_tail = np.mean(cl_history[-n_tail:])
        fx_tail = np.mean(fx_history[-n_tail:])
        fy_tail = np.mean(fy_history[-n_tail:])
        
        # 创建 2x2 子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 阻力系数 Cd (红=全量均值, 橙=稳态)
        axes[0, 0].plot(steps, cd_history, 'b-', linewidth=0.8)
        axes[0, 0].axhline(cd_mean, color='r', linestyle='--', linewidth=1.5, 
                          label=f'All: {cd_mean:.4f}±{cd_std:.4f}')
        axes[0, 0].axhline(cd_tail, color='orange', linestyle=':', linewidth=1.5, 
                          label=f'Last {n_tail}: {cd_tail:.4f}')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Cd')
        axes[0, 0].set_title('Drag Coefficient')
        axes[0, 0].legend(loc='best', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 升力系数 Cl
        axes[0, 1].plot(steps, cl_history, 'g-', linewidth=0.8)
        axes[0, 1].axhline(cl_mean, color='r', linestyle='--', linewidth=1.5,
                          label=f'All: {cl_mean:.4f}±{cl_std:.4f}')
        axes[0, 1].axhline(cl_tail, color='orange', linestyle=':', linewidth=1.5,
                          label=f'Last {n_tail}: {cl_tail:.4f}')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cl')
        axes[0, 1].set_title('Lift Coefficient')
        axes[0, 1].legend(loc='best', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 阻力 Fx
        axes[1, 0].plot(steps, fx_history, 'b-', linewidth=0.8)
        axes[1, 0].axhline(fx_mean, color='r', linestyle='--', linewidth=1.5,
                          label=f'All: {fx_mean:.4f}')
        axes[1, 0].axhline(fx_tail, color='orange', linestyle=':', linewidth=1.5,
                          label=f'Last {n_tail}: {fx_tail:.4f}')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Fx')
        axes[1, 0].set_title('Drag Force')
        axes[1, 0].legend(loc='best', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 升力 Fy
        axes[1, 1].plot(steps, fy_history, 'g-', linewidth=0.8)
        axes[1, 1].axhline(fy_mean, color='r', linestyle='--', linewidth=1.5,
                          label=f'All: {fy_mean:.4f}')
        axes[1, 1].axhline(fy_tail, color='orange', linestyle=':', linewidth=1.5,
                          label=f'Last {n_tail}: {fy_tail:.4f}')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Fy')
        axes[1, 1].set_title('Lift Force')
        axes[1, 1].legend(loc='best', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片（与 CSV 同目录）
        fig_path = raw_dir / 'force_coeffs.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        LOGGER.info(f"[Runner] 力系数图已保存: {fig_path}")
        
        # 打印统计摘要
        print(f"\n========== 力系数统计（{n} 个采样点） ==========")
        print(f"  Cd = {cd_mean:.4f} ± {cd_std:.4f} (稳态: {cd_tail:.4f})")
        print(f"  Cl = {cl_mean:.4f} ± {cl_std:.4f} (稳态: {cl_tail:.4f})")
        print(f"  Fx = {fx_mean:.4f}")
        print(f"  Fy = {fy_mean:.4f}")
        
        # 若含 torque，绘制单独图
        if self.cfg.force_coeffs.include_torque and 'Mz' in self.force_coeffs[0]:
            mz_history = [r['Mz'] for r in self.force_coeffs]
            mz_mean, mz_std = np.mean(mz_history), np.std(mz_history)
            print(f"  Mz = {mz_mean:.6f} ± {mz_std:.6f}")
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(steps, mz_history, 'purple', linewidth=0.8)
            ax2.axhline(mz_mean, color='r', linestyle='--', linewidth=1.5,
                       label=f'Mean={mz_mean:.6f}±{mz_std:.6f}')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Mz')
            ax2.set_title('Torque (z-axis)')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            fig2.savefig(raw_dir / 'torque.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)
            LOGGER.info(f"[Runner] 力矩图已保存: {raw_dir / 'torque.png'}")
        
        print("=============================================")
    
    def _generate_videos(self):
        """尝试使用 ffmpeg 将各 quantities 的图片合成 mp4 视频
        
        注意: 使用相对路径避免 Windows 上 ffmpeg concat 对中文路径的兼容问题
        """
        import subprocess
        import glob
        
        vis_cfg = self.cfg.visuals
        frames_dir = self.output_dir / vis_cfg.folder
        fps = vis_cfg.video_fps
        
        for qty in vis_cfg.quantities:
            qty_dir = frames_dir / qty.lower()
            if not qty_dir.exists():
                continue
            
            # 获取所有 png 文件并按名称排序
            png_files = sorted(glob.glob(str(qty_dir / 'step*.png')))
            png_files = [f for f in png_files if '_final' not in f]
            if not png_files:
                continue
            
            # 创建临时文件列表 (使用相对路径避免中文路径问题)
            list_file = qty_dir / 'filelist.txt'
            with open(list_file, 'w', encoding='utf-8') as f:
                for png in png_files:
                    # 只使用文件名（相对路径）
                    png_name = Path(png).name
                    f.write(f"file '{png_name}'\n")
                    f.write(f"duration {1.0/fps}\n")
                # 最后一帧需要重复一次（ffmpeg concat 要求）
                if png_files:
                    f.write(f"file '{Path(png_files[-1]).name}'\n")
            
            output_video = frames_dir / f'{qty.lower()}.mp4'
            encoder_candidates = self._get_ffmpeg_encoder_candidates()
            
            try:
                # 在 qty_dir 目录下执行，确保相对路径有效
                success = False
                had_failure = False      # 是否触发过回退
                failed_encoders = []     # 已尝试失败的编码器
                last_stderr = ''
                for encoder in encoder_candidates:
                    cmd = [
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', 'filelist.txt',  # 使用相对路径
                        '-c:v', encoder, '-pix_fmt', 'yuv420p',
                        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                        str(output_video.resolve())
                    ]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, cwd=str(qty_dir))
                        if had_failure:
                            # 发生过回退且最终成功 → 升级为 WARNING，确保与失败信息同级可见
                            LOGGER.warning(
                                f"[Runner] 视频已生成 ({qty}): "
                                f"{failed_encoders} → {encoder} 回退成功")
                        else:
                            # 首选编码器直接成功 → INFO 即可
                            LOGGER.info(f"[Runner] 视频已生成: {output_video} (encoder={encoder})")
                        success = True
                        break
                    except subprocess.CalledProcessError as e:
                        last_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
                        had_failure = True
                        failed_encoders.append(encoder)
                        reason = "编码器不可用" if 'Unknown encoder' in last_stderr else "编码失败"
                        LOGGER.warning(f"[Runner] ffmpeg {qty}: {encoder} {reason}，尝试下一个")
                        continue

                if not success:
                    LOGGER.warning(f"[Runner] 视频生成失败 ({qty}), "
                                   f"已尝试编码器: {failed_encoders}: {last_stderr[:200]}")
            except FileNotFoundError:
                LOGGER.warning("[Runner] ffmpeg 未找到，跳过视频生成")
                return
            finally:
                list_file.unlink(missing_ok=True)  # 确保临时文件总被清理（含 ffmpeg 缺失情况）

    def _get_ffmpeg_encoder_candidates(self):
        """返回当前环境可用的 ffmpeg 编码器候选列表（按优先级）

        优先级:
        1) libx264 (质量与兼容性最佳)
        2) libopenh264 (部分 conda/linux 构建可用)
        3) mpeg4 (通用兜底)
        """
        import subprocess

        preferred = ['libx264', 'libopenh264', 'mpeg4']
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                check=False,
                capture_output=True,
                text=True,
            )
            text = ((result.stdout or '') + '\n' + (result.stderr or '')).lower()
            available = [enc for enc in preferred if enc.lower() in text]
            # 若检测不到，则保留 mpeg4 兜底
            return available if available else ['mpeg4']
        except Exception:
            # 检测失败时保留多候选，运行时重试
            return preferred

    # ---- 辅助方法 ----
    def plot_domain_overview(self, save_path: Optional[str] = None):
        """
        绘制求解域概览图
        
        包含: 域边界、障碍物、边界条件类型、射流喷嘴位置（精确网格区域）、探针位置
        
        喷口可视化原理：
        - 使用 PhiFlow 原生 SDF 计算障碍物距离场
        - 结合角度范围确定喷口作用的网格单元
        - 支持任意几何形状（圆柱、方柱、旋转方柱等）
        
        Args:
            save_path: 保存路径，None=显示窗口
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.transforms import Affine2D
        from matplotlib.collections import PatchCollection
        
        solver = self.solver
        cfg = solver.cfg
        ov_cfg = self.cfg.domain_overview
        Lx, Ly = cfg.domain.size
        Nx, Ny = cfg.domain.resolution
        dx, dy = Lx / Nx, Ly / Ny
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-Lx * 0.05, Lx * 1.05)
        ax.set_ylim(-Ly * 0.1, Ly * 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Domain Overview')
        
        # 网格线
        if ov_cfg.show_grid:
            for i in range(Nx + 1):
                ax.axvline(i * dx, color='lightgray', linewidth=0.3, zorder=0)
            for j in range(Ny + 1):
                ax.axhline(j * dy, color='lightgray', linewidth=0.3, zorder=0)
        
        # 域边框
        domain_rect = mpatches.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(domain_rect)
        
        # 边界条件标注
        bc_labels = {
            'x-': (0, Ly/2, 'left'),
            'x+': (Lx, Ly/2, 'right'),
            'y-': (Lx/2, 0, 'bottom'),
            'y+': (Lx/2, Ly, 'top'),
        }
        for side, (x, y, va) in bc_labels.items():
            bc = cfg.boundary.velocity.get(side, {})
            bc_type = bc.get('type', 'unknown')
            ha = 'center' if 'y' in side else ('right' if side == 'x-' else 'left')
            offset = (-5, 0) if side == 'x-' else (5, 0) if side == 'x+' else (0, -5 if side == 'y-' else 5)
            ax.annotate(f'{side}: {bc_type}', (x, y), textcoords='offset points', xytext=offset,
                       ha=ha, va='center', fontsize=8, color='blue')
        
        # 障碍物
        if cfg.obstacle.enabled:
            obs = cfg.obstacle
            cx, cy = obs.center[0] * Lx, obs.center[1] * Ly
            
            if obs.shape.lower() == 'sphere':
                circle = mpatches.Circle((cx, cy), obs.radius, facecolor='gray', edgecolor='black', alpha=0.7)
                ax.add_patch(circle)
            elif obs.shape.lower() == 'box':
                rect = mpatches.Rectangle((-obs.size[0]/2, -obs.size[1]/2), obs.size[0], obs.size[1],
                                         facecolor='gray', edgecolor='black', alpha=0.7)
                t = Affine2D().rotate(obs.rotation).translate(cx, cy) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)
            
            # 射流喷嘴 - 精确绘制喷口网格区域
            if cfg.jets.enabled and cfg.jets.nozzles and solver.jet_obstacle is not None:
                show_nozzles = getattr(ov_cfg, 'show_nozzles', True)
                
                if show_nozzles:
                    self._plot_jet_nozzle_cells(ax, solver, cfg, Lx, Ly, Nx, Ny, dx, dy, cx, cy)
                
                # 喷嘴角度文字标签
                if ov_cfg.show_nozzle_labels:
                    r = obs.radius if obs.shape.lower() == 'sphere' else max(obs.size) / 2
                    for nozzle in cfg.jets.nozzles:
                        arc_deg = nozzle.get('arc_deg', [0, 10])
                        arc_center = (arc_deg[0] + arc_deg[1]) / 2
                        nx = cx + r * 1.5 * np.cos(np.radians(arc_center))
                        ny = cy + r * 1.5 * np.sin(np.radians(arc_center))
                        ax.annotate(f'{arc_deg[0]}°-{arc_deg[1]}°', (nx, ny), fontsize=7,
                                   textcoords='offset points', xytext=(5, 5))
        
        # 探针
        if ov_cfg.show_probes and self.cfg.probes.enabled:
            show_probes_labels = getattr(ov_cfg, 'show_probes_labels', True)
            for loc in self.cfg.probes.locations:
                name = loc.get('name', '')
                if 'fraction' in loc:
                    px, py = loc['fraction'][0] * Lx, loc['fraction'][1] * Ly
                elif 'units' in loc:
                    px, py = loc['units']
                else:
                    continue
                # 探针位置标记（show_probes 控制）
                ax.plot(px, py, 'go', markersize=ov_cfg.probe_size)
                # 探针名称文字标签（show_probes_labels 控制）
                if show_probes_labels:
                    ax.annotate(name, (px, py), fontsize=7, textcoords='offset points', xytext=(3, 3))
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            LOGGER.info(f"[Runner] 域概览图已保存: {save_path}")
        else:
            plt.show()
    
    def _plot_jet_nozzle_cells(self, ax, solver, cfg, Lx, Ly, Nx, Ny, dx, dy, cx, cy):
        """
        精确绘制喷口作用的网格单元
        
        直接使用 jet_obstacle._nozzle_cache，这是唯一的数据源，
        因为缓存中的 mask 正是射流实际作用的网格点。
        
        缓存基于 StaggeredGrid 节点坐标 (Nx+1, Ny+1)，可视化时需要映射到单元格。
        
        可视化模式（由配置参数控制）：
        - nozzle_edge_only (默认 True): 仅显示喷口核心区域（nozzle_core_masks）
        - nozzle_display_mode (默认 'cells'):
            - 'cells': 绘制矩形单元格（显示网格结构）
            - 'points': 绘制圆点（更贴合曲面形状，视觉效果更好）
        
        注意：此方法仅用于可视化，使用 numpy 即可，不需要可微分。
        
        Raises:
            RuntimeError: 如果 _nozzle_cache 未初始化
        """
        import matplotlib.patches as mpatches
        
        jet_obstacle = solver.jet_obstacle
        ov_cfg = self.cfg.domain_overview
        
        # 获取配置参数
        nozzle_edge_only = getattr(ov_cfg, 'nozzle_edge_only', True)
        nozzle_display_mode = getattr(ov_cfg, 'nozzle_display_mode', 'cells')
        
        # 为每个喷嘴收集作用的网格单元
        nozzle_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        
        # ========== 必须使用缓存 ==========
        cache = getattr(jet_obstacle, '_nozzle_cache', None)
        if cache is None:
            LOGGER.error(
                "[Runner] jet_obstacle._nozzle_cache 为 None！"
                "请确保在可视化前已调用 set_velocity_template() 或在 JetObstacle 构造时传入 velocity_template。"
            )
            raise RuntimeError(
                "[Runner] 无法绘制喷嘴网格：_nozzle_cache 未初始化。"
                "请检查 JetObstacle 的初始化流程。"
            )
        
        LOGGER.debug(f"[Runner] 使用 jet_obstacle._nozzle_cache 绘制喷嘴网格 (mode={nozzle_display_mode})")
        
        # 选择使用核心 mask 还是完整 mask
        if nozzle_edge_only:
            masks_to_use = cache.nozzle_core_masks
        else:
            masks_to_use = cache.nozzle_masks
        
        for idx, mask in enumerate(masks_to_use):
            # 将 torch tensor 转为 numpy
            if hasattr(mask, 'cpu'):
                mask_np = mask.cpu().numpy()
            elif hasattr(mask, 'numpy'):
                mask_np = mask.numpy()
            else:
                mask_np = np.asarray(mask)
            
            cache_Nx, cache_Ny = mask_np.shape  # (Nx+1, Ny+1)
            color = nozzle_colors[idx % len(nozzle_colors)]
            
            if nozzle_display_mode == 'points':
                # ===== 圆点模式：更贴合曲面形状 =====
                # 收集所有喷嘴节点的物理坐标
                node_x_list = []
                node_y_list = []
                
                for i in range(cache_Nx):
                    for j in range(cache_Ny):
                        if mask_np[i, j]:
                            node_x_list.append(i * dx)
                            node_y_list.append(j * dy)
                
                if node_x_list:
                    # 计算合适的点大小（基于网格尺寸）
                    point_size = max(10, min(50, 500 * dx))  # 自适应大小
                    ax.scatter(
                        node_x_list, node_y_list,
                        c=color, s=point_size, marker='o',
                        edgecolors='darkred', linewidths=0.5,
                        alpha=0.7, label=f'Nozzle {idx+1}' if idx == 0 else None,
                        zorder=10,  # 确保在其他元素之上
                    )
            else:
                # ===== 单元格模式（默认）：显示网格结构 =====
                # 缓存 mask 形状是 (Nx+1, Ny+1)，对应 StaggeredGrid 节点
                # 可视化时绘制以节点为中心的单元格，更准确地表示节点位置
                patches = []
                
                for i in range(cache_Nx):
                    for j in range(cache_Ny):
                        if mask_np[i, j]:
                            # 绘制以节点为中心的单元格
                            # 节点坐标
                            node_x = i * dx
                            node_y = j * dy
                            # 单元格以节点为中心，大小为 dx × dy
                            cell_x = node_x - dx / 2
                            cell_y = node_y - dy / 2
                            
                            rect = mpatches.Rectangle(
                                (cell_x, cell_y), dx, dy,
                                facecolor=color,
                                edgecolor='darkred',
                                alpha=0.5,
                                linewidth=0.5,
                            )
                            patches.append(rect)
                
                # 添加到图中
                for patch in patches:
                    ax.add_patch(patch)

    def _compute_reference_length(self, inflow_vec: List[float]) -> float:
        """
        计算参考长度：结构物在垂直于来流方向上的投影长度
        
        原理:
        - 来流方向单位向量: d = (Ux, Uy) / |U|
        - 垂直方向单位向量: n = (-Uy, Ux) / |U| （逆时针旋转90度）
        - 投影长度 = 结构物在 n 方向上的宽度
        
        对于旋转后的矩形:
        - 矩形四个顶点相对中心的局部坐标（旋转后）
        - 将顶点投影到垂直方向，取最大 - 最小
        
        Args:
            inflow_vec: 来流速度向量 [Ux, Uy]
        
        Returns:
            参考长度 L_ref
        """
        solver = self.solver
        obs = solver.cfg.obstacle
        
        # 计算来流方向和垂直方向
        U_mag = np.sqrt(inflow_vec[0]**2 + inflow_vec[1]**2)
        if U_mag > 0:
            # 垂直方向单位向量（逆时针旋转90度）
            perp_x, perp_y = -inflow_vec[1] / U_mag, inflow_vec[0] / U_mag
        else:
            # 默认垂直于 x 方向，即 y 方向
            perp_x, perp_y = 0.0, 1.0
        
        shape_type = obs.shape.lower()
        
        if shape_type == 'sphere':
            # 圆柱：任意方向投影都是直径
            return 2.0 * obs.radius
        
        elif shape_type == 'box':
            # 方柱：考虑旋转后的顶点投影
            sx, sy = obs.size[0], obs.size[1]
            theta = obs.rotation  # 旋转角度（弧度）
            
            # 旋转矩阵: R = [[cos, -sin], [sin, cos]]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            # 矩形四个顶点相对中心的局部坐标（未旋转）
            # 右上、左上、左下、右下
            half_sx, half_sy = sx / 2, sy / 2
            corners_local = [
                (half_sx, half_sy),
                (-half_sx, half_sy),
                (-half_sx, -half_sy),
                (half_sx, -half_sy),
            ]
            
            # 旋转后的顶点，并投影到垂直方向
            projections = []
            for lx, ly in corners_local:
                # 旋转变换
                rx = lx * cos_t - ly * sin_t
                ry = lx * sin_t + ly * cos_t
                # 投影到垂直方向: proj = (rx, ry) · (perp_x, perp_y)
                proj = rx * perp_x + ry * perp_y
                projections.append(proj)
            
            # 投影长度 = max - min
            L_ref = max(projections) - min(projections)
            return L_ref
        
        else:
            # 其他形状：尝试获取 size 或 radius
            if hasattr(obs, 'radius') and obs.radius:
                return 2.0 * obs.radius
            elif hasattr(obs, 'size') and obs.size:
                # 保守估计：对角线长度
                sx, sy = obs.size[0], obs.size[1]
                return np.sqrt(sx**2 + sy**2)
            else:
                LOGGER.warning("[Runner] 无法确定障碍物尺寸，使用默认参考长度 1.0")
                return 1.0
