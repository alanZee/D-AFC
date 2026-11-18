"""
SolverRunner 最小使用示例

演示如何使用 KarmanTorchSolver + SolverRunner 运行圆柱绕流仿真

用法:
    python run_karman_with_runner.py [config.yaml] [--num_steps N]
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# 修复 OpenMP 运行时库重复加载问题（NumPy/PyTorch/MKL 冲突）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 抑制 C 库级别的警告（libpng iCCP 警告）
os.environ['QT_LOGGING_RULES'] = '*=false'

# 抑制 Python 警告
warnings.filterwarnings('ignore', category=UserWarning)

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# 设置 matplotlib 后端（避免 libpng 警告）
import matplotlib
matplotlib.use('Agg')

from myafc.env_kernels.solvers.karman_torch_solver import KarmanTorchSolver
from myafc.env_kernels.solver_wrappers import SolverRunner


def parse_args():
    parser = argparse.ArgumentParser(description='运行 Karman 涡街仿真')
    parser.add_argument('config', nargs='?', default=None,
                        help='配置文件路径（默认使用 karman_torch.yaml）')
    parser.add_argument('--num_steps', '-n', type=int, default=None,
                        help='仿真步数（覆盖配置文件中的 num_steps）')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 配置文件路径
    if args.config:
        cfg_path = Path(args.config)
    else:
        cfg_path = Path(__file__).parent.parent / 'src/myafc/configs/cfgs4env/karman_torch.yaml'
    
    # 创建求解器
    solver = KarmanTorchSolver(str(cfg_path))
    
    # 创建 Runner
    runner = SolverRunner(solver, str(cfg_path))
    
    # 运行仿真（支持命令行覆盖 num_steps）
    result = runner.run(
        num_steps=args.num_steps,  # None 时从配置文件读取
    )

    # 输出结果
    print(f"\n☑️   完成! 当前步: {solver.current_step}, 时间: {solver.current_time:.4f}")
    print(f"力系数记录数: {len(runner.force_coeffs)}")
    print(f"探针数据: {list(runner.probes.keys())}")
    print(f"场数据缓存数: {len(runner.fields)}")


if __name__ == '__main__':
    main()
