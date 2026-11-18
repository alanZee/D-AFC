#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超参数优化收尾脚本

功能：
    为已完成迭代但未生成可视化结果的超参数优化案例补充收尾功能
    
    1. 从 checkpoint.json 恢复优化状态
    2. 从 trials 目录重新计算缺失的多目标指标（val_loss, loss_stability 等）
    3. 调用 GeneticOptimizer.finalize() 执行统一的收尾流程：
       - 生成 all_experiments.json, best_hyperparams.yaml, experiments_summary.csv
       - 生成 optimization_results.png, pareto_front_*.png

使用方法：
    python finalize_hyperopt.py --dir <hyperopt结果目录>
    
    例如：
    python finalize_hyperopt.py --dir path/to/result

注意事项：
    收尾功能的核心实现在 GeneticOptimizer.finalize() 方法中，本脚本仅负责：
    - 恢复优化器状态
    - 重新计算缺失指标
    - 调用统一的 finalize() 方法
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict

# 添加路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent.parent))

# 从主脚本导入配置和优化器类
from myafc.scripts.afc_supervision.hyperopt.hyperopt_genetic import (
    HyperoptConfig, parse_hyperopt_config, GeneticOptimizer
)


def compute_trial_metrics(trial_dir: Path, stability_weights: List[float] = None) -> Dict:
    """从 trial 目录计算训练指标
    
    用于重新计算旧 checkpoint 中缺失的多目标指标。
    
    Args:
        trial_dir: trial 目录路径（如 gen000_ind000/）
        stability_weights: 稳定性指标权重 [CV权重, (1-monotonicity)权重]，默认 [0.7, 0.3]
    
    Returns:
        包含 val_loss, loss_stability 等指标的字典
        
    稳定性指标设计说明:
        combined_stability = w_cv * CV + w_mono * (1 - monotonicity)
        - CV (变异系数) = std(loss) / mean(loss)，衡量波动幅度
        - monotonicity = count(下降步) / 总步数，衡量收敛趋势
        - 越小越好
    """
    if stability_weights is None:
        stability_weights = [0.7, 0.3]
    
    metrics = {
        'val_loss': None,
        'validation_loss': None,
        'loss_stability': None,
        'monotonicity': None,
        'primary_loss_min': None,
        'primary_loss_final': None,
        'reg_loss_min': 0.0,
        'reg_loss_final': 0.0,
    }
    
    # 1. 尝试从 training_data.npz 加载 loss 历史
    npz_path = trial_dir / 'training_data.npz'
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # 获取 primary_loss_history
            primary_loss_history = None
            if 'primary_loss_history' in data:
                primary_loss_history = data['primary_loss_history'].tolist()
            elif 'loss_history' in data:
                primary_loss_history = data['loss_history'].tolist()
            
            if primary_loss_history and len(primary_loss_history) >= 3:
                # 计算变异系数 (CV)
                loss_mean = np.mean(primary_loss_history)
                loss_std = np.std(primary_loss_history)
                cv = loss_std / (loss_mean + 1e-6)
                
                # 计算单调性
                diffs = np.diff(primary_loss_history)
                monotonicity = np.sum(diffs < 0) / len(diffs) if len(diffs) > 0 else 0.0
                metrics['monotonicity'] = monotonicity
                
                # 加权整合稳定性指标
                w_cv, w_mono = stability_weights[:2] if len(stability_weights) >= 2 else (0.7, 0.3)
                metrics['loss_stability'] = w_cv * cv + w_mono * (1 - monotonicity)
                
                # 最小和最终 loss
                metrics['primary_loss_min'] = min(primary_loss_history)
                metrics['primary_loss_final'] = primary_loss_history[-1]
            
            # 获取 reg_loss
            if 'reg_loss_history' in data:
                reg_loss_history = data['reg_loss_history'].tolist()
                if reg_loss_history:
                    metrics['reg_loss_min'] = min(reg_loss_history)
                    metrics['reg_loss_final'] = reg_loss_history[-1]
            
            # 获取 validation_loss
            if 'validation_loss' in data:
                val_loss = float(data['validation_loss'])
                if np.isfinite(val_loss):
                    metrics['validation_loss'] = val_loss
                    metrics['val_loss'] = val_loss
                    
        except Exception as e:
            print(f"  警告: 加载 {npz_path} 失败: {e}")
    
    # 2. 如果没有 validation_loss，尝试从 validation 目录获取
    if metrics['val_loss'] is None:
        val_data_path = trial_dir / 'validation' / 'validation_data.npz'
        if val_data_path.exists():
            try:
                val_data = np.load(val_data_path, allow_pickle=True)
                if 'validation_loss' in val_data:
                    val_loss = float(val_data['validation_loss'])
                    if np.isfinite(val_loss):
                        metrics['validation_loss'] = val_loss
                        metrics['val_loss'] = val_loss
            except Exception:
                pass
    
    # 3. 如果还没有 val_loss，使用 primary_loss_min 作为替代
    if metrics['val_loss'] is None and metrics['primary_loss_min'] is not None:
        metrics['val_loss'] = metrics['primary_loss_min']
    
    return metrics


def restore_optimizer_from_checkpoint(result_dir: Path, recompute_metrics: bool = True) -> GeneticOptimizer:
    """从 checkpoint 恢复优化器状态
    
    Args:
        result_dir: hyperopt 结果目录（包含 checkpoint.json）
        recompute_metrics: 是否从 trials 目录重新计算缺失的多目标指标
        
    Returns:
        恢复状态后的 GeneticOptimizer 实例
    """
    result_dir = Path(result_dir)
    checkpoint_path = result_dir / 'checkpoint.json'
    config_path = result_dir / 'hyperopt_config.yaml'
    trials_dir = result_dir / 'trials'
    
    # 检查必要文件
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint.json: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"未找到 hyperopt_config.yaml: {config_path}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    config = parse_hyperopt_config(cfg_dict)
    
    # 创建优化器实例
    optimizer = GeneticOptimizer(config, resume=True)
    optimizer.output_dir = result_dir
    optimizer.checkpoint_path = checkpoint_path
    
    # 从 checkpoint 加载状态
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    optimizer.population = checkpoint['population']
    optimizer.fitness_history = checkpoint['fitness_history']
    optimizer.best_individual = checkpoint['best_individual']
    optimizer.best_fitness = checkpoint['best_fitness'] if checkpoint['best_fitness'] is not None else float('inf')
    
    # 恢复实验记录
    optimizer.all_experiments = []
    metrics_recomputed_count = 0
    
    for exp in checkpoint.get('all_experiments', []):
        restored_exp = {
            'generation': exp['generation'],
            'individual_idx': exp['individual_idx'],
            'hyperparams': exp['hyperparams'],
            'fitness': exp['fitness'] if exp['fitness'] is not None else float('inf'),
            'val_loss': exp.get('val_loss'),
            'validation_loss': exp.get('validation_loss'),
            'loss_stability': exp.get('loss_stability'),
            'monotonicity': exp.get('monotonicity'),
            'primary_loss_min': exp.get('primary_loss_min'),
            'primary_loss_final': exp.get('primary_loss_final'),
            'reg_loss_min': exp.get('reg_loss_min', 0.0),
            'reg_loss_final': exp.get('reg_loss_final', 0.0),
            'objective_values': exp.get('objective_values', {}),
            'loss_history': exp.get('loss_history', []),
            'amplitude_history': exp.get('amplitude_history', []),
        }
        
        # 如果启用指标重算且关键字段缺失，从 trial 目录加载
        if recompute_metrics:
            needs_recompute = (
                restored_exp['loss_stability'] is None or 
                restored_exp['val_loss'] is None or
                restored_exp['monotonicity'] is None
            )
            if needs_recompute and trials_dir.exists():
                trial_name = f"gen{exp['generation']:03d}_ind{exp['individual_idx']:03d}"
                trial_dir = trials_dir / trial_name
                
                if trial_dir.exists():
                    computed = compute_trial_metrics(trial_dir, config.stability_weights)
                    
                    # 填充缺失字段
                    for key, value in computed.items():
                        if restored_exp.get(key) is None and value is not None:
                            restored_exp[key] = value
                    
                    # 更新 objective_values
                    if not restored_exp['objective_values']:
                        restored_exp['objective_values'] = {}
                    for obj_name in config.objectives:
                        if obj_name not in restored_exp['objective_values']:
                            restored_exp['objective_values'][obj_name] = restored_exp.get(obj_name)
                    
                    metrics_recomputed_count += 1
        
        optimizer.all_experiments.append(restored_exp)
    
    if metrics_recomputed_count > 0:
        print(f"从 trials 目录重新计算指标: {metrics_recomputed_count} 个实验")
    
    return optimizer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='超参数优化收尾脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python finalize_hyperopt.py --dir path/to/result
    python finalize_hyperopt.py --dir /absolute/path/to/result --skip-pareto
        """
    )
    parser.add_argument('--dir', type=str, required=True,
                        help='超参数优化结果目录（包含 checkpoint.json）')
    parser.add_argument('--skip-pareto', action='store_true',
                        help='跳过帕累托前沿可视化')
    
    args = parser.parse_args()
    
    # 解析结果目录路径
    result_dir = Path(args.dir)
    if not result_dir.is_absolute():
        candidates = [
            Path.cwd() / result_dir,
            SCRIPT_DIR.parent.parent.parent.parent.parent / 'examples' / result_dir,
        ]
        for candidate in candidates:
            if candidate.exists():
                result_dir = candidate
                break
    
    if not result_dir.exists():
        print(f"错误: 结果目录不存在: {result_dir}")
        return 1
    
    # 恢复优化器状态
    print(f"{'='*60}")
    print(f"收尾脚本: 从断点恢复并生成结果")
    print(f"{'='*60}")
    print(f"结果目录: {result_dir}")
    
    try:
        optimizer = restore_optimizer_from_checkpoint(result_dir, recompute_metrics=True)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1
    
    print(f"恢复实验数: {len(optimizer.all_experiments)}")
    print(f"历史最佳适应度: {optimizer.best_fitness:.6f}")
    if optimizer.best_individual:
        print(f"最佳超参数:")
        for k, v in optimizer.best_individual.items():
            print(f"  {k}: {v}")
    
    # 调用统一的收尾方法
    success = optimizer.finalize(remove_checkpoint=False, skip_pareto=args.skip_pareto)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
