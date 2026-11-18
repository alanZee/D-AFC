#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用超参数优化脚本（基于遗传算法）

参考 YOLOv5 的超参数进化方法。

功能：
1. 支持通过 YAML 配置选择待调参数及其取值范围
2. 使用遗传算法进行超参数搜索
3. 自动记录所有实验结果
4. 提供可视化分析

使用方法：
    # 正常运行
    python hyperopt_genetic.py --config hyperopt_config.yaml
    python hyperopt_genetic.py --config hyperopt_config.yaml --generations 20 --population 10
    # 从断点恢复
    python hyperopt_genetic.py --config hyperopt_config.yaml --resume
"""

# ============================================================
# 解决 OpenMP 运行时库冲突问题
# 必须在任何其他 import 之前设置，否则无效
# 原因：PyTorch、NumPy、MKL 等可能各自链接了 libiomp5md.dll
# ============================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import yaml
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from copy import deepcopy

# 添加路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent.parent))


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class HyperparamRange:
    """单个超参数的搜索范围
    
    Attributes:
        name: 参数名（支持嵌套格式，如 "training.learning_rate"）
        min: 搜索范围最小值
        max: 搜索范围最大值
        log_scale: 是否使用对数缩放（适合学习率等跨多个数量级的参数）
        discrete: 是否离散化（取整）
        step: 离散步长（仅 discrete=True 时有效）
    """
    name: str
    min: float
    max: float
    log_scale: bool = False
    discrete: bool = False
    step: Optional[float] = None


@dataclass
class HyperoptConfig:
    """超参数优化配置"""
    # 基础训练配置文件路径
    base_config_path: str = ""
    
    # 遗传算法参数
    population_size: int = 10          # 种群大小
    generations: int = 20              # 进化代数
    mutation_rate: float = 0.2         # 变异概率
    mutation_scale: float = 0.3        # 变异幅度（相对于范围）
    crossover_rate: float = 0.8        # 交叉概率
    elite_ratio: float = 0.2           # 精英保留比例
    
    # 评估参数
    num_interactions: int = 10         # 每次评估的交互次数（控制训练长度）
    
    # ============================================================
    # 多目标优化配置
    # ============================================================
    # 优化目标列表（都是最小化）
    # 可选: "val_loss", "loss_stability", "final_loss", "primary_loss_min", "monotonicity"
    objectives: List[str] = field(default_factory=lambda: ["val_loss", "loss_stability"])
    
    # 目标权重（加权求和计算综合适应度）
    objective_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # ============================================================
    # 稳定性指标配置
    # ============================================================
    # loss_stability 指标设计说明：
    #   combined_stability = w_cv * CV + w_mono * (1 - monotonicity)
    #   其中:
    #     - CV (变异系数) = std(loss) / (mean(loss) + 1e-6)，衡量波动幅度
    #     - monotonicity = count(下降步) / 总步数，衡量收敛趋势
    #     - w_cv, w_mono 为加权权重（默认 0.7, 0.3）
    #   越小表示训练越稳定
    # 
    # stability_weights: [CV权重, (1-monotonicity)权重]
    stability_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])

    
    # 待优化参数
    hyperparams: List[HyperparamRange] = field(default_factory=list)
    
    # 输出
    output_dir: str = "runs/hyperopt"
    experiment_name: str = "genetic_opt"


def parse_hyperopt_config(cfg_dict: Dict) -> HyperoptConfig:
    """解析超参数优化配置"""
    config = HyperoptConfig()
    
    # 基础参数
    config.base_config_path = cfg_dict.get('base_config_path', '')
    config.population_size = cfg_dict.get('population_size', 10)
    config.generations = cfg_dict.get('generations', 20)
    config.mutation_rate = cfg_dict.get('mutation_rate', 0.2)
    config.mutation_scale = cfg_dict.get('mutation_scale', 0.3)
    config.crossover_rate = cfg_dict.get('crossover_rate', 0.8)
    config.elite_ratio = cfg_dict.get('elite_ratio', 0.2)
    config.num_interactions = cfg_dict.get('num_interactions', 10)
    config.output_dir = cfg_dict.get('output_dir', 'runs/hyperopt')
    config.experiment_name = cfg_dict.get('experiment_name', 'genetic_opt')
    
    # 多目标优化配置
    config.objectives = cfg_dict.get('objectives', ['val_loss'])
    config.objective_weights = cfg_dict.get('objective_weights', [1.0] * len(config.objectives))
    
    # 稳定性权重配置: [CV权重, (1-monotonicity)权重]
    config.stability_weights = cfg_dict.get('stability_weights', [0.7, 0.3])
    
    # 解析超参数范围
    # 优先使用 min/max 字段，兼容旧格式 min_val/max_val（用于读取旧 checkpoint）
    hyperparams = []
    for hp in cfg_dict.get('hyperparams', []):
        min_val = hp.get('min', hp.get('min_val'))
        max_val = hp.get('max', hp.get('max_val'))
        if min_val is None or max_val is None:
            raise ValueError(f"超参数 {hp.get('name', 'unknown')} 缺少 min/max 配置")
        hyperparams.append(HyperparamRange(
            name=hp['name'],
            min=float(min_val),
            max=float(max_val),
            log_scale=hp.get('log_scale', False),
            discrete=hp.get('discrete', False),
            step=hp.get('step', None),
        ))
    config.hyperparams = hyperparams
    
    return config


# ============================================================
# 遗传算法核心
# ============================================================
class GeneticOptimizer:
    """遗传算法超参数优化器"""
    
    def __init__(self, config: HyperoptConfig, resume: bool = False):
        self.config = config
        self.population: List[Dict[str, float]] = []
        self.fitness_history: List[List[float]] = []  # 每代的适应度
        self.best_individual: Optional[Dict[str, float]] = None
        self.best_fitness: float = float('inf')
        
        # 实验记录
        self.all_experiments: List[Dict] = []
        
        # 断点续训状态
        self.current_generation: int = 0
        self.current_individual_idx: int = 0
        self.resume_mode: bool = resume
        
        # 输出目录
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # checkpoint 文件路径
        self.checkpoint_path = self.output_dir / 'checkpoint.json'
        
        # 尝试恢复状态
        if resume and self.checkpoint_path.exists():
            self._load_checkpoint()
        else:
            # 保存配置
            with open(self.output_dir / 'hyperopt_config.yaml', 'w') as f:
                yaml.safe_dump(asdict(config), f, sort_keys=False)
    
    def _save_checkpoint(self, gen: int, ind_idx: int, current_fitnesses: List[float] = None):
        """保存断点状态（包含完整的多目标优化字段）"""
        checkpoint = {
            'generation': gen,
            'individual_idx': ind_idx,
            'population': self.population,
            'fitness_history': self.fitness_history,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness if np.isfinite(self.best_fitness) else None,
            'all_experiments': [
                {
                    'generation': exp['generation'],
                    'individual_idx': exp['individual_idx'],
                    'hyperparams': {k: float(v) for k, v in exp['hyperparams'].items()},
                    'fitness': float(exp['fitness']) if np.isfinite(exp['fitness']) else None,
                    # 多目标优化相关字段
                    'val_loss': float(exp['val_loss']) if exp.get('val_loss') is not None and np.isfinite(exp['val_loss']) else None,
                    'loss_stability': float(exp['loss_stability']) if exp.get('loss_stability') is not None and np.isfinite(exp['loss_stability']) else None,
                    'monotonicity': float(exp['monotonicity']) if exp.get('monotonicity') is not None else None,
                    'primary_loss_min': float(exp['primary_loss_min']) if exp.get('primary_loss_min') is not None and np.isfinite(exp['primary_loss_min']) else None,
                    'primary_loss_final': float(exp['primary_loss_final']) if exp.get('primary_loss_final') is not None and np.isfinite(exp['primary_loss_final']) else None,
                    'reg_loss_min': float(exp.get('reg_loss_min', 0.0)),
                    'reg_loss_final': float(exp.get('reg_loss_final', 0.0)),
                    'validation_loss': float(exp['validation_loss']) if exp.get('validation_loss') is not None and np.isfinite(exp['validation_loss']) else None,
                }
                for exp in self.all_experiments
            ],
            'current_fitnesses': current_fitnesses,
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"[Checkpoint] 已保存: gen={gen}, ind={ind_idx}")
    
    def _load_checkpoint(self):
        """从断点恢复状态"""
        print(f"[Resume] 从断点恢复: {self.checkpoint_path}")
        with open(self.checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        self.current_generation = checkpoint['generation']
        self.current_individual_idx = checkpoint['individual_idx']
        self.population = checkpoint['population']
        self.fitness_history = checkpoint['fitness_history']
        self.best_individual = checkpoint['best_individual']
        self.best_fitness = checkpoint['best_fitness'] if checkpoint['best_fitness'] is not None else float('inf')
        
        # 恢复实验记录
        self.all_experiments = []
        for exp in checkpoint.get('all_experiments', []):
            self.all_experiments.append({
                'generation': exp['generation'],
                'individual_idx': exp['individual_idx'],
                'hyperparams': exp['hyperparams'],
                'fitness': exp['fitness'] if exp['fitness'] is not None else float('inf'),
                'loss_history': [],  # 恢复时无法获取详细历史
                'amplitude_history': [],
            })
        
        self._current_fitnesses = checkpoint.get('current_fitnesses', [])
        
        print(f"[Resume] 恢复到: Generation {self.current_generation + 1}, Individual {self.current_individual_idx + 1}")
        print(f"[Resume] 历史最佳: {self.best_fitness:.6f}")
    
    def _sample_value(self, hp: HyperparamRange) -> float:
        """从参数范围中采样一个值"""
        if hp.log_scale:
            # 对数缩放：在 log 空间均匀采样
            log_min = np.log(hp.min)
            log_max = np.log(hp.max)
            value = np.exp(random.uniform(log_min, log_max))
        else:
            # 线性缩放
            value = random.uniform(hp.min, hp.max)
        
        # 离散化
        if hp.discrete:
            if hp.step:
                value = round(value / hp.step) * hp.step
            else:
                value = round(value)
        
        # 确保在范围内
        value = max(hp.min, min(hp.max, value))
        return value
    
    def _mutate_value(self, value: float, hp: HyperparamRange) -> float:
        """变异单个值"""
        scale = self.config.mutation_scale
        
        if hp.log_scale:
            # 对数空间变异
            log_val = np.log(value)
            log_range = np.log(hp.max) - np.log(hp.min)
            log_val += random.gauss(0, scale * log_range)
            value = np.exp(log_val)
        else:
            # 线性空间变异
            range_size = hp.max - hp.min
            value += random.gauss(0, scale * range_size)
        
        # 离散化和边界处理
        if hp.discrete:
            if hp.step:
                value = round(value / hp.step) * hp.step
            else:
                value = round(value)
        
        value = max(hp.min, min(hp.max, value))
        return value
    
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.config.population_size):
            individual = {}
            for hp in self.config.hyperparams:
                individual[hp.name] = self._sample_value(hp)
            # 应用参数约束
            individual = self._apply_constraints(individual)
            self.population.append(individual)
        print(f"[GeneticOpt] 初始化种群: {len(self.population)} 个个体")
    
    def _apply_constraints(self, individual: Dict[str, float]) -> Dict[str, float]:
        """应用超参数约束，确保参数之间的关系合理
        
        约束规则：
        1. control_interval >= action_duration（若 control_interval 被优化）
        
        修复策略：向上调整被违反的参数
        
        注意：num_of_interaction_per_agent_update 是独立参数，不需要约束
        """
        # 获取参数范围信息
        hp_ranges = {hp.name: hp for hp in self.config.hyperparams}
        
        # 约束 1: control_interval >= action_duration
        if 'training.action_duration' in individual and 'training.control_interval' in individual:
            action_dur = individual['training.action_duration']
            control_int = individual['training.control_interval']
            if control_int < action_dur:
                # 向上调整 control_interval
                individual['training.control_interval'] = action_dur
        
        # 确保离散参数是整数
        for hp_name, value in individual.items():
            if hp_name in hp_ranges and hp_ranges[hp_name].discrete:
                individual[hp_name] = round(value)
        
        return individual
    
    def evaluate_individual(self, individual: Dict[str, float], gen: int, idx: int) -> float:
        """评估单个个体的适应度（运行训练）
        
        适应度计算：
        - 计算多个目标值，使用加权求和得到综合适应度
        
        支持的目标：
        - val_loss: 验证 loss（泛化能力）
        - loss_stability: 训练 loss 稳定性（标准差越小越稳定）
        - final_loss: 训练最终 loss
        - primary_loss_min: 训练最小 loss
        """
        from myafc.afc_supervision.engines.simple_trainer import SimpleTrainer
        
        # 加载基础配置
        base_config_path = self.config.base_config_path
        if not Path(base_config_path).is_absolute():
            base_config_path = SCRIPT_DIR / base_config_path
        
        with open(base_config_path, 'r', encoding='utf-8') as f:
            train_config = yaml.safe_load(f)
        
        # 应用超参数
        for hp_name, hp_value in individual.items():
            self._set_nested_value(train_config, hp_name, hp_value)
        
        # 固定训练参数
        train_config['training']['num_interaction_per_mini_episode'] = self.config.num_interactions
        # 启用验证（用于计算 fitness）
        if 'validation_steps' not in train_config['training'] or train_config['training']['validation_steps'] == 0:
            train_config['training']['validation_steps'] = 50  # 默认验证步数
        train_config['output']['project'] = str(self.output_dir / 'trials')
        train_config['output']['name'] = f'gen{gen:03d}_ind{idx:03d}'
        
        # 运行训练
        try:
            trainer = SimpleTrainer(train_config)
            result = trainer.train()
            
            # 获取训练和验证数据
            primary_loss_history = result.get('primary_loss_history', trainer.primary_loss_history)
            reg_loss_history = result.get('reg_loss_history', trainer.reg_loss_history)
            loss_history = result.get('loss_history', trainer.loss_history)  # total_loss，向后兼容
            validation_loss = result.get('validation_loss', None)  # 验证 loss
            
            if not primary_loss_history:
                # 向后兼容：如果没有 primary_loss_history，使用 loss_history
                primary_loss_history = loss_history
            
            # ============================================================
            # 计算各类指标
            # ============================================================
            
            # 1. 验证 loss（泛化能力）
            val_loss = validation_loss if validation_loss is not None and not np.isnan(validation_loss) else float('inf')
            
            # 2. 训练 loss 稳定性指标
            # ============================================================
            # 设计说明：
            #   combined_stability = w_cv * CV + w_mono * (1 - monotonicity)
            #   其中：
            #     - CV (变异系数) = std(loss) / (mean(loss) + 1e-6)
            #       衡量波动幅度，越小表示训练曲线越平稳
            #     - monotonicity = count(下降步) / 总步数
            #       衡量收敛趋势，越高表示 loss 越单调下降
            #     - (1 - monotonicity) 使其可最小化
            #   权重默认: w_cv=0.7, w_mono=0.3 (CV 更能反映波动稳定性)
            # ============================================================
            if primary_loss_history and len(primary_loss_history) >= 3:
                # 计算变异系数 (CV)
                loss_mean = np.mean(primary_loss_history)
                loss_std = np.std(primary_loss_history)
                cv = loss_std / (loss_mean + 1e-6)
                
                # 计算单调性
                diffs = np.diff(primary_loss_history)
                monotonicity = np.sum(diffs < 0) / len(diffs) if len(diffs) > 0 else 0.0
                
                # 加权整合稳定性指标
                w_cv, w_mono = self.config.stability_weights[:2] if len(self.config.stability_weights) >= 2 else (0.7, 0.3)
                loss_stability = w_cv * cv + w_mono * (1 - monotonicity)
            else:
                loss_stability = float('inf')
                monotonicity = 0.0
                cv = float('inf')
            
            # 3. 最终和最小训练 loss
            final_loss = primary_loss_history[-1] if primary_loss_history else float('inf')
            primary_loss_min = min(primary_loss_history) if primary_loss_history else float('inf')
            
            # 4. 正则化损失统计
            reg_loss_min = min(reg_loss_history) if reg_loss_history else 0.0
            reg_loss_final = reg_loss_history[-1] if reg_loss_history else 0.0
            
            # ============================================================
            # 计算适应度
            # ============================================================
            
            # 多目标模式：计算各目标值，加权求和
            objective_values = {}
            for obj_name in self.config.objectives:
                if obj_name == 'val_loss':
                    objective_values[obj_name] = val_loss if np.isfinite(val_loss) else primary_loss_min
                elif obj_name == 'loss_stability':
                    objective_values[obj_name] = loss_stability
                elif obj_name == 'final_loss':
                    objective_values[obj_name] = final_loss
                elif obj_name == 'primary_loss_min':
                    objective_values[obj_name] = primary_loss_min
                elif obj_name == 'monotonicity':
                    # 单调性越高越好，取负数使其可最小化
                    objective_values[obj_name] = -monotonicity
                else:
                    objective_values[obj_name] = float('inf')
            
            # 加权求和（所有目标都是最小化）
            weights = self.config.objective_weights
            if len(weights) < len(self.config.objectives):
                weights = weights + [1.0] * (len(self.config.objectives) - len(weights))
            
            fitness = 0.0
            for i, obj_name in enumerate(self.config.objectives):
                fitness += weights[i] * objective_values.get(obj_name, 0.0)
            
            # 记录实验（增加多目标相关字段）
            experiment = {
                'generation': gen,
                'individual_idx': idx,
                'hyperparams': individual.copy(),
                'fitness': fitness,  # 综合适应度
                'validation_loss': validation_loss,
                'primary_loss_history': primary_loss_history,
                'reg_loss_history': reg_loss_history,
                'loss_history': loss_history,
                'amplitude_history': trainer.amplitude_history,
                # 各目标值
                'val_loss': val_loss if np.isfinite(val_loss) else None,
                'loss_stability': loss_stability if np.isfinite(loss_stability) else None,
                'monotonicity': monotonicity,
                'primary_loss_min': primary_loss_min,
                'primary_loss_final': final_loss,
                'reg_loss_min': reg_loss_min,
                'reg_loss_final': reg_loss_final,
                # 多目标优化相关
                'objective_values': objective_values if objective_values else None,
            }
            self.all_experiments.append(experiment)
            
            return fitness
            
        except Exception as e:
            print(f"[GeneticOpt] 个体评估失败: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """设置嵌套配置值（支持 "training.learning_rate" 格式）"""
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        # 转换 numpy 类型为 Python 原生类型
        if hasattr(value, 'item'):
            value = value.item()
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.integer):
            value = int(value)
        d[keys[-1]] = value
    
    def select_parents(self, fitnesses: List[float]) -> List[int]:
        """锦标赛选择"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(self.population)):
            # 随机选择 tournament_size 个个体
            candidates = random.sample(range(len(self.population)), min(tournament_size, len(self.population)))
            # 选择适应度最好的
            best = min(candidates, key=lambda i: fitnesses[i])
            selected.append(best)
        
        return selected
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """交叉操作"""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        # 交叉后应用约束
        child1 = self._apply_constraints(child1)
        child2 = self._apply_constraints(child2)
        
        return child1, child2
    
    def mutate(self, individual: Dict) -> Dict:
        """变异操作"""
        mutated = individual.copy()
        
        for hp in self.config.hyperparams:
            if random.random() < self.config.mutation_rate:
                mutated[hp.name] = self._mutate_value(mutated[hp.name], hp)
        
        # 变异后应用约束
        mutated = self._apply_constraints(mutated)
        
        return mutated
    
    def evolve(self):
        """执行遗传算法优化"""
        print(f"\n{'='*60}")
        print(f"开始遗传算法超参数优化")
        print(f"  种群大小: {self.config.population_size}")
        print(f"  进化代数: {self.config.generations}")
        print(f"  待优化参数: {[hp.name for hp in self.config.hyperparams]}")
        if self.resume_mode and hasattr(self, '_current_fitnesses'):
            print(f"  [Resume模式] 从 Gen {self.current_generation + 1}, Ind {self.current_individual_idx + 1} 继续")
        print(f"{'='*60}\n")
        
        # 初始化或恢复种群
        if not self.population:
            self.initialize_population()
        
        # 确定起始代数
        start_gen = self.current_generation if self.resume_mode else 0
        
        for gen in range(start_gen, self.config.generations):
            print(f"\n[Generation {gen+1}/{self.config.generations}]")
            
            # 评估所有个体
            if self.resume_mode and gen == start_gen and hasattr(self, '_current_fitnesses') and self._current_fitnesses:
                # 恢复模式：继续当前代的评估
                fitnesses = self._current_fitnesses.copy()
                start_idx = self.current_individual_idx
                self.resume_mode = False  # 恢复后切换为正常模式
            else:
                fitnesses = []
                start_idx = 0
            
            for idx in range(start_idx, len(self.population)):
                individual = self.population[idx]
                print(f"  评估个体 {idx+1}/{len(self.population)}: {individual}")
                fitness = self.evaluate_individual(individual, gen, idx)
                fitnesses.append(fitness)
                print(f"    -> 适应度: {fitness:.6f}")
                
                # 每个个体评估后保存断点
                self._save_checkpoint(gen, idx + 1, fitnesses)
            
            # 记录本代适应度
            self.fitness_history.append(fitnesses.copy())
            
            # 更新最佳个体
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            print(f"  本代最佳: {min(fitnesses):.6f}, 历史最佳: {self.best_fitness:.6f}")
            
            # 最后一代不需要进化
            if gen == self.config.generations - 1:
                break
            
            # 选择、交叉、变异
            new_population = []
            
            # 精英保留
            elite_count = max(1, int(self.config.elite_ratio * len(self.population)))
            sorted_indices = np.argsort(fitnesses)
            for i in range(elite_count):
                new_population.append(self.population[sorted_indices[i]].copy())
            
            # 生成新个体
            selected = self.select_parents(fitnesses)
            while len(new_population) < self.config.population_size:
                p1, p2 = random.sample(selected, 2)
                c1, c2 = self.crossover(self.population[p1], self.population[p2])
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_population.extend([c1, c2])
            
            self.population = new_population[:self.config.population_size]
            
            # 代结束后保存断点
            self._save_checkpoint(gen + 1, 0, [])
        
        # 执行收尾功能
        self.finalize(remove_checkpoint=True)
        
        return self.best_individual, self.best_fitness
    
    def finalize(self, remove_checkpoint: bool = False, skip_pareto: bool = False):
        """执行收尾功能：保存结果、生成可视化
        
        统一的收尾入口，可被 evolve() 或外部脚本调用。
        
        Args:
            remove_checkpoint: 是否删除 checkpoint 文件
            skip_pareto: 是否跳过帕累托前沿可视化
            
        Returns:
            bool: 收尾是否成功
        """
        print(f"\n{'='*60}")
        print("执行收尾功能...")
        print(f"{'='*60}")
        
        success = True
        
        # 1. 保存实验结果
        try:
            print("[1/3] 保存实验结果...")
            self.save_results()
            print("  ✓ all_experiments.json")
            print("  ✓ best_hyperparams.yaml")
            print("  ✓ experiments_summary.csv")
        except Exception as e:
            print(f"  ✗ 保存结果失败: {e}")
            success = False
        
        # 2. 生成优化结果可视化
        try:
            print("[2/3] 生成优化结果可视化...")
            self.visualize_results()
            print("  ✓ optimization_results.png")
        except Exception as e:
            print(f"  ✗ 可视化失败: {e}")
            success = False
        
        # 3. 生成帕累托前沿图
        if not skip_pareto:
            try:
                print("[3/3] 生成帕累托前沿可视化...")
                self.visualize_pareto_front()
                print("  ✓ pareto_front_*.png")
            except Exception as e:
                print(f"  ✗ 帕累托前沿可视化失败: {e}")
                success = False
        else:
            print("[3/3] 跳过帕累托前沿可视化")
        
        # 删除或备份 checkpoint
        if remove_checkpoint and self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"\n[Checkpoint] 优化完成，已清理断点文件")
        elif self.checkpoint_path.exists():
            # 备份 checkpoint
            backup_path = self.output_dir / 'checkpoint_backup.json'
            if not backup_path.exists():
                import shutil
                shutil.copy(self.checkpoint_path, backup_path)
                print(f"\n已备份 checkpoint 到: {backup_path}")
        
        print(f"\n{'='*60}")
        print("收尾完成！" if success else "收尾完成（部分失败）")
        print(f"{'='*60}")
        
        return success
    
    def save_results(self):
        """保存实验结果"""
        # 保存所有实验记录
        results_path = self.output_dir / 'all_experiments.json'
        with open(results_path, 'w') as f:
            # 转换为可序列化格式
            serializable = []
            for exp in self.all_experiments:
                s = exp.copy()
                s['hyperparams'] = {k: float(v) for k, v in exp['hyperparams'].items()}
                s['fitness'] = float(exp['fitness']) if np.isfinite(exp['fitness']) else None
                # 处理 objective_values
                if 'objective_values' in s and s['objective_values']:
                    s['objective_values'] = {k: float(v) if np.isfinite(v) else None 
                                            for k, v in s['objective_values'].items()}
                serializable.append(s)
            json.dump(serializable, f, indent=2)
        
        # 保存最佳配置
        if self.best_individual:
            best_path = self.output_dir / 'best_hyperparams.yaml'
            with open(best_path, 'w') as f:
                yaml.safe_dump({
                    'best_fitness': float(self.best_fitness),
                    'best_hyperparams': {k: float(v) for k, v in self.best_individual.items()}
                }, f)
        
        # 保存 CSV 汇总（包含多目标优化相关字段）
        df_data = []
        for exp in self.all_experiments:
            row = {
                'generation': exp['generation'],
                'individual_idx': exp['individual_idx'],
                'fitness': exp['fitness'],  # 综合适应度
                'validation_loss': exp.get('validation_loss', None),
                # 新增：loss 稳定性指标
                'loss_stability': exp.get('loss_stability', None),
                'monotonicity': exp.get('monotonicity', None),
                # 训练 loss 分解字段
                'primary_loss_min': exp.get('primary_loss_min', None),
                'primary_loss_final': exp.get('primary_loss_final', None),
                'reg_loss_min': exp.get('reg_loss_min', 0.0),
                'reg_loss_final': exp.get('reg_loss_final', 0.0),
            }
            row.update(exp['hyperparams'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(self.output_dir / 'experiments_summary.csv', index=False)
        
        print(f"\n结果已保存到: {self.output_dir}")
    
    def visualize_results(self):
        """可视化优化结果"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 适应度进化曲线
        ax1 = axes[0, 0]
        generations = range(1, len(self.fitness_history) + 1)
        best_per_gen = [min(f) for f in self.fitness_history]
        mean_per_gen = [np.mean([x for x in f if np.isfinite(x)]) for f in self.fitness_history]
        
        ax1.plot(generations, best_per_gen, 'b-o', label='Best', linewidth=2)
        ax1.plot(generations, mean_per_gen, 'r--', label='Mean', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (Loss)')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 所有实验的适应度分布
        ax2 = axes[0, 1]
        all_fitness = [exp['fitness'] for exp in self.all_experiments if np.isfinite(exp['fitness'])]
        ax2.hist(all_fitness, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(self.best_fitness, color='r', linestyle='--', label=f'Best: {self.best_fitness:.4f}')
        ax2.set_xlabel('Fitness')
        ax2.set_ylabel('Count')
        ax2.set_title('Fitness Distribution')
        ax2.legend()
        
        # 3. 超参数重要性（基于与适应度的相关性）
        ax3 = axes[1, 0]
        hp_names = [hp.name for hp in self.config.hyperparams]
        correlations = []
        
        for hp_name in hp_names:
            hp_values = [exp['hyperparams'][hp_name] for exp in self.all_experiments]
            fitness_values = [exp['fitness'] for exp in self.all_experiments]
            # 过滤无效值
            valid = [(h, f) for h, f in zip(hp_values, fitness_values) if np.isfinite(f)]
            if len(valid) > 2:
                hp_v, fit_v = zip(*valid)
                # 检查标准差是否为 0（固定参数），避免 corrcoef 警告
                if np.std(hp_v) < 1e-10:
                    corr = 0.0  # 固定参数没有相关性可言
                else:
                    corr = np.corrcoef(hp_v, fit_v)[0, 1]
                    corr = abs(corr) if np.isfinite(corr) else 0
                correlations.append(corr)
            else:
                correlations.append(0)
        
        colors = ['green' if c > 0.3 else 'orange' if c > 0.1 else 'gray' for c in correlations]
        ax3.barh(hp_names, correlations, color=colors)
        ax3.set_xlabel('|Correlation with Fitness|')
        ax3.set_title('Hyperparameter Importance')
        ax3.set_xlim(0, 1)
        
        # 4. 最佳超参数值
        ax4 = axes[1, 1]
        if self.best_individual:
            names = list(self.best_individual.keys())
            values = list(self.best_individual.values())
            
            # 归一化显示（处理 min == max 的固定参数情况）
            normalized = []
            for hp in self.config.hyperparams:
                val = self.best_individual[hp.name]
                # 处理固定参数（min == max）的情况，避免除零
                if abs(hp.max - hp.min) < 1e-10:
                    norm = 0.5  # 固定参数显示在中间位置
                elif hp.log_scale:
                    log_range = np.log(hp.max) - np.log(hp.min)
                    norm = (np.log(val) - np.log(hp.min)) / log_range if log_range > 1e-10 else 0.5
                else:
                    norm = (val - hp.min) / (hp.max - hp.min)
                normalized.append(norm)
            
            ax4.barh(names, normalized, color='steelblue')
            ax4.set_xlabel('Normalized Value (0=min, 1=max)')
            ax4.set_title(f'Best Hyperparameters (Fitness: {self.best_fitness:.4f})')
            ax4.set_xlim(0, 1)
            
            # 添加实际值标注
            for i, (name, val) in enumerate(zip(names, values)):
                ax4.text(0.02, i, f'{val:.4g}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimization_results.png', dpi=150)
        plt.close()
        
        print(f"可视化结果已保存: {self.output_dir / 'optimization_results.png'}")
    
    def visualize_pareto_front(self, mode: str = 'auto'):
        """可视化帕累托前沿
        
        根据配置的优化目标自动选择绘制方式：
        - 多目标优化（objectives 包含 2+ 个目标）: 绘制配置中第一对目标的帕累托前沿
        - 单目标优化: 绘制 主损失 vs 正则化损失 的帕累托前沿
        
        Args:
            mode: 'auto' (推荐), 'multi_objective', 'loss_tradeoff'
        """
        import matplotlib.pyplot as plt
        
        # 准备有效数据
        valid_exps = [exp for exp in self.all_experiments if np.isfinite(exp['fitness'])]
        if len(valid_exps) < 3:
            print("有效实验数量不足，无法生成帕累托前沿图")
            return
        
        # 自动选择模式：根据 objectives 配置决定
        if mode == 'auto':
            if len(self.config.objectives) >= 2:
                # 多目标优化：使用配置中的目标对
                mode = 'multi_objective'
            else:
                # 单目标优化：使用 主损失 vs 正则化损失
                mode = 'loss_tradeoff'
        
        if mode == 'multi_objective':
            # ===== 多目标优化帕累托前沿 =====
            # 使用配置中的前两个目标
            obj_x = self.config.objectives[0] if len(self.config.objectives) > 0 else 'val_loss'
            obj_y = self.config.objectives[1] if len(self.config.objectives) > 1 else 'loss_stability'
            self._visualize_multi_objective_pareto(valid_exps, obj_x, obj_y)
            
        elif mode == 'loss_tradeoff':
            # ===== 主损失 vs 正则化损失 帕累托前沿 =====
            self._visualize_loss_tradeoff_pareto(valid_exps)
    
    def _get_objective_value(self, exp: Dict, obj_name: str) -> Optional[float]:
        """从实验记录中获取指定目标的值
        
        Args:
            exp: 实验记录字典
            obj_name: 目标名称，如 'val_loss', 'loss_stability' 等
        
        Returns:
            目标值，如果无效则返回 None
        """
        # 目标名到字段的映射
        value = None
        if obj_name == 'val_loss':
            value = exp.get('val_loss') or exp.get('validation_loss') or exp.get('primary_loss_min')
        elif obj_name == 'loss_stability':
            value = exp.get('loss_stability')
        elif obj_name == 'final_loss':
            value = exp.get('primary_loss_final')
        elif obj_name == 'primary_loss_min':
            value = exp.get('primary_loss_min')
        elif obj_name == 'monotonicity':
            value = exp.get('monotonicity')
        else:
            # 尝试直接从 exp 或 objective_values 获取
            value = exp.get(obj_name) or (exp.get('objective_values') or {}).get(obj_name)
        
        # 验证有效性
        if value is not None and np.isfinite(value):
            return float(value)
        return None
    
    def _get_objective_label(self, obj_name: str) -> str:
        """获取目标的可读标签"""
        labels = {
            'val_loss': 'Validation Loss (验证损失)',
            'loss_stability': 'Loss Stability (训练稳定性, 越小越稳定)',
            'final_loss': 'Final Loss (最终损失)',
            'primary_loss_min': 'Min Primary Loss (最小主损失)',
            'monotonicity': 'Monotonicity (单调性)',
        }
        return labels.get(obj_name, obj_name)
    
    def _visualize_multi_objective_pareto(self, valid_exps: List[Dict], 
                                          obj_x: str = 'val_loss', 
                                          obj_y: str = 'loss_stability'):
        """可视化多目标优化帕累托前沿
        
        根据配置的两个目标绘制帕累托前沿图，展示:
        1. 两个目标之间的权衡
        2. 帕累托前沿上的解集
        3. 推荐解（兼顾两个目标的平衡点）
        
        Args:
            valid_exps: 有效的实验记录列表
            obj_x: X 轴目标名称（来自 config.objectives[0]）
            obj_y: Y 轴目标名称（来自 config.objectives[1]）
        """
        import matplotlib.pyplot as plt
        
        # 获取两个目标值
        x_values_list = []
        y_values_list = []
        valid_indices = []
        
        for i, exp in enumerate(valid_exps):
            x_val = self._get_objective_value(exp, obj_x)
            y_val = self._get_objective_value(exp, obj_y)
            
            if x_val is not None and y_val is not None:
                x_values_list.append(x_val)
                y_values_list.append(y_val)
                valid_indices.append(i)
        
        if len(x_values_list) < 3:
            print(f"有效的多目标数据不足 (目标: {obj_x} vs {obj_y})，回退到 loss_tradeoff 模式")
            self._visualize_loss_tradeoff_pareto(valid_exps)
            return
        
        x_values = np.array(x_values_list)
        y_values = np.array(y_values_list)
        
        # 创建 2x2 图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # ===== 图1: 主帕累托前沿图 =====
        ax1 = axes[0, 0]
        
        # 计算帕累托前沿（两个目标都最小化）
        pareto_mask = self._compute_pareto_front(x_values, y_values, minimize_both=True)
        
        # 按代数着色
        generations = np.array([valid_exps[i]['generation'] for i in valid_indices])
        
        # 绘制被支配点
        dominated_mask = ~pareto_mask
        if dominated_mask.sum() > 0:
            scatter = ax1.scatter(x_values[dominated_mask], y_values[dominated_mask], 
                                c=generations[dominated_mask], cmap='Blues', alpha=0.6, 
                                label='Dominated', s=50, edgecolors='gray', linewidths=0.5)
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Generation', fontsize=10)
        
        # 绘制帕累托前沿点
        pareto_x = x_values[pareto_mask]
        pareto_y = y_values[pareto_mask]
        ax1.scatter(pareto_x, pareto_y, c='red', s=120, 
                  label=f'Pareto Front ({pareto_mask.sum()} pts)', 
                  edgecolors='darkred', linewidths=1.5, zorder=5)
        
        # 连接帕累托前沿点
        if len(pareto_x) > 1:
            sort_idx = np.argsort(pareto_x)
            ax1.plot(pareto_x[sort_idx], pareto_y[sort_idx], 'r--', alpha=0.6, linewidth=2)
        
        # 找到推荐解（帕累托前沿上距离理想点最近的解）
        if pareto_mask.sum() > 0:
            ideal_x, ideal_y = np.min(pareto_x), np.min(pareto_y)
            x_range = np.max(x_values) - np.min(x_values)
            y_range = np.max(y_values) - np.min(y_values)
            if x_range > 0 and y_range > 0:
                distances = np.sqrt(((pareto_x - ideal_x) / x_range) ** 2 + 
                                   ((pareto_y - ideal_y) / y_range) ** 2)
                best_pareto_idx = np.argmin(distances)
                ax1.scatter(pareto_x[best_pareto_idx], pareto_y[best_pareto_idx], 
                          c='lime', s=250, marker='*', label='Recommended', 
                          edgecolors='darkgreen', linewidths=2, zorder=15)
        
        # 标记各目标最优点
        best_x_idx = np.argmin(x_values)
        ax1.scatter(x_values[best_x_idx], y_values[best_x_idx], 
                  c='gold', s=200, marker='*', label=f'Best {obj_x}', 
                  edgecolors='black', linewidths=1.5, zorder=10)
        
        best_y_idx = np.argmin(y_values)
        ax1.scatter(x_values[best_y_idx], y_values[best_y_idx], 
                  c='cyan', s=200, marker='^', label=f'Best {obj_y}', 
                  edgecolors='darkblue', linewidths=1.5, zorder=10)
        
        ax1.set_xlabel(self._get_objective_label(obj_x), fontsize=11)
        ax1.set_ylabel(self._get_objective_label(obj_y), fontsize=11)
        ax1.set_title(f'Multi-Objective Pareto Front: {obj_x} vs {obj_y}', fontsize=12)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # ===== 图2: 目标值进化曲线 =====
        ax2 = axes[0, 1]
        gen_best_x = {}
        gen_best_y = {}
        for i, idx in enumerate(valid_indices):
            gen = valid_exps[idx]['generation']
            if gen not in gen_best_x or x_values[i] < gen_best_x[gen]:
                gen_best_x[gen] = x_values[i]
            if gen not in gen_best_y or y_values[i] < gen_best_y[gen]:
                gen_best_y[gen] = y_values[i]
        
        gens = sorted(gen_best_x.keys())
        ax2.plot(gens, [gen_best_x[g] for g in gens], 'b-o', label=f'Best {obj_x}', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(gens, [gen_best_y[g] for g in gens], 'r-s', label=f'Best {obj_y}', linewidth=2)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel(obj_x, color='blue')
        ax2_twin.set_ylabel(obj_y, color='red')
        ax2.set_title('Multi-Objective Evolution')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # ===== 图3: 帕累托前沿解对比 =====
        ax3 = axes[1, 0]
        if pareto_mask.sum() > 0:
            pareto_indices = np.where(pareto_mask)[0]
            pareto_data = [{'x': x_values[pi], 'y': y_values[pi], 
                           'gen': valid_exps[valid_indices[pi]]['generation'],
                           'idx': valid_exps[valid_indices[pi]]['individual_idx']}
                          for pi in pareto_indices]
            pareto_data.sort(key=lambda d: d['x'])
            
            x_pos = range(len(pareto_data))
            ax3.bar(x_pos, [d['x'] for d in pareto_data], width=0.4, label=obj_x, alpha=0.7, color='blue')
            ax3_twin = ax3.twinx()
            ax3_twin.bar([x + 0.4 for x in x_pos], [d['y'] for d in pareto_data], 
                        width=0.4, label=obj_y, alpha=0.7, color='red')
            
            ax3.set_xticks([x + 0.2 for x in x_pos])
            ax3.set_xticklabels([f"G{d['gen']}I{d['idx']}" for d in pareto_data], rotation=45)
            ax3.set_xlabel('Pareto Solutions')
            ax3.set_ylabel(obj_x, color='blue')
            ax3_twin.set_ylabel(obj_y, color='red')
            ax3.set_title('Pareto Front Solutions Comparison')
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
        else:
            ax3.text(0.5, 0.5, 'No Pareto solutions found', ha='center', va='center', transform=ax3.transAxes)
        
        # ===== 图4: 适应度分布 =====
        ax4 = axes[1, 1]
        fitness_values = [exp['fitness'] for exp in valid_exps if np.isfinite(exp['fitness'])]
        ax4.hist(fitness_values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax4.set_xlabel('Fitness (综合适应度)')
        ax4.set_ylabel('Count')
        ax4.set_title('Fitness Distribution')
        if fitness_values:
            ax4.axvline(np.min(fitness_values), color='g', linestyle='--', label=f'Best: {np.min(fitness_values):.4f}')
            ax4.axvline(np.mean(fitness_values), color='r', linestyle='--', label=f'Mean: {np.mean(fitness_values):.4f}')
        ax4.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / 'pareto_front_multi_objective.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"多目标优化帕累托前沿图已保存: {save_path}")
        
        # 输出帕累托前沿解的推荐
        if pareto_mask.sum() > 0:
            print(f"\n{'='*60}")
            print(f"帕累托前沿解推荐 ({obj_x} vs {obj_y}):")
            print(f"{'='*60}")
            for i, pi in enumerate(np.where(pareto_mask)[0]):
                exp_idx = valid_indices[pi]
                exp = valid_exps[exp_idx]
                print(f"  [{i+1}] Gen{exp['generation']}Ind{exp['individual_idx']}: "
                      f"{obj_x}={x_values[pi]:.4f}, {obj_y}={y_values[pi]:.4f}")
        
        return save_path
    
    def _visualize_loss_tradeoff_pareto(self, valid_exps: List[Dict]):
        """可视化主损失 vs 正则化损失帕累托前沿"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # 获取 primary_loss 和 reg_loss，过滤无效值
        primary_losses = []
        reg_losses = []
        filtered_exps = []
        
        for exp in valid_exps:
            # 获取 primary_loss
            primary = None
            if exp.get('primary_loss_min') is not None:
                primary = exp['primary_loss_min']
            elif 'primary_loss_history' in exp and exp['primary_loss_history']:
                primary = min(exp['primary_loss_history'])
            elif exp.get('fitness') is not None:
                primary = exp['fitness']
            
            # 获取 reg_loss
            reg = None
            if exp.get('reg_loss_min') is not None:
                reg = exp['reg_loss_min']
            elif 'reg_loss_history' in exp and exp['reg_loss_history']:
                reg = min(exp['reg_loss_history'])
            else:
                reg = 0.0
            
            # 只保留有效数据
            if primary is not None and reg is not None and np.isfinite(primary) and np.isfinite(reg):
                primary_losses.append(primary)
                reg_losses.append(reg)
                filtered_exps.append(exp)
        
        if len(primary_losses) < 3:
            print("有效的 loss tradeoff 数据不足，跳过帕累托前沿可视化")
            plt.close()
            return None
        
        x_values = np.array(primary_losses)
        y_values = np.array(reg_losses)
        
        # 计算帕累托前沿
        pareto_mask = self._compute_pareto_front(x_values, y_values, minimize_both=True)
        
        # 按代数着色（使用过滤后的实验列表）
        generations = np.array([exp['generation'] for exp in filtered_exps])
        
        # 绘制
        scatter = ax.scatter(x_values[~pareto_mask], y_values[~pareto_mask], 
                           c=generations[~pareto_mask], cmap='Blues', alpha=0.6, 
                           label='Dominated', s=50, edgecolors='gray', linewidths=0.5)
        
        pareto_x = x_values[pareto_mask]
        pareto_y = y_values[pareto_mask]
        ax.scatter(pareto_x, pareto_y, c='red', s=120, 
                  label=f'Pareto Front ({pareto_mask.sum()} pts)', 
                  edgecolors='darkred', linewidths=1.5, zorder=5)
        
        if len(pareto_x) > 1:
            sort_idx = np.argsort(pareto_x)
            ax.plot(pareto_x[sort_idx], pareto_y[sort_idx], 'r--', alpha=0.6, linewidth=2)
        
        best_primary_idx = np.argmin(x_values)
        ax.scatter(x_values[best_primary_idx], y_values[best_primary_idx], 
                  c='gold', s=200, marker='*', label='Best Primary Loss', 
                  edgecolors='black', linewidths=1.5, zorder=10)
        
        ax.set_xlabel('Primary Loss (主损失)', fontsize=11)
        ax.set_ylabel('Regularization Loss (正则化损失)', fontsize=11)
        ax.set_title('Primary Loss vs Regularization Loss Pareto Front', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Generation', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / 'pareto_front_loss_tradeoff.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"主损失 vs 正则化损失帕累托前沿图已保存: {save_path}")
        return save_path
    
    def _visualize_param_analysis_pareto(self, valid_exps: List[Dict], 
                                         objective_pairs: List[Tuple[str, str]] = None):
        """可视化参数-性能分析帕累托前沿"""
        import matplotlib.pyplot as plt
        
        if objective_pairs is None:
            hp_names = [hp.name for hp in self.config.hyperparams]
            correlations = []
            for hp_name in hp_names:
                hp_values = [exp['hyperparams'][hp_name] for exp in valid_exps]
                fitness_values = [exp['fitness'] for exp in valid_exps]
                corr = abs(np.corrcoef(hp_values, fitness_values)[0, 1])
                correlations.append((hp_name, corr if np.isfinite(corr) else 0))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_params = [c[0] for c in correlations[:3]]
            objective_pairs = [('fitness', p) for p in top_params]
        
        n_pairs = len(objective_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]
        
        for ax, (obj_x, obj_y) in zip(axes, objective_pairs):
            if obj_x == 'fitness':
                x_values = np.array([exp['fitness'] for exp in valid_exps])
                x_label = 'Fitness'
            else:
                x_values = np.array([exp['hyperparams'][obj_x] for exp in valid_exps])
                x_label = obj_x
            
            if obj_y == 'fitness':
                y_values = np.array([exp['fitness'] for exp in valid_exps])
                y_label = 'Fitness'
            else:
                y_values = np.array([exp['hyperparams'][obj_y] for exp in valid_exps])
                y_label = obj_y
            
            pareto_mask = self._compute_pareto_front(x_values, y_values, minimize_both=True)
            
            ax.scatter(x_values[~pareto_mask], y_values[~pareto_mask], 
                      c='lightgray', alpha=0.5, label='Dominated', s=30)
            
            pareto_x = x_values[pareto_mask]
            pareto_y = y_values[pareto_mask]
            ax.scatter(pareto_x, pareto_y, c='red', s=80, 
                      label='Pareto Front', edgecolors='darkred', linewidths=1)
            
            if len(pareto_x) > 1:
                sort_idx = np.argsort(pareto_x)
                ax.plot(pareto_x[sort_idx], pareto_y[sort_idx], 'r--', alpha=0.5, linewidth=1.5)
            
            best_idx = np.argmin([exp['fitness'] for exp in valid_exps])
            ax.scatter(x_values[best_idx], y_values[best_idx], c='gold', s=150, 
                      marker='*', label='Best', edgecolors='black', linewidths=1, zorder=10)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'Pareto Front: {obj_x} vs {obj_y}')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'pareto_front_params.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"参数分析帕累托前沿图已保存: {save_path}")
        return save_path
    
    @staticmethod
    def _compute_pareto_front(x: np.ndarray, y: np.ndarray, minimize_both: bool = True) -> np.ndarray:
        """计算帕累托前沿
        
        Args:
            x: 第一个目标的值
            y: 第二个目标的值
            minimize_both: 是否两个目标都是最小化（默认 True）
        
        Returns:
            布尔数组，True 表示该点在帕累托前沿上
        """
        n = len(x)
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if minimize_both:
                        # 如果 j 在两个目标上都不比 i 差，且至少一个目标更好
                        if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                            pareto_mask[i] = False
                            break
                    else:
                        # 最大化第一个目标，最小化第二个目标
                        if x[j] >= x[i] and y[j] <= y[i] and (x[j] > x[i] or y[j] < y[i]):
                            pareto_mask[i] = False
                            break
        
        return pareto_mask


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='遗传算法超参数优化')
    parser.add_argument('--config', type=str, default='hyperopt_config.yaml',
                        help='超参数优化配置文件路径')
    parser.add_argument('--generations', type=int, default=None,
                        help='进化代数（覆盖配置文件）')
    parser.add_argument('--population', type=int, default=None,
                        help='种群大小（覆盖配置文件）')
    parser.add_argument('--resume', action='store_true',
                        help='从断点继续训练（如果存在checkpoint.json）')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path
    
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        print("请创建配置文件或使用 --config 指定路径")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    
    config = parse_hyperopt_config(cfg_dict)
    
    # 命令行参数覆盖
    if args.generations:
        config.generations = args.generations
    if args.population:
        config.population_size = args.population
    
    # 运行优化
    optimizer = GeneticOptimizer(config, resume=args.resume)
    best_params, best_fitness = optimizer.evolve()
    
    print(f"\n{'='*60}")
    print("优化完成!")
    print(f"{'='*60}")
    if best_params is not None:
        print(f"最佳适应度: {best_fitness:.6f}")
        print(f"最佳超参数:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        print("未找到有效的超参数配置（所有实验均失败）")
        print("请检查训练配置是否正确")


if __name__ == '__main__':
    main()
