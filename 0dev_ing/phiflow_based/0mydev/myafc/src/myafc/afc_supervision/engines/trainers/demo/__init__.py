"""
集成式训练器示例

包含:
- SimpleTrainer: 基于可微分仿真的 MLP 训练器

使用方法：
    from myafc.afc_supervision.engines.trainers.demo import SimpleTrainer
"""
from .simple_trainer import (
    SimpleTrainer,
    SimpleTrainerConfig,
    MLPConfig,
    InputFeatureConfig,
    LossConfig,
    InletConfig,
    TrainingConfig,
    OutputConfig,
)

__all__ = [
    'SimpleTrainer',
    'SimpleTrainerConfig',
    'MLPConfig',
    'InputFeatureConfig',
    'LossConfig',
    'InletConfig',
    'TrainingConfig',
    'OutputConfig',
]
