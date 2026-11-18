"""
afc_supervision.engines 模块

包含监督学习训练引擎。

提供:
- SimpleTrainer: 集成式训练示例
- BaseTrainer: 训练器引擎抽象基类
- JetTrainer: 射流控制训练器

使用方法：
    from myafc.afc_supervision.engines import JetTrainer
    
    # 从 YAML 文件
    trainer = JetTrainer("config.yaml")
    
    # 训练
    results = trainer.train()
"""
# 集成式训练示例
from .trainers.demo import (
    SimpleTrainer,
    SimpleTrainerConfig,
    MLPConfig,
    InputFeatureConfig,
    LossConfig,
    InletConfig,
    TrainingConfig,
    OutputConfig,
)

# 射流训练器与基类
from .trainers import (
    BaseTrainer,
    TrainingBaseConfig,
    OutputBaseConfig,
    JetTrainer,
    JetTrainerConfig,
    JetTrainingConfig,
    InputFeatureConfig as JetInputFeatureConfig,
    InletConfig as JetInletConfig,
    LossConfig as JetLossConfig,
)

__all__ = [
    # 集成式训练示例
    'SimpleTrainer',
    'SimpleTrainerConfig',
    'MLPConfig',
    'InputFeatureConfig',
    'LossConfig',
    'InletConfig',
    'TrainingConfig',
    'OutputConfig',
    # 射流训练器与基类
    'BaseTrainer',
    'TrainingBaseConfig',
    'OutputBaseConfig',
    'JetTrainer',
    'JetTrainerConfig',
    'JetTrainingConfig',
    'JetInputFeatureConfig',
    'JetInletConfig',
    'JetLossConfig',
    # validators
    'AFCValidator',
    'ValidatorConfig',
]
