"""
AFC Validators - 通用验证引擎

支持加载 JetTrainer (.pt) 和 DRLTrainer (.zip) 训练的策略模型进行验证。

Example:
    from myafc.engines.validators import AFCValidator
    
    validator = AFCValidator(
        trainer_config="configs/jet_trainer_benchmark.yaml",
        model_path="runs/exp/checkpoints/best.pt"
    )
    results = validator.run()
"""
from .afc_validator import AFCValidator, ValidatorConfig, ValidationVisualsConfig

__all__ = ['AFCValidator', 'ValidatorConfig', 'ValidationVisualsConfig']
