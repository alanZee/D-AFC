#!/usr/bin/env python
"""
JetTrainer 训练示例

端到端轻量级训练脚本 - 仅需配置文件即可运行

Usage:
    python train_jet.py                              # 使用默认配置
    python train_jet.py path/to/config.yaml          # 使用自定义配置
    python train_jet.py --epochs 200 --lr 0.001      # 命令行覆盖参数
"""
import os
import sys
import argparse

# Windows 环境兼容
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    parser = argparse.ArgumentParser(description='AFC JetTrainer 训练脚本')
    parser.add_argument('config', nargs='?', default=None,
                        help='配置文件路径 (YAML)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--env', type=str, default=None,
                        help='环境配置文件路径')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称')
    args = parser.parse_args()
    
    # 导入 (放在这里避免导入时间过长)
    from myafc.afc_supervision import JetTrainer
    from myafc.afc_supervision.engines.trainers.jet_trainer import JetTrainingConfig
    from pathlib import Path
    import yaml
    
    # 确定配置
    if args.config:
        config_path = Path(args.config)
    else:
        # 使用默认配置
        config_path = Path(__file__).parent.parent / 'src/myafc/configs/jet_trainer_default.yaml'
    
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)
    
    print(f"[INFO] 使用配置: {config_path}")
    
    # 加载并修改配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    
    # 命令行参数覆盖
    if args.epochs:
        cfg_dict.setdefault('training', {})['epochs'] = args.epochs
    if args.lr:
        cfg_dict.setdefault('training', {})['learning_rate'] = args.lr
    if args.env:
        cfg_dict['env_config_path'] = args.env
    if args.name:
        cfg_dict.setdefault('output', {})['name'] = args.name
    
    # 创建并运行 Trainer
    trainer = JetTrainer(cfg_dict)
    
    try:
        results = trainer.train()
        print(f"\n[SUCCESS] 训练完成!")
        print(f"  - 最终 Loss: {results['final_loss']:.6f}")
        print(f"  - 实验目录: {trainer.exp_dir}")
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断")
        trainer._save_checkpoint('interrupted')
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
