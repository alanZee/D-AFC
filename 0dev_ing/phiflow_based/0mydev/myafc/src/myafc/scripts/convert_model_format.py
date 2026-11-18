#!/usr/bin/env python
"""
模型格式转换工具: (类监督学习)可微分优化AFC框架(如 JetTrainer)下的 checkpoint (.pt) ↔ SB3 模型 (.zip)
注意：归一化参数不会迁移，转换后的模型适合作为迁移学习起点，而非精确还原

双向转换工具，支持:
1. JetTrainer → SB3: 将 AFCAgent 权重转换为 SB3 SAC/PPO/TD3 模型
2. SB3 → JetTrainer: 从 SB3 模型提取 Actor 权重转换为 JetTrainer checkpoint

注意事项:
- 转换需要网络结构兼容（输入/输出维度必须匹配）
- SB3 和 JetTrainer 使用不同的网络架构，仅能转换兼容层
- 推荐用于迁移学习起点，而非精确权重还原

使用方法:
    # JetTrainer → SB3
    python convert_model_format.py \\
        --source checkpoints/best.pt \\
        --target output/model.zip \\
        --direction sup2sb3 \\
        --trainer-config configs/jet_trainer.yaml \\
        --env-config cfgs4env/karman_torch_drl_square.yaml

    # SB3 → JetTrainer
    python convert_model_format.py \\
        --source final_model.zip \\
        --target output/converted.pt \\
        --direction sb32sup \\
        --trainer-config configs/jet_trainer.yaml \\
        --env-config cfgs4env/karman_torch_drl_square.yaml
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from collections import OrderedDict

import torch
import numpy as np

# 设置环境
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from myafc.utils import get_logger

LOGGER = get_logger()


def load_jet_trainer_checkpoint(path: Path) -> Dict[str, Any]:
    """加载 JetTrainer checkpoint
    
    Returns:
        包含 'model', 'ema', 'config' 等字段的字典
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    LOGGER.info(f"[Converter] 加载 JetTrainer checkpoint: {path}")
    LOGGER.info(f"  - Keys: {list(ckpt.keys())}")
    
    if 'model' in ckpt:
        LOGGER.info(f"  - Model layers: {len(ckpt['model'])} 个")
    if 'epoch' in ckpt:
        LOGGER.info(f"  - Epoch: {ckpt['epoch']}")
    
    return ckpt


def load_sb3_model(path: Path, env=None, algo_type: str = 'sac'):
    """加载 SB3 模型
    
    Args:
        path: SB3 模型路径 (.zip)
        env: Gymnasium 环境 (可选)
        algo_type: 算法类型 ('sac', 'ppo', 'td3')
    
    Returns:
        SB3 模型实例
    """
    from stable_baselines3 import SAC, PPO, TD3
    
    algo_class = {'sac': SAC, 'ppo': PPO, 'td3': TD3}[algo_type.lower()]
    model = algo_class.load(str(path), env=env)
    
    LOGGER.info(f"[Converter] 加载 SB3 模型: {path}")
    LOGGER.info(f"  - Algorithm: {algo_type.upper()}")
    
    # 打印 actor 网络结构
    if hasattr(model, 'actor'):
        actor = model.actor
        LOGGER.info(f"  - Actor type: {type(actor).__name__}")
        if hasattr(actor, 'mu'):
            LOGGER.info(f"  - Actor.mu: {actor.mu}")
    
    return model


def extract_sb3_actor_weights(model) -> Dict[str, torch.Tensor]:
    """从 SB3 模型提取 Actor 网络权重
    
    SB3 SAC 模型结构:
    - model.actor.latent_pi (mlp_extractor)
    - model.actor.mu (mean action network)
    - model.actor.log_std (log std for exploration)
    
    Returns:
        权重字典
    """
    weights = OrderedDict()
    
    if hasattr(model, 'actor'):
        actor = model.actor
        
        # 提取 features_extractor
        if hasattr(actor, 'features_extractor'):
            fe = actor.features_extractor
            for name, param in fe.named_parameters():
                weights[f'features_extractor.{name}'] = param.data.clone()
        
        # 提取 latent_pi (mlp_extractor for policy)
        if hasattr(actor, 'latent_pi'):
            for name, param in actor.latent_pi.named_parameters():
                weights[f'latent_pi.{name}'] = param.data.clone()
        
        # 提取 mu (mean network)
        if hasattr(actor, 'mu'):
            for name, param in actor.mu.named_parameters():
                weights[f'mu.{name}'] = param.data.clone()
        
        # 提取 log_std (可能是 Tensor 或 Linear 层)
        if hasattr(actor, 'log_std'):
            log_std = actor.log_std
            if isinstance(log_std, torch.nn.Module):
                # log_std 是一个 Linear 层
                for name, param in log_std.named_parameters():
                    weights[f'log_std.{name}'] = param.data.clone()
            elif isinstance(log_std, torch.Tensor):
                weights['log_std'] = log_std.data.clone()
    
    LOGGER.info(f"[Converter] 提取 SB3 Actor 权重: {len(weights)} 个参数")
    for name, tensor in list(weights.items())[:5]:
        LOGGER.debug(f"  - {name}: {tensor.shape}")
    
    return weights


def convert_sup2sb3(
    source_path: Path,
    target_path: Path,
    env_config_path: Path,
    trainer_config_path: Optional[Path] = None,
    algo_type: str = 'sac',
    use_ema: bool = False,
):
    """将 可微分优化AFC框架(如 JetTrainer)下的 checkpoint 转换为 SB3 模型
    
    注意:
    - JetTrainer 使用 AFCAgent (MLPPolicy)
    - SB3 使用 SACPolicy/PPOPolicy
    - 由于架构不同，此转换创建新的 SB3 模型并尝试迁移兼容层
    
    转换策略:
    1. 创建新的 SB3 模型 (使用相同的网络架构配置)
    2. 尝试将 JetTrainer 的 hidden layers 权重复制到 SB3 的 mlp_extractor
    3. 输出层可能维度不匹配，需要重新初始化
    """
    import yaml
    from stable_baselines3 import SAC, PPO, TD3
    from myafc.afc_drl_sb3.envs import PhiFlowEnv
    
    # 加载 JetTrainer checkpoint
    ckpt = load_jet_trainer_checkpoint(source_path)
    
    # 选择权重
    if use_ema and 'ema' in ckpt and ckpt['ema'] is not None:
        ema_data = ckpt['ema']
        if 'ema_model_state_dict' in ema_data:
            state_dict = ema_data['ema_model_state_dict']
            LOGGER.info("使用 EMA 权重")
        else:
            state_dict = ckpt['model']
    else:
        state_dict = ckpt['model']
    
    LOGGER.info(f"JetTrainer state_dict keys: {list(state_dict.keys())[:10]}...")
    
    # 创建环境配置
    # 需要合并 solver_config (env_config_path) 和 trainer_config 中的 input_features
    if env_config_path and env_config_path.exists():
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_cfg_dict = yaml.safe_load(f)
        # Solver 配置格式，需要添加 env_config_path 引用自身
        if 'solver' in env_cfg_dict and 'env_config_path' not in env_cfg_dict:
            env_cfg_dict['env_config_path'] = str(env_config_path)
    elif trainer_config_path and trainer_config_path.exists():
        with open(trainer_config_path, 'r', encoding='utf-8') as f:
            env_cfg_dict = yaml.safe_load(f)
    else:
        raise ValueError("需要提供 env_config_path 或 trainer_config_path")
    
    # 从 trainer_config 合并 input_features 到 env 配置
    if trainer_config_path and trainer_config_path.exists():
        with open(trainer_config_path, 'r', encoding='utf-8') as f:
            trainer_cfg = yaml.safe_load(f)
        # 复制 input_features 到 env 配置 (PhiFlowEnv 使用 env.input_features)
        input_features = trainer_cfg.get('input_features')
        if input_features:
            env_cfg_dict.setdefault('env', {})['input_features'] = input_features
            LOGGER.info(f"[Converter] 合并 input_features: main_input_type={input_features.get('main_input_type')}")
    
    env = PhiFlowEnv(config=env_cfg_dict)
    
    # 分析 JetTrainer 网络结构
    jet_layers = {}
    for name, param in state_dict.items():
        jet_layers[name] = param.shape
    LOGGER.info(f"JetTrainer 网络层:")
    for name, shape in list(jet_layers.items())[:10]:
        LOGGER.info(f"  - {name}: {shape}")
    
    # 创建新的 SB3 模型
    algo_class = {'sac': SAC, 'ppo': PPO, 'td3': TD3}[algo_type.lower()]
    
    # 尝试从 JetTrainer config 推断网络结构
    hidden_dims = [512, 512]  # 默认
    if 'hidden_dims' in ckpt.get('config', {}):
        hidden_dims = ckpt['config']['hidden_dims']
    
    policy_kwargs = {'net_arch': hidden_dims}
    
    sb3_model = algo_class(
        policy='MlpPolicy',
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    
    # 尝试权重迁移
    # JetTrainer (AFCAgent) 使用: mlp_extractor.0.weight, mlp_extractor.2.weight, action_net.weight/bias
    # SB3 SAC 使用: actor.latent_pi.0.weight, latent_pi.2.weight, mu.weight/bias
    
    actor = sb3_model.actor
    transferred = 0
    
    # 映射 JetTrainer mlp_extractor → SB3 latent_pi
    if hasattr(actor, 'latent_pi'):
        latent_pi = actor.latent_pi
        latent_pi_dict = dict(latent_pi.named_parameters())
        
        LOGGER.debug(f"SB3 latent_pi keys: {list(latent_pi_dict.keys())}")
        
        for jet_name, jet_param in state_dict.items():
            # JetTrainer: "mlp_extractor.0.weight" → SB3: "0.weight"
            if jet_name.startswith('mlp_extractor.'):
                sb3_name = jet_name.replace('mlp_extractor.', '')
                if sb3_name in latent_pi_dict:
                    sb3_param = latent_pi_dict[sb3_name]
                    if jet_param.shape == sb3_param.shape:
                        sb3_param.data.copy_(jet_param)
                        transferred += 1
                        LOGGER.info(f"迁移: {jet_name} → latent_pi.{sb3_name} {jet_param.shape}")
                    else:
                        LOGGER.warning(f"维度不匹配: {jet_name} {jet_param.shape} vs {sb3_param.shape}")
                else:
                    LOGGER.debug(f"SB3 中未找到对应层: {sb3_name}")
            # 向后兼容: "layers.0.weight" 格式
            elif jet_name.startswith('layers.'):
                sb3_name = jet_name.replace('layers.', '')
                if sb3_name in latent_pi_dict:
                    sb3_param = latent_pi_dict[sb3_name]
                    if jet_param.shape == sb3_param.shape:
                        sb3_param.data.copy_(jet_param)
                        transferred += 1
                        LOGGER.info(f"迁移: {jet_name} → latent_pi.{sb3_name} {jet_param.shape}")
                    else:
                        LOGGER.warning(f"维度不匹配: {jet_name} {jet_param.shape} vs {sb3_param.shape}")
    
    # 映射 JetTrainer action_net → SB3 mu (输出层)
    if hasattr(actor, 'mu'):
        mu = actor.mu
        mu_dict = dict(mu.named_parameters())
        for jet_name, jet_param in state_dict.items():
            # JetTrainer: "action_net.weight" → SB3: "weight"
            if jet_name.startswith('action_net.'):
                sb3_name = jet_name.replace('action_net.', '')
                if sb3_name in mu_dict:
                    sb3_param = mu_dict[sb3_name]
                    if jet_param.shape == sb3_param.shape:
                        sb3_param.data.copy_(jet_param)
                        transferred += 1
                        LOGGER.info(f"迁移: {jet_name} → mu.{sb3_name} {jet_param.shape}")
                    else:
                        LOGGER.warning(f"维度不匹配: {jet_name} {jet_param.shape} vs {sb3_param.shape}")
    
    LOGGER.info(f"[Converter] 成功迁移 {transferred} 个参数")
    
    # 保存 SB3 模型
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sb3_model.save(str(target_path))
    LOGGER.info(f"[Converter] SB3 模型保存至: {target_path}")
    
    # 清理
    env.close()
    
    return {
        'transferred_params': transferred,
        'total_jet_params': len(state_dict),
        'output_path': str(target_path),
    }


def convert_sb32sup(
    source_path: Path,
    target_path: Path,
    trainer_config_path: Path,
    algo_type: str = 'sac',
    env_config_path: Optional[Path] = None,
):
    """将 SB3 模型转换为 可微分优化AFC框架(如 JetTrainer)下的 checkpoint
    
    转换策略:
    1. 从 SB3 模型提取 actor 网络权重
    2. 将 mlp_extractor 权重映射到 JetTrainer 的 PolicyNetwork
    3. 生成兼容 JetTrainer 的 checkpoint 格式
    
    Args:
        source_path: SB3 模型路径 (.zip)
        target_path: 输出 checkpoint 路径 (.pt)
        trainer_config_path: Trainer 配置路径 (用于读取 input_features 等)
        algo_type: SB3 算法类型 ('sac', 'ppo', 'td3')
        env_config_path: 环境配置路径 (可选，用于创建兼容的环境)
    
    注意:
    - SB3 输出层 (mu) 通常与 JetTrainer 输出层不同
    - 此转换适合作为预训练起点
    """
    from stable_baselines3 import SAC, PPO, TD3
    from myafc.afc_drl_sb3.envs import PhiFlowEnv
    import yaml
    
    # 加载 trainer_config 以获取 input_features
    with open(trainer_config_path, 'r', encoding='utf-8') as f:
        trainer_cfg = yaml.safe_load(f)
    
    # 确定环境配置路径
    if env_config_path and env_config_path.exists():
        env_cfg_path = env_config_path
    elif 'env_config_path' in trainer_cfg:
        env_cfg_path = Path(trainer_cfg['env_config_path'])
    else:
        # 尝试使用 trainer_config_path 目录下的 solver_config.yaml
        env_cfg_path = trainer_config_path.parent / 'solver_config.yaml'
    
    if not env_cfg_path.exists():
        raise FileNotFoundError(f"找不到环境配置文件: {env_cfg_path}")
    
    # 加载环境配置并合并 input_features
    with open(env_cfg_path, 'r', encoding='utf-8') as f:
        env_cfg_dict = yaml.safe_load(f)
    
    # 从 trainer_config 合并 input_features
    input_features = trainer_cfg.get('input_features')
    if input_features:
        env_cfg_dict.setdefault('env', {})['input_features'] = input_features
        LOGGER.info(f"[Converter] 合并 input_features: main_input_type={input_features.get('main_input_type')}")
    
    # 确保 env_config_path 被设置 (PhiFlowEnv 需要)
    env_cfg_dict['env_config_path'] = str(env_cfg_path)
    
    # 创建环境加载 SB3 模型
    env = PhiFlowEnv(config=env_cfg_dict)
    sb3_model = load_sb3_model(source_path, env=env, algo_type=algo_type)
    
    # 提取权重
    sb3_weights = extract_sb3_actor_weights(sb3_model)
    
    # 构建 JetTrainer 格式的 state_dict
    jet_state_dict = OrderedDict()
    transferred = 0
    
    # 权重映射规则:
    # SB3 latent_pi.*.weight/bias → JetTrainer mlp_extractor.*.weight/bias
    # SB3 mu.weight/bias → JetTrainer action_net.weight/bias
    for name, param in sb3_weights.items():
        if name.startswith('latent_pi.'):
            # latent_pi.0.weight → mlp_extractor.0.weight
            jet_name = 'mlp_extractor.' + name.replace('latent_pi.', '')
            jet_state_dict[jet_name] = param
            transferred += 1
            LOGGER.info(f"迁移: {name} → {jet_name} {param.shape}")
        elif name.startswith('mu.'):
            # mu.weight → action_net.weight
            jet_name = 'action_net.' + name.replace('mu.', '')
            jet_state_dict[jet_name] = param
            transferred += 1
            LOGGER.info(f"迁移: {name} → {jet_name} {param.shape}")
    
    LOGGER.info(f"[Converter] 成功迁移 {transferred} 个参数")
    
    # 创建 JetTrainer checkpoint 格式
    checkpoint = {
        'epoch': 0,
        'step': 0,
        'update_count': 0,
        'best_fitness': float('inf'),
        'model': jet_state_dict,
        'ema': None,
        'optimizer': None,
        'scaler': None,
        'date': None,
        'version': 'converted_from_sb3',
        '_source': str(source_path),
        '_conversion_info': {
            'source_type': 'sb3',
            'algo_type': algo_type,
            'transferred_params': transferred,
            'total_sb3_params': len(sb3_weights),
        }
    }
    
    # 保存
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target_path)
    LOGGER.info(f"[Converter] JetTrainer checkpoint 保存至: {target_path}")
    
    # 清理
    env.close()
    
    return {
        'transferred_params': transferred,
        'total_sb3_params': len(sb3_weights),
        'output_path': str(target_path),
    }


def analyze_compatibility(
    sup_path: Optional[Path] = None,
    sb3_path: Optional[Path] = None,
    env_config_path: Optional[Path] = None,
    algo_type: str = 'sac',
):
    """分析两种模型格式的兼容性
    
    输出各层的维度对比，帮助用户判断转换可行性
    """
    print("\n" + "=" * 60)
    print("模型兼容性分析")
    print("=" * 60)
    
    if sup_path and sup_path.exists():
        print(f"\n📁 JetTrainer Checkpoint: {sup_path}")
        ckpt = load_jet_trainer_checkpoint(sup_path)
        
        if 'model' in ckpt:
            print("\n  网络层:")
            for name, param in ckpt['model'].items():
                print(f"    {name}: {param.shape}")
        
        if 'ema' in ckpt and ckpt['ema']:
            print("\n  EMA 权重: ✓ 存在")
    
    if sb3_path and sb3_path.exists():
        print(f"\n📁 SB3 Model: {sb3_path}")
        
        if env_config_path:
            from myafc.afc_drl_sb3.envs import PhiFlowEnv
            env = PhiFlowEnv(config=str(env_config_path))
            model = load_sb3_model(sb3_path, env=env, algo_type=algo_type)
            
            print("\n  Actor 网络层:")
            if hasattr(model, 'actor'):
                for name, param in model.actor.named_parameters():
                    print(f"    {name}: {param.shape}")
            
            env.close()
        else:
            print("  ⚠️ 需要 env_config_path 来加载 SB3 模型")
    
    print("\n" + "=" * 60)
    print("转换建议:")
    print("-" * 60)
    print("• JetTrainer → SB3: 适合将监督学习预训练的策略用于 DRL 微调")
    print("• SB3 → JetTrainer: 适合将 DRL 策略用于可微分优化微调")
    print("• 输出层维度不同时，转换后需要额外训练来适应")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='模型格式转换工具: JetTrainer ↔ SB3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # JetTrainer → SB3
  python convert_model_format.py \\
      --source checkpoints/best.pt \\
      --target output/model.zip \\
      --direction sup2sb3 \\
      --trainer-config configs/jet_trainer.yaml

  # SB3 → JetTrainer
  python convert_model_format.py \\
      --source final_model.zip \\
      --target output/converted.pt \\
      --direction sb32sup \\
      --trainer-config configs/jet_trainer.yaml

  # 分析兼容性
  python convert_model_format.py \\
      --analyze \\
      --sup-path checkpoints/best.pt \\
      --sb3-path final_model.zip \\
      --env-config cfgs4env/karman_torch_drl_square.yaml
        """
    )
    
    parser.add_argument('--source', type=Path, help='源模型路径')
    parser.add_argument('--target', type=Path, help='目标模型路径')
    parser.add_argument(
        '--direction', 
        choices=['sup2sb3', 'sb32sup'],
        help='转换方向'
    )
    parser.add_argument('--trainer-config', type=Path, help='Trainer 配置路径')
    parser.add_argument('--env-config', type=Path, help='环境配置路径 (sup2sb3 需要)')
    parser.add_argument(
        '--algo-type', 
        choices=['sac', 'ppo', 'td3'],
        default='sac',
        help='SB3 算法类型 (默认: sac)'
    )
    parser.add_argument('--use-ema', action='store_true', help='使用 EMA 权重 (sup2sb3)')
    
    # 分析模式
    parser.add_argument('--analyze', action='store_true', help='分析模型兼容性')
    parser.add_argument('--sup-path', type=Path, help='类监督学习式可微分优化AFC框架(如 JetTrainer) checkpoint 路径 (分析)')
    parser.add_argument('--sb3-path', type=Path, help='SB3 模型路径 (分析)')
    
    args = parser.parse_args()
    
    # 分析模式
    if args.analyze:
        analyze_compatibility(
            sup_path=args.sup_path,
            sb3_path=args.sb3_path,
            env_config_path=args.env_config or args.trainer_config,
            algo_type=args.algo_type,
        )
        return 0
    
    # 转换模式
    if not args.source or not args.target or not args.direction:
        parser.error("转换需要 --source, --target, --direction 参数")
    
    if args.direction == 'sup2sb3':
        if not args.trainer_config and not args.env_config:
            parser.error("sup2sb3 需要 --trainer-config 或 --env-config")
        
        result = convert_sup2sb3(
            source_path=args.source,
            target_path=args.target,
            env_config_path=args.env_config or args.trainer_config,
            trainer_config_path=args.trainer_config,
            algo_type=args.algo_type,
            use_ema=args.use_ema,
        )
    
    elif args.direction == 'sb32sup':
        if not args.trainer_config:
            parser.error("sb32sup 需要 --trainer-config")
        
        result = convert_sb32sup(
            source_path=args.source,
            target_path=args.target,
            trainer_config_path=args.trainer_config,
            algo_type=args.algo_type,
            env_config_path=args.env_config,
        )
    
    print("\n转换结果:")
    for key, val in result.items():
        print(f"  {key}: {val}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
