#!/usr/bin/env python3
"""
将配置文件中探针的绝对坐标 (units) 转换为相对坐标 (fraction)

用法:
    python convert_probes_to_fraction.py <config_file> [--dry-run]
    
    --dry-run: 只显示更改，不写入文件
"""
import re
import sys
import argparse
from pathlib import Path
import yaml


def extract_domain_size(config_path: Path) -> tuple:
    """从配置文件中提取域大小"""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 domain.size 配置
    # 匹配格式: size: [Lx, Ly]
    match = re.search(r'domain:.*?size:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', content, re.DOTALL)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    raise ValueError(f"无法从 {config_path} 中提取域大小 (domain.size)")


def convert_probes_in_file(config_path: Path, dry_run: bool = False):
    """将配置文件中所有 units 格式的探针坐标转换为 fraction 格式"""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取域大小
    try:
        Lx, Ly = extract_domain_size(config_path)
        print(f"域大小: Lx={Lx}, Ly={Ly}")
    except ValueError as e:
        print(f"警告: {e}，跳过此文件")
        return 0
    
    # 匹配 units: [x, y] 格式并转换为 fraction
    # 正则表达式匹配: units: [数字, 数字]
    pattern = r'units:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    
    def replace_units(match):
        x = float(match.group(1))
        y = float(match.group(2))
        # 转换为相对坐标
        fx = x / Lx
        fy = y / Ly
        return f'fraction: [{fx:.6f}, {fy:.6f}]'
    
    new_content, count = re.subn(pattern, replace_units, content)
    
    if count > 0:
        print(f"转换了 {count} 个探针坐标")
        
        if dry_run:
            print("(dry-run 模式，未写入文件)")
        else:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"已更新: {config_path}")
    else:
        print("未找到需要转换的 units 坐标")
    
    return count


def main():
    parser = argparse.ArgumentParser(description='将探针的 units 坐标转换为 fraction 坐标')
    parser.add_argument('files', nargs='+', help='配置文件路径')
    parser.add_argument('--dry-run', action='store_true', help='只显示更改，不写入文件')
    args = parser.parse_args()
    
    total_count = 0
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"文件不存在: {path}")
            continue
        
        print(f"\n处理: {path}")
        print("-" * 60)
        count = convert_probes_in_file(path, dry_run=args.dry_run)
        total_count += count
    
    print(f"\n总计转换: {total_count} 个探针坐标")


if __name__ == '__main__':
    main()
