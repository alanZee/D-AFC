#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
涡脱周期计算工具

功能：计算圆柱/方柱绕流的涡脱周期、Reynolds 数、Strouhal 数等相关物理量

公式：
    Re = U × D / ν        (Reynolds 数)
    St = f × D / U        (Strouhal 数)
    f = St × U / D        (涡脱频率)
    T = 1/f = D / (St × U) (涡脱周期)
    
其中:
    - Re: Reynolds 数（惯性力/粘性力）
    - St: Strouhal 数（无量纲频率）
    - f: 涡脱频率 [Hz]
    - T: 涡脱周期 [s]
    - D: 特征长度（障碍物直径/边长）[m]
    - U: 来流速度 [m/s]
    - ν: 运动粘度 [m²/s]

典型 Strouhal 数:
    - 圆柱: St ≈ 0.21 (Re = 40-200)
    - 方柱: St ≈ 0.13-0.14 (Re = 50-200)

===========================================================================================
DRLinFluids 开源库案例 (GitHub: venturi123/DRLinFluids)
===========================================================================================

【案例 1: 圆柱绕流 - cylinder2D_multiprocessing】
  参数 (来自 forceCoeffsIncompressible / transportProperties):
    - D = 0.1 m (lRef, 圆柱直径)
    - U = 1.0 m/s (magUInf, 入口速度)
    - ν = 0.001 m²/s (nu)
  计算:
    Re = U × D / ν = 1.0 × 0.1 / 0.001 = 100
    St ≈ 0.165 (Re=100 时圆柱的典型值)
    T = D / (St × U) = 0.1 / (0.165 × 1.0) = 0.606 s
    f = 1/T = 1.65 Hz

【案例 2: 方柱绕流 - square2D_multiprocessing】
  参数 (来自 forceCoeffsIncompressible / transportProperties):
    - D = 0.0508 m (lRef, 方柱边长)
    - U = 2.0 m/s (magUInf, 入口速度)
    - ν = 0.001 m²/s (nu)
  计算:
    Re = U × D / ν = 2.0 × 0.0508 / 0.001 ≈ 101.6
    St ≈ 0.13 (Re≈100 时方柱的典型值)
    T = D / (St × U) = 0.0508 / (0.13 × 2.0) = 0.195 s
    f = 1/T = 5.12 Hz

===========================================================================================
DRLinFluids 论文案例 (Physics of Fluids 34, 081801, 2022)
===========================================================================================

【案例 1: 圆柱绕流 (论文 Fig. 2-4)】
  参数 (论文标准设置):
    - D = 1.0 m (标准化直径)
    - U = 1.0 m/s (标准化入口速度)
    - ν = 0.01 m²/s (对应 Re=100)
  计算:
    Re = U × D / ν = 1.0 × 1.0 / 0.01 = 100
    St ≈ 0.165 (Re=100 圆柱涡脱 Strouhal 数)
    T = D / (St × U) = 1.0 / (0.165 × 1.0) = 6.06 s

【案例 2: 方柱绕流 (论文 Fig. 5-6)】
  参数 (论文标准设置):
    - D = 1.0 m (标准化边长)
    - U = 1.0 m/s (标准化入口速度)
    - ν = 0.01 m²/s (对应 Re=100)
  计算:
    Re = U × D / ν = 1.0 × 1.0 / 0.01 = 100
    St ≈ 0.13 (Re=100 方柱涡脱 Strouhal 数)
    T = D / (St × U) = 1.0 / (0.13 × 1.0) = 7.69 s

注: 论文中使用无量纲化参数，实际仿真时间与开源库有所不同，
    但 Re 和 St 保持一致。

===========================================================================================

Strouhal 数说明:
    St 是无量纲数，定义为 St = f × D / U，表征涡脱频率的无量纲形式。
    St 本身需要通过实验或仿真测量涡脱频率后计算得到，无法从第一性原理直接推导。
    但在工程估算中，可以根据 Re 数和障碍物形状查表获得经验值，常用经验公式包括：
    
    - 圆柱 Roshko 公式 (Re=40-150): St ≈ 0.212 × (1 - 21.2/Re)
    - 圆柱 Re=100 附近: St ≈ 0.165
    - 圆柱 Re>300: St ≈ 0.21 (近似常数)
    - 方柱 Re=50-200: St ≈ 0.13
    - 方柱 Re>200: St ≈ 0.14

===========================================================================================

使用方法:
    # 计算涡脱周期 (需指定 St)
    python calc_vortex_shedding_period.py --D 0.05 --U 2.0 --St 0.13
    
    # 自动估算 St 并计算涡脱周期 (根据 Re 和障碍物类型)
    python calc_vortex_shedding_period.py --D 0.1 --U 1.0 --nu 0.001 --shape cylinder
    python calc_vortex_shedding_period.py --D 0.05 --U 2.0 --nu 0.001 --shape square
    
    # 仅计算 Reynolds 数
    python calc_vortex_shedding_period.py --calc-re --D 0.1 --U 1.0 --nu 0.001
    
    # 演示 DRLinFluids 案例
    python calc_vortex_shedding_period.py --demo
"""

import argparse


def calc_reynolds(D: float, U: float, nu: float) -> float:
    """计算 Reynolds 数
    
    Args:
        D: 特征长度 [m]
        U: 来流速度 [m/s]
        nu: 运动粘度 [m²/s]
    
    Returns:
        Reynolds 数 (无量纲)
    """
    return U * D / nu


def estimate_strouhal(Re: float, shape: str = 'cylinder') -> tuple:
    """根据 Re 数和障碍物形状估算 Strouhal 数
    
    使用经验公式/查表估算，非精确值。
    
    Args:
        Re: Reynolds 数
        shape: 障碍物形状 ('cylinder' 或 'square')
    
    Returns:
        tuple: (St 估算值, 估算方法说明)
    
    参考文献:
        - Roshko, A. (1954). On the development of turbulent wakes from vortex streets. NACA Report 1191.
        - Williamson, C.H.K. (1996). Annu. Rev. Fluid Mech. 28: 477-539
        - Norberg, C. (1993). J. Wind Eng. Ind. Aerodyn. 49: 187-196
    """
    shape = shape.lower()
    
    if shape in ('cylinder', 'cyl', 'c', '圆柱'):
        if Re < 40:
            return (None, f"Re={Re:.1f} < 40, 尚未形成稳定涡脱")
        elif Re < 150:
            # Roshko 公式适用于 Re=40-150
            St = 0.212 * (1 - 21.2 / Re)
            return (St, f"Roshko 公式: St = 0.212×(1-21.2/Re)")
        elif Re < 300:
            # 过渡区，使用插值
            St = 0.165 + (Re - 100) / 200 * (0.21 - 0.165)
            return (St, f"过渡区插值 (Re=100→300)")
        else:
            # Re > 300, St 近似为常数
            return (0.21, f"高 Re 经验值 (Re>300)")
    
    elif shape in ('square', 'sq', 's', '方柱'):
        if Re < 50:
            return (None, f"Re={Re:.1f} < 50, 方柱涡脱不稳定")
        elif Re <= 200:
            return (0.13, f"经验值 (Re=50-200)")
        else:
            return (0.14, f"经验值 (Re>200)")
    
    else:
        return (None, f"未知形状: {shape}, 支持 'cylinder' 或 'square'")


def calc_vortex_period(D: float, U: float, St: float) -> dict:
    """计算涡脱周期和频率
    
    Args:
        D: 特征长度 [m]
        U: 来流速度 [m/s]
        St: Strouhal 数
    
    Returns:
        dict: {frequency, period}
    """
    f = St * U / D  # 频率 [Hz]
    T = 1 / f       # 周期 [s]
    return {'frequency': f, 'period': T}


def calc_steps_for_periods(T: float, dt: float, n_periods: float) -> int:
    """计算覆盖指定涡脱周期数所需的仿真步数
    
    Args:
        T: 涡脱周期 [s]
        dt: 时间步长 [s]
        n_periods: 涡脱周期数
    
    Returns:
        仿真步数
    """
    return int(n_periods * T / dt)


def print_reynolds_result(D: float, U: float, nu: float):
    """打印 Reynolds 数计算结果"""
    Re = calc_reynolds(D, U, nu)
    print("=" * 70)
    print("Reynolds 数计算")
    print("=" * 70)
    print(f"公式: Re = U × D / ν")
    print("-" * 70)
    print(f"输入参数:")
    print(f"  特征长度 D = {D} m")
    print(f"  来流速度 U = {U} m/s")
    print(f"  运动粘度 ν = {nu} m²/s")
    print("-" * 70)
    print(f"计算结果:")
    print(f"  Re = {U} × {D} / {nu} = {Re:.2f}")
    print("=" * 70)
    return Re


def print_strouhal_estimate(D: float, U: float, nu: float, shape: str):
    """打印 Strouhal 数估算结果"""
    Re = calc_reynolds(D, U, nu)
    St, method = estimate_strouhal(Re, shape)
    
    print("=" * 70)
    print(f"Strouhal 数估算 ({shape})")
    print("=" * 70)
    print(f"输入参数:")
    print(f"  特征长度 D = {D} m")
    print(f"  来流速度 U = {U} m/s")
    print(f"  运动粘度 ν = {nu} m²/s")
    print(f"  障碍物形状 = {shape}")
    print("-" * 70)
    print(f"计算 Reynolds 数:")
    print(f"  Re = U × D / ν = {U} × {D} / {nu} = {Re:.2f}")
    print("-" * 70)
    print(f"估算 Strouhal 数:")
    print(f"  方法: {method}")
    if St is not None:
        print(f"  St ≈ {St:.4f}")
        # 同时计算涡脱周期
        result = calc_vortex_period(D, U, St)
        print("-" * 70)
        print(f"涡脱周期:")
        print(f"  f = St × U / D = {St:.4f} × {U} / {D} = {result['frequency']:.4f} Hz")
        print(f"  T = 1/f = {result['period']:.4f} s")
    else:
        print(f"  无法估算 St")
    print("=" * 70)
    return St


def print_vortex_period_results(D: float, U: float, St: float, dt: float = 0.005, 
                                 nu: float = None, case_name: str = None):
    """打印涡脱周期计算结果"""
    result = calc_vortex_period(D, U, St)
    T = result['period']
    f = result['frequency']
    
    print("=" * 70)
    if case_name:
        print(f"涡脱周期计算 - {case_name}")
    else:
        print("涡脱周期计算结果")
    print("=" * 70)
    print(f"输入参数:")
    print(f"  特征长度 D = {D} m")
    print(f"  来流速度 U = {U} m/s")
    if nu is not None:
        Re = calc_reynolds(D, U, nu)
        print(f"  运动粘度 ν = {nu} m²/s")
        print(f"  Reynolds 数 Re = {Re:.1f}")
    print(f"  Strouhal 数 St = {St}")
    print(f"  时间步长 dt = {dt} s")
    print("-" * 70)
    print(f"计算结果:")
    print(f"  涡脱频率 f = St × U / D = {St} × {U} / {D} = {f:.4f} Hz")
    print(f"  涡脱周期 T = 1/f = {T:.4f} s")
    print(f"  每周期步数 = T/dt = {T/dt:.1f} steps")
    print("-" * 70)
    print(f"仿真步数建议 (验证至少需要 3-5 个周期):")
    for n in [1, 3, 5, 10, 13]:
        steps = calc_steps_for_periods(T, dt, n)
        sim_time = steps * dt
        print(f"  {n:2d} 个周期: {steps:5d} 步 ({sim_time:.3f}s)")
    print("=" * 70)


def demo_drlinfluids_cases():
    """演示 DRLinFluids 开源库和论文的案例"""
    
    print("\n" + "=" * 70)
    print("  DRLinFluids 开源库案例 (GitHub: venturi123/DRLinFluids)")
    print("=" * 70)
    
    # 开源库 - 圆柱案例
    print("\n>>> 开源库案例 1: 圆柱绕流 (cylinder2D_multiprocessing)")
    print("    配置来源: forceCoeffsIncompressible, transportProperties")
    print_vortex_period_results(
        D=0.1, U=1.0, St=0.165, dt=0.0005, nu=0.001,
        case_name="DRLinFluids 开源库 - 圆柱 (Re=100)"
    )
    
    # 开源库 - 方柱案例
    print("\n>>> 开源库案例 2: 方柱绕流 (square2D_multiprocessing)")
    print("    配置来源: forceCoeffsIncompressible, transportProperties")
    print_vortex_period_results(
        D=0.0508, U=2.0, St=0.13, dt=0.0005, nu=0.001,
        case_name="DRLinFluids 开源库 - 方柱 (Re≈102)"
    )
    
    print("\n" + "=" * 70)
    print("  DRLinFluids 论文案例 (Physics of Fluids 34, 081801, 2022)")
    print("=" * 70)
    
    # 论文 - 圆柱案例 (标准化参数)
    print("\n>>> 论文案例 1: 圆柱绕流 (无量纲化参数)")
    print("    说明: 论文采用标准化几何 D=1, U=1")
    print_vortex_period_results(
        D=1.0, U=1.0, St=0.165, dt=0.02, nu=0.01,
        case_name="DRLinFluids 论文 - 圆柱 (Re=100, 标准化)"
    )
    
    # 论文 - 方柱案例 (标准化参数)
    print("\n>>> 论文案例 2: 方柱绕流 (无量纲化参数)")
    print("    说明: 论文采用标准化几何 D=1, U=1")
    print_vortex_period_results(
        D=1.0, U=1.0, St=0.13, dt=0.02, nu=0.01,
        case_name="DRLinFluids 论文 - 方柱 (Re=100, 标准化)"
    )
    
    print("\n" + "=" * 70)
    print("  典型 Strouhal 数参考表")
    print("=" * 70)
    print("""
    ┌──────────────┬──────────────┬──────────────────────────────────┐
    │ 障碍物类型    │ Re 范围       │ Strouhal 数 St                   │
    ├──────────────┼──────────────┼──────────────────────────────────┤
    │ 圆柱         │ 40-50        │ ~0.12                             │
    │ 圆柱         │ 100          │ ~0.165                            │
    │ 圆柱         │ 200          │ ~0.19                             │
    │ 圆柱         │ 300-10⁵      │ ~0.21                             │
    ├──────────────┼──────────────┼──────────────────────────────────┤
    │ 方柱         │ 50-100       │ ~0.13                             │
    │ 方柱         │ 100-200      │ ~0.13-0.14                        │
    │ 方柱         │ >200         │ ~0.14                             │
    └──────────────┴──────────────┴──────────────────────────────────┘
    
    参考文献:
    - Williamson, C.H.K. (1996). Annu. Rev. Fluid Mech. 28: 477-539
    - Norberg, C. (1993). J. Wind Eng. Ind. Aerodyn. 49: 187-196
    """)


def main():
    parser = argparse.ArgumentParser(
        description='涡脱周期、Reynolds 数、Strouhal 数计算工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算涡脱周期 (手动指定 St)
  python calc_vortex_shedding_period.py --D 0.05 --U 2.0 --St 0.13
  
  # 自动估算 St 并计算涡脱周期 (根据 Re 和障碍物类型)
  python calc_vortex_shedding_period.py --D 0.1 --U 1.0 --nu 0.001 --shape cylinder
  python calc_vortex_shedding_period.py --D 0.05 --U 2.0 --nu 0.001 --shape square
  
  # 仅计算 Reynolds 数
  python calc_vortex_shedding_period.py --calc-re --D 0.1 --U 1.0 --nu 0.001
  
  # 演示 DRLinFluids 案例
  python calc_vortex_shedding_period.py --demo
        """
    )
    parser.add_argument('--D', type=float, default=0.05, help='特征长度 [m]')
    parser.add_argument('--U', type=float, default=2.0, help='来流速度 [m/s]')
    parser.add_argument('--St', type=float, default=0.13, help='Strouhal 数')
    parser.add_argument('--dt', type=float, default=0.005, help='时间步长 [s]')
    parser.add_argument('--nu', type=float, default=0.001, help='运动粘度 [m²/s]')
    parser.add_argument('--shape', type=str, choices=['cylinder', 'square'], help='障碍物形状 (cylinder/square)，用于自动估算 St')
    parser.add_argument('--demo', action='store_true', help='演示 DRLinFluids 案例')
    parser.add_argument('--calc-re', action='store_true', help='计算 Reynolds 数')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_drlinfluids_cases()
    elif args.calc_re:
        print_reynolds_result(D=args.D, U=args.U, nu=args.nu)
    elif args.shape:
        # 自动估算 St 并计算涡脱周期
        St = print_strouhal_estimate(D=args.D, U=args.U, nu=args.nu, shape=args.shape)
        if St is not None:
            print()
            print_vortex_period_results(D=args.D, U=args.U, St=St, dt=args.dt, nu=args.nu)
    else:
        print_vortex_period_results(D=args.D, U=args.U, St=args.St, dt=args.dt, nu=args.nu)


if __name__ == '__main__':
    main()
