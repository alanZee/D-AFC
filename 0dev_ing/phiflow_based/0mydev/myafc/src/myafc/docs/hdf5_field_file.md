# HDF5 字段规范（myafc）

简要说明 Runner 写入的 `fields.h5` 格式与读取约定。

---

## fields.h5 文件内部结构示例
fields.h5
├── attrs (文件级属性)
│   ├── resolution: [Nx, Ny]
│   ├── domain_size: [Lx, Ly]
│   ├── dt: 时间步长
│   └── array_dim: 'x,y'
├── step_0 (时间步 0 组)
│   ├── velocity (维度(Nx, Ny, 2)) (数据集)
│   ├── pressure (维度(Nx, Ny)) (数据集)
│   ├── vorticity (维度(Nx, Ny)) (可选数据集)
│   └── ... (其他可选场)
├── step_10 (时间步 10 组)
└── ...

---

## 快速概览
- 布局：step-group（组名 `step_<N>`，如 `step_10`）。
- 必含：`velocity`（`(Nx, Ny, 2)`）、`pressure`（`(Nx, Ny)`）。
- 可选：`vorticity`, `velocity_magnitude`, `sdf`, `mask`（均为 `(Nx, Ny)`）。
- 文件属性（attrs）：`resolution=[Nx,Ny]`, `domain_size=[Lx,Ly]`, `dt`, `array_dim='x,y'`。

## 配置（示例）
```yaml
runner:
  fields:
    save:
      enabled: true
      interval: 10
      filename: 'fields.h5'
      quantities:              # 要保存的场类型（velocity/pressure 始终保存）
        - velocity             # 速度场
        - pressure             # 压力场
        - vorticity            # 涡量场
        # - velocity_magnitude # 速度大小场（可选）
```

## 读取约定（示例）
- 按数值解析并排序 step 名（避免字典序错误），选取最新或指定 step。
- 读取时把数组从 `(Nx, Ny[, C])` 转为常用可视化顺序 `(Ny, Nx[, C])`（即做 .T 或 transpose）。

示例代码（返回以可视化顺序排列的数组）：

```python
import h5py, numpy as np

def read_last_frame(path):
    with h5py.File(path,'r') as f:
        steps = sorted(int(k.split('_',1)[1]) for k in f.keys() if k.startswith('step_'))
        if not steps:
            raise RuntimeError('no step groups')
        grp = f[f'step_{steps[-1]}']
        out = {}
        for name in grp.keys():
            a = grp[name][()]
            out[name] = a.transpose(1,0,2) if a.ndim==3 else a.T
        return out
```

## Resume 行为（要点）
- 严格续算（`resume_strict: true`）：从指定 step 恢复，**自动删除**该 step 之后的所有输出（H5、CSV、图像），并在同一目录追加新数据；时间步计数接续。
- 热启动（`resume_strict: false`）：从指定 step 读场初始化，但不改原始文件，输出到新目录，时间步从 0 重新开始。

## 最佳实践
- 写入 `resolution` 与 `array_dim='x,y'`；读取端应以此为准并在需要时转置；
- 只保存必要的 `quantities` 以节约存储；
- 由 Runner 管理 resume 清理规则，用户只需配置 `initial.resume_*`。

