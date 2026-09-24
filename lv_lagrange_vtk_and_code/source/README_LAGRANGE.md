# 左心室拉格朗日乘子版本

在原模型上新增不可压缩 P2/P1 Taylor–Hood 混合有限元：位移 u 为二次连续场，压力 p 为一次连续场。这里将“拉格朗日乘子”解释为不可压缩约束的压力乘子；基底约束仍使用原程序的边界罚项。

混合残量的新增项为

- 位移方程：`-∫ p cof(F):grad(w) dx`
- 压力方程：`-∫ q (J-1) dx = 0`

`F=I+grad(u)`，`J=det(F)`，正 p 表示压缩压力，单位 Pa。原体积罚能 W_compress 被移除，并加入一致的增强项 `β/2·(J-1)^2`，默认 β=1e6 Pa。该项对精确不可压缩解为零，作用于离散场中尚未消除的局部体积变化；压力测试方程仍为 `∫q(J-1)dx=0`，没有 `p/κ` 松弛项。保留原主动应力、黏性、惯性、压力载荷、心外膜阻尼和基底约束。由于内膜/外膜有自然边界，未任意固定乘子平均值。

纯乘子版本在本粗网格上出现很大的局部体积误差，因此交付 VTK 采用增强拉格朗日版本。设置 `SIMPLE_3D0D_LM_AUGMENTATION=0` 可运行纯乘子版本；任意非负值以 Pa 为单位。

约束在 P1 压力测试空间中弱满足；不保证 J 在每个点严格等于 1。输出包含积分计算的 `J_minus_1_RMS` 和五阶求积点的 J 范围。网格较粗，结果仍需网格和时间步收敛研究。

## 安装与运行（Linux x86_64）

```
bash install_miniconda.sh
```

该脚本安装 Miniconda、创建 DOLFINx 0.11.0 环境、运行完整周期并打包。本次实际环境锁定在 `environment-linux-64-explicit.txt`，可用 `conda create -p ./env_lagrange --file environment-linux-64-explicit.txt` 精确重建 Linux 环境。

已有兼容环境时：

```
bash run_lagrange.sh
```

短步检查：

```
SIMPLE_3D0D_STEPS=3 SIMPLE_3D0D_OUTPUT_DIR=smoke_lagrange bash run_lagrange.sh
```

完整周期默认 dt=0.005 s、200 步，到 t=1 s。仅支持单 MPI 进程导出。任何 SNES 不收敛、非有限状态或求积点 J≤0 会立即报错停止；失败运行不会自动打包。

## 新文件

- `solidmechanics_lagrange.py`：混合求解器，继承原有载荷和动力学。
- `run_lagrange_vtk_cycle.py`：完整周期驱动，二次四面体 VTK 导出。
- `run_lagrange.sh`：运行并生成结果 ZIP。
- `install_miniconda.sh`：Miniconda 环境安装与运行。

## VTK

在 ParaView 打开 `vtk/lagrange.pvd`。坐标为参考构形，单位 m；使用 Warp By Vector / displacement、Scale Factor=1 显示变形。

- `displacement`：m，P2 节点值。
- `velocity`：m/s，P2 节点值。
- `lagrange_pressure_Pa`：P1 乘子压力，插值到 P2 输出节点；区别于规定的内膜压力。增强模式中，总体积应力为 `(-p+β(J-1))cof(F)`，导出的乘子 p 不包括增强项。
- `J_cell_center`：单元中心采样 J（不是单元平均）。
- `f0`、`s0`：参考纤维/片层方向。

监控 CSV 中 `V_mL` 沿用附件的 Delaunay 封腔体积算法，改为 P2 内膜节点；它是原算法的体积估计，并非曲面精确积分，因此与原 P1 数值不应视为纯粹的离散格式对照。原 reference_output 是罚函数参考结果，不是本次混合结果。

安装依据：https://fenicsproject.org/download/
求解器接口：https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html
