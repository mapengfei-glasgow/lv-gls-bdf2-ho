# 左心室动态 P1/P1 GLS 移植试验

本次将上传的 `DynamicConsistentGLSBDF2Solver` 中 **GLS 两项、tau 公式和 BE/BDF2 时间离散**移植到 DOLFINx 0.11.0 左心室框架。默认启用 `gls`，不启用 OSS，不叠加此前的增强拉格朗日体积项。

这是附件公式的可运行移植试验，不代表该 GLS 公式已经完成量纲、网格独立性或物理有效性验证。尤其不能把 SNES 收敛等同于逐点不可压缩。

## 与上传代码的对应

| 上传代码 | 本次实现 |
|---|---|
| legacy `dolfin` / NonlinearVariationalSolver | DOLFINx `fem.petsc.NonlinearProblem` / PETSc SNES + MUMPS |
| P1/P1 速度和压力未知量 | 保留；位移由速度的时间离散关系更新 |
| 第一步 BE，之后 BDF2 | 保留；可用 `LV_GLS_TIME_SCHEME=be` 全程 BE 作对照 |
| GLS 的 LSIC 型项和 PSPG 型项 | 按附件的正号和同一个 tau 直接移植 |
| `c1=1, c2=2`（驱动值） | 默认相同，可配置 |
| `mu=G+Gf` | LV Holzapfel–Ogden 没有同名常数，显式用参考尺度代理，见下文 |
| StandardReinforcedModel | 保留当前 LV 的 Holzapfel–Ogden、主动应力和黏性 |
| 柱底固定、自由扭转 | 保留当前 LV 内膜压力、心外膜阻尼、基底罚约束 |
| OSS 投影 | 本次未移植，仅实现所请求的 GLS |
| 线性外推初值 | 使用前一步解作为较保守初值；不改变 BDF2 历史公式 |

上传驱动中的 `stab_mode = "none_p1p1"` 实际关闭稳定化，不能把该驱动的默认结果直接称为 GLS 结果。本次默认是 `gls`。

## 方程

令 `F=I+Grad(u)`、`J=det(F)`、`H=cofac(F)`。正 p 表示压缩压力。

第一步：

```
a = (v - v_n)/dt
u = u_n + dt*v
```

第二步及以后：

```
a = (3*v - 4*v_n + v_nn)/(2*dt)
u = (4*u_n - u_nn + 2*dt*v)/3
```

体积应力由压力乘子承担，原 `W_compress` 被移除，没有 `p/kappa` 松弛项，也没有增强拉格朗日项：

```
P = d(W_iso)/dF - p*H + P_viscous + P_active
R_mom = rho*a - Div(P)
R_p = inner(H, Grad(v))
G_v = ∫ rho*a·w + P:Grad(w) dx + 原有边界项
G_p = ∫ q*R_p dx
```

显式选用不可压缩速率残差 R_p，不按 `hasattr(material, 'kappa')` 自动分支。否则 LV 材料对象中保留的 kappa 属性会使附件分支引入 `J-1+p/kappa`，从而改变模型。

附件 GLS 项按原样移植为：

```
tau = 1/(c1*mu_ref/h**2 + c2*rho/dt**2)
S_v = ∫ tau * inner(H, Grad(w)) * R_p dx
S_p = ∫ tau * inner(R_mom, H*Grad(q)) dx
Residual = G_v + G_p + S_v + S_p
```

Jacobian 对完整残量自动求导，包含稳定化项的非线性依赖。动量强残差使用与 Galerkin 相同的主动、黏性和被动应力。

## 必须保留的解释边界

1. **附件稳定化量纲待核对。** 按 SI 量纲，tau 的单位是 m²/Pa。对本速度—压力形式，`q R_p` 的量纲为 `[q]/s`，而 `tau R_mom·H Grad(q)` 为 `[q]`，两者相差时间尺度；同一 tau 用在 S_v 也不能直接与动量弱式配平。本次没有悄悄加入 dt 因子、重新定义 LSIC 系数或无量纲化，因此结果标为“reference-form 实验”。在用于定量研究前应结合原论文明确缩放。本附件没有提供其 [27,28] 文献信息。
2. **速率约束会有体积漂移。** 它与上一版直接使用 `q(J-1)` 的约束不同。GLS 压力方程是 G_p+S_p=0，而不是 G_p 单独为零；输出 J_mean 和 J_RMS 跟踪实际变化。
3. **参考剪切尺度不是各向异性切线模量。** 默认 `mu_ref=a+a_f+a_s+a_fs=3597.941 Pa`，仅为显式、可复现的正数尺度代理。它不是从 Holzapfel–Ogden 切线张量严格推导的 mu，也不是材料参数替换。可通过 `LV_GLS_MU_REF` 指定并开展敏感性检查。
4. **强残差按单元内部计算。** P1 位移/速度在每个单元内梯度常数，当前纤维为 DG0；`Div(P)` 不包含跨单元的应力跳跃分布项。未额外加入界面或边界 GLS 项。
5. **基底没有改为完全固定。** 长轴 x 和截面切向位移仍由原罚项限制，径向自由；内膜压力、外膜法向速度阻尼保留。
6. `V_mL` 继续使用原框架的 Delaunay 封腔体积估计，非精确曲面体积积分。

## 运行

使用上一轮已安装的 Miniconda / DOLFINx 环境：

```bash
conda activate /your/path/lv_env
bash run_gls.sh
```

完整周期默认 dt=0.005 s、200 步，输出到 `out_gls_bdf2/`。使用新的输出目录，程序拒绝覆盖已有 VTU。

```bash
# 3 步检查
SIMPLE_3D0D_STEPS=3 SIMPLE_3D0D_OUTPUT_DIR=out_gls_smoke bash run_gls.sh
# 同空间、同时间格式，关闭 GLS 的诊断对照（可能有不稳定压力）
LV_GLS_MODE=none_p1p1 SIMPLE_3D0D_OUTPUT_DIR=out_no_gls bash run_gls.sh
# P2/P1 无稳定化的 BDF2 对照
LV_GLS_MODE=none SIMPLE_3D0D_OUTPUT_DIR=out_p2p1_bdf2 bash run_gls.sh
# 配置 tau 参数
LV_GLS_C1=1 LV_GLS_C2=2 LV_GLS_MU_REF=3597.941 SIMPLE_3D0D_OUTPUT_DIR=out_gls_custom bash run_gls.sh
```

可用 `environment-linux-64-explicit.txt` 重建此前 Linux 环境：

```bash
conda create -p ./env_gls --file environment-linux-64-explicit.txt
conda run -p ./env_gls --no-capture-output bash run_gls.sh
```

`run_gls.sh` 使用单 MPI 进程；输出程序主动拒绝多进程，避免文件互相覆盖。SNES 不收敛或发现非正 J 时立即停止，元数据写为 failed，保留此前已通过检查的帧，绝不标成完整周期。

## 结果

- `vtk/gls.pvd`：ParaView 时间序列；Warp By Vector 选择 displacement，Scale Factor=1。
- `vtk/gls_XXXXX.vtu`：displacement [m]、velocity [m/s]、lagrange_pressure_Pa；单元场 J、tau_reference_GLS、f0、s0。
- `monitor_gls.csv`：J 范围/RMS/均值、压力范围、动能、位移/速度、迭代次数、GLS 子残量诊断。
- `run_metadata.json`：实际参数、格式和完成/失败状态。

诊断范数是代数向量范数，受测试空间尺度和量纲影响，不可直接拿不同方程的范数比较物理大小。`pspg_norm`/`lsic_norm` 在关闭稳定化的模式中也计算候选项，用于诊断，但不加入该模式的方程。
