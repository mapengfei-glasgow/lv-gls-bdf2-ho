# HO PDF 对照修改版

本目录是针对 HO_GLS_stabilization_rewritten(1).pdf 的修改版本。
原 README_GLS.md、reference_output、versions*.txt、environment-linux-64-explicit.txt
是上传包的历史说明/数据，不能作为本次运行结果。以本文件及本次 results/ 为准。

## 与上传包各方法的关系

- plain P1：位移单场、体积罚能；不是 PDF 方法。
- Lagrange：P2/P1 位移-压力，支持纯乘子和增强拉格朗日；不是 PDF 方法。
- GLS：P1/P1 速度-压力，PSPG + LSIC；是本次修改基础。
- none_p1p1 / none：同一 GLS 驱动中的 P1/P1 / P2/P1 无稳定化对照。

## 实际修改

1. 新增 material_ho_pdf.HolzapfelOgdenPDF；只由 GLS 驱动使用。
   被动能量只将 I1 等容化，I4f/I4s/I8fs 由原始 C 计算。
   W4 使用平滑正部 xi=(x+sqrt(x*x+1e-12))/2，其中 x=I4-1。
   W1 比 PDF 多减去一个常数，对应力及刚度没有影响。
2. mu_eff = a exp(b(I1bar-3)) + 2 af xi_f exp(bf xi_f^2)
   + 2 as xi_s exp(bs xi_s^2)。可选额外 2 afs sqrt(I8^2+eps) exp(bfs I8^2)。
   该可选开关只影响稳定尺度；被动能量始终保留 fs 耦合。
3. omega=ct*rho/dt+cs*mu_eff*dt/h^2；tau_m=Ctau/omega；tau_c=Cc*h^2*omega。
   默认 ct=1.5, cs=3, Ctau=0.03, Cc=0.1。
   PSPG 使用 tau_m，加入压力方程；LSIC 使用 tau_c，加入速度方程。
   完整残量自动求导包含 mu_eff 的变形依赖，未冻结系数。
4. 保留原始速度约束 Rp=cofac(F):grad(v)，不引入 p/kappa 或增强体积项。
   PDF 的 J-1+p/kappa 是可选模型举例，不是对旧速度约束的明确替换指令。
   当前 tau_c 的量纲为 Pa*s，配合速率残差；若换成无量纲的 J-1+p/kappa，
   必须重新明确压力方程及 LSIC 的时间尺度，不能只替换 Rp。
5. 原有主动应力、黏性、边界载荷、材料数值、BE 起步和 BDF2 历史更新保留。
   PDF 未定义主动应力新公式；主动应力内部原有等容化不变量仍保留。
6. 每步输出当前 mu_eff_Pa、tau_m、tau_c，先评估再移动 BDF2 历史。
   verify_gls_results.py 独立重建 F/J、时间递推、动能及新稳定系数。

## Miniconda 安装和运行（Linux x86_64）

使用项目原来指定的 DOLFINx 0.11.0 / Python 3.12。
官方 conda 安装说明：https://fenicsproject.org/download/

```bash
# 安装 Miniconda、创建 dolfinx 环境并运行 3 步
SIMPLE_3D0D_STEPS=3 SIMPLE_3D0D_OUTPUT_DIR=out_pdf_smoke bash install_miniconda.sh
./miniconda3/bin/conda run -p ./env_gls python verify_gls_results.py out_pdf_smoke

# 完整 200 步，dt=0.005 s，总时长 1 s
SIMPLE_3D0D_OUTPUT_DIR=out_pdf_cycle ./miniconda3/bin/conda run -p ./env_gls --no-capture-output bash run_gls.sh
./miniconda3/bin/conda run -p ./env_gls python verify_gls_results.py out_pdf_cycle

# 可选：把 fs 耦合加入稳定尺度
LV_GLS_INCLUDE_FS=1 SIMPLE_3D0D_STEPS=3 SIMPLE_3D0D_OUTPUT_DIR=out_pdf_fs ./miniconda3/bin/conda run -p ./env_gls --no-capture-output bash run_gls.sh
```

可通过 MINICONDA_PREFIX、GLS_ENV_PREFIX 指定安装脚本的路径。
稳定参数：LV_GLS_CT、LV_GLS_CS、LV_GLS_CTAU、LV_GLS_CC、LV_GLS_INCLUDE_FS。
旧 LV_GLS_C1/C2/MU_REF 已拒绝使用，避免误以为旧参数仍生效。
所有输出目录必须是新目录；不要覆盖已经保存的 VTU。

## 解释范围

单份 PDF 无法证明其相对更早 PDF 的编辑历史；上述为 PDF 与上传代码的差别。
默认选用 PDF 允许的不计 fs 刚度形式。速率不可压缩允许离散体积漂移，
SNES 收敛不等于逐点 J=1。P1 单元强残差未加跨单元应力跳跃项。
验证一次周期不能代替网格/时间步收敛性和材料/实验验证。
