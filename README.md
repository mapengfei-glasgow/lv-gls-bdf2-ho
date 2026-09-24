# LV 动态固体力学：P1 罚函数 / P2-P1 Lagrange / P1-P1 GLS

理想化左心室（`mesh/lv_aha17_v2_plain`，972 个 P1 四面体，cm → m）在给定载荷下的一个心动周期
（$\Delta t = 0.005$ s，200 步）。三种体积约束离散共用同一个模型，差别只在单元、未知量和不可压缩处理。

## 共同模型

$$
\rho\,\ddot{\mathbf u} - \nabla\!\cdot\!\big(\mathbf P_{\text{pas}} + \mathbf P_{\text{vis}} + \mathbf P_{\text{act}}\big) = \mathbf 0,
\qquad \mathbf F = \mathbf I + \nabla \mathbf u,\ J = \det \mathbf F,\ \mathbf H = \operatorname{cof}\mathbf F .
$$

- 被动：Holzapfel–Ogden，$W = W_1(\bar I_1) + W_4(I_{4f}) + W_4(I_{4s}) + W_8(I_{8fs})$。
  - `ho`：$I_{4f}, I_{4s}, I_{8fs}$ 用等容 $\bar{\mathbf C}$，纤维开关为光滑 Heaviside（原 LV 程序）。
  - `ho_pdf`：只有 $I_1$ 等容，其余用原始 $\mathbf C$；开关为 $\xi = \tfrac12\big(x+\sqrt{x^2+\varepsilon}\big)$，$x = I_4-1$，$\varepsilon=10^{-12}$（GLS PDF）。
- 黏性：$W_{\text{vis}} = \tfrac{\eta}{2}\operatorname{tr}(\dot{\mathbf E}^2)$，$\eta = 100$ Pa·s。
- 主动：$\mathbf P_{\text{act}} = T_a(t)\,\big(1 + 4.9(\sqrt{\bar I_{4f}}-1)\big)\,\mathbf F\,\mathbf f_0\otimes\mathbf f_0$。
- 载荷：内膜随动压力 $p_{\text{endo}}(t)$（式 79），主动张力 $T_a(t)$（式 80），见 `lvsolver/model.py`。
- 边界：心外膜法向阻尼 $\beta_{\text{epi}} = 5\times10^3$；基底罚项 $\gamma = 10^6$，约束轴向位移，面内只允许沿参考径向运动。
- $\rho = 10^3\ \text{kg/m}^3$，材料常数见 `model.CONSTITUTIVE_PARAMS`。

## 三种方法

| `--method` | 未知量 / 单元 | 体积项 | 时间离散 |
|---|---|---|---|
| `p1` | $\mathbf u\in P_1$ | 罚能 $W_c = \tfrac{\kappa}{4}(J^2-1-2\ln J)$，$\kappa=10^6$ Pa | $\mathbf a = (\mathbf u-\mathbf u_n-\Delta t\,\mathbf v_n)/\Delta t^2$ |
| `lagrange` | $(\mathbf u,p)\in P_2\times P_1$ | $-p\,\mathbf H + \tfrac{\beta}{2}(J-1)^2$，$\int q\,(J-1)=0$ | 同上 |
| `gls` | $(\mathbf v,p)\in P_1\times P_1$ | 速率约束 $R_p = \mathbf H:\nabla\mathbf v = 0$，加 PSPG + LSIC | 首步 BE，之后 BDF2 |

`lagrange`：`--beta 1e6`（增强，默认）或 `--beta 0`（纯乘子）。

`gls`：强残量 $\mathbf R_m = \rho\mathbf a - \nabla\!\cdot\!\mathbf P$，稳定项
$S_c = \int \tau_c\,(\mathbf H:\nabla\mathbf w)\,R_p$（加在速度方程），
$S_m = \int \tau_m\,(\mathbf H\nabla q)\cdot\mathbf R_m$（加在压力方程）。$h$ 取单元直径。

- `--tau pdf`（默认，材料 `ho_pdf`）：
  $\omega = c_t\rho/\Delta t + c_s\,\mu_{\text{eff}}\Delta t/h^2$，$\tau_m = C_\tau/\omega$，$\tau_c = C_c h^2\omega$，
  $\mu_{\text{eff}} = a e^{b(\bar I_1-3)} + 2a_f\xi_f e^{b_f\xi_f^2} + 2a_s\xi_s e^{b_s\xi_s^2}$（`--include-fs` 再加 fs 项）。
  默认 $c_t=1.5,\ c_s=3,\ C_\tau=0.03,\ C_c=0.1$。
- `--tau reference`（材料 `ho`）：上传 GLS 求解器的原式 $\tau = 1/(c_1\mu_{\text{ref}}/h^2 + c_2\rho/\Delta t^2)$，
  $\mu_{\text{ref}} = a+a_f+a_s+a_{fs}$，两项共用；量纲未配平，只作对照。
- `--no-stab`：稳定项只作诊断，不进方程（Galerkin 对照）；此时可用 `--velocity-degree 2`（P2/P1）。

## 运行

```bash
conda env create -n lv0d3d -f environment.yml
bash run_all.command            # 或在 Finder 里双击
```

`run_all.command` 依次运行 10 个算例、逐个校验，然后写 `results/summary.md`，
并与上一次提交的 `results/*/monitor.csv` 逐列比较（`tools/compare_runs.py`，默认容差 $10^{-10}$）。
在 macOS arm64 上总共约 5 分钟。

| 算例 | 命令参数 |
|---|---|
| `p1` | `--method p1` |
| `lagrange` | `--method lagrange --beta 1e6` |
| `lagrange_pure` | `--method lagrange --beta 0` |
| `gls_pdf` | `--method gls --tau pdf` |
| `gls_pdf_fs` | `--method gls --tau pdf --include-fs`（PDF §2.2，$\mu_{\text{eff}}$ 计入 fs） |
| `gls_pdf_cc1` / `cc10` / `cc100` | `--method gls --tau pdf --cc 1` / `10` / `100`（LSIC 强度扫描，默认 $C_c=0.1$） |
| `gls_reference` | `--method gls --tau reference` |
| `gls_reference_nostab_3steps` | `--method gls --tau reference --no-stab --steps 3` |

单个算例：`python -m lvsolver.run --method gls --tau pdf --out results/my_case`，
全部参数见 `python -m lvsolver.run -h`。

## 输出（`results/<case>/`）

- `monitor.csv`：每步的 $V$、$J$ 在单元中心和五阶求积点上的极值、$\mathrm{RMS}(J-1)$、$\bar J$、$|\mathbf u|_{\max}$、$|\mathbf v|_{\max}$、压力范围、动能、Newton 次数；GLS 另有各残量范数。
- `metadata.json`：方法、参数、网格、状态。
- `validation.json`：`python -m lvsolver.verify` 的独立检查结果。对 P1 输出，从 VTU 节点场重建 BE/BDF2 递推、$J$、$\bar J$、$\mathrm{RMS}(J-1)$、动能和稳定尺度。
- `vtk/<method>.pvd`：ParaView 打开，含位移、速度、压力、$J$、纤维，GLS 另有 $\mu_{\text{eff}}, \tau_m, \tau_c$ 或 $\tau$。不纳入 git。

## 复现、取数据与分享

见 `REPRODUCE.md`：环境、三个运行脚本、结果核对、每类数据对应的文件，以及如何打包发给别人。

## 结构

```
lvsolver/model.py      载荷曲线、材料常数、SNES/MUMPS 选项
lvsolver/mesh.py       XDMF 网格与纤维读取、基底中心、腔体体积
lvsolver/material.py   HolzapfelOgden / HolzapfelOgdenPDF
lvsolver/problems.py   P1Penalty / Lagrange / GLS
lvsolver/run.py        命令行、逐步诊断、CSV/VTU 输出
lvsolver/verify.py     独立校验与 summary.md
tools/compare_runs.py  两次运行的 monitor.csv 对比
tools/convergence.py   网格加密汇总、精确腔体体积 cavity_volume
tools/generate_mesh.py 生成 lv_aha17_v2_plain 网格（需 gmsh），例：
                       python tools/generate_mesh.py --mesh-size 1.0 \
                         --xdmf mesh/lv_plain_h1.0/lv_aha17_v2_plain_mesh.xdmf --out /tmp/lv_bin
mesh/                  lv_aha17_v2_plain（972 单元）、dalton_geom_0000（真实 LV）、download_dalton.command
convergence.command    网格加密研究 -> results_convergence/
paper/sec46_lv/        论文 4.6 节：tex、图、数据、run_paper_case.command -> results_paper/
```

## 解释范围

- 腔体体积 $V$ 是对内膜自由度点做 Delaunay 剖分后得到的估计值。$P_2$ 内膜点集是退化点集，剖分不唯一，所以 `lagrange` 的 $V$ 在不同平台（qhull 版本）之间可相差约 4%。在同一平台上结果可以逐位复现。
- 速率约束和弱约束都允许局部体积误差：SNES 收敛不代表逐点 $J=1$，$\bar J\approx1$ 也掩盖不了局部误差，要看 $J$ 的极值和 $\mathrm{RMS}(J-1)$。
- P1 单元内强残量没有计入单元间应力跳跃项。单次周期不能替代网格和时间步收敛研究，也不能替代实验验证。
- 敏感性（`run_all` 的 10 个算例）：
  - 在 $\mu_{\text{eff}}$ 中计入 fs 项（`gls_pdf_fs`）几乎不影响解：$V$ 相差 < 0.1%，$J$ 极值和 $\mathrm{RMS}(J-1)$ 相差 < 1%，压力极值相差 < 5%。
  - GLS 结果对 LSIC 强度很敏感：$C_c$ 从 0.1 增大到 100 时，$\max\mathrm{RMS}(J-1)$ 从 0.198 降到 0.102，末时刻 $V$ 变化 10%。可见在这套网格上，GLS 的解还受稳定参数控制，没有达到网格无关。
  - 局部体积误差主要由局部体积刚度决定，和单元类型关系不大：`lagrange_pure`（P2/P1，$\beta=0$）的 $\max\mathrm{RMS}(J-1)=0.33$，比 `gls_pdf` 还大；`lagrange`（$\beta=10^6$）和 `p1`（$\kappa=10^6$）好，是因为有 $10^6$ Pa 的局部刚度。约束写成速率形式还是 $J-1$ 形式影响有多大，现有算例还不能单独区分。
- 网格加密（`convergence.command`，`gls --tau pdf`，$C_c=0.1$，5 级网格 899–21707 个单元，见 `results_convergence/convergence.md`）：
  - 全局量大致收敛：用曲面积分计算的腔体体积 $V(1\,\mathrm s)$ 依次为 355 / 372 / 376 / 388 / 390 mL，末两级相差 1.3%。
  - 局部不可压缩性不收敛：$\max\mathrm{RMS}(J-1)$ 依次为 0.197 / 0.187 / 0.166 / 0.361 / 0.371，$J_{\max}$ 从 1.85 增大到 3.03。在最细一级上，误差分布在收缩期约 29% 的心肌体积里，不是只集中在基底或心尖；舒张期的 RMS 仍在 0.08 左右。
  - `monitor.csv` 里的 `V_mL` 是 Delaunay 估计，大变形下不可靠（加密研究中各级之间最多相差 2 倍），改用 `tools/convergence.py` 中的 `cavity_volume`。

## 代理模型（surrogate/）

训练数据由本文 GLS 求解器生成（`lvsolver.run --no-vtk --save-fields`），网格来自 HF Dalton 数据集，纤维用 `tools/rule_based_fibers.py` 生成（+60°/−60°），不使用数据集自带纤维。
除几何外还变化 8 个参数：$a,b,a_f,b_f$（名义值 ×[2/3, 1.5]，对数均匀），$T_a$ 幅值、$p_{\mathrm{ED}}$、$p$ 收缩幅值、周期 $T$（范围见 `surrogate/params.py`）。

```bash
bash surrogate/run_smoke.command          # 下载 10 个几何 → Sobol 参数 → 求解 → 训练 → 评估
python -m surrogate.sample_params --ids 0000-2966 --out params_all.csv    # 服务器全量
python -m surrogate.generate params_all.csv surrogate_data --jobs 32
```
训练环境：`conda env create -f environment_ml.yml`（lvml）。数据格式见 `lvsolver/export.py`。
