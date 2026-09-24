# 复现、取数据与分享

## 1. 仓库里有什么、没有什么

| 在仓库里（约 12 MB） | 不在仓库里（可重新生成，本地约 1.9 GB） |
|---|---|
| 代码 `lvsolver/`、`tools/`、运行脚本 `*.command` | VTK 帧 `results*/*/vtk/`（ParaView 用，占 1.9 GB 的绝大部分） |
| 网格 `mesh/lv_aha17_v2_plain/`（972 单元）、`mesh/dalton_geom_0000/`（真实 LV，59940 单元） | 生成的加密网格 `mesh/lv_plain_h*/`（`convergence.command` 自动生成） |
| 每个算例的 `monitor.csv`、`metadata.json`、`validation.json`、汇总 `summary.md`、`convergence.md` | 日志 `log.txt`、`verify.txt` |
| 论文 4.6 节：`paper/sec46_lv/`（tex、图、数据、作图脚本） | |

## 2. 环境

```bash
conda env create -n lv0d3d -f environment.yml     # DOLFINx 0.11.0, PETSc (real), meshio 5.3.5, gmsh
```

已在 macOS arm64 上测试（DOLFINx 0.11.0 / PETSc 3.25.5 / Python 3.12.13）。Linux 也应能运行，但未测试。
`*.command` 文件是普通 bash 脚本：在 macOS 上双击即可；在 Linux 上用 `bash xxx.command lv0d3d` 运行。

## 3. 重新跑出结果

| 脚本 | 内容 | 输出 | macOS 上耗时 |
|---|---|---|---|
| `run_all.command` | 10 个算例（P1 / Lagrange / GLS 各变体，972 单元网格） | `results/` | 约 5 分钟 |
| `convergence.command` | 用 gmsh 生成 5 级网格，GLS（$C_c=0.1$）网格加密 | `mesh/lv_plain_h*/`、`results_convergence/` | 约 15 分钟 |
| `paper/sec46_lv/run_paper_case.command` | 论文 4.6 节算例（h = 0.35，$\mu_{\rm eff}$ 含 fs 项），并重新作图 | `results_paper/`、`paper/sec46_lv/figures/` | 约 6 分钟 |

跑完后，与仓库里提交的结果逐列比较：

```bash
python tools/compare_runs.py results --ref-git HEAD      # 默认容差 1e-10
```

`run_all.command` 会自动做这一步。在同一平台上结果应逐位一致。换平台时，`V_mL`（Delaunay 体积估计）可能相差几个百分点，其余量应在舍入误差量级。

**注意：** 加密网格由 gmsh 现场生成，gmsh 版本不同时网格可能不同（例如 h = 1.5 在一个版本下是 899 单元，另一个版本是 972 单元）。
复现前先核对单元数：h = 1.5 / 1.0 / 0.7 / 0.5 / 0.35 应分别为 899 / 1921 / 3786 / 8954 / 21707 个单元（见 `mesh/lv_plain_h*/info.json`）。
如果单元数不同，结果会有差别但结论相同。若要逐位复现，需要另外传递 `mesh/lv_plain_h*/`（h = 0.35 约 1 MB）。

## 4. 自己取数据重新画图

| 想画什么 | 用哪个文件 |
|---|---|
| 某个算例的时间曲线（体积、$J$、位移、压力、Newton 次数、残量） | `results*/<case>/monitor.csv`，一行对应一个时间步，列名见表头 |
| 算例的参数和配置 | `results*/<case>/metadata.json` |
| 各算例的汇总统计 | `results*/<case>/validation.json`、`results/summary.md` |
| 网格加密对比 | `results_convergence/*/monitor.csv`、`results_convergence/convergence.md`；精确腔体体积用 `tools/convergence.py` 里的 `cavity_volume(<result_dir>)` |
| 论文 4.6 节的曲线 | `paper/sec46_lv/data/lv_timeseries.csv`（含精确腔体体积、长轴长度、赤道壁厚），作图脚本 `paper/sec46_lv/make_figures.py` |
| 从别的算例重新提取 4.6 节那种数据 | `python paper/sec46_lv/extract_data.py <result_dir> out.csv` |
| 三维场（位移、速度、压力、$J$、纤维、$\mu_{\rm eff}$、$\tau_m$、$\tau_c$） | `results*/<case>/vtk/<method>.pvd`（ParaView），第 k 帧对应 $t=0.005(k+1)$ s |

注意：`monitor.csv` 里的 `V_mL` 是 Delaunay 估计，大变形时不可靠；腔体体积请用 `cavity_volume` 或 `lv_timeseries.csv` 里的 `V_cavity_mL`。

## 5. 真实 LV 网格（Dalton 数据集）

`mesh/dalton_geom_0000/` 来自 huggingface.co/datasets/mapengfei/dalton-lv-meshes（geom_0000）。其他几何可以用 `bash mesh/download_dalton.command 0007` 下载。
这个网格的坐标单位是 cm，长轴为 $x$，基底在 $x=0$；面标签 5 = 基底、6 = 内膜、7 = 外膜；纤维和 sheet 逐单元存储，都是单位向量且相互正交，所以参考构形无应力。
它和默认参数兼容，运行命令（尚未运行过）：

```bash
python -m lvsolver.run --method gls --tau pdf --include-fs --mesh mesh/dalton_geom_0000 --out results_dalton/geom_0000
```

## 6. 分享给别人

- **快照压缩包（推荐）：** 在仓库根目录运行 `git archive --format=zip -o dist/lvsolver.zip HEAD`，得到只含已提交文件的 zip（约 10 MB）。对方解压后按第 2、3 节操作即可。
- **带完整历史：** `git bundle create dist/lvsolver.bundle --all`；对方运行 `git clone lvsolver.bundle lvsolver` 即可得到完整仓库。
- **长期协作：** 推送到 GitHub 或 GitLab 的私有仓库。
- VTK 结果不要打包，对方运行脚本就能重新生成。如果对方只需要看 ParaView 图而不想运行，可以单独打包某一个算例的 `vtk/` 文件夹（例如 `results_paper/gls_pdf_fs_h0.35/vtk`，约 570 MB，压缩后会小一些）。
