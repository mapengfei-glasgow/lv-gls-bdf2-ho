# 文献检索记录：GNN × 单元（cell/element）× 固体力学 / 心脏力学

- 检索日期：2026-09-25
- 来源：**arXiv API**（`export.arxiv.org/api/query`），覆盖 cs.LG / cs.CE / physics.* 等分类
- 局限：仅 arXiv。CMAME / JCP / IJNME 等期刊可能有同类工作未收录；建议用 Google Scholar 补查：
  `"node-element" graph "finite element"`、`element-wise message passing solid`

## 检索式（供复现）

| # | 查询 |
|---|---|
| 1 | `all:"simplicial neural network"` |
| 2 | `all:"hypergraph" AND all:"mechanical simulation"` |
| 3 | `all:"MeshGraphNets"` |
| 4 | `all:"tetrahedral mesh" AND all:"graph neural network"` |
| 5 | `all:"left ventricle" AND all:"graph neural network"` |
| 6 | `all:"element-based" AND all:"graph neural network"` |
| 7 | `all:"graph neural network" AND all:"solid mechanics"` |
| 8 | `all:"hypergraph" AND all:"finite element"` |
| 9 | `all:"node-element"` |
| 10 | `all:"graph neural network" AND all:"elasticity" AND all:"finite element" AND all:"surrogate"` |
| 11 | `ti:"Message Passing Simplicial Networks"`（逐字核对） |
| 12 | `ti:"Learning Mesh-Based Simulation with Graph Networks"`（逐字核对） |

---

## A. 单元级消息传递（"去掉边、以单元为传递单位"）

| 论文 | 作者 | 出处 / arXiv | 要点 | 与本项目相关性 |
|---|---|---|---|---|
| **CAttBiGNN**：A finite-element-inspired bipartite graph learned simulator for manufacturability assessment in large-deformation sheet forming | Zhao, Li, Zhou, **Pfaff**, Li | [2605.22845](https://arxiv.org/abs/2605.22845) (2026) | 节点与单元为两类实体的**二部图**，有向节点-单元边；交叉注意力处理器；节点位移增量 + **单元级变形状态**在原生离散上预测；有下采样/上采样变体 | ⭐ 任务最像：大变形固体、节点+单元双输出。MGN 作者下场 |
| **FHGNN**：FEM-Informed Hypergraph Neural Networks for Efficient Elastoplasticity | Yang, Chen, Zhao | [2602.07364](https://arxiv.org/abs/2602.07364) (2026) | **节点-单元超图**；把 FEM 节点/高斯点计算嵌入消息传递层；纯物理驱动无标签；3D 弹塑性、循环加载 | ⭐ 固体+单元超图直接先例（物理内嵌路线） |
| **HGTO**：A Unified Graph-Based Physics-Informed Formulation for Structural Topology Optimization | Liu, Punna, Ma | [2609.15001](https://arxiv.org/abs/2609.15001) (2026) | 单元图上参数化密度；节点-单元超图求结构状态；FEM 运动学/积分/材料/装配作为可微计算图中的显式算子；支持有限变形/弹塑性 | 固体+单元图（拓扑优化场景） |
| **FEIH(φ)-GNN**：A Finite Element-Inspired Hypergraph Neural Network: Application to Fluid Dynamics Simulations | Rui Gao, Indu Kant Deo, Rajeev K. Jaiman | **JCP 504:112866 (2024)**，[2212.14545](https://arxiv.org/abs/2212.14545) | 开山作："nodes connected by **elements rather than edges**"；φ 权重模仿局部刚度矩阵装配；旋转等变 | ⭐ 版本 B 的架构原型（流体场景） |
| Mesh-based Super-Resolution of Fluid Flows with Multiscale GNN | Barwey et al. | [2409.07769](https://arxiv.org/abs/2409.07769) | 直接以**单元为图单位**的 GNN 层；"coincident node 同步"兼容单元型连接 | 单元级实现的工程技巧参考 |
| FSI 系列（同一超图框架） | Gao, Jaiman | [2210.04193](https://arxiv.org/abs/2210.04193)（Phys. Fluids 36 (2024)）、[2507.02242](https://arxiv.org/abs/2507.02242)、[2511.22012](https://arxiv.org/abs/2511.22012) | 超图框架扩展到流固耦合；含**超弹性板**/柔性翼等结构耦合 | 结构耦合的扩展路径参考 |

## B. 单纯复形 / 高阶消息传递（理论工具箱）

| 论文 | 作者 | 出处 / arXiv | 内容 |
|---|---|---|---|
| Simplicial Neural Networks | Ebli, Defferrard, Spreemann | [2010.03633](https://arxiv.org/abs/2010.03633) | 把卷积推广到单纯复形（点/边/三角形均为成员） |
| Principled SNN for Trajectory Prediction | Roddenberry, Glaze, Segarra | ICML 2021, [2102.10058](https://arxiv.org/abs/2102.10058) | 置换/定向等变 + 单纯复形感知卷积 |
| **MPSN**：Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks | Bodnar et al. | ICML 2021, [2103.03212](https://arxiv.org/abs/2103.03212) | 单纯复形上的消息传递 + 表达能力理论（SWL） |
| **EMPSN**：E(n) Equivariant Message Passing Simplicial Networks | Eijkelboom, Hesselink, Bekkers | ICML 2023, [2305.07100](https://arxiv.org/abs/2305.07100) | 把几何/等变信息引入单纯复形消息传递；抗过平滑 |
| Simplicial Attention Networks | Goh, Bodnar, Liò | [2204.09455](https://arxiv.org/abs/2204.09455) | 带（有符号）注意力的单纯复形网络 |
| SCRaWl | Frantzen, Schaub | ICLR 2024, [2404.03434](https://arxiv.org/abs/2404.03434) | 随机游走 + 1D 卷积，低成本替代 |
| Topo-MLP | Ramamurthy et al. | [2312.11862](https://arxiv.org/abs/2312.11862) | 完全不靠消息传递的单纯网络 |

## C. MeshGraphNets 家族（基线与对比）

| 论文 | 出处 / arXiv | 备注 |
|---|---|---|
| **MeshGraphNets**：Learning Mesh-Based Simulation with Graph Networks | ICLR 2021, [2010.03409](https://arxiv.org/abs/2010.03409)（Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia） | **我们的基线**（含结构力学实验） |
| MultiScale MeshGraphNets | ICML'22 workshop, [2210.00612](https://arxiv.org/abs/2210.00612) | 双分辨率消息传递 |
| X-MeshGraphNet | [2411.17164](https://arxiv.org/abs/2411.17164) | 分区 + halo + 多尺度；NVIDIA Modulus |
| MeshGraphNet-Transformer（固体） | [2601.23177](https://arxiv.org/abs/2601.23177) | Transformer 全局处理器；冲击动力学/塑性 |
| HI-MeshGraphNets | [2608.13827](https://arxiv.org/abs/2608.13827) | 分层插值多尺度 |
| PI-MGNs | CMAME 2024, [2402.10681](https://arxiv.org/abs/2402.10681) | 物理信息 + MGN（无标签） |

## D. 心脏力学 GNN（本课题域）

| 论文 | 出处 / arXiv | 要点 |
|---|---|---|
| **HeartSimSage**：Attention-Enhanced GNN for Accelerating Cardiac Mechanics Modeling | [2504.18968](https://arxiv.org/abs/2504.18968) (2025) | 双心室被动位移 FEA emulator；**显式处理纤维方向**、多网格拓扑、生理边界条件；GraphSAGE 式邻域 + Laplace-Dirichlet 空间编码；~13000× GPU 加速 | 
| Rapid prediction of cardiac activation（CRT） | [2506.08987](https://arxiv.org/abs/2506.08987)（npj Digit. Med.） | LV 电生理，GNN vs GINO；非力学但同一几何族 |
| EchoGLAD | MICCAI 2023, [2307.12229](https://arxiv.org/abs/2307.12229) | 超声 LV 关键点检测（边缘相关） |
| EchoGNN | MICCAI 2022, [2208.14003](https://arxiv.org/abs/2208.14003) | 超声 EF 估计（边缘相关） |

## E. 固体力学 GNN（点-边体系，邻域参照）

| 论文 | 出处 / arXiv | 要点 |
|---|---|---|
| A review of GNN applications in mechanics-related domains | [2407.11060](https://arxiv.org/abs/2407.11060) | **领域综述**（含固体力学分类、数据/代码汇总）——索引入口 |
| ROBIN: Diffusion-Based Hierarchical GNN for Simulating Nonlinear Solid Mechanics | [2506.06045](https://arxiv.org/abs/2506.06045) | 扩散+分层消息传递；滚动推理加速；2D/3D 固体基准 |
| Physics-Informed Coarsening for Multigrid Graph Neural Surrogates | ICML 2026, [2605.31013](https://arxiv.org/abs/2605.31013) | 按局部应变/应力活动度粗化；长 rollout 稳定性 |
| A variational physics-informed GNN for heterogeneous solid mechanics | [2609.10983](https://arxiv.org/abs/2609.10983) | 逐单元本构；能量泛函为单一损失；无惩罚界面 |
| Thermodynamics-informed graph neural networks | [2203.01874](https://arxiv.org/abs/2203.01874) | GENERIC 结构约束 + GNN（固体/流体） |
| Energy-conserving equivariant GNN for elasticity of lattice metamaterials | ICLR 2024, [2401.16914](https://arxiv.org/abs/2401.16914) | SE(3) 等变 + 能量守恒；四阶张量 |
| GNN grain-scale anisotropic elastic behavior | [2205.06324](https://arxiv.org/abs/2205.06324) | 晶体弹性 FEM 数据 + 迁移学习 |

---

## F. 结论与对本项目的意义（2026-09-25 检索时的判断）

1. **"GNN + 单元 + 固体力学"不止一篇**：直接同类至少 3–4 篇（CAttBiGNN、FHGNN、HGTO；流体开山是 FEIH），且 2026 年集中出现，方向正在升温。
2. **仍存在的空位**：上述工作分别缺少我们关心的组合要素——"**单元超图 + 跨几何时序 rollout 预测 + 心脏/软组织**"未见已发表工作（检索范围内）。CAttBiGNN 单/少几何、FHGNN 无标签物理驱动、HGTO 拓扑优化。
3. **若走"版本 B"（点保留、单元当超边），精读顺序**：
   **CAttBiGNN**（任务最像）→ **FEIH**（架构与 φ 权重细节）→ **FHGNN**（物理内嵌与损失设计）。
4. **顺带可用**：HeartSimSage（纤维怎么作为输入特征处理、空间编码）；综述 [2407.11060]（找更多同类）。
