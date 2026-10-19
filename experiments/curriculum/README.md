# 自动串联几何课程：±0.25σ → ±0.5σ → ±1.0σ

`run.py` 是可断点续跑的控制器，部署在共享仓库的 euclid-17 上。它按顺序执行：

1. 等待当前窄几何模型的 `last.pt` 到达该运行计划的最终迭代次数。
2. 检查 ±0.5σ 联合采样数据。当前 600 个案例已经验证，因此直接复用；若没有，自动生成、求解、重试并验证。
3. 从窄模型**已保存检查点中训练损失最低**的权重开始，训练 ±0.5σ 模型。
4. 只有上一步训练完成后，才开始 ±1.0σ 的联合采样、求解、重试和独立验证。
5. 从 ±0.5σ 阶段最低训练损失的已保存模型开始，训练 ±1.0σ 模型。

两个新阶段默认各 600 个联合样本、120,000 次训练迭代，batch=4，Adam，
学习率余弦退火 1e-4→1e-6，位置噪声 5e-5 m。前 16 个 PCA 系数分别位于
±0.5σ / ±1.0σ；其余 24 个系数为零。物理参数范围保持不变。每个样本均
联合采样几何与八个物理参数，不使用旧的 30×20 交叉采样。±0.5σ 种子为
20261020；±1.0σ 为 20261022。每阶段分别划分 456/72/72 个训练/验证/测试案例。
测试集不参与训练和检查点选择。

## 独立输出

| 阶段 | 数据目录 | 模型目录 |
|---|---|---|
| 已有 ±0.25σ | `surrogate_data.geometry_narrow` | `surrogate_runs/geometry_narrow_from_mean_loss` |
| ±0.5σ | `surrogate_data.geometry_joint` | `surrogate_runs/geometry_joint_sigma050` |
| ±1.0σ | `surrogate_data.geometry_joint_sigma100` | `surrogate_runs/geometry_joint_sigma100` |

已有单几何和窄几何模型只读。初始化权重会复制到控制目录中的 `sigma050/initial.pt`
或 `sigma100/initial.pt`，记录来源、训练损失、迭代数和 SHA256。每阶段保留
`best_loss.pt`（训练损失）、`best.pt`（验证 rollout）和 `last.pt`。下一阶段按
**训练损失**选择，而不是按 rollout 选择。保留权重、图结构和 normaliser，重新开始
优化器与学习率计划。当前模型没有新增纤维输入，本脚本也不改变网络结构。

## 启动与恢复

在共享 Euclid 仓库执行：

```bash
export TRAIN_PY="$HOME/.venvs/lv-mean-base/bin/python"
export SOLVER_PY="$HOME/miniconda3/envs/lv0d3d/bin/python"
mkdir -p surrogate_runs/geometry_curriculum_control
nohup env CUDA_VISIBLE_DEVICES=0 bash experiments/curriculum/run.sh \
    >> surrogate_runs/geometry_curriculum_control/controller.log 2>&1 < /dev/null &
echo $! > surrogate_runs/geometry_curriculum_control/controller.pid
```

可以关闭 VS Code；控制器和后续子进程均在远程机器运行。重启时使用同一命令。
控制目录的 `pipeline.lock` 防止重复启动；`plan.json` 固定数据/模型路径、样本数、
迭代数、训练配置版本和机器分配。已完成阶段跳过；未完成训练从该阶段自己的
`last.pt` 恢复。不要改变原运行的参数或手动删除检查点。

```bash
cat surrogate_runs/geometry_curriculum_control/status.json
tail -f surrogate_runs/geometry_curriculum_control/controller.log
tail -f surrogate_runs/geometry_curriculum_control/sigma050/training.log
```

训练在 euclid-17 的 GPU 0 上执行。启动前用独立进程实际测试 CUDA，并等待该 GPU
已用显存小于 1 GiB；其他任务占用时继续等待。该检查不依赖 `nvidia-smi`，因为
NVML 查询可能因驱动库版本不一致失败，而 CUDA 仍可正常工作。不会重启机器或终止
其他训练任务。

模拟沿用 `experiments/geometry_joint/machines.json` 中的公共机器：euclid-01
（12 CPU jobs，shards 0/1）、euclid-10（6 jobs，shard 2）、euclid-21（12 jobs，
shards 3/4）。各机器必须能通过 BatchMode SSH 登录，并共享相同绝对仓库路径和
求解器环境。`--machines` 可指定另一个同格式分配文件（固定五个 shards）。

## 失败时的行为

- 新几何必须通过正四面体和严格纤维检查，否则停止，不开始训练。
- 模拟失败先保留原目录，再执行同 PDE、同时间网格的 robust 非线性恢复。
- 仍有不完整案例或验证失败时，控制器写入 `status.json` 的 failed 状态并退出。
  不会跳过失败样本后继续训练。日志位于相应数据目录 `_campaign/curriculum_*.log`。
- 处理失败后重启控制器即可；已完成案例保留，未完成模拟会归档后重新生成。
- CUDA 暂不可用时持续等待；训练子进程出错则保留检查点并退出，修复后重启恢复。
- 所有阶段完成后写入 `surrogate_runs/geometry_curriculum_control/complete.done`。
