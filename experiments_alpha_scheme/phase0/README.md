# Phase 0 — tabular reference

本目录仅复用 `stream_rl_grid.environment.ContinualWindyGridWorld` 和环境配置接口；tabular
TD/Sarsa、adaptive step-size、运行与指标代码均在本目录独立实现。

两个子实验分别运行 README 冻结的四种环境和四种 step-size 机制：

```powershell
python -m experiments.phase0.p0_1_prediction
python -m experiments.phase0.p0_2_control
```

快速检查可减少 step 和 seed：

```powershell
python -m experiments.phase0.p0_1_prediction --steps 1000 --seeds 0
python -m experiments.phase0.p0_2_control --steps 1000 --seeds 0
```

每个 run 保存一个压缩 `npz`，包含 reward、squared TD error、collision、goal、访问状态、mode、
policy entropy 和冻结 A probe；同目录的 `summary.json` 保存分段、切换后 AUEC/recovery、recurrence、
step-size 分布和数值稳定性。完整主实验至少需要 3000 steps，才能覆盖 seasonal wind 的 mode 0
再次出现；短运行只用于 smoke test。

完成后统一绘图：

```powershell
python -m experiments.plotting experiments\phase0\results\p0_1_prediction\summary.json
```

图会写入同级 `figures/`。
