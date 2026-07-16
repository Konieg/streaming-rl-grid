# Phase 2

D=55 LFA Differential Sarsa(0) control。正式运行前先完成 Phase 1 机制检查；不得改变共同 feature、
环境矩阵、step-size 机制或评价协议。

```powershell
python -m experiments.phase2.p2_1_fixed_control
python -m experiments.phase2.p2_2_adaptive_control
```

两者完成后可把 fixed 与 adaptive summary 一次传给 `experiments.plotting`，无需重复运行 baseline。
