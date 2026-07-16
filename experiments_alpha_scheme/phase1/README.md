# Phase 1

D=55 LFA Differential TD(0) prediction。本阶段复用 Phase 0 的
环境矩阵、step-size 机制、seed 和评价协议。

```powershell
python -m experiments.phase1.p1_1_fixed_prediction
python -m experiments.phase1.p1_2_adaptive_prediction
```

两者完成后可把 fixed 与 adaptive summary 一次传给 `experiments.plotting`，无需重复运行 baseline。
