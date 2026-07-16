# Phase 3

D=71 nuisance-feature 与冻结消融实验。代码已实现，但只有 Phase 1 或 Phase 2 已显示稳定的
adaptive step-size 机制信号后才正式运行。

先根据 Phase 1/2 结果选择一个代表性 condition，再运行 D=71；该入口不会重复运行 D=55 baseline：

```powershell
python -m experiments.phase3.p3_1_nuisance --condition seasonal_wind
python -m experiments.phase3.p3_2_ablation experiments\phase3\results\p3_1_nuisance\prediction\summary.json
python -m experiments.phase3.p3_2_ablation experiments\phase3\results\p3_1_nuisance\control\summary.json
```
