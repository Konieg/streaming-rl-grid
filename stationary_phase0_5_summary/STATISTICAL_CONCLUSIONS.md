# Phase 0–5 Stationary 实验统计结论

## 问题定义核对

所有实验都是单一 continuing stream；目标命中后立即随机重启到合法非目标状态，但 `terminated=False`、`truncated=False`，learner、平均奖励估计与模型均不重置。主指标是 average reward/gain，不存在 episodic return。

## 主要结果

- 最佳 learned-model 方法是 **Dyna empirical (10)**，tail exact gain 为 `1.1612 ± 0.0595`（95% CI），为 oracle 的 `78.3%`。
- Dyna empirical(10) 相对 Q-learning 的 paired exact-gain 差为 `0.4627 ± 0.2332`；95% CI不跨过 0。
- 等 update-budget 的 Replay-Q 相对 Q-learning 差为 `0.0289 ± 0.2649`，用于区分 model structure 与单纯增加更新次数。
- Oracle-model Dyna 是 representation/planning upper bound，不计作可部署算法。
- latest-transition 与 empirical stochastic model 的差异用于检验：随机 dynamics 下，只记最后一次结果是否会产生系统性 model bias。

## 可支持的结论边界

Phase 0–5 只能回答 stationary competence、表示上限、模型正确性和 clean planning advantage。它不能单独支持‘MBRL 更适应环境漂移’；该命题必须在 Phase 6 以后通过 post-change regret、model tracking error 与 recurrence/composition 实验检验。

数值来源：`aggregate_summary.csv` 与 `paired_vs_q_learning.csv`。误差为跨独立随机种子的 95% 正态近似置信区间；方法差异使用相同 seed 的 paired difference。
