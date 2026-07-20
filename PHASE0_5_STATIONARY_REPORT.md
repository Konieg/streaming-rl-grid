# Phase 0–5 Stationary 研究报告

日期：2026-07-20  
Canonical protocol：v4-fixed（每 100 real steps 进行一次只读 diagnostic）  
研究对象：continuing、average-reward、linear function approximation 下的 clean MBRL diagnostic

> 高频记录更新：最初 v3 每 1000 real steps 记录一次。v4-fixed 锁定 v3 的全部算法超参数，只把记录频率提高到每 100 steps，共 300 个检查点。210 个 runs 在所有公共 1000-step 检查点的 stream reward 完全一致，exact gain 最大差 `2.4e-11`，确认 diagnostic 不影响 continuing training trajectory。下文的环境、表示和模型设计不变；时间分辨率相关结论以本节和 `stationary_phase0_5_summary_v4_fixed` 为准。

高分辨率结果进一步表明：

- Dyna empirical(5) 相对 Q-learning 的 gain AUC 差为 `+0.0941 ± 0.0263`，前 5000 步 mean gain 差为 `+0.5170 ± 0.0688`，达到 80% oracle 提前 `337 ± 216` 步；
- Dyna empirical(10) 的对应差异为 `+0.0900 ± 0.0272`、`+0.5194 ± 0.0911` 和提前 `263 ± 278` 步；前两项显著，threshold-time CI 略跨 0；
- 高频 tail exact gain 显示 Q-learning 与 empirical Dyna 的渐近差异不显著，MBRL 的可靠优势主要是早期 sample efficiency，而不是更高 asymptote；
- 高频曲线与逐 step 表见 [`stationary_phase0_5_summary_v4_fixed`](stationary_phase0_5_summary_v4_fixed)。

## 1. 结论摘要

Phase 0–5 已完成。最终 stationary competence setting 不使用 episode：目标命中后，环境在同一个 transition 内把智能体均匀放回合法非目标状态，始终返回 `terminated=False, truncated=False`。learner weights、eligibility traces、平均奖励估计和 world model 均不重置。

最终 v3 结果支持以下结论：

1. Q-learning、SARSA(λ)、Replay-Q 和 empirical Dyna 都能解决主 stationary diagnostic，tail exact policy gain 达到 deterministic oracle 的 81%–86%。
2. Empirical Dyna(10) 相比 Q-learning 的全程 exact-gain AUC 提高 `0.0755 ± 0.0474`，并提前 `667 ± 509` 个真实环境步达到 80% oracle gain；paired 95% CI 均排除 0。
3. 使用相同额外 update budget 的 Replay-Q 没有显著改善，因此 Dyna 的早期优势不能仅归因于“每个真实步更新更多次”。
4. Latest-transition model 在 stochastic wind 下的 tail total-variation error 为 `0.260 ± 0.006`，empirical model 为 `0.092 ± 0.003`。Latest Dyna 的最终表现也低于 competence gate，证明旧式“只记最后一次结果”的模型不适合作为 stochastic MBRL 主实现。
5. 这些结果建立了 stationary competence 和 clean planning advantage，但还不能证明 MBRL 对 drift 更有优势；该主张属于 Phase 6 以后。

本文中的 `±` 均为跨 30 个独立 seed 的 95% 正态近似置信区间；方法差异使用相同 seed 的 paired difference。

## 2. 问题定义

主实验优化 continuing average reward：

\[
\delta_t = R_{t+1}-\bar R_t + Q(S_{t+1}, A^*)-Q(S_t,A_t),
\]

其中：

- Q-learning 和 Dyna 使用 greedy target；
- SARSA(λ) 使用 behavior next action；
- `bar R` 只由真实环境 transition 更新；
- planning samples 不重复更新 `bar R`；
- 没有 discount factor；
- 没有 episode return、terminal bootstrap 或 episode-level reset。

目标 transition 的完整语义是：

```text
到达 goal
→ 产生 reward_goal
→ 立即采样合法非目标 restart state
→ learner 将它视为普通 next_state
→ 继续同一条无限数据流
```

## 3. Phase 0：旧实验审计

审计对象为旧 `eight_algorithm_comparison` 的 200 个 runs。

- 200/200 个 `summary.json` 均存在；
- 所有 run 的 `reward_auc / completed_steps` 与 `stream_average_reward` 一致；
- 旧 `metrics.goals_per_1000_steps` 是 trailing-window 指标，而不是完整数据流累计目标率；
- wind-only Dyna-Q+ 五个 seed 的 60k-step 总目标数为 `[9, 0, 0, 0, 1]`，完整数据流率为 `[0.150, 0, 0, 0, 0.0167]` goals/1000 steps；
- 旧 `metrics.csv` 每 50 步采样一次，不能独立重建每一次 goal，因此 v3 同时记录 interval goal count、full-stream total 和 exact frozen-policy goal rate。

产物：

- [`phase0_audit.json`](experiment_results/stationary_phase0_5_v3/phase0_audit.json)
- [`phase0_legacy_run_checks.csv`](experiment_results/stationary_phase0_5_v3/phase0_legacy_run_checks.csv)

## 4. Phase 1：Exact average-reward oracle

实现内容：

- 枚举每个 `(state, action)` 的 stochastic transition distribution；
- goal transition 精确展开为 uniform continuing restart distribution；
- 使用 relative value iteration 求 average-reward optimal bias、gain 和 greedy policy；
- 使用 stationary distribution 独立复算 fixed-policy gain。

四级环境的 oracle gain：

| Task | States | Oracle gain |
|---|---:|---:|
| A_open5 | 24 | 4.0400 |
| B_wall5 | 20 | 3.3750 |
| C_windy7 | 42 | 2.3901 |
| D_corridor9 | 66 | 1.4832 |

Oracle 的 RVI residual 均小于 `1e-12`，greedy policy gain 与 stationary-distribution evaluation 一致。

## 5. Phase 2：Feature representation

最终 `MultiGroupTileCoder` 使用：

| Group | Tilings | Resolution |
|---|---:|---:|
| absolute position | 8 | 4 × 4 |
| relative goal | 8 | 4 × 4 |
| joint position/relative goal | 4 | 3 × 3 × 3 × 3 |
| local wall mask | 1 categorical | 16 masks |
| action bias | 1 categorical | 5 actions |

每个 `(state, action)` nominally 激活 22 个 sparse features；`previous_action` 不进入 feature 或 model key。

表示选择经历了三个可复现版本：

- v1：10/10/5 resolution，在 D task 上使用 4950 个 features，离线可实现但在线覆盖慢；
- v2：3/3/2 resolution，使用 640 个 features，但 online interference 过强；
- v3：4/4/3 resolution，使用 1865 个 features，在共享与局部可分辨性之间取得较好平衡。

在最难的 D task 上，v3 fitted greedy policy 达到 100% oracle gain；D55 fitted greedy policy gain 为 `−2.0`，说明 D55 无法表达该任务所需的绕路决策。

产物：

- [`feature_realizability.csv`](experiment_results/stationary_phase0_5_v3/feature_realizability.csv)
- [`phase1_oracle_and_phase2_features.json`](experiment_results/stationary_phase0_5_v3/phase1_oracle_and_phase2_features.json)

## 6. Phase 3：Stationary calibration

### 主 competence setting

最终主任务选择 `C_windy7`，而不是把更难的 `D_corridor9` 当作基础 gate：

- 7 × 7 grid；
- 固定 wall，保留一个通道；
- goal `(6, 6)`；
- 向右随机风，概率 0.20；
- `reward_goal=20`、`reward_step=-1`、`reward_collision=-2`；
- 42 个 continuing states。

它既能让主要算法在合理预算内学会，又保留了 stochastic dynamics，足以诊断 latest 与 empirical world model 的差别。D task 被保留为 stress test，不用于基础 competence gate。

### 紧凑超参数搜索

使用 3 development seeds、8k steps，在 `alpha_eff ∈ {0.05, 0.1, 0.2, 0.4, 0.8}` 中选择：

| Family | Selected alpha_eff |
|---|---:|
| Q-learning | 0.2 |
| SARSA(λ) | 0.1 |
| Empirical Dyna | 0.2 |

所有配置统一使用 `epsilon=0.05`、`eta_R=0.005`、`lambda=0.8`；planning update 的步长为 real-update 步长的 0.25 倍。development seeds 与 30 个 final seeds 分开使用。

产物：

- [`pilot_summary.csv`](experiment_results/stationary_phase0_5_v3/pilot_summary.csv)
- [`selected_hyperparameters.json`](experiment_results/stationary_phase0_5_v3/selected_hyperparameters.json)
- [`stationary_ladder_summary.csv`](experiment_results/stationary_phase0_5_v3/stationary_ladder_summary.csv)

## 7. Phase 4：World-model framework

已实现统一模型接口和以下模型：

- latest transition；
- full empirical categorical distribution；
- exponential-recency categorical distribution；
- sliding-window distribution；
- exact oracle model。

Research Dyna 明确分离：

- `alpha_real` 与 `alpha_plan`；
- real update count 与 planning update count；
- real transition 的 reward-rate update；
- planning transition 的 value-only update。

`Dyna(planning_steps=0)` 与相同配置 Q-learning 的 weights 和 reward-rate estimate 逐步完全一致，已由单元测试覆盖。

## 8. Phase 5：正式 stationary comparison

正式实验为 7 methods × 30 seeds × 30k real steps，共 210 个 continuing runs：

- Q-learning；
- SARSA(λ)；
- Replay-Q，10 个额外 replay updates；
- Dyna latest，10 planning updates；
- Dyna empirical，5 planning updates；
- Dyna empirical，10 planning updates；
- Dyna oracle，10 planning updates，作为 upper bound。

### 8.1 最终性能

| Method | Tail average reward | Tail exact gain | Oracle ratio | Steps to 80% oracle |
|---|---:|---:|---:|---:|
| Q-learning | 2.123 ± 0.025 | 2.035 ± 0.028 | 85.1% | 923 ± 187 |
| SARSA(λ) | 2.031 ± 0.207 | 1.933 ± 0.202 | 80.9% | 2003 ± 1986 |
| Replay-Q(10) | 2.039 ± 0.208 | 1.989 ± 0.206 | 83.2% | 2170 ± 2158 |
| Dyna latest(10) | 1.859 ± 0.196 | 1.764 ± 0.191 | 73.8% | 1843 ± 1951 |
| Dyna empirical(5) | 2.117 ± 0.020 | 2.037 ± 0.027 | 85.2% | 587 ± 90 |
| Dyna empirical(10) | 2.107 ± 0.023 | 2.005 ± 0.028 | 83.9% | 660 ± 181 |
| Dyna oracle(10) | 2.104 ± 0.014 | 2.045 ± 0.021 | 85.5% | 373 ± 30 |

Q-learning、SARSA(λ)、Replay-Q 与两个 empirical Dyna 均通过 80% oracle stationary competence threshold。Latest Dyna 是故意保留的 misspecified-model negative control，不属于通过 gate 的主算法。

### 8.2 Paired sample-efficiency comparison

相对于同 seed Q-learning：

| Method | Gain AUC difference | Early gain difference | Steps-to-80% difference |
|---|---:|---:|---:|
| Replay-Q(10) | −0.025 ± 0.223 | +0.277 ± 0.268 | +1247 ± 2172 |
| Dyna empirical(5) | **+0.094 ± 0.026** | **+0.517 ± 0.069** | **−337 ± 216** |
| Dyna empirical(10) | **+0.090 ± 0.027** | **+0.519 ± 0.091** | −263 ± 278 |
| Dyna oracle(10) | **+0.135 ± 0.023** | **+0.676 ± 0.078** | **−550 ± 192** |

粗体结果的 paired 95% CI 排除 0。Replay-Q 使用与 Dyna(10) 相同的额外 update 数量，却没有显著 sample-efficiency gain。

### 8.3 Model accuracy

| Model | Tail TV error |
|---|---:|
| latest | 0.260 ± 0.006 |
| empirical, planning 5 | 0.094 ± 0.003 |
| empirical, planning 10 | 0.092 ± 0.003 |

Planning count 不影响 learned model 的数据，因此两个 empirical error 接近；它们和 latest 的差异来自 stochastic distribution estimation，而不是 planning budget。

## 9. Gate 判定

| Gate | 结果 | 证据 |
|---|---|---|
| Continuing semantics | 通过 | 环境与测试始终不终止；goal 后不重置 learner |
| Average-reward objective | 通过 | differential TD；oracle 与指标均为 gain/reward rate |
| Representation realizability | 通过 | v3 fitted policy 在 D task 达到 oracle gain |
| Stationary competence | 通过 | 主 learned algorithms 达到 81%–86% oracle |
| Planning=0 identity | 通过 | 精确逐步单元测试 |
| Perfect-model clean gain | 通过 | oracle Dyna AUC 与 threshold time 显著优于 Q-learning |
| Learned-model clean gain | 通过 | empirical Dyna AUC 与 threshold time 显著优于 Q-learning |
| Not only more updates | 通过 | equal-budget Replay-Q 无显著改善 |
| Stochastic model validity | 通过 | empirical TV error 显著低于 latest |

## 10. 产物与复现

Canonical 高频原始结果：[`experiment_results/stationary_phase0_5_v4_fixed`](experiment_results/stationary_phase0_5_v4_fixed)

统计与绘图：[`stationary_phase0_5_summary_v4_fixed`](stationary_phase0_5_summary_v4_fixed)

主要文件：

- [`experiment_manifest.json`](experiment_results/stationary_phase0_5_v4_fixed/experiment_manifest.json)
- [`final_run_summary.csv`](experiment_results/stationary_phase0_5_v4_fixed/final_run_summary.csv)
- [`aggregate_summary.csv`](stationary_phase0_5_summary_v4_fixed/aggregate_summary.csv)
- [`paired_vs_q_learning.csv`](stationary_phase0_5_summary_v4_fixed/paired_vs_q_learning.csv)
- [`model_error_summary.csv`](stationary_phase0_5_summary_v4_fixed/model_error_summary.csv)
- [`learning_curve_stepwise_summary.csv`](stationary_phase0_5_summary_v4_fixed/learning_curve_stepwise_summary.csv)
- [`stationary_learning_curves.png`](stationary_phase0_5_summary_v4_fixed/stationary_learning_curves.png)
- [`stationary_learning_curves_early.png`](stationary_phase0_5_summary_v4_fixed/stationary_learning_curves_early.png)
- [`stationary_final_policy_gain.png`](stationary_phase0_5_summary_v4_fixed/stationary_final_policy_gain.png)
- [`stationary_model_error.png`](stationary_phase0_5_summary_v4_fixed/stationary_model_error.png)

在 `RLSS` 环境中复现：

```bash
MPLCONFIGDIR=/tmp/rlss-mpl conda run -n RLSS \
  python run_stationary_phase0_5.py \
  --output experiment_results/stationary_phase0_5_v4_fixed \
  --summary stationary_phase0_5_summary_v4_fixed \
  --selected-hyperparameters \
    experiment_results/stationary_phase0_5_v3/selected_hyperparameters.json \
  --workers 24 --seeds 30 --steps 30000
```

任务是 resumable 的：已有 per-run JSON 会被加载，不会重复运行。

## 11. 结论边界与下一步

Phase 0–5 证明的是：在一个所有主要算法都具备 stationary competence 的 stochastic continuing task 上，正确估计 stochastic transition distribution 的 Dyna 可以带来 clean sample-efficiency advantage；而 latest-transition model 会产生持久 model bias。

尚未证明的是“MBRL 更适应 continual drift”。Phase 6 应固定 v3 representation、empirical Dyna 和通过 competence gate 的 model-free baselines，依次加入：

1. smooth wind drift；
2. policy-relevant local obstacle change；
3. LoCA reward change；
4. recurring/compositional contexts；
5. recency、factorization、change-aware prioritization 消融。

主要指标应改为 post-change/dynamic regret、model tracking error、recovery time 和 recurring-context reacquisition，而不是只比较 stationary tail performance。
