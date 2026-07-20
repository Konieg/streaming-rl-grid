# 持续强化学习 MBRL 研究执行计划

## 1. 目标与原则

本文档把 [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md) 拆解成可以逐项实施、测试和验收的研究计划。核心目标是：

> 在继续使用 function approximation 的前提下，先建立所有主要算法都能解决的 stationary diagnostic setting；随后逐步引入 world-model learning、环境变化、变化感知和因子化规划，最终检验 MBRL 是否具有超越样本效率的 continual adaptation advantage。

本计划遵循四条原则：

1. **先验证表示能力，再比较学习算法。** 如果最优权重都无法表示好策略，继续调 TD 或 Dyna 没有意义。
2. **先解决 stationary task，再加入 non-stationarity。** 每个变化前后的 frozen MDP 都必须可解。
3. **一次只增加一个研究变量。** 依次隔离 representation、model learning、planning、forgetting、context inference。
4. **每个阶段设置 gate。** 未通过 gate 时停止扩展，先定位当前问题。

本项目不把 tabular learning agent 作为主方法。允许使用 exact finite-state solver，但它只作为：

- stationary/frozen MDP oracle；
- feature representation 的表达能力诊断工具；
- scenario policy-relevance validator；
- perfect-model planning upper bound。

主学习算法继续采用 sparse linear function approximation。

---

## 2. 总体路线

```text
Phase 0  冻结并审计现有实验
   ↓
Phase 1  Exact average-reward oracle
   ↓
Phase 2  Multi-group tile coding 与表示能力测试
   ↓
Phase 3  Stationary online-learning calibration
   ↓
Phase 4  统一 stochastic world-model API
   ↓
Phase 5  Clean stationary Dyna diagnostic
   ↓
Phase 6  Diagnostic non-stationary environments
   ↓
Phase 7  Recency-aware / prioritized Dyna baselines
   ↓
Phase 8  Change-Aware Factored Dyna
   ↓
Phase 9  消融、最终实验与统计报告
```

---

## 3. 全局通过标准

### 3.1 Stationary competence gate

在固定地图、目标和 dynamics 下，进入 continual comparison 的主要算法应满足：

- final average reward 至少达到 oracle 的 80%；
- goals per 1000 steps 至少达到 oracle 的 70%；
- mean path length 不超过 oracle 的 1.5 倍；
- 至少 90% 的 test runs 不出现“后半程完全不再到达目标”；
- state occupancy 不集中于与目标无关的局部循环；
- weights、trace、reward-rate estimate 保持数值稳定。

阈值可以在第一次 pilot 后统一调整一次，之后写入 manifest 并锁定。

### 3.2 Clean MBRL gate

在 fully observable stationary setting 中：

- perfect-model Dyna 应比相同 representation 的 Q-learning 更快达到目标性能；
- `planning_steps=0` 的 Dyna 必须与对应 Q-learning 逐步一致；
- empirical stochastic model 的 prediction error 应随数据增加而下降；
- planning advantage 不能只由更多 update 数解释。

### 3.3 Continual-learning gate

最终方法至少应在一个明确 setting 中同时表现出：

- 更低 post-change/dynamic regret；
- 更低 model error；
- 更准确的目标或路线选择；
- 更快 recurring-context reacquisition；
- 优势能够通过 factorization、forgetting 或 planning ablation 解释。

---

## 4. 建议代码结构

保留现有模块和旧算法语义，新增独立 research package：

```text
stream_rl_grid/
  research/
    __init__.py
    observation.py
    environments.py
    scenarios.py
    oracle.py
    representations.py
    model_api.py
    metrics.py
    execution.py
    analysis.py
    models/
      latest.py
      empirical.py
      recency.py
      factored.py
      oracle_model.py
    planning/
      uniform.py
      prioritized.py
    agents/
      research_dyna.py
      cafd.py
```

对应测试：

```text
tests/research/
  test_observation.py
  test_oracle.py
  test_representations.py
  test_models.py
  test_planning.py
  test_scenarios.py
  test_stationary_calibration.py
  test_cafd.py
```

建议实验入口：

```text
run_stationary_calibration.py
run_model_diagnostics.py
run_loca_experiment.py
run_wind_drift_experiment.py
run_context_recurrence_experiment.py
run_cafd_ablation.py
```

所有新结果写入 `research_results/`，不覆盖已有 `experiment_results/`。

---

## 5. Phase 0：冻结和审计现有结果

### 目标

将当前 200-run comparison 固化为可信的 negative baseline，并补全行为层面的解释。

### Step 0.1：冻结元数据

- 保存 eight-algorithm manifest、selected configs 和 aggregate CSV 的 snapshot/hash；
- 记录 git commit、Python/NumPy version；
- 明确当前 seed 同时改变 environment instance 和 algorithm randomness；
- 不修改旧 run 文件。

产出：

```text
research_results/phase0_audit/manifest_snapshot.json
research_results/phase0_audit/environment_metadata.json
```

### Step 0.2：行为审计

从现有 `summary.json` 和 `metrics.csv` 重建：

- 全程 goals per 1000；
- collision rate；
- reward decomposition；
- seed-level goal count；
- model size 和 planning update count。

对关键配置 deterministic replay，新增记录：

- action histogram 和 stay rate；
- state occupancy heatmap；
- recurrent state/action cycles；
- 每 5,000 或 10,000 步的 goal count。

产出：

```text
research_results/phase0_audit/behavior_summary.csv
research_results/phase0_audit/seed_level_summary.csv
research_results/phase0_audit/occupancy_*.png
```

### Gate 0

- 可从 raw results 重建 200-run summary；
- 至少一个 Dyna-Q+ seed 的 goal、collision、reward checksum 完全一致；
- 当前 negative findings 形成独立短报告。

---

## 6. Phase 1：Exact oracle 与 scenario validator

### Step 1.1：Frozen-MDP enumerator

对给定固定环境参数枚举：

- 所有合法 physical states；
- 所有 actions；
- `P(s'|s,a)`；
- `R(s,a)`。

Stochastic wind 必须枚举全部 outcome，不能调用 RNG 采样。Goal transition 直接连接到 restart distribution，保持 continuing semantics。

### Step 1.2：Average-reward oracle

用 relative value iteration 或 policy iteration 输出：

```text
optimal gain g*
differential value h*(s)
optimal Q*(s,a)
optimal action set
```

统一固定参考状态 `h(s_ref)=0`，或在分析前中心化 differential values。

### Step 1.3：Oracle tests

至少测试：

- deterministic 3×3 grid；
- 单障碍绕路；
- stochastic transition probability sum 为 1；
- goal restart distribution 正确归一化；
- oracle-policy rollout 的长期 reward 与 `g*` 一致；
- 同一 MDP 重复求解完全确定。

### Step 1.4：Scenario validator

对每个 proposed change 计算：

- frozen optimal gain difference；
- reachable states 中 optimal action set 改变比例；
- decision bottleneck action 是否改变；
- old policy 在 new MDP 中的 performance loss；
- transition/reward variation magnitude。

初始接纳条件：

```text
至少 15% reachable states 改变 optimal action
或指定 bottleneck 的 optimal action 必须改变
old policy 在 new MDP 中损失至少 20% oracle gain
```

### Gate 1

- 所有 oracle tests 通过；
- rollout 与 oracle gain 在容差内一致；
- 至少一个 hand-designed stationary map 可以人工核对最优路线。

---

## 7. Phase 2：Multi-group tile coding

### 目标

构造既能表达局部障碍导航，又保留 goal generalization 的 sparse linear representation。

### Step 2.1：Research observation schema

定义显式 observation，而不是继续依赖位置含义不透明的 tuple：

```text
agent_x, agent_y
goal_x, goal_y
local_wall_mask        # 可选
context_belief         # 后续可选
```

支持三种 mode：

```text
position_goal
local_geometry
oracle_context
```

新 features 和 model key 默认移除 `previous_action`，因为当前 dynamics 不依赖它。

### Step 2.2：实现 MultiGroupTileCoder

主 representation：

\[
\phi(s,a)=
[\phi_{pos},\phi_{goal},\phi_{joint},
\phi_{local},\phi_{bias}].
\]

#### Absolute-position group

```text
TC(x, y, action)
8 tilings
8～12 tiles / spatial dimension
```

用于表达局部墙体、边界和通道。

#### Relative-goal group

```text
TC(goal_x-x, goal_y-y, action)
8 tilings
8～12 tiles / relative dimension
```

用于跨目标位置共享一般性导航方向。

#### Joint group

```text
TC(x, y, goal_x-x, goal_y-y, action)
4 tilings
4～6 tiles / dimension
```

用于表达“在这个局部位置且目标位于这个方向时”的绕路决策。使用较粗 resolution 控制 4D feature growth。

#### Local-geometry group

在 `local_geometry` mode 中加入：

```text
categorical(four-neighbor wall mask, action)
```

四邻域 mask 只有 16 种可能，第一版不使用更复杂的视觉输入。

#### Bias group

```text
one categorical bias per action
```

初始 nominal active count 为：

```text
8 + 8 + 4 + 1 + 1 = 22
```

### Step 2.3：Feature tests

- 相同 input 产生确定性 indices；
- 不同 action 正确隔离；
- `previous_action` 不改变 features；
- goal translation 只按设计改变 relative/joint group；
- wall change 只改变 local group；
- active count 和 norm 正确；
- readonly query 不分配 IHT entries；
- checkpoint 后精确恢复；
- 正常容量下 IHT collision 为零或可接受。

### Step 2.4：Representation realizability test

这一步不训练 RL agent，只测试 representation 上限：

1. 用 oracle 获得 `Q*` 或 optimal advantage；
2. 为全部 state-action 生成 design matrix `X`；
3. 求最佳线性拟合：

\[
w^*=\arg\min_w\|Xw-A^*\|_2^2;
\]

4. 评估：

- advantage RMSE；
- greedy-action agreement；
- bottleneck action agreement；
- fitted greedy policy 的 actual average reward；
- obstacle-adjacent state error；
- held-out goal generalization。

比较：

```text
D=55 handcrafted LFA
现有 DualTileCoder
MultiGroupTileCoder without joint
MultiGroupTileCoder with joint
MultiGroupTileCoder with local geometry
```

### Step 2.5：小规模 feature sweep

```text
position tilings: 4, 8, 16
relative tilings: 4, 8
joint tilings: 0, 4, 8
local geometry: off, on
```

同时考虑 oracle fit、held-out generalization、active feature count 和 IHT usage，不以容量最大者自动获胜。

### Gate 2A

- fitted greedy policy 至少达到 95% oracle gain；
- greedy action agreement 至少 90%；
- decision bottleneck actions 全部正确；
- 明显优于 D=55；
- held-out goal 达到预设 generalization threshold。

未通过时先修改 tilings/resolution/group structure，不能通过增加训练步数掩盖表示不足。

---

## 8. Phase 3：Stationary online-learning calibration

### Step 3.1：Stationary ladder

按顺序通过四级环境：

#### Level A：基础导航

```text
fixed goal, no obstacles, no wind
```

#### Level B：局部绕路

```text
fixed goal, one hand-designed obstacle map, no wind
```

#### Level C：固定随机 dynamics

```text
fixed goal, fixed map, fixed stochastic wind distribution
```

#### Level D：Goal-conditioned generalization

```text
fixed map/dynamics
multiple observable training goals
held-out evaluation goals
```

### Step 3.2：校准主要算法

第一轮：

```text
Q-learning
Q(lambda)
SARSA(lambda)
TIDBD
Dyna-Q with planning=0
```

通过后加入：

```text
vanilla Dyna-Q
Dyna-Q(lambda)
legacy Dyna-Q+
reward-calibrated Dyna-Q+
```

保留 legacy Dyna-Q+ 对应旧结果；新增 calibrated 版本，把 untried self-loop prior reward 设为 step-reward estimate 或可配置 prior，而不是固定 0。

### Step 3.3：Hyperparameter protocol

Development/test seeds 分开。初始搜索：

```text
effective alpha:  0.02, 0.05, 0.10, 0.20
lambda:           0.0, 0.5, 0.8, 0.95
reward-rate step: 0.001, 0.005, 0.01
epsilon:          0.05, 0.10
```

用 3～5 个 development seeds 搜索，锁定后在至少 10 个 test seeds 上验证。

### Step 3.4：行为日志

每个 run 必须记录：

- cumulative/rolling goals per 1000；
- collision 和 stay rate；
- action histogram；
- path length between goals；
- oracle action agreement；
- state occupancy；
- cycle length/frequency；
- TD error、weight norm 和 trace norm。

### Gate 2B

进入 continual comparison 的算法必须达到第 3.1 节的 stationary competence gate。失败时依次检查：

1. numerical/step-size；
2. representation learning curve；
3. on/off-policy linear-FA stability；
4. average-reward update。

不得通过降低全部方法标准来掩盖某个算法的机制性失败；无法通过的算法可保留为 negative baseline。

---

## 9. Phase 4：统一 stochastic world-model framework

### Step 4.1：Model API

```python
update(observation, action, reward, next_observation)
predict_distribution(observation, action)
sample(observation, action, rng)
expected_reward(observation, action)
uncertainty(observation, action)
known_state_actions()
predecessors(observation)
diagnostics()
state_dict()
load_state_dict()
```

Model 使用 canonical physical state，不直接使用 value coder indices，也不包含 nuisance variable。

### Step 4.2：实现五种 model

1. `LatestOutcomeModel`：复现当前行为，作为 negative control。
2. `EmpiricalCategoricalModel`：保存 `N(s,a,s')` 和 reward mean/variance。
3. `ExponentialRecencyModel`：使用 `N_t=rho*N_{t-1}+new_count`。
4. `SlidingWindowModel`：每个 state-action 只保留最近 `W` 个 outcomes。
5. `OracleModel`：查询 frozen environment distribution，作为 upper bound。

### Step 4.3：Model tests

- distribution sum 为 1；
- deterministic setting 收敛到单一 outcome；
- stochastic wind empirical frequency 正确；
- forgetting 后旧 context probability 按预期下降；
- checkpoint 精确恢复；
- model state 不随 `previous_action`/nuisance 改变；
- OracleModel 与 enumerator 一致。

### Step 4.4：分离 real/planning update

Research Dyna 使用：

```text
alpha_real
alpha_plan
planning_steps
planning_sampler
backup_type = sampled | expected
```

实现 `planning_steps=0` identity test：相同 transition/action RNG 下，Research Dyna 与 Q-learning 的 weights、reward rate 和 actions 逐步一致。

### Step 4.5：Model diagnostics

在枚举产生的独立 query set 上记录：

- transition NLL；
- Brier score；
- expected displacement error；
- reward RMSE；
- calibration；
- stale-entry fraction；
- model age distribution。

---

## 10. Phase 5：Clean stationary Dyna diagnostic

### Step 5.1：方法

```text
Q-learning
Dyna + LatestOutcomeModel
Dyna + EmpiricalCategoricalModel
Dyna + OracleModel
```

### Step 5.2：公平性 protocol

同时报告：

```text
performance vs environment steps
performance vs total updates
performance vs wall-clock time
```

前者衡量 sample efficiency，后两者防止把更多 planning compute 当成算法机制优势。

### Step 5.3：Planning sweep

```text
planning steps:            0, 1, 5, 20
alpha_plan / alpha_real:   0.1, 0.25, 0.5, 1.0
backup:                    sampled, expected
```

### Gate 3

- OracleModel Dyna 在 environment-step curve 上快于 Q-learning；
- Empirical Dyna 位于 Q-learning 与 OracleModel Dyna 之间；
- LatestOutcomeModel 在 stochastic wind 下表现出可测量 model bias；
- planning gain 能由 model quality 而非 update count 解释。

如果 OracleModel Dyna 都没有优势，暂停后续 CAFD，先检查 average-reward planning、representation、behavior/target mismatch 和 reward-rate handling。

---

## 11. Phase 6：Diagnostic non-stationary environments

### Step 6.1：Two-goal/two-corridor map

```text
common restart region
decision bottleneck
short risky corridor -> goal A
long safe corridor  -> goal B
```

要求两条路线在 bottleneck 前共享状态；局部 reward/obstacle change 必须改变 bottleneck optimal action；每个 frozen variant 都通过 stationary gate。

### Step 6.2：Reward calibration

候选初值：

```text
ordinary step:  -0.05
stay:           -0.10
collision:      -0.25
goal A/B:       +1.0 / +4.0，按 context 交换
```

用 oracle 验证：

- goal seeking 显著优于局部循环；
- 少量 exploration collision 不掩盖 goal benefit；
- 两个 context 的最优目标/路线确实交换；
- 变化仍需要 value propagation，而不是一步即可解决。

### Step 6.3：LoCA abrupt reward change

```text
Phase A: 完整 state distribution 上学习旧任务
Phase B: 只在 goal A 附近观察新 reward
Phase C: 从公共 restart region 评估 bottleneck choice
```

阶段切换不清空 weights、trace、model 或 average reward。

### Step 6.4：Smooth wind drift

实现：

```text
sinusoidal
triangular
bounded random walk
T = 500, 2000, 8000
```

训练和测试使用不同 schedule family，避免只记忆周期。

### Step 6.5：Obstacle change

- Abrupt version：改变 decision-critical corridor 的一个 edge/cell；
- Smooth version：把对应 edge block probability 从 0 逐渐提高到 1。

两者都必须通过 scenario validator。

### Step 6.6：Recurring/compositional contexts

```text
wind   ∈ {left, right}
map    ∈ {upper open, lower open}
reward ∈ {A preferred, B preferred}
```

训练只覆盖部分组合；测试包括旧 context 再现和已知 factor 的未见组合。

### Step 6.7：Observability tiers

每个 scenario 先做 observable tier，再做 hidden tier：

```text
observable: local geometry 或 context parameter 可见
hidden:     只有 physical observation 与 interaction history
```

所有学习方法获得相同原始 observation；只有 oracle-context upper bound 可访问真实 context。

### Gate 4

- 每个 change 都改变 oracle policy；
- change 前后 frozen variants 都可解；
- old policy 在 new context 中产生显著 regret；
- stay/局部循环不再具有竞争力；
- continuing state 不在 phase transition 时被清空。

---

## 12. Phase 7：Adaptive Dyna baselines

### Step 7.1：Recency models

```text
Empirical Dyna without forgetting
SlidingWindow Dyna: W = 25, 100, 500
Exponential Dyna:   rho = 0.90, 0.97, 0.99, 0.997
```

预期 sliding window 更适合 abrupt change，慢 EMA 更适合 smooth drift；实验应验证而非预设这一结论。

### Step 7.2：TD-error prioritized sweeping

\[
p(s,a)=|r-\bar R+\max_{a'}Q(s',a')-Q(s,a)|.
\]

实现 predecessor index、priority queue、阈值和 checkpoint restoration。

### Step 7.3：Model-change priority

\[
p_{change}(s,a)
=|T_{M_t}Q(s,a)-T_{M_{t-1}}Q(s,a)|.
\]

比较 uniform、TD-error priority 和 model-change priority。

### Gate 5

- LoCA change 中 prioritized planning 比 uniform 更快传播；
- recency advantage 能在 NLL/stale fraction 中观察；
- abrupt/smooth condition 对 forgetting rate 的偏好不同；
- 所有比较使用相同 representation 和公平 update budget。

---

## 13. Phase 8：Change-Aware Factored Dyna

### Step 8.1：CAFD-Lite

第一版只实现：

```text
stable motion model
global categorical wind model
local obstacle edge model
separate reward heads
fixed exponential forgetting
model-change prioritized planning
```

暂不加入复杂 latent context bank。

### Step 8.2：Stable motion model

学习 action intended displacement 与 boundary behavior。使用 slow update，context change 后保留。

### Step 8.3：Global wind model

根据 intended/realized displacement residual 更新：

\[
P_t(w\in\{none,up,right,down,left\}).
\]

一次局部 wind observation 应更新全地图 predicted transitions。

### Step 8.4：Local obstacle model

维护 directed edge 或 target cell 的 blocked probability。成功移动和 collision 只更新涉及的局部参数。

### Step 8.5：Factored reward model

分别维护：

```text
ordinary-step reward
stay reward
collision reward
goal-A reward
goal-B reward
```

Reward change 不得修改 motion/wind/obstacle parameters。

### Step 8.6：CAFD-Surprise

定义：

\[
u_t=-\log\hat p(r_t,s_{t+1}|s_t,a_t)
\]

\[
\eta_t=\eta_{min}+(\eta_{max}-\eta_{min})
\sigma(c(u_t-h)).
\]

记录 surprise、effective learning rate、detected change time、false alarms 和每个 factor 的 parameter update magnitude。

### Step 8.7：Confidence-weighted planning

\[
p(s,a)=p_{change}(s,a)c_t(s,a),
\]

其中 confidence 来自 posterior count、entropy 或 predictive variance。通过消融检查它是否减少 model exploitation。

### Step 8.8：Hidden-context belief

先实现轻量版本：

- 小型 discrete context posterior；或
- continuous wind/reward parameter belief；
- 根据 recent transition likelihood 在线过滤。

Value representation 接收 belief summary，不使用 ground-truth context。

### Step 8.9：Optional context bank

仅在前述机制通过后加入 context prototype storage/retrieval，并与 monolithic context table bank 比较。

### Gate 6

- CAFD-Lite 至少在一个 local change 和一个 drift setting 中优于 unfactored recency Dyna；
- 优势同时体现在 regret、goal choice 和 model error；
- 去掉 factorization 或 change-priority 后优势缩小；
- CAFD-Surprise 在 abrupt/smooth 平均表现上优于单一固定 forgetting rate，或清楚展示失败边界。

---

## 14. Phase 9：最终实验

### 14.1 核心方法

```text
Q-learning + MultiGroupTileCoder
Q(lambda) 或 SARSA(lambda) + MultiGroupTileCoder
LatestOutcome Dyna
Empirical Dyna
best fixed-recency Dyna
prioritized recency Dyna
CAFD-Lite
CAFD-Surprise
perfect-model Dyna upper bound
oracle-context CAFD upper bound
```

### 14.2 核心 experiments

| Experiment | 回答的问题 |
|---|---|
| A. Stationary calibration | Representation 与 clean planning 是否工作？ |
| B. LoCA reward change | 局部 reward 能否传播到远端决策点？ |
| C. Abrupt obstacle change | 局部 dynamics change 能否快速传播？ |
| D. Smooth wind drift | 不同 timescale 下能否跟踪 latent parameter？ |
| E. Recurring contexts | 旧知识能否保留并快速恢复？ |
| F. Held-out composition | Factorization 是否学习可重组结构？ |

### 14.3 Pilot

```text
development environment seeds: 3～5
algorithm seeds / environment:  2～3
steps:                          20k～50k 起步
```

Pilot 只用于 debug、缩小超参数范围、验证 gate 和估计 variance。

### 14.4 Final runs

```text
至少 5 个 test environment instances
每个 instance 至少 4 个 algorithm seeds
至少 20 paired runs / method / condition
```

拆分 RNG：

```text
map_seed
schedule_seed
exogenous_noise_seed
behavior_seed
planning_seed
```

各方法共享前三类，后两类独立但可复现。

### 14.5 Hyperparameter selection

- 只在 development environments 上选择；
- 主结果使用一个跨 setting 的 robust configuration；
- appendix 可报告 per-setting tuned upper performance；
- test seeds 不重新调参；
- selection metric 使用 normalized dynamic regret，并施加 stationary competence constraint。

### 14.6 Compute accounting

记录：

```text
environment steps
real updates
planning updates
model updates
wall-clock time
peak memory/model size
```

同时报告 environment-step matched 和 update/compute matched 结果。

---

## 15. 指标、图表和统计

### 15.1 Control/behavior metrics

```text
stream average reward
goals / 1000 by goal identity
collision rate
stay rate
optimal route/goal choice fraction
path inefficiency
dynamic regret
50/250/500/1000-step post-change regret AUC
recovery/reacquisition time
```

### 15.2 Model metrics

```text
transition NLL
Brier score
reward RMSE
wind parameter RMSE
obstacle prediction error
stale-entry fraction
uncertainty calibration
unvisited-state prediction error
```

### 15.3 必需图表

- stationary reward 与 goal-rate learning curves；
- performance relative to oracle；
- event-aligned 和 cumulative dynamic regret；
- true wind parameter 与 posterior estimate；
- model NLL over time；
- state occupancy heatmap；
- optimal corridor/goal choice fraction；
- first-acquisition/reacquisition comparison；
- held-out composition performance；
- component ablations；
- performance vs steps/updates/wall-clock 三联图。

### 15.4 统计方法

- paired bootstrap confidence interval；
- mean 和 median paired difference；
- recovery time 报告 median/IQR；
- 未在 horizon 内恢复的 run 作为 censored data，不直接删除；
- 不用 marginal CI overlap 代替 paired test；
- 保留 seed-level scatter，避免均值掩盖地图级坍塌。

---

## 16. 建议时间表

| 周期 | 工作 | 必须产出 |
|---|---|---|
| Week 1 | Phase 0、oracle 基础 | Audit、oracle tests |
| Week 2 | MultiGroupTileCoder、realizability | Feature report、Gate 2A |
| Week 3 | Stationary online calibration | Stationary report、Gate 2B |
| Week 4 | Model API、empirical/recency/oracle Dyna | Model diagnostics、Gate 3 |
| Week 5 | LoCA/two-corridor、wind drift | Scenario validation、Gate 4 |
| Week 6 | Prioritized Dyna、CAFD-Lite | Local/drift pilots、Gate 5/6 |
| Week 7 | CAFD-Surprise、消融、final runs | Final raw results |
| Week 8 | 统计、绘图、报告 | Figures、tables、draft |

时间不足时优先完成：

```text
oracle
new representation
stationary calibration
LoCA environment
empirical/recency/prioritized Dyna
CAFD-Lite
```

Stretch items：

```text
neural latent model
context bank
完整 composition grid
复杂 Bayesian change-point detector
```

---

## 17. 任务依赖表

| ID | 任务 | 依赖 | 完成标准 |
|---|---|---|---|
| P0.1 | 冻结旧结果 | 无 | 可重建 200-run summary |
| P0.2 | 行为审计 | P0.1 | Goal/reward checksum 一致 |
| P1.1 | Frozen MDP enumerator | 无 | Transition tests 通过 |
| P1.2 | Average-reward oracle | P1.1 | Rollout gain 一致 |
| P1.3 | Scenario validator | P1.2 | 自动识别 policy-relevant change |
| P2.1 | Observation schema | 无 | 三种 mode 可序列化 |
| P2.2 | MultiGroupTileCoder | P2.1 | Feature/checkpoint tests 通过 |
| P2.3 | Realizability study | P1.2, P2.2 | Fitted policy ≥95% oracle |
| P2.4 | Feature selection | P2.3 | 锁定主表示配置 |
| P3.1 | Stationary ladder | P1, P2 | 四级环境可复现 |
| P3.2 | Online calibration | P3.1 | 主要算法通过 gate |
| P4.1 | Model API | P2.1 | Model 可替换/checkpoint |
| P4.2 | Empirical/recency/oracle models | P4.1 | Accuracy tests 通过 |
| P4.3 | Research Dyna | P4.2 | Planning=0 identity 通过 |
| P5.1 | Stationary Dyna diagnostic | P3, P4 | Perfect-model clean gain |
| P6.1 | Two-goal/corridor env | P1.3 | 两 context 最优路线不同 |
| P6.2 | LoCA protocol | P6.1 | Phase semantics 测试通过 |
| P6.3 | Wind drift | P1.3 | Schedules 可复现 |
| P6.4 | Recurrence/composition | P6.1 | Train/test contexts 分离 |
| P7.1 | Sliding/EMA Dyna | P4 | Tracking 可比较 |
| P7.2 | Prioritized planning | P4 | Propagation tests 通过 |
| P8.1 | CAFD-Lite | P6, P7 | Local 与 drift pilot 改善 |
| P8.2 | CAFD-Surprise | P8.1 | Fixed-rate ablation 完成 |
| P8.3 | Context belief/bank | P8.2 | Hidden/recurring test 完成 |
| P9.1 | Final manifest | 核心 gates | Methods/settings/metrics 锁定 |
| P9.2 | Final runs | P9.1 | 无 missing/failed run |
| P9.3 | Statistical analysis | P9.2 | Paired CI、seed scatter 完成 |
| P9.4 | Final report | P9.3 | Claims 与消融证据对应 |

---

## 18. 每次提交前检查清单

### 代码

- [ ] 不修改旧 Phase 1 algorithm semantics；
- [ ] 新 config 能 checkpoint round-trip；
- [ ] RNG streams 独立且可恢复；
- [ ] nuisance/无关 history 不进入 model key；
- [ ] probability distributions 正确归一化；
- [ ] 数值有限性检查存在；
- [ ] tests 覆盖正常和失败路径。

### 环境

- [ ] Frozen variants 可被 oracle 和学习算法解决；
- [ ] change 确实改变 optimal policy；
- [ ] goal/collision/restart/schedule 顺序明确；
- [ ] phase transition 不清空 continuing learner；
- [ ] observable/hidden tier 无信息泄漏。

### 实验

- [ ] development/test seeds 分离；
- [ ] environment/algorithm RNG 分离；
- [ ] test 前锁定 hyperparameters；
- [ ] reward、goal、collision、stay、regret 同时记录；
- [ ] model error 与 control performance 同时记录；
- [ ] compute/update budget 可追踪；
- [ ] run 可断点恢复；
- [ ] summary 可从 raw files 重建。

### 结论

- [ ] 不把低 pre-change baseline 误判为快速 recovery；
- [ ] 不把非目标导向策略误判为 robustness；
- [ ] 不把 representation failure 归因于 MBRL 原理；
- [ ] 不把更多 updates 归因于 model-based advantage；
- [ ] 每个主要 claim 都有 ablation 或 upper bound。

---

## 19. 最终论文/汇报结构

1. 为什么 sample efficiency 不足以刻画 continual MBRL；
2. 现有 negative result：stale latest-transition Dyna 的失败；
3. Benchmark audit：D=55 和 reward metric 如何掩盖基础导航失败；
4. Diagnostic redesign：stationary competence、LoCA、drift、recurrence、composition；
5. 方法：Change-Aware Factored Dyna；
6. 机制实验：factorization、forgetting、prioritization、confidence、context inference；
7. 结果：哪些 change type/timescale 下 MBRL 优于 model-free tracking；
8. 失败边界：何时简单 recency 足够，何时 world model 放大误差；
9. 结论：continual advantage 来自结构、更新选择和 replanning，而不只是“拥有模型”。

