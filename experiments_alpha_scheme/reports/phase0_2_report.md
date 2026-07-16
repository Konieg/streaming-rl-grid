# Phase 0–2 实验结果审计报告

## 1. 范围与冻结协议

本报告只分析 `experiments/phase0–2/results` 中的正式 `summary.json`、对应 `npz` trace 和已生成图；未修改或重跑实验。

共同协议为：四种环境（`stationary`、`seasonal_wind`、`hidden_context`、`moving_goal`），3,000 steps，seeds 0–4，fixed alpha 0.01/0.05/0.10，adaptive 初值 0.05，θ=0.01，边界 [10⁻⁴, 0.5]，η_r̄=0.01，ε=0.1，switch interval 500，metric window 100，recovery smoothing 25、tolerance 10%，probe interval/length 100/100。Phase 1–2 使用 D=55、无 nuisance、λ=0。

表中的 “±” 是跨 seed 标准误（SE）。正式 aggregate 另含 95% CI；未进行额外显著性检验。

## 2. Artifact 与有效性审计

六个 formal bundle 均为 schema 1.0，并包含规定的 top-level fields 和 run metadata/status/metrics/analysis/diagnostics/trace path。

| Bundle | runs | 完整性 |
|---|---:|---|
| P0.1 tabular prediction | 80 | 4 environments × 4 methods × 5 seeds |
| P0.2 tabular control | 80 | 4 × 4 × 5 |
| P1.1 D55 fixed prediction | 60 | 4 × 3 × 5 |
| P1.2 D55 adaptive prediction | 20 | 4 × 1 × 5 |
| P2.1 D55 fixed control | 60 | 4 × 3 × 5 |
| P2.2 D55 adaptive control | 20 | 4 × 1 × 5 |

320 个 formal runs 均为 3,000/3,000 steps、`numerically_stable=true`、`failure=null`。全部 trace 存在，主数组长度均为 3,000；summary 和 NPZ 数值数组未发现 NaN/Inf。LFA run 均记录 D=55。

`online_squared_td_error` 是在线 δ²，不是 MSVE。没有独立 reference value function 或 `reference_msve`，因此本报告不使用“真值误差”。

### 排除项

`experiments/phase0/results/smoke_p0_1` 是 10-step、seed 0 的 smoke bundle，共 16 runs；全部排除。

另有 10 个 moving-goal formal runs 的实际 `change_steps` 偏离
([500,1000,1500,2000,2500,3000])，trace mode 转换也确认该偏离，故从 moving-goal 定量声明中排除：

1. P0.2：`fixed_0.01/seed3`、`fixed_0.05/seed3`、`fixed_0.10/seed3`、`adaptive/seed3`，均缺 step 500；
2. P2.1：`fixed_0.01/seed0`、`fixed_0.05/seed0` 缺 step 500；
3. P2.1：`fixed_0.10/seed1` 缺 step 3000，`fixed_0.10/seed3` 缺 step 2500；
4. P2.2：`adaptive/seed0` 缺 step 500，`adaptive/seed1` 缺 step 3000。

moving-goal control 的合规 n 因此为：P0.2 各方法 n=4；P2.1 fixed 0.01/0.05 为 n=4、fixed 0.10 为 n=3；P2.2 adaptive 为 n=3。其余结果使用 n=5。

### Artifact 限制与补充分析

- formal protocol 未记录 exact wind strength/direction、两张 context maps 和 goal path；mode 标签只能部分佐证，无法从 artifact 独立复核全部环境配置。
- Phase 0 无逐步 alpha trace 和 `alpha_dynamics.png`；Phase 1–2 的五类共享图齐全。
- 无 run-note 文件。
- 原始 aggregate 只覆盖 whole-run metrics。现已用统一的只读 `experiments.retention` 定义，从现有
  trace 和 A-probe 生成六个 `retention.json`；未增加或重跑训练。它补充了初次 A acquisition、首次
  A recurrence recovery、两者差值和 B 期间冻结 A-probe retention loss，并排除 schedule 不合规 run。
- `retention.json` 只分析第一次 A→B→A；未恢复保留为 `null` 并通过统计中的 n 显示，不静默删除。
- 完整 visitation distribution、空间 error heatmap、action-block weight norm 和 alpha/reward 时间先后关系未形成可直接审计的 summary 指标。

## 3. Phase 0：tabular baseline

### Prediction：whole-run online δ²

| 环境 | fixed .01 | fixed .05 | fixed .10 | adaptive |
|---|---:|---:|---:|---:|
| stationary | 3.686±.143 | **3.060±.106** | 3.543±.138 | 3.307±.155 |
| seasonal wind | 5.038±.066 | **4.762±.081** | 5.488±.110 | 5.108±.104 |
| hidden context | 3.686±.091 | **3.215±.081** | 3.587±.110 | 3.411±.109 |
| moving goal | 4.048±.086 | 3.220±.061 | **3.163±.068** | 3.214±.062 |

adaptive 在四个环境均未低于最佳 fixed mean。prediction 的行为流相同，因此 reward/goal 不是方法性能差异。

### Control：mean reward；goal/1,000 steps

moving-goal 行已排除 schedule-deviating runs。

| 环境 | fixed .01 | fixed .05 | fixed .10 | adaptive |
|---|---|---|---|---|
| stationary | -1.075±.110；9.8±9.3 | -.977±.187；17.6±17.1 | **-.787±.243；32.7±22.8** | -.948±.215；20.1±19.6 |
| seasonal wind | -1.142±.043；19.5±6.2 | -.968±.109；38.8±16.5 | **-.876±.063；46.7±9.7** | -.950±.136；39.7±18.9 |
| hidden context | -1.150±.029；1.9±1.0 | -1.029±.111；12.3±10.9 | -1.136±.019；1.8±.9 | **-.997±.142；15.0±13.7** |
| moving goal | -1.186±.030；1.4±1.0 | -1.148±.039；4.0±2.7 | **-.880±.287；28.9±26.6** | -1.100±.074；7.0±5.7 |

control seed 方差很大，尤其 goal rate。adaptive 只在 hidden context 有最高均值，但不确定性很大；其余环境最佳均值来自 fixed .10。因此只支持“control 对 alpha 敏感”，不支持 adaptive 普遍占优。

## 4. Phase 1：D=55 fixed-policy prediction

whole-run online δ²：

| 环境 | fixed .01 | fixed .05 | fixed .10 | adaptive |
|---|---:|---:|---:|---:|
| stationary | 4.028±.162 | **3.978±.141** | 4.304±.151 | 4.048±.145 |
| seasonal wind | **5.239±.071** | 5.415±.076 | 5.799±.088 | 5.487±.076 |
| hidden context | **3.864±.105** | 3.914±.109 | 4.243±.124 | 3.986±.117 |
| moving goal | **4.115±.086** | 4.116±.083 | 4.433±.105 | 4.171±.088 |

adaptive 在四个环境均未低于最佳 fixed mean。seasonal wind 中 adaptive 明显高于 fixed .01；hidden/moving 的均值接近且区间重叠，应判为未显示优势。

### 与 P0 prediction 的对齐比较

同环境、同 seed、同 fixed alpha 下，P1 whole-run δ² 全部高于 P0。例如：

- stationary fixed .05：3.060±.106 → 3.978±.141；
- seasonal fixed .05：4.762±.081 → 5.415±.076；
- hidden fixed .05：3.215±.081 → 3.914±.109；
- moving fixed .10：3.163±.068 → 4.433±.105。

这是 tabular 与 D=55 的 representation/parameter-sharing gap，包含 approximation/projection error；不能称为纯 step-size effect，也不能在缺空间 error artifact 时单独断言 interference。

### 完整 retention 分析

`recurrence−initial` 为负表示第一次回到 A 后，比初次学习 A 更快达到同一个初始-A baseline；
probe loss 为正表示训练非 A 模式期间冻结 A-probe 变差。比较 adaptive 与 whole-run TD error 最低的
fixed .01：

| 环境 | 方法 | initial A steps | recurrent A steps | recurrence−initial | A-probe error loss |
|---|---|---:|---:|---:|---:|
| seasonal | fixed .01 | 47.6±11.2 | 25.0±0.0 | -22.6±11.2 | +0.172±0.188 |
| seasonal | adaptive | 47.6±11.2 | 25.0±0.0 | -22.6±11.2 | +0.650±0.373 |
| hidden | fixed .01 | 63.8±15.1 | 30.4±4.1 | -33.4±16.3 | -0.177±0.096 |
| hidden | adaptive | 62.8±14.9 | 28.6±3.6 | -34.2±15.9 | +0.089±0.121 |
| moving | fixed .01 | 63.8±15.1 | 28.2±2.7 | -35.6±15.9 | -0.130±0.044 |
| moving | adaptive | 62.8±14.9 | 30.8±3.6 | -32.0±16.1 | +0.325±0.229 |

所有方法在 recurrence 上通常比 initial acquisition 快，但这不是 adaptive 独有。adaptive 的 frozen-A
probe loss 在三个环境均为正，并且均值高于 fixed .01；因此 Phase 1 不支持 adaptive 带来更好的
retention。该结论同时使用 recurrence recovery 和 B 期间不学习 A 的冻结 probe，而不再把“回到 A 后
重新学得快”单独当作 retention。

## 5. Phase 2：D=55 control

每格为 mean reward；goal/1,000 steps。moving-goal 行仅含合规 runs。

| 环境 | fixed .01 | fixed .05 | fixed .10 | adaptive |
|---|---|---|---|---|
| stationary | -1.119±.013；.7±.2 | -1.105±.024；4.2±3.7 | **-.972±.042；25.5±6.8** | -1.074±.043；8.1±6.5 |
| seasonal wind | -1.433±.088；28.5±7.8 | -1.135±.040；49.9±6.2 | **-1.095±.084；63.0±13.8** | -1.318±.038；36.7±7.3 |
| hidden context | -1.103±.007；.8±.2 | -1.103±.011；2.5±2.1 | **-1.063±.015；12.4±5.0** | -1.099±.013；4.0±3.6 |
| moving goal | -1.114±.020；.9±.3，n=4 | -1.111±.023；4.6±3.5，n=4 | **-1.062±.028；14.3±6.9，n=3** | -1.117±.028；1.6±1.0，n=3 |

adaptive 在四个环境均未超过最佳 fixed mean reward/goal。seasonal 中 adaptive 为 -1.318±.038、36.7±7.3，fixed .10 为 -1.095±.084、63.0±13.8，是最清楚的均值差距。

control 中 TD error 与 reward 排序不同：seasonal fixed .01 的 δ²=5.192±.811，低于 fixed .10 的
9.104±1.988，但 reward/goal 更差。TD error 衡量当前 value 对自身 bootstrap target 的一致性；reward
衡量由该 value 诱导的策略实际控制效果。一个保守、变化慢的 value 可以有较小 TD error，却仍给出较差
动作排序；较大的 alpha 也可能更快改变策略并提高 reward，同时因 target/visitation 持续改变而产生更大
TD error。因此 TIDBD 使用 TD-error meta signal，不等于它直接优化 reward，也不保证 whole-run TD error
必然低于所有 fixed alpha。

### 与 P0 control 的对齐比较

两者在 task/environment/alpha/seed 上对齐，但 tabular→LFA 会改变 generalization，进而改变 epsilon-greedy policy 和 visitation。结果也非单向：stationary fixed .10 reward 为 -.787±.243 → -.972±.042；hidden fixed .10 为 -1.136±.019 → -1.063±.015。缺完整 visitation distribution，故只能报告 representation/parameter sharing 与 policy–data coupling 共同改变，不能拆成纯因果效应。

### 完整 retention 分析

以 whole-run reward/goal 最佳的 fixed .10 为 comparator；probe reward loss 为正表示训练非 A 模式时
冻结 A-policy reward 下降：

| 环境 | 方法 | initial A steps | recurrent A steps | recurrence−initial | A-probe reward loss |
|---|---|---:|---:|---:|---:|
| seasonal | fixed .10 | 105.2±13.5 | 33.8±4.2 | -71.4±15.0 | +1.724±0.969 |
| seasonal | adaptive | 107.0±37.5 | 78.6±27.1 | -28.4±18.3 | +0.396±0.538 |
| hidden | fixed .10 | 59.8±9.6 | 25.0±0.0 | -34.8±9.6 | -0.768±0.768 |
| hidden | adaptive | 64.0±11.8 | 25.0±0.0 | -39.0±11.8 | -0.714±0.714 |
| moving | fixed .10 | 46.7±7.4 | 25.0±0.0 | -21.7±7.4 | +0.053±2.263，n=3 |
| moving | adaptive | 79.3±11.9 | 25.0±0.0 | -54.3±11.9 | 0.000±0.000，n=3 |

Phase 2 的 retention 是混合结果：seasonal adaptive 的 probe loss 小于 fixed .10，但 recurrence recovery
更慢，且 whole-run reward/goal 更差；hidden 两者相近。moving 只剩 3 个 schedule-compliant runs，
不作强结论。因此 adaptive 没有跨环境、跨 retention 指标的一致优势。

## 6. Adaptive alpha 证据

P1 terminal alpha 分位数：

| 环境 | p10 | median | p90 |
|---|---:|---:|---:|
| stationary | .0492±.0001 | .0519±.0003 | .0717±.0016 |
| seasonal wind | .0489±.0002 | .0526±.0003 | .0734±.0015 |
| hidden context | .0492±.0001 | .0513±.0003 | .0711±.0021 |
| moving goal | .0495±.0001 | .0521±.0003 | .0690±.0014 |

所有 P1–2 adaptive runs 的 lower/upper bound hit count 均为 0。P1 还显示粗 group 差异，例如 seasonal absolute-position median 约 .0569，relative-goal median 约 .0505。

但 trace 只保存 alpha p10/median/p90 时间序列，summary 只保存终点分位和两个粗 group；没有 55 个具名 feature 各自的 alpha trajectory。因此可支持“alpha distribution 非退化”和“组级差异”，不能验证同一具体 feature 跨 seeds 稳定分化。

## 7. 结论分级

### Supported

1. **存在 fixed-alpha trade-off。** Prediction 的最佳 fixed alpha 随环境和表示变化；control 中较大 alpha 往往提高 reward/goal，同时增大在线 TD error。
2. **D=55 与 tabular 的 prediction gap 可观测。** 同 fixed alpha/environment 的 P1 whole-run δ² 均高于 P0；这属于表示/参数共享组合差异。
3. **adaptive alpha distribution 非退化且未撞边界。** P1 的跨 seed 分位和粗 group 统计稳定显示 p90 高于 p10。

### Unsupported

1. **adaptive 优于所有 fixed baseline：不支持。** P1 的 adaptive δ² 和 P2 的 adaptive reward/goal 均未超过各环境最佳 fixed mean。
2. **adaptive 同时改善 adaptation 与 retention：不支持。** Phase 1 的 adaptive A-probe loss 均高于
   comparator；Phase 2 的 recurrence recovery 与 probe loss 呈混合排序，没有跨环境一致优势。
3. **低 alpha 等于可删除 feature：不支持。** Phase 0–2 没有冻结消融。

### Inconclusive

1. moving-goal control retention 只有 3 个 schedule-compliant runs，不能作强结论。
2. 无法从 artifact 分离 projection error、parameter-sharing interference 与 control visitation shift。
3. 无法验证 feature-identity 层面的跨 seed alpha 稳定性。

## 8. Phase 3 决策

**建议暂时停止在 Phase 3 之前。**

alpha 分位和粗 group 确实稳定分化，但严格门槛要求“跨 seed 的逐 feature adaptive-alpha difference”。现有 artifact 没有保存 55 个具体 feature 的 alpha identity/trajectory，只能验证分位和 group 层面；同时 P1/P2 adaptive 没有显示超过最佳 fixed 的稳定性能或 retention 优势。

因此 Phase 3 gate 当前应判为 **inconclusive**，不能把粗粒度 alpha 分化当作已满足逐 feature 稳定性条件，也不能宣称 feature-selection 机制成立。
