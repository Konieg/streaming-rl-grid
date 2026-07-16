# Step-size、feature selection 与 continual learning：实验计划

## 研究问题

在严格 streaming 的 continual RL 中，研究 step-size 如何影响：

1. 对环境变化的 adaptation；
2. 学习新经验时对已有知识的 interference；
3. 环境模式再次出现时的 retention；
4. 逐 feature step-size 能否成为可靠的 feature-selection 信号。

不比较 tile coding 或不同 feature engineering。本项目不用 tile coder、IHT 或 hash；所有
LFA 实验使用同一套显式、低维、无碰撞的 feature vector。每个 transition 只使用一次，不用
replay、batch 或 eligibility trace。

## 环境 non-stationarity

所有环境直接使用 `stream_rl_grid.environment.ContinualWindyGridWorld`。真正的模式变化由
`_advance_schedules()` 在 transition 后更新，下一次 transition 起生效：

| 来源 | Profile | 改变的对象 | agent 是否可见 |
|---|---|---|---:|
| 季节 | `seasonal_wind` | 风 phase 与奖励倍率 phase | 否 |
| 地图 context | `hidden_context` | 障碍物地图 | 否 |
| 移动目标 | `moving_goal` | goal 坐标 | 是 |

主实验只使用以下条件，不做周期 sweep：

1. `stationary`；
2. `seasonal_wind`：`manual_wind_direction="auto"`、`w_strength=0.3`、
   `wind_period=500`；
3. `hidden_context`：两张显式不同、合法且连通的 `context_maps`，
   `context_switch_interval=500`，形成 A→B→A；
4. `moving_goal`：显式给定至少两个合法 waypoint 的 `goal_path`，
   `target_move_interval=500`。环境沿该路径往返移动，因此可观察 goal A→B→A 的 recurrence。

以上三种 non-stationarity 在主实验中共享**唯一固定 schedule**：切换间隔统一为 500 个 global
steps，3,000-step formal run 的预定切换点固定为
`[500, 1000, 1500, 2000, 2500, 3000]`。切换周期、首次切换时刻或不同 change step 都不是实验轴；
不得为它们建立额外 run、sweep 或子实验。每个 Phase 只在这一条固定 schedule 上比较 alpha 机制。

formal runner 必须核对实际 mode change 是否与上述 schedule 一致。`seasonal_wind` 和
`hidden_context` 在每个固定点直接推进 phase/context；当前 `moving_goal` 环境会在下一个 waypoint
与 `agent_state` 重合时跳过该 waypoint，因此“每 500 step 尝试移动”未必等于“每 500 step 实际
改变 goal”。这属于无效的 protocol run，不构成另一种切换步长条件；在 moving-goal 结果进入正式
比较前，必须保证目标在固定点确实切换，否则排除该 run。

主实验由一个 stationary 对照和三类性质不同的 non-stationarity 组成：

- `seasonal_wind` 改变风和 reward phase；
- `hidden_context` 改变障碍物地图，同时不向 agent 提供 context identity。
- `moving_goal` 改变目标位置；新 goal 坐标出现在 observation 中，属于可观测的任务变化。

`stationary` 只是不发生变化的对照条件。以上四个条件必须完整地用于 Phase 0、Phase 1 和
Phase 2，不能为某个 Phase 单独增加或删除环境条件。

`combined` 同时改变多种因素，不进入主机制结论，留作后续外部验证。

注意：默认 `hidden_context` 未必真的改变地图。当前环境在 `context_maps=None` 且使用默认
`obstacle_coordinates` 时会将同一张地图复制给各 context。因此 hidden-context 实验必须显式
传入不同 `context_maps`；否则只有 context 编号变化，动力学并未变化。

`w_strength` 是每步施加风的概率，本身只是随机性；季节 phase 的改变才是 non-stationarity。
到达目标后的随机重生同样是 continuing MDP 的随机转移，而不是模式变化。

## 表示与变量规格

### Tabular baseline

算法输入不使用 raw observation 中的 `previous_action`；定义：

\[
s=(x,y,g_x,g_y),\qquad a\in\{0,1,2,3,4\}
\]

tabular 表直接以 `(s,a)` 为索引。固定目标时，表项数为：

\[
25\text{ positions}\times5\text{ actions}=125
\]

若 `goal_path` 含有 \(K\) 个不同 waypoint，则 moving-goal 条件最多需要：

\[
25\text{ agent positions}\times K\text{ goal positions}\times5\text{ actions}=125K
\]

个表项（非法或不可达组合不会被访问）。所以状态定义本身已经适用于 moving goal；需要改变的是
LFA feature，而不是额外加入不可观测状态。

这与后续 TD/Sarsa 使用相同更新形式，只是 one-hot parameterization。hidden-context 中，所有
Phase 的 learner 都只能使用共同的 observation \((s,a)\)，不能额外接收 `context_index`。否则就
改变了输入信息和 feature，无法再把差异归因于 step-size 或参数共享。

### Phase 1–2 的固定 LFA：D = 55

同样忽略 `previous_action`。将 agent 与 goal 的绝对位置，以及 agent 相对 goal 的位移归一化为：

\[
\tilde x=2x/(W-1)-1,\qquad
\tilde y=2y/(H-1)-1,
\]

\[
\tilde d_x=(x-g_x)/(W-1),\qquad
\tilde d_y=(y-g_y)/(H-1).
\]

这里不单独重复加入 \(\tilde g_x,\tilde g_y\)，因为 \((x,y,d_x,d_y)\) 已经唯一确定
\((g_x,g_y)\)。定义 11 个明确、连续有界的基函数：

\[
\psi(s)=[1,\tilde x,\tilde y,\tilde x^2,\tilde y^2,\tilde x\tilde y,
\tilde d_x,\tilde d_y,\tilde d_x^2,\tilde d_y^2,\tilde d_x\tilde d_y]
\]

前六项保留 agent 的绝对位置，用于表达固定地图、障碍物和位置相关风的影响；后五项表达 agent
与当前 goal 的关系，使同一 agent 位置在不同 goal 下可以产生不同 value。若仍使用原 D=30，
moving goal 的不同状态会被强制映射到同一 feature vector，误差将主要来自表示 aliasing，无法用于
干净研究 step-size。

将它归一化为 \(\bar\psi=\psi/\lVert\psi\rVert_2\)。常数项保证 \(\psi\) 永不为零，故归一化
总是有定义。

对每个 candidate action 使用一套独立系数：

\[
\phi(s,a)=\bar\psi(s)\otimes\operatorname{onehot}(a)
\]

因此：

\[
D=11\text{ basis}\times5\text{ actions}=55
\]

例如动作 `right` 时，只有 right 对应的 11 个分量取 \(\bar\psi(s)\)，其余 44 个为零。所有
feature 都在 \([-1,1]\) 内，最多 11 个非零，且 \(\lVert\phi(s,a)\rVert_2=1\)。action-specific 参数使“同一位置向上”和
“同一位置向右”可以有不同 value；不需要 tile coding。

LFA action-value 为：

\[
Q_w(s,a)=w^\top\phi(s,a),\qquad w\in\mathbb R^{55}
\]

### 后续 feature-selection 扩展：nuisance group

Phase 0–2 **先只使用 D=55**。在 Phase 3 才增加预先定义的 nuisance group：每步采样一个
与环境、动作、reward 和 context 独立的随机类别 \(u_t\in\{0,\ldots,15\}\)，并编码成 16 维
one-hot 特征。它是人为提供给 learner 的已知无关输入，用来检验“per-feature alpha 是否会
压低无关 feature 的更新”。

加入后：

\[
\phi_{\mathrm{selection}}(s,a,u_t)=
\frac{1}{\sqrt 2}[\phi(s,a),\operatorname{onehot}(u_t)]
\in\mathbb R^{71}
\]

系数 \(1/\sqrt2\) 使扩展后的向量仍满足 \(\lVert\phi_{\mathrm{selection}}\rVert_2=1\)，避免把
feature 数量造成的整体尺度变化误认为 step-size 效应。该扩展不改变环境、不比较另一种 feature
design；它只向固定 pool 加入已知无关的候选特征。

### 学习变量与边界

所有 Phase 使用 \(\lambda=0\)，即无 eligibility trace。

| 变量 | 形式 |
|---|---|
| tabular value | 固定目标为 \(Q\in\mathbb R^{125}\)；含 \(K\) 个 waypoint 时最多为 \(\mathbb R^{125K}\) |
| LFA weight | \(w\in\mathbb R^{55}\)，Phase 3 为 \(\mathbb R^{71}\) |
| reward-rate estimate | \(\bar r\in\mathbb R\)，步长 \(\eta_{\bar r}=0.01\) |
| fixed alpha | \(\alpha\in\{0.01,0.05,0.10\}\)，tabular 与单位范数 LFA 都直接使用该值 |
| TIDBD state | \(\beta,h\in\mathbb R^D\)，\(\alpha_i=\exp(\beta_i)\) |
| TIDBD 初值 | \(\alpha_i=0.05\)，\(\theta=0.01\) |
| TIDBD bounds | \(\beta_i\in[\log(10^{-4}),\log(0.5)]\) |
| 数值安全 | 不 clip weight；NaN/Inf 或 \(\max_i|w_i|>10^6\) 记为数值不稳定并停止该 run |

\(\bar r\) 不是 value weight，也不是 TIDBD 学习的 alpha。它是 continuing average-reward
任务对当前平均 reward 的估计：

\[
\bar r\leftarrow\bar r+\eta_{\bar r}\delta
\]

它出现在 \(\delta=r-\bar r+Q(s',a')-Q(s,a)\) 中，用来移除持续任务的非零平均奖励；若没有
它，value 会整体漂移。\(\eta_{\bar r}=0.01\) 在所有方法和环境中固定，只作为共同的慢时间尺度，
不属于本项目比较的 alpha 机制。该值先在 Phase 0 stationary 条件下做一次小型校准后冻结。

fixed alpha 是权重 \(w\) 的学习率。由于 tabular one-hot feature 和本项目归一化后的 LFA feature
都满足 \(\lVert\phi\rVert_2=1\)，二者可直接使用同一 alpha；不再需要“除以 6”。候选值
\(0.01,0.05,0.10\) 是预注册的有限比较，而不是对每个结果无限搜索。TIDBD 的 \(\beta\) 更新
仅作用于当前 active feature；Phase 0 的 tabular 不做 feature selection，只作为相同 TD/Sarsa
更新规则的无共享参数 baseline。

### 为什么这些 fixed alpha 合理

Sutton 与 Barto 对线性 SGD/function approximation 给出的经验规则是：

\[
\alpha\approx\frac{1}{\tau\,\mathbb E[\phi^\top\phi]}
\]

其中 \(\tau\) 是希望一个典型 feature 经过多少次**被更新**后基本完成适应的时间常数。常见的
单样本近似是 \(\alpha_t\approx1/(\tau\,\phi_t^\top\phi_t)\)。本项目刻意令
\(\lVert\phi\rVert_2^2=1\)，故规则简化为 \(\alpha\approx1/\tau\)：

| fixed alpha | 对应的名义 \(\tau\)（该 feature 被更新的次数） |
|---:|---:|
| 0.01 | 100 |
| 0.05 | 20 |
| 0.10 | 10 |

这三值不是声称的最优值，而是覆盖慢、中、快三种明确时间常数的有限基线。环境的 context
切换周期为 500 个全局 step；某一 feature 的实际更新次数还取决于访问频率，通常远少于 500。
这正是 global alpha 难以同时适合所有 feature、而 per-feature alpha 值得研究的原因。

同理，\(\eta_{\bar r}=0.01\) 对标量平均奖励估计对应约 100 个 TD 更新的时间常数。它在
Phase 0 的 stationary 条件下做一次小型校准后，对所有方法和环境冻结；否则改变 \(\eta_{\bar r}\)
会与研究对象 alpha 混淆。TIDBD 的初值 0.05 对应 \(\tau\approx20\)，位于 fixed baselines 中间。
\(\theta=0.01\) 不是理论最优值，须在 Phase 0 的一个独立 pilot 中于 \(0.001,0.01\) 两个值
之间选择一次，然后冻结。


本计划不参考仓库中已有的 `agent`、`algo`、`tile_coder` 或 `AgentConfig` 的任何算法实现或
超参数。它们属于此前的算法原型；本项目只复用 `environment.py` 所定义的 GridWorld 动力学、
奖励、目标移动和 context 调度。这里的 alpha、\(\eta_{\bar r}\)、\(\theta\)、\(D=55\) 与
\(\lambda=0\) 都是本实验计划独立定义、后续应独立实现和验证的设定。

## 三个共享实验轴

### 轴 A：alpha 机制

1. fixed scalar alpha；
2. learned per-feature alpha（TIDBD/IDBD-style，无 trace）。

固定 group-wise 与 learned scalar 暂不进入主实验；待上述对比已清楚后再做消融。

### 轴 B：环境条件

使用 `stationary`、`seasonal_wind`、显式地图切换的 `hidden_context` 和 `moving_goal`。
风 phase、地图或 goal 的切换属于轴 B，不是 retention 本身。

### 轴 C：adaptation 与 retention 评价

- prediction 的在线 \(\delta^2\)（TD-error proxy）、control 的 reward/goal rate/collision；
- 变化前窗口 baseline、变化后恢复时间、变化后 AUEC；
- A→B→A 时，A 首次和再次出现时的恢复时间；
- 训练 B 时冻结 A probe stream 上的误差/回报；
- active alpha 分位数、权重漂移和更新能量；
- control 阶段的平均 reward、目标率、碰撞率、策略熵与状态访问分布。

轴 C 不制造地图切换；它规定如何评价轴 B 中 A→B→A 的 retention。恢复快但 A probe 已退化，
只能称为 tracking，不能称为 retention。

retention 统一从已有 formal trace 离线计算，不增加训练 run：初始 A 的最后一个 metric window 定义
共同 baseline；分别记录初次学习 A 和第一次回到 A 后达到该 baseline（含既定 10% tolerance）的
recovery steps。冻结 A probe 的 retention loss 从“离开 A 前最后一个 probe”计算到“A 第一次返回前
最后一个 probe”；prediction 中 error 上升为正 loss，control 中 reward/goal-rate 下降为正 loss。
只有 schedule 合规的 run 进入 aggregate，未恢复的 run 保留为 `null` 并单独计数，不能静默删除。
统一命令为：

```powershell
python -m experiments.retention path\to\summary.json
```

它只读取现有 `summary.json`/trace 并写出同目录的 `retention.json`，不会重跑 learner。

### Phase 0–2 的严格对齐矩阵

| 比较项 | Phase 0 | Phase 1 | Phase 2 |
|---|---|---|---|
| 表示 | tabular | D=55 LFA | D=55 LFA |
| 学习问题 | prediction 与 control 两条 reference | prediction | control |
| 对应关系 | prediction 对齐 Phase 1；control 对齐 Phase 2 | 对齐 P0 prediction | 对齐 P0 control |
| 环境 | stationary、seasonal wind、hidden context、moving goal | 完全相同 | 完全相同 |
| step-size | 三种 fixed alpha；learned per-parameter alpha | 完全相同机制 | 完全相同机制 |
| 数据与评价 | 配对 seed、切换点、probe 和指标 | 完全相同 | 完全相同 |

这里的“对齐”是指除表示以及 prediction/control 所必需的策略差异外，不允许改变环境、输入信息、
alpha 机制或评价协议。

## Phase 0 — Tabular 机制 baseline

使用与后续对应的两种 tabular 算法：

1. 固定行为策略 \(\mu\) 下的 Differential TD(0) action-value prediction；
2. epsilon-greedy 的 Differential Sarsa(0) control。

两者与后续 LFA 的 TD error 形式相同：

\[
\delta_t=r_t-\bar r_t+Q(s_{t+1},a_{t+1})-Q(s_t,a_t)
\]

区别仅在于 prediction 的 \(a_{t+1}\sim\mu\) 固定，而 control 的 \(a_{t+1}\) 来自当前
epsilon-greedy policy。Phase 0 使 tabular 能分别作为 Phase 1 与 Phase 2 的公平 baseline。

### 目的

先在**没有函数近似和参数共享**的条件下，测出同一类 step-size 机制面对环境变化时的基本
tracking--interference--retention 权衡。tabular prediction 可作为同一 observation-state space
上的高容量 value reference；它也与后续使用相同更新规则，因此是 LFA 的主要对照。需要注意的是，
tabular 与 D=55 LFA 之间的误差差异包含 LFA 的表示/投影误差，不能全部称为 step-size 算法误差。

### P0.1：固定策略下的 tabular prediction

- **设置**：固定、覆盖所有动作的行为策略 \(\mu\)（例如每个动作有非零概率）；运行
  Differential TD(0) action-value prediction。对 `stationary`、`seasonal_wind`、
  `hidden_context` 和 `moving_goal` 的完整矩阵，比较 \(\alpha\in\{0.01,0.05,0.10\}\) 与 learned
  per-parameter alpha。tabular 中每个 \(Q(s,a)\) 表项视为一个独立参数，因此 learned alpha
  是 per-table-entry step-size，不把它解释为 feature selection。
- **主要观测量**：每 step TD error 平方的滑动均值；各模式稳定窗口内的 prediction error；切换后
  error 的峰值、恢复到切换前稳定误差的 \(1+\rho\) 倍所需步数（预先固定 \(\rho\)，如 10%）；
  切换后固定长度窗口的 AUEC；第二次 A 的恢复时间；训练 B 期间固定 A probe stream 的 error。
  若要报告“真值误差”，只对固定策略、固定模式离线用独立的长轨迹 reference estimate 计算，
  不把该 reference 输入 learner。
- **预期与判定**：较大 fixed alpha 应较快压低切换后误差、但在 stationary 窗口有更高方差，且更容易
  使 A probe 退化；较小 alpha 相反。若第二次 A 的恢复更快且 B 期间 A probe 未显著恶化，才有
  tabular retention 的证据；只要 probe 已恶化，即使恢复快也只能说明 tracking。若所有 alpha 的
  曲线在给定置信区间内无差异，则此环境强度下尚未暴露 step-size trade-off，应先检查测量窗口与
  切换强度，而不是直接引入复杂算法。

### P0.2：tabular control

- **设置**：保持同一环境矩阵与全部 fixed / learned alpha 机制，运行 Differential Sarsa(0)，使用固定的
  \(\epsilon\)-greedy 探索率；每个 seed 在相同随机数协议下比较不同 alpha。
- **主要观测量**：平均 reward、每 1,000 step 达到目标次数、碰撞率、策略熵、状态--动作访问
  分布；以及与 P0.1 同定义的切换后恢复时间、AUEC 和 A probe。A probe 必须使用冻结的评估策略和
  固定环境随机种子，不能被训练中的 \(\epsilon\)-greedy 行为替代。
- **预期与判定**：相对于 P0.1，control 的数据分布会随策略改变，因此性能下降可能来自 value
  改变、访问分布改变或两者。只有在同时报告访问分布时，才能将较慢恢复归因于学习机制；若 reward
  恢复但 A probe / A 模式评估不恢复，则不能宣称 retention。

### Phase 0 的产出

确定 \(\eta_{\bar r}\) 的一次性校准、固定 \(\epsilon\)、评价窗口 \(\rho\) 和统计协议；这些设置随后
冻结。Phase 0 还给出“在无共享参数时，环境是否已足以产生可观测 step-size 现象”的答案。

## Phase 1 — D=55 LFA Differential TD(0) prediction

固定行为策略、环境条件、alpha 机制与 Phase 0 对应；仅将 tabular one-hot 表替换为上面的
55 维显式 LFA。这样 Phase 0/1 的差异可解释为 parameter sharing / representation 的影响，
而不是算法不同。

### 目的

在固定数据分布下，分离两件事：D=55 的参数共享如何改变 fixed alpha 的取舍，以及 per-feature
alpha 能否在**不改变 feature pool**的前提下改善这种取舍。因为行为策略固定，这一阶段不把策略
诱导的数据分布变化混入结论。

### P1.1：fixed-alpha 的表示效应

- **设置**：对 P0.1 的全部环境和三种 fixed alpha，使用同一固定行为策略与 D=55 LFA。
  tabular 与 LFA 使用相同的 seed、切换时刻和评估 stream；不针对 LFA 重调 alpha。
- **主要观测量**：沿用 P0.1 的 prediction 指标；另记录每个 action block 的权重范数、
  \(\lVert\Delta w_t\rVert_2^2\)（update energy）、不同位置 prediction error 的空间热图。
- **预期与判定**：若 LFA 相比同 alpha 的 tabular 有更低的切换后 AUEC，却在未访问位置也出现
  明显误差改变，说明共享带来更快传播及 interference；若 A probe 的退化同时增大，则不能把较快
  适应视为无代价的 continual learning 改进。若 LFA 与 tabular 的 trade-off 一致，则在当前 D=55
  下表示共享不是主要瓶颈。

### P1.2：per-feature alpha 的机制检验

- **设置**：仅在 P1.1 已显示有明确 trade-off 的环境条件中，将 fixed alpha 与无 trace 的
  TIDBD/IDBD-style per-feature alpha 比较。TIDBD 使用预注册的初值、边界和冻结后的 \(\theta\)；
  不按每个环境或 seed 重调。报告所有数值不稳定 run，不能静默剔除。
- **主要观测量**：除 P1.1 指标外，记录 55 个 \(\alpha_i\) 的中位数、10/90 分位、每个 action
  block 的分布，\(\beta_i\) 是否撞到边界，以及每个 feature 的累计 \(|\Delta w_i|\)。
- **预期与判定**：支持“learnable step-size 有益”的最小证据是：相对所有 fixed-alpha 基线，
  TIDBD 在同一条件下同时降低或不恶化稳定窗口误差，并降低恢复时间或 AUEC，且 A probe 不更差；
  同时 alpha 分布出现可重复的非退化差异。仅仅某些 \(\alpha_i\) 变小、或只在一次切换后更快，均
  不是 feature selection / retention 结论。若 TIDBD 仅相当于某个 fixed alpha，则结论应是当前
  feature pool 中它未展示出超越全局步长的机制优势。

## Phase 2 — D=55 LFA Differential Sarsa(0) control

与 Phase 1 使用完全相同的 D=55 feature、环境和 alpha 条件；仅将固定行为策略替换为学习中的
epsilon-greedy policy。Phase 2 相比 Phase 1 只增加 policy-induced distribution shift。

### 目的

检验 Phase 1 观察到的 step-size 现象在控制问题中是否仍成立；并明确区分 value 学习的遗忘与
策略改变导致的访问变化。Phase 2 不增加新 feature，也不改变环境矩阵。

### P2.1：fixed-alpha control

- **设置**：对 P1.1 的相同环境、D=55 和 alpha 运行 Differential Sarsa(0)。训练时固定
  \(\epsilon\)，评估时使用贪婪策略（或预注册的同一评估 \(\epsilon\)）；训练和评估必须分开记录。
- **主要观测量**：每模式的平均 reward、goal rate、碰撞率、策略熵、状态--动作访问分布；切换后
  reward/goal-rate 的恢复时间与 AUEC；冻结 A probe 上的相同指标；权重漂移和 update energy。
- **预期与判定**：如果 P1 的 alpha 排序在控制性能和 value 指标上都保留，说明它不只限于固定数据
  流；若 reward 结果与 P1 相反而访问分布也明显改变，应报告“策略--数据耦合改变了结果”，而不是
  声称 prediction 结论失效。A→B→A 的第二次 A 若比首次恢复快、并保持 A probe，才是 control
  中的 retention。

### P2.2：per-feature alpha control

- **设置**：在与 P1.2 相同的代表性条件下，比较 fixed alpha 与 TIDBD；其余设置、随机种子配对和
  失败处理完全相同。
- **主要观测量**：P2.1 的全部控制指标，加上 P1.2 的 alpha 分布和边界命中率；特别记录 alpha
  分布变化发生在 reward 改善之前还是之后。
- **预期与判定**：若 TIDBD 同时改善 adaptation 和 A probe / A recurrence，且没有通过显著提高
  碰撞率或降低探索来“伪改善”平均 reward，则可把它作为 continual-control 候选机制。若只提升
  当前 B 模式 reward 而 A probe 恶化，则其作用是更快 tracking，不是保留记忆。

## Phase 3 — D=71 feature-selection 因果检验

在 Phase 1–2 的机制结论稳定后，加入固定 16 维 nuisance group，训练 per-feature TIDBD。
冻结训练完成的权重和 alpha，不再学习，分别删除：

- nuisance group；
- 低 / 中 / 高 alpha 分位的 feature。

在 stationary、变化后与 A/B recurrence probes 上评价。只有删除后所有预注册指标基本不变，
才能称 feature 冗余；小 alpha 本身不等于无用。

### 目的

把“某 feature 的 alpha 小”从相关现象提升为可检验的 feature-selection 主张。由于 16 维
nuisance 的生成过程已知且与任务独立，它提供了一个受控的负例；Phase 3 仍不比较 feature
engineering。

### P3.1：nuisance 对适应和 alpha 的影响

- **设置**：在 P1.2/P2.2 中确实观察到 TIDBD 机制信号的代表性环境上，将 D=55 扩展为 D=71。
  比较 D=55 fixed alpha、D=71 fixed alpha 与 D=71 TIDBD；每步的 \(u_t\) 必须由独立随机流产生，
  并在算法之间配对。
- **主要观测量**：主 feature 与 nuisance group 的 alpha、累计更新量和权重范数分布；以及相同的
  prediction/control、adaptation、A probe 和 recurrence 指标。
- **预期与判定**：一个有意义的选择信号应表现为：TIDBD 下 nuisance group 的累计更新量和典型
  alpha 系统性低于相关的 55 维主 group，同时 D=71 的表现不显著劣于 D=55。若 nuisance alpha
  未下降但表现仍好，说明算法可能只是鲁棒而非在筛选；若下降却表现恶化，说明压低步长也可能错误地
  关闭有用容量。

### P3.2：冻结后的消融检验

- **设置**：训练结束后冻结 \(w,\alpha,\bar r\) 与策略；不再更新。分别将 nuisance group 置零，
  并按 alpha 分位数删除低、中、高 alpha 的 feature（删除数量预先相同、删除规则跨 seed 固定）。
  在 stationary、刚切换后和 A→B→A probe 上重放固定评估 stream。
- **主要观测量**：相对于未消融模型的 prediction error、平均 reward、goal rate、AUEC 与 A probe
  差值，并报告置信区间。
- **预期与判定**：若删除 nuisance 或低-alpha 组的所有预注册指标都在等价界内（等价界须在运行前按
  Phase 0 方差设定），而删除中/高-alpha 组产生明确性能下降，才支持“per-feature alpha 是可用的
  feature-importance / selection 信号”。若低-alpha 组删除也损害性能，则 alpha 不能被解释为可安全
  剪枝的选择信号。

## 实施顺序

环境审计已经完成。以下是后续 agent 应遵循的主线；不再执行 pilot、\(\theta\)/\(\eta_{\bar r}\) 调参、
特征维度 sweep、nuisance 消融或 `combined` 环境测试。除明确比较的 alpha 机制外，所有设定直接采用
上文固定值：\(D=55\)、\(\lambda=0\)、\(\eta_{\bar r}=0.01\)、\(\epsilon\) 与既定的评价窗口/统计协议。

1. **先实现并运行 Phase 0（tabular 对照）。** 对每个环境条件分别运行 P0.1 固定策略 prediction 与
   P0.2 epsilon-greedy control；每项只比较 fixed scalar alpha 与 per-table-entry adaptive alpha。
   输出共同指标：TD-error、切换后的恢复/AUEC、A probe retention，以及 control 的 reward 与 goal rate。
   这一步确认环境变化本身能产生 adaptation、interference 和 recurrence 现象，并给出无参数共享的基线。
2. **在完全相同协议下完成 Phase 1（D=55 prediction）。** 复用 Phase 0 的环境种子、切换时刻、
   行为策略、alpha 候选值和固定评估 stream；仅把 tabular one-hot 表换为 D=55 LFA，运行 P1.1 fixed
   alpha 与 P1.2 per-feature TIDBD。与 Phase 0 配对比较，以分离 parameter sharing 对 step-size 取舍的影响。
3. **再完成 Phase 2（D=55 control）。** 不改 feature、环境、alpha 机制或指标，只把固定行为策略换为
   Sarsa(0) 的 epsilon-greedy control，运行 P2.1 fixed alpha 与 P2.2 per-feature TIDBD。对照 Phase 1，
   判断 prediction 中的规律在策略和访问分布随学习改变时是否仍成立。
4. **按同一表格汇总主结论。** 每个 Phase × 环境 × alpha 机制均报告同一组 adaptation、retention 与
   steady-state 指标及跨 seed 统计；只在 stationary、seasonal wind、hidden context、moving goal 的结果上
   下结论。结论必须同时说明其适用的 prediction/control 情形，以及相对 tabular 基线的变化。
5. **Phase 3 留作条件性后续工作。** 它只在 Phase 1/2 已稳定观察到 per-feature alpha 差异后启动，
   用于检验 alpha 能否作为 feature-selection 信号；不属于当前主实验、不得阻塞 Phase 0–2 的实现与分析。
