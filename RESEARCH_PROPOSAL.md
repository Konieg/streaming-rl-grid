# 先理解结构，再进行回放

## 面向持续强化学习的变化感知因子化 Dyna 研究计划

## 1. 摘要

本项目希望研究一个比“基于模型的强化学习（MBRL）是否具有更高样本效率”更具体的问题：

> **当环境持续发生结构化变化时，一个能够区分稳定环境规律与漂移隐变量的智能体，能否比无模型 TD 方法和传统 Dyna 更快、更可靠地适应？**

当前仓库已经为这个问题提供了很好的工程基础：它实现了 continuing average-reward grid world，允许风、目标、障碍物和奖励独立变化；包含七种 differential TD/Dyna 算法以及 Dyna-Q+；支持精确 checkpoint 恢复和基于变化事件对齐的评估。

但是，现有完整实验表明，当前 benchmark 尚不能有效识别 MBRL 的潜在优势。实验结果并不是“所有算法都一样”，而是：

- vanilla Dyna 在 transition shift 下明显差于 model-free 方法；
- Dyna-Q+ 能显著缓解 vanilla Dyna 的退化，但仍未超过最佳 model-free baseline；
- Dyna-Q+ 的表面稳定性主要来自低碰撞、非目标导向的行为，而不是成功导航或快速重新规划；
- 共享的 D=55 表示没有让多数算法真正解决基础导航任务，因此 adaptation comparison 建立在一个尚未校准的 stationary learning problem 上。

基于这些发现，下一阶段应当：

1. 先建立所有主要算法都能解决的 stationary diagnostic setting；
2. 确保每一种环境变化确实改变最优策略；
3. 分离 observability、context inference、model learning 和 planning 四个问题；
4. 将 latest-transition table 替换为随机、可遗忘、可评估的 world model；
5. 提出 **Change-Aware Factored Dyna（CAFD，变化感知因子化 Dyna）**：分别学习稳定运动规律和快速漂移的风、障碍物、奖励参数，并根据模型变化进行有针对性的规划；
6. 使用 dynamic/LoCA regret、模型误差、目标到达行为和组合泛化，而不只使用全程平均奖励进行评估。

本项目希望验证一个可证伪的核心观点：

> MBRL 并不会天然获得持续学习优势。只有当 world model 表达了正确的因果结构、能够局部遗忘过期信息，并把模型变化转化为有针对性的重新规划时，它才可能比 model-free learning 更快地适应环境变化。

---

## 2. 当前仓库与实验设置

### 2.1 Continuing environment

环境始终不终止。智能体的 observation 为：

```text
(agent_x, agent_y, goal_x, goal_y, previous_action)
```

到达目标后，智能体获得目标奖励，并立即被随机传送到另一个合法非目标格。四种非平稳因素可以独立启用：

- 风向变化；
- 目标移动；
- 障碍物地图切换；
- 奖励倍率变化。

风是随机的：以 `w_strength` 的概率在当前风向上增加一个单位位移。调度事件发生在边界 step 的环境转移之后。具体实现见 [`environment.py`](stream_rl_grid/environment.py) 和 [`config.py`](stream_rl_grid/config.py)。

### 2.2 Value representation

已完成的主要实验统一使用 D=55 hand-crafted linear representation。每个动作对应一个独立的十一维特征块：

```text
[1, x, y, x^2, y^2, xy, dx, dy, dx^2, dy^2, dxdy]
```

这是一个全局、平滑的二阶函数逼近。它没有显式的局部障碍物特征，也没有 map/context representation。可选的 D=71 表示还会附加一个随机 nuisance category。具体实现见 [`features.py`](stream_rl_grid/features.py)。

### 2.3 当前 Dyna model

Vanilla Dyna 为每个原始 observation-action pair 保存最近一次转移：

```text
(observation, action) -> (latest_reward, latest_next_observation)
```

Planning 从所有 model keys 中均匀采样，并执行与真实 transition 相同的一步 differential Q-learning update。Dyna-Q+ 还会把已观察状态的所有未尝试动作初始化为零奖励自环，并添加经典的 time-since-tried bonus。具体实现见 [`dyna_q.py`](stream_rl_grid/algo/dyna_q.py) 和 [`dyna_q_plus.py`](stream_rl_grid/algo/dyna_q_plus.py)。

### 2.4 已完成的最终比较

新的 eight-algorithm comparison 已完成全部 200 个 run：

```text
5 个 setting × 8 种算法 × 5 个 paired seed = 200 runs
```

没有 failed 或 missing run。五个 setting 分别是：

- wind-only；
- goal-only；
- obstacles-only；
- reward-only；
- combined。

汇总结果位于 [`eight_algorithm_summary`](eight_algorithm_summary/)，原始结果位于 [`experiment_results/eight_algorithm_comparison`](experiment_results/eight_algorithm_comparison/)。

---

## 3. 完整八算法实验所提供的证据

### 3.1 全程平均奖励

下表给出每个 setting 中的最佳方法，以及三种 Dyna 方法的 mean reward per environment step。数值越高越好。

| Setting | 最佳方法 | 最佳均值 | Dyna-Q | Dyna-Q(λ) | Dyna-Q+ |
|---|---|---:|---:|---:|---:|
| Wind only | TIDBD | -1.041 | -1.231 | -1.142 | -1.056 |
| Goal only | Q-learning | -0.981 | -1.085 | -1.082 | -1.044 |
| Obstacles only | SARSA(λ) | -1.021 | -1.081 | -1.078 | -1.048 |
| Reward only | SARSA(λ) | -1.001 | -1.023 | -1.034 | -1.022 |
| Combined | SARSA | -1.084 | -1.365 | -1.250 | -1.161 |

这些结果修正并强化了此前的判断：

- Vanilla Dyna 并非与 model-free 方法持平，而是在 transition changes 下明显更差。
- Dyna-Q+ 始终优于 vanilla Dyna，在 wind-only 中提升约 `0.175`，在 combined 中提升约 `0.203`。
- Dyna-Q+ 没有在任何一个全程平均奖励比较中获胜。
- 所有入选的 Dyna-Q+ 配置都是 `planning_steps=1` 和最小的 `kappa=0.0001`；在 transition 和 combined setting 中还使用最小的 `alpha=0.01`。

超参数选择持续偏向最少 planning、最小 bonus 和通常最小的 step size。这说明 sweep 更倾向于抑制 model-generated update，而不是积极利用它。

### 3.2 Wind-only 中的表面 adaptation advantage 具有误导性

在 wind-only 的变化后前 250 步中，Dyna-Q+ 的描述性均值最高：

| Method | 250-step post-change mean reward |
|---|---:|
| Dyna-Q+ | -1.064 |
| TIDBD | -1.068 |
| Q-learning | -1.079 |
| SARSA(λ) | -1.081 |
| Vanilla Dyna-Q | -1.281 |

由于只有五个 seed，置信区间大量重叠，因此这不能被解释为统计上已经建立的胜利。更重要的是，行为分解表明 Dyna-Q+ 几乎从不到达目标：

| Setting | Method | Goals / 1000 steps | Collision rate |
|---|---|---:|---:|
| Wind only | Dyna-Q+ | 0.03 | 1.42% |
| Wind only | Q-learning | 6.51 | 3.08% |
| Wind only | Q(λ) | 8.74 | 4.04% |
| Goal only | Dyna-Q+ | 0.35 | 1.21% |
| Goal only | Q-learning | 16.69 | 4.12% |
| Combined | Dyna-Q+ | 0.19 | 4.01% |

对没有 reward multiplier 的 setting，每一步只能是普通 step、collision 或 goal，因此全程平均奖励可以精确写成：

\[
\bar r=-1+11p_{\text{goal}}-4p_{\text{collision}}.
\]

所以，一个极少产生有效目标进展、同时避免碰撞的策略，其平均奖励会稳定在 -1 附近。Dyna-Q+ 对 wind change 表现得“不敏感”，可能只是因为它原本就没有稳定地执行会被风显著影响的目标导航行为。

需要强调的是，独立复现并不支持“Dyna-Q+ 主要选择 stay”的说法。在 wind-only seed 0 中，它的 `stay` 比例约为 23%，但约 54% 的时间集中在远离目标的两个中心格子，并在前 8,046 步后不再到达目标。更准确的描述是：

> Dyna-Q+ 学到了低碰撞但非目标导向的局部循环，而不是成功适应风变化。

### 3.3 基础导航任务并没有被解决

所有已学习方法的 goal-reaching rate 都远低于拥有当前地图信息的 shortest-path reference。在同一批地图和 schedule 上，参考策略每 1000 步约到达 124--130 次目标，而学习算法通常只有 0--17 次。

最终实验实际使用的是 10×7 grid 和 8 个障碍物，而不是默认 GUI 中的 5×5。不过，这仍无法解释每次到达目标需要 100--400 步。在一个具体 seed 中，所有合法起点到固定目标的平均无风最短距离只有 7.23，最大距离为 15。

D=55 表示能够表达平滑的方向偏好，却无法准确表达局部墙体、狭窄通道、绕行和 context-specific detour。不同 seed 上还出现了明显的“部分地图能学到、部分地图完全坍塌”的现象。例如 wind-only Q-learning 五个 seed 的目标次数为：

```text
528, 1, 463, 955, 7
```

这里 seed 同时改变训练随机性和随机生成的地图，因此方差还混合了 environment difficulty。

这是一个 benchmark identifiability 问题：当多数方法都陷入弱策略时，adaptation curve 无法说明算法是否正确地通过 world model 传播了环境变化。

### 3.4 Latest-transition model 为什么会有害

当前 model 是 tabular 的，但问题的核心不是 tabularity。在这个小规模环境中，tabular stochastic model 恰好是最适合排查问题的 baseline。真正的问题包括：

1. **随机 outcome 被过度确定化。** 一次 sampled wind outcome 被保存为确定性转移，并在 planning 中反复 replay。
2. **Hidden-context aliasing。** Wind、obstacle 和 reward phase 不在 observation 中，不同 MDP 的 transition 被存到同一个 key 下。
3. **Stale model entry。** 环境变化后，旧 entry 会一直保留，直到完全相同的 key 被再次访问。
4. **Uniform replay dilution。** Combined run 中 vanilla model 约有 2,900 个 entries，新变化的局部 transition 很难获得足够 planning probability。
5. **冗余 model state。** `previous_action` 不影响当前 dynamics，却使 model 碎片化；D=71 的随机 nuisance category 也会进入 model key，进一步扩大这一问题。
6. **外生变化被错误归因给 action。** 定时 goal move 会出现在 `next_observation` 中，并被记录成仿佛是当前 action 导致目标移动。这个虚假因果 transition 随后可能被长期 replay。
7. **Update budget 不公平。** Dyna 每个真实 step 执行一次 real update 和 `planning_steps` 次相同步长的 update；Q-learning 只有一次。增加 planning 同时改变了模型使用程度、优化速度和计算量。
8. **Dyna-Q+ reward scale 不匹配。** 未尝试动作被初始化为 zero-reward self-loop，但真实普通 step 的 reward 是 -1。在 differential update 中，即使 `kappa=0`，零奖励相对于当前 average reward 已经具有很强的 optimism；之后才会额外添加 `kappa * sqrt(tau)`。因此，入选最小 `kappa` 并不意味着实际 optimism 很小。

完整实验与“model bias 被 planning 放大”这一解释一致。目前没有证据表明直接把 tabular model 换成通用神经网络就能解决问题。

### 3.5 当前非平稳因素与指标的问题

- Goal coordinate 是直接可观察的。Goal movement 主要测试 goal-conditioned generalization；wind/map/reward changes 则产生 hidden-context partial observability。这不是同一个问题。
- Reward multiplier 较小，而且常常不会改变最优策略的身份。
- 随机障碍物地图不能保证发生变化的障碍物处在决策关键路径上。
- 超参数按 60,000 步 stream-average reward 选择，而不是按 adaptation regret 选择。
- Wind/goal/obstacle isolated setting 复用了在 composite transition-shift distribution 上选出的配置，而不是针对每个 isolated setting 单独调优。这是合理的 robustness protocol，但不能被解释为 per-setting oracle tuning。
- Recovery time 相对于 agent 自己的 pre-change reward 定义。表现长期很差的 agent 因为 baseline 很低，反而可能看起来恢复得更快。
- 仅看 250--1000 step reward window，无法区分真正的 replanning 与从未认真追逐受影响目标或路线的策略。
- `goals_per_1000_steps` 是最后一个 rolling window 的指标；全程行为应当另外由 `total_goals / total_steps` 计算，避免只看到训练末尾。

---

## 4. 研究问题与假设

### 4.1 核心研究问题

> 一个具有 adaptive forgetting 和 change-prioritized planning 的因子化 world model，能否在 abrupt、gradual、recurring 和 compositionally novel changes 下，比 model-free TD、vanilla Dyna 和 unfactored recency-aware Dyna 获得更低的 dynamic regret？

### 4.2 如何操作性定义“理解环境本质”

“理解环境”不能只作为比喻，而需要对应可测量的能力。在本项目中，如果一个智能体能够做到以下几点，就认为它学到了有用的环境结构：

1. 准确预测 held-out transitions 和 rewards；
2. 从地图局部观察到的风，推断全局 wind parameter 的变化；
3. 通过 planning 把这个变化传播到尚未重新访问的状态；
4. 在更新变化因子的同时保留稳定的 motion/topology knowledge；
5. 再次遇到旧 context 时，比第一次学习更快地恢复；
6. 在过去未见过、但由已学习因素重新组合而成的环境中成功规划。

### 4.3 研究假设

**H1：模型误差放大。** 在当前 latest-outcome model 下，当 dynamics 是随机或 context 隐藏时，增加 planning updates 会增大 post-change regret，除非显著减小 planning step size。

**H2：干净条件下的 Dyna advantage。** 在 fully observable stationary MDP 中，如果 value representation 足够表达任务，empirical stochastic Dyna 应当比 Q-learning 具有更快的早期学习；perfect-model planner 给出可达到的上界。

**H3：局部变化传播。** 在 LoCA-style local change 中，prioritized planning 应当优于 uniform planning，因为它能够在 agent 重新访问所有相关状态前传播新的 reward/transition information。

**H4：漂移跟踪。** 在不同 drift speed 下，fast/slow factored model 应当比 fixed-window model 具有更低的 parameter estimation error 和 dynamic regret。

**H5：组合迁移。** 当 agent 已分别见过不同 wind、obstacle 和 reward factor 时，factorized model 应当比 monolithic table 更快地适应未见过的 factor combination。

**H6：Abrupt 与 gradual 的机制差异。** Hard change-point reset 应当更适合突变，smooth filter 更适合连续漂移，而 surprise-gated mixture 应当在两类变化中更加稳健。

---

## 5. Environment v2：具有诊断能力的 continual grid world

下一版环境的目标不是单纯变大，而是让任务能够识别不同算法机制。

### 5.1 基础布局

使用一到两张人工设计的 10×10 或 15×15 地图，包含：

- 一个公共 restart region；
- 两个 goal region；
- 一条较短但危险的 corridor；
- 一条较长但安全的 corridor；
- 一个位于路线分叉前的 decision bottleneck；
- 若干对风或障碍物变化敏感的关键边。

两个目标使 reward change 能够改变最优目标；两条路线使 wind 和 obstacle change 能够改变最优路线。人工拓扑也使 adaptation failure 的因果原因可以被解释。

在接受一个 scenario 前，应当对每个 frozen MDP 做 exact planning，并验证：

- 最优 goal 或 route 确实发生改变；
- 足够比例的 reachable state 改变了 greedy action；
- 最优 average reward 与非目标导向行为之间存在明显差距；
- 所有主要 value representation 都能够解决 stationary variants。

### 5.2 Reward calibration

当前 `stay` 或局部循环能够稳定获得接近 -1 的 reward，是很有吸引力的保守解。应当重新校准 reward，使 goal seeking 明显优于停留，同时避免少量 exploratory collision 造成过大的早期损失。一个候选尺度是：

```text
ordinary step:       -0.05
stay:                -0.10
collision:           -0.25
low-value goal:      +1.0
high-value goal:     +4.0
```

这些数值不能直接照搬，应当先用 exact oracle 和 random-policy baseline 验证。单独设置 stay cost 比删除 stay action 更合适，因为它保留了研究 stay 受风影响的能力，同时降低退化解的吸引力。

### 5.3 三类非平稳变化

#### A. Abrupt local change：LoCA-style revaluation

- 预训练阶段令 goal A 的价值高于 goal B；
- 只改变 goal A 的局部 reward，使 goal B 成为新的最优目标；
- 在一个受控阶段，只允许 agent 在能观察到 A 新 reward、但无法回到早期 decision bottleneck 的区域活动；
- 随后从公共 restart distribution 评估。

Model-based agent 应当更新局部 reward model，并通过 planning 改变远端 bottleneck 上的选择。Model-free agent 则需要通过真实访问逐步传播新的 value。

[LoCA regret](https://arxiv.org/abs/2007.03158) 正是为了把这种 model-based propagation 与普通 single-task sample efficiency 区分开来。

#### B. Smooth global drift：wind distribution

把风表示为以下类别上的概率分布：

```text
{none, up, right, down, left}
```

随后连续改变其参数。例如：

\[
p_t(\text{right})=\rho\frac{1+\sin(2\pi t/T)}{2},\qquad
p_t(\text{left})=\rho-p_t(\text{right}).
\]

对 drift timescale `T` 做 sweep，例如 500、2,000 和 8,000 步。同时在测试中加入 triangular schedule 和 bounded random walk，避免方法仅通过记忆正弦周期取得成功。

#### C. Recurring and compositional contexts

构造因子化 context：

```text
wind:      {left, right}
map:       {upper corridor open, lower corridor open}
reward:    {goal A preferred, goal B preferred}
```

训练时只提供其中一部分组合，之后重新出现旧组合，并至少保留一个未见组合用于测试。这能够区分“记忆完整 MDP”与“学习可复用环境因素”。

### 5.4 不应删除 abrupt changes

Smooth drift 应当被加入，而不是替代全部突变：

- abrupt local change 测试模型能否传播一条新事实；
- gradual change 测试 online filtering 和 prediction；
- recurring switch 测试 retention 和 retrieval；
- obstacle topology 本来就是离散的，不过可以用逐渐增加 edge-block probability 作为 smooth analogue。

一个有价值的 benchmark 应当覆盖 change magnitude 和 change timescale，而不是只选择一种 non-stationarity。

### 5.5 Observability tiers

每个 scenario 都应当包含两个版本：

1. **Observable-context diagnostic：**直接提供当前 wind/reward/map parameter 或 context ID，用于隔离 representation、model learning 和 planning。
2. **Hidden-context continual task：**只提供物理状态和近期 interaction history，或者由 agent 学习 context belief，用于额外测试 inference。

如果不进行这一拆分，性能差异无法被清楚归因于 model learning、planning 还是 partial observability。

---

## 6. 提议算法：Change-Aware Factored Dyna

### 6.1 Canonical state

Model 使用最小物理状态：

```text
(agent_x, agent_y, goal configuration)
```

除非显式加入 inertia 或其他 action-history-dependent dynamics，否则应从 model key 中移除 `previous_action`。随机 nuisance variable 绝不能进入 model key。

在 hidden-context experiment 中，应当使用 context belief `b_t` 扩展 policy/planner，而不是使用 ground-truth context。

### 6.2 因子化 world model

把 world model 分解为：

\[
M_t=\{M_{\text{motion}},M_{\text{wind},t},
M_{\text{obstacle},t},M_{\text{reward},t},M_{\text{context},t}\}.
\]

#### 稳定 motion model

学习环境中长期稳定的 controlled movement law 和 boundary behavior。这个模块使用较慢更新，并在 context change 后保留。

#### Wind model

利用 intended movement 与 realized movement 的残差推断风。维护 categorical wind distribution 的 discounted Dirichlet counts 或等价的 exponential moving estimate。由于风是全局变量，一个位置上的 observation 可以更新整个地图上的 predicted transition。

#### Obstacle model

维护每个 cell 或 directed edge 被阻挡的概率。只有被成功移动或 collision 涉及的局部 entry 才更新，从而支持 local forgetting，而不删除其他区域仍然准确的知识。

#### Reward model

分别估计 ordinary step、collision 和每个 goal 的 reward。局部 goal-reward change 不应覆盖无关 dynamics 或其他 reward components。

#### Exogenous context model

Goal、wind 或 map 的变化应被表示为 latent variable `z_t` 的演化，而不是被错误表示为 selected action 的结果：

\[
p(s_{t+1},z_{t+1}\mid s_t,z_t,a_t)
=p(s_{t+1}\mid s_t,a_t,z_t)p(z_{t+1}\mid z_t).
\]

第一版可以使用小规模 discrete posterior，或者由 continuous wind/reward parameters 构成的向量。Neural latent-state model 是 stretch goal，而不是项目成立的前提。

### 6.3 Surprise-gated fast/slow learning

同时维护 slow structural estimate 和 fast adaptive estimate。定义 transition surprise：

\[
u_t=-\log\hat p(r_t,s_{t+1}\mid s_t,a_t).
\]

用它在 slow 和 fast update rate 之间插值：

\[
\eta_t=\eta_{\min}+
(\eta_{\max}-\eta_{\min})\sigma(c(u_t-h)).
\]

持续的大 surprise 触发快速局部遗忘或 context switch；普通 sampling noise 则保留 slow model。Page-Hinkley、CUSUM 或 Bayesian online change-point detection 可以作为替代消融。

这一机制直接针对 stale replay。已有研究表明，局部删除过期样本可以恢复 Dyna 和 deep world-model agent 的 adaptation，同时不破坏其他区域的模型知识（[Rahimi-Kalahroudi et al., 2023](https://arxiv.org/abs/2303.08690)）。

### 6.4 Confidence-weighted prioritized planning

用模型变化对 value 的影响替代 uniform planning priority：

\[
\operatorname{priority}(s,a)
=\left|T_{M_t}Q(s,a)-T_{M_{t-1}}Q(s,a)\right|
\times c_t(s,a),
\]

其中 `c_t(s,a)` 随 posterior/model uncertainty 增加而减小。第一项选择 Bellman target 因模型变化而显著改变的状态；confidence term 防止 agent 重复利用不可靠模型。

当 global wind posterior 更新时，即使没有新的环境交互，也可以重新计算所有物理状态的 priority。这正是希望展示的 MBRL advantage：在一个位置观察到风以后，可以对其他尚未访问的位置进行 counterfactual replanning。

### 6.5 可选 context memory

作为 stretch component，可以保存一个小型 recurring-context bank 及其 model parameters。当新 evidence 与旧 context 匹配时，直接恢复相应参数，而不是从头学习。Factorized retrieval 应当与 monolithic context table bank 进行比较。

---

## 7. Baselines 与消融实验

### 7.1 最小 baseline set

1. One-hot/tabular differential Q-learning；
2. One-hot/tabular differential Q(λ) 或 SARSA(λ)；
3. 当前 latest-transition Dyna-Q；
4. 使用 undiscounted counts 的 empirical stochastic Dyna；
5. Sliding-window 或 exponentially discounted Dyna；
6. 不带 factorization 的 prioritized Dyna；
7. Change-Aware Factored Dyna；
8. Perfect-model planner，作为 model-learning upper bound；
9. Oracle-context CAFD，作为 context-inference upper bound。

D=55 representation 应当作为 representation-stress ablation 保留，但不能继续作为评价 planning 的唯一表示。

### 7.2 必要消融

| 移除的组件 | 回答的问题 |
|---|---|
| Factorization | Compositional transfer 是否真的来自结构分解？ |
| Adaptive forgetting | 仅保证 freshness 是否已经足够？ |
| Prioritization | Post-change gain 是否来自有针对性的传播？ |
| Confidence weighting | Uncertainty 是否能避免 harmful model exploitation？ |
| Context inference | 瓶颈在识别环境还是在当前环境内规划？ |
| Context memory | Recurrence advantage 是否来自 retrieval？ |

### 7.3 Planning fairness

引入相互独立的参数：

```text
alpha_real
alpha_plan
planning_steps
```

报告两种 protocol：

- **Environment-step matched：**真实环境交互次数相同；
- **Update/compute matched：**总 TD/gradient update 数或 wall-clock compute 可比。

必须包含 `planning_steps=0`，验证 Dyna implementation 与对应 model-free update 完全等价。增加 planning steps 时，应当单独调节或归一化 `alpha_plan`，否则实验混淆了 planning 与更大的 effective learning rate。

---

## 8. 评估协议

### 8.1 主要 control metrics

#### Dynamic regret

对每个 frozen context parameter `z`，用 relative value iteration 精确求解有限 grid，获得 optimal average reward `g*(z)`。报告：

\[
\operatorname{DynRegret}(T)
=\sum_{t=1}^{T}\left(g^*(z_t)-r_t\right).
\]

Stochastic realized reward 会使该指标有噪声，但在 paired seeds 下是合理的。可以同时报告基于 oracle differential action value 的低方差 pseudo-regret。

#### Local-change regret

对 abrupt event，报告 50、250、500 和 1000 step post-change regret AUC，而不只报告 raw reward mean。LoCA protocol 特别适合识别快速 model-based propagation（[van Seijen et al., 2020](https://arxiv.org/abs/2007.03158)）。

#### Behavioral success

始终同时报告：

- 每 1000 步到达目标的次数，并按 goal identity 分开；
- 选择当前 optimal corridor/goal 的比例；
- collision rate；
- stay-action rate；
- 相对于 oracle 的 path inefficiency；
- state occupancy heatmap 和 recurrent cycle statistics。

这可以防止 change-insensitive、non-goal-directed policy 被错误标记为 adaptive。

### 8.2 Model 与 inference metrics

- Transition negative log likelihood 和 Brier score；
- 按 reward component 分解的 prediction RMSE；
- Wind parameter RMSE；
- Obstacle edge-classification error；
- Context posterior accuracy/calibration；
- Change-detection delay 和 false-positive rate；
- 自变化后尚未访问状态上的 prediction error；
- Held-out factor combination 上的模型误差。

### 8.3 Continual-learning metrics

- Recurring context 的 first-acquisition 与 reacquisition time；
- 对 unseen factor combination 的 forward transfer；
- 未变化 model component 的 retention；
- 返回旧 context 后的 backward transfer/interference；
- Regret 随 drift speed 和 variation budget 的变化。

### 8.4 Randomness 与统计方法

- 五个 paired seed 只用于 smoke test；最终结论至少使用二十个 paired seed。
- 超参数在独立 development maps/seeds 上选择，随后锁定用于 test runs。
- 对 method difference 报告 paired bootstrap confidence interval。
- 预生成 exogenous schedule 和 wind draws，使 policy 分化后的算法仍面对可比的环境随机性。
- 将“per-setting tuned configuration”与“在 development distribution 上选择的单一 robust configuration”分开报告。
- 不使用 marginal confidence interval 是否重叠来代替 paired significance test。
- 将 environment instance seed 与 algorithm/exploration seed 分开，避免五个 seed 同时改变地图难度和训练随机性。

---

## 9. 分阶段实现与 decision gates

### Stage 0：保存并补充当前结果

- 冻结现有 200-run manifest 和 summary，作为 Phase 1 negative result；
- 增加 goals、collisions、stay rate、state occupancy 和 model size 汇总；
- 记录当前发现：stale、aliased、unfactored planning 可能使 continual Dyna 明显差于 model-free TD；
- 明确 Dyna-Q+ 的部分提升来自低碰撞但非目标导向的行为。

### Stage 1：Stationary calibration

实现：

- One-hot/tabular value representation；
- 不含 `previous_action` 或 nuisance variable 的 canonical model key；
- Empirical categorical transition counts；
- `planning_steps=0` identity test；
- Perfect-model Dyna；
- 独立的 real/planning step size；
- Action histogram、state occupancy 和 cycle diagnostics。

Decision gate：

> 在 fully observable stationary task 中，所有主要 agent 必须达到预设比例的 oracle performance，并且 perfect-model Dyna 必须表现出可测量的 early-learning benefit。否则，应先修复 representation/planning，而不是继续增加 non-stationarity。

### Stage 2：Diagnostic environment

实现 two-goal/two-corridor map、exact average-reward oracle、reward calibration 和 observable/hidden context tiers。加入 scenario validation，自动拒绝不会改变 oracle policy 的环境变化。

### Stage 3：Adaptive model baselines

依次实现并比较：

- Empirical Dyna；
- Sliding-window Dyna；
- Exponential-forgetting Dyna；
- Prioritized Dyna。

在加入完整 factorized contribution 前，先确定性能改善有多少仅来自 freshness 和 planning。

### Stage 4：Change-Aware Factored Dyna

加入 motion/wind/obstacle/reward factorization、surprise-gated fast/slow update、confidence-weighted priority 和 latent-context belief。

### Stage 5：最终实验

运行：

1. Stationary sample-efficiency calibration；
2. Abrupt LoCA reward revaluation；
3. Abrupt local obstacle change；
4. 三种 timescale 的 smooth wind drift；
5. Recurring contexts；
6. Held-out factor compositions；
7. Component ablations；
8. Compute-matched comparisons。

---

## 10. 预期结果与可证伪性

即使新方法没有在所有条件下获胜，只要实验能够隔离原因，这个项目仍然可以形成有价值的研究结论。

### 支持核心假设的结果

CAFD 具有更低的 LoCA/dynamic regret，在未重新访问状态上具有更低的 model error，并表现出更快的 reacquisition 和 held-out composition adaptation。移除 factorization 或 prioritized planning 后，这些优势显著缩小。

### 只支持 adaptive forgetting 的结果

Sliding-window 或 exponential Dyna 与 CAFD 表现相同。这意味着在当前问题规模上，freshness 比 causal factorization 更重要。

### 支持 model-free tracking 的结果

在 slow drift 下 model-free TD 与 CAFD 相同，但在 abrupt local change 下落后。这将确定 model-based propagation 发挥价值的边界条件。

### 否定所提 MBRL advantage 的结果

在完成 stationary calibration 且使用准确模型后，perfect-model/prioritized Dyna 仍不能改善 adaptation。这意味着当前 task 或 average-reward planning formulation 并不提供预期的传播优势，应当在提出 continual-MBRL claim 前重新审视问题定义。

---

## 11. 可行的项目范围

### Minimum viable contribution

如果时间有限，只实现：

1. 一个 two-goal LoCA grid；
2. Tabular Q-learning、vanilla Dyna、empirical Dyna、prioritized Dyna 和 perfect-model Dyna；
3. 带 exponential forgetting 的 factored reward/transition model；
4. LoCA regret 和 model prediction error；
5. Observable context 和一个 hidden-context condition。

这已经能够形成一个完整的 final project：解释当前 Dyna 为什么失败，以及哪些最小 model/planning modification 可以恢复 adaptive behavior。

### Stretch contribution

- Continuous latent wind filtering；
- Surprise-gated fast/slow model；
- Recurring context bank；
- Held-out factor composition；
- 一个小型 neural latent model，与 structured model 对比。

只有在 tabular/factored diagnostic 正常工作后，才应加入 neural model。现代 deep MBRL 并不会自动适应局部变化：PlaNet 和 DreamerV2 等方法也曾表现出较差的 local-change adaptation，而经过修改的 linear Dyna 可以成功（[Wan et al., 2022](https://arxiv.org/abs/2204.11464)）。

---

## 12. 预期研究贡献

最终最有力的研究叙事是：

1. **Negative finding：**从 stale、aliased latest-transition model 中进行 uniform replay，可能使 continual Dyna 明显差于 model-free TD。
2. **Benchmark finding：**当保守、非目标导向的策略对变化不敏感时，只看 reward adaptation curve 会产生误导。
3. **Method contribution：**分离稳定环境结构与漂移因素，以不同 timescale 更新它们，并利用 model change 驱动 prioritized planning。
4. **Evaluation contribution：**同时评估 control regret、world-model accuracy、behavioral success、recurrence 和 compositional transfer。

这一解释与 broader continual MBRL literature 一致：non-stationarity 更适合被建模为低维 latent process，而不是任意、互不相关的 MDP 序列。相关形式化包括 context-conditioned dynamics（[Lee et al., 2020](https://proceedings.mlr.press/v119/lee20g.html)）、block contextual MDP（[Sodhani et al., 2022](https://proceedings.mlr.press/v168/sodhani22a.html)）和 dynamic parameter MDP（[Xie et al., 2021](https://proceedings.mlr.press/v139/xie21c/xie21c.pdf)）。

---

## 13. 参考文献

1. van Seijen, H., Nekoei, H., Racah, E., & Chandar, S. (2020).  
   [The LoCA Regret: A Consistent Metric to Evaluate Model-Based Behavior in Reinforcement Learning](https://arxiv.org/abs/2007.03158).
2. Wan, Y., Rahimi-Kalahroudi, A., Rajendran, J., Momennejad, I., Chandar, S., & van Seijen, H. (2022).  
   [Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods](https://arxiv.org/abs/2204.11464).
3. Rahimi-Kalahroudi, A., Rajendran, J., Momennejad, I., van Seijen, H., & Chandar, S. (2023).  
   [Replay Buffer with Local Forgetting for Adapting to Local Environment Changes in Deep Model-Based Reinforcement Learning](https://arxiv.org/abs/2303.08690).
4. Lee, K., Seo, Y., Lee, S., Lee, H., & Shin, J. (2020).  
   [Context-aware Dynamics Model for Generalization in Model-Based Reinforcement Learning](https://proceedings.mlr.press/v119/lee20g.html).
5. Sodhani, S., Meier, F., Pineau, J., & Zhang, A. (2022).  
   [Block Contextual MDPs for Continual Learning](https://proceedings.mlr.press/v168/sodhani22a.html).
6. Xie, A., Harrison, J., & Finn, C. (2021).  
   [Deep Reinforcement Learning amidst Continual Structured Non-Stationarity](https://proceedings.mlr.press/v139/xie21c/xie21c.pdf).

