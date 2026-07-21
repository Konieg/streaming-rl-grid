# Beyond Sample Efficiency: Learning Reusable Environment Mechanisms for Continual Reinforcement Learning

# 超越样本效率：为持续强化学习学习可复用的环境机制

**Recommended subtitle / 推荐副标题**

Change-Aware Factored Dyna for Rapid Adaptation to Local Change, Drift, and Recurrence

变化感知因子化 Dyna：快速适应局部变化、漂移与复现

**Alternative titles / 备选标题**

- Beyond Sample Efficiency: Can Model-Based RL Learn the Structure of a Changing World?
- What Should a World Model Learn for Continual Reinforcement Learning?

**Why the recommended title works / 为什么推荐这个标题**

EN: “Beyond Sample Efficiency” states the motivating question, while “Learning Reusable Environment Mechanisms” explains the idea before introducing an unfamiliar acronym. “Continual Reinforcement Learning” fixes the setting. The subtitle then names the concrete contribution and its target changes without claiming a universally causal representation.

中：标题前半句直接对应研究动机：MBRL 除了 sample efficiency 还有什么？“Learning Reusable Environment Mechanisms”先用直观语言说明核心思想，避免观众一开始就被陌生算法名阻挡；“Continual Reinforcement Learning”限定研究场景。副标题再引出 CAFD 及其适应对象，同时不夸大为“已经学到了普适因果模型”。

**Thirty-second overview / 30 秒总述**

EN: We ask whether model-based RL can offer more than sample efficiency in a continuing, non-stationary task. Our hypothesis is that a model becomes useful for continual adaptation only when it represents reusable mechanisms. We therefore separate stable grid mechanics from changing wind, corridor, and reward factors, then use model revisions to prioritize planning. Across four diagnostic changes and 20 paired seeds, factorization is the dominant improvement; targeted planning adds a smaller benefit, and the resulting lower regret also produces higher real online reward.

中：我们研究的问题是：在持续、非平稳任务中，MBRL 除了样本效率之外还能提供什么？我们的假设是，world model 只有在表示可复用环境机制时，才会真正帮助持续适应。因此，我们将稳定的网格运动规律与会变化的风、通道和目标奖励分离，并根据模型变化分配 planning。四个诊断场景、20 个 paired seeds 的结果表明：factorization 是主要增益来源，针对性 planning 提供较小的追加收益，而且较低的 regret 最终确实转化成了更高的真实在线 reward。

---

## 1. Motivation

## 1. 研究动机

### Beyond sample efficiency

### 超越样本效率

#### Core question / 核心问题

EN: The usual advantage attributed to model-based reinforcement learning is sample efficiency: the agent can reuse observed transitions through planning. Our project asks a different question: if the environment changes continually, can an explicit model help the agent identify what changed, preserve what did not change, and revise decisions that have not yet been revisited?

中：MBRL 最常见的优势是样本效率：智能体可以通过 planning 重复利用已经观察到的 transition。我们的项目提出了另一个问题：当环境持续变化时，显式 world model 能否帮助智能体识别“什么变了”、保留“什么没变”，并在尚未重新访问相关状态之前就修改决策？

#### Operational meaning of “understanding” / “理解环境本质”的可操作定义

EN: We do not use “understanding” as a philosophical claim. In this project it has three measurable meanings:

1. **Parameter tracking:** the learned wind, obstacle, and reward factors follow the true latent factors.
2. **Systematic generalization:** one local observation changes predictions at other state–action pairs governed by the same factor.
3. **Compositional reuse:** previously learned factors can be combined in a context that was not observed as a whole.

中：这里的“理解”不是哲学意义上的主张，而是三个可测量的能力：

1. **参数跟踪：** 学到的风、障碍和奖励因子能够跟踪真实 latent factors；
2. **系统性泛化：** 一次局部观察能修改所有受同一因子支配的 state-action predictions；
3. **组合复用：** 已经分别学习过的 factors 能在一个整体未见过的 context 中重新组合。

#### Hypothesis / 假设

EN: A monolithic state–action model can become stale because evidence at one location only updates one table entry. A factored model can instead revise a global wind distribution, one corridor edge, or one goal-reward head. Planning can then propagate the model revision to distant decisions.

中：整体式 state-action model 很容易陈旧，因为一个位置上的新证据通常只更新一个表项。因子化模型则能把证据解释为“全局 wind distribution 变了”“某条 corridor edge 变了”或“某个 goal reward head 变了”。随后 planning 可以把这种模型修正传播到远处的决策状态。

#### What would falsify the idea? / 什么结果会否定这个想法？

EN: The hypothesis would be weakened if ordinary recency-aware Dyna matched the factored model, if lower model error did not reduce control regret, or if benefits disappeared under smooth drift and recurring contexts. The negative surprise-adaptation result is therefore important: extra model complexity is not automatically useful.

中：如果普通 recency-aware Dyna 与 factored model 一样好、如果更低的模型误差不能转化为更低的控制 regret，或者优势在 smooth drift 与 recurring context 中消失，那么我们的假设就会被削弱。因此，Surprise 变体的负结果很重要：增加模型复杂度不会自动带来优势。

#### Likely question: “Is this just sample efficiency?” / 可能追问：“这不还是样本效率吗？”

EN: Partly, but the diagnostic distinction is sharper. In LoCA, the agent observes the changed reward only near one goal and is then tested at a distant choice point. The benefit is not merely extracting more updates from the same local transition; it is using model structure to infer how that transition changes decisions elsewhere. Smooth wind tracking and held-out factor composition test the same idea beyond one abrupt switch.

中：部分相关，但我们的诊断更具体。LoCA 中，agent 只在一个目标附近观察奖励变化，然后立刻回到远端 choice point 接受测试。优势不只是从同一条 transition 中做更多次更新，而是利用模型结构推断这条局部证据如何改变其他位置的决策。Smooth wind tracking 和未见 factor composition 又把这个问题扩展到了单次突变之外。

---

## 2. Environment Design

## 2. 环境设计

![Continuing two-goal grid](poster_assets/grid_world.png)

### Topology and continuing semantics / 拓扑与 continuing 语义

EN: The environment is a $7\times7$ grid. A vertical wall at $x=3$ leaves two passages, at $y=1$ and $y=5$. Goal A is at $(6,1)$ and Goal B is at $(6,5)$. Removing five wall cells and two goal cells leaves 42 ordinary Markov states. The action set is up, right, down, left, and stay.

中：环境是一个 $7\times7$ grid。$x=3$ 位置有一堵竖墙，只在 $y=1$ 和 $y=5$ 留出上下两个通道。Goal A 位于 $(6,1)$，Goal B 位于 $(6,5)$。去除五个墙体格和两个目标格后，共有 42 个普通 Markov states。动作包括上、右、下、左和原地不动。

EN: Reaching a goal produces its context-dependent reward and immediately samples the next state uniformly from the seven cells on the left boundary. The environment still returns `terminated=False` and `truncated=False`. This restart is part of the transition kernel, not an episode reset. Parameters, traces, learned model, value weights, and average-reward estimate all continue across goal events and context changes.

中：到达目标后，agent 获得该 context 下的目标奖励，并立刻从左边界的七个格子中均匀采样下一个 state。环境始终返回 `terminated=False` 和 `truncated=False`。这个 restart 是 transition kernel 的一部分，而不是 episode reset。参数、trace、world model、value weights 和 average-reward estimate 都会跨越目标事件和 context change 保留。

### Transition and reward semantics / Transition 与 reward 语义

EN: The chosen action is applied first. If the intended move is illegal or a stochastic corridor edge blocks it, the agent remains in the previous state and receives $-0.25$. Otherwise, wind may add one extra displacement. A legal non-goal transition receives $-0.05$. Reaching A or B receives the corresponding goal reward, between $+1$ and $+6$ in the tested contexts.

中：首先执行 agent 选择的动作。如果目标格非法，或者 stochastic corridor edge 阻挡了移动，agent 留在原 state 并获得 $-0.25$。否则，wind 可能再施加一次额外位移。合法的非目标 transition 获得 $-0.05$；到达 A 或 B 时获得对应 goal reward，在实验 contexts 中取 $+1$ 到 $+6$。

### Observation / 观测

The agent receives

$$
o_t=(x_t,y_t,g_x^*,g_y^*,m_t),
$$

where $(g_x^*,g_y^*)=(6,3)$ is a fixed central reference between the goals and $m_t$ is a four-bit local wall mask indicating which cardinal moves are immediately illegal.

中：Agent 的 observation 如上式，其中 $(g_x^*,g_y^*)=(6,3)$ 是两个目标之间的固定中央参考点，$m_t$ 是四位 local wall mask，表示四个方向中哪些立即不可行。

EN: The observation does not expose the current wind probability, wind context, goal-reward preference, or dynamic edge probability. Goal identity is available when a goal transition occurs; because the goal cells have fixed coordinates, this is structured event information rather than access to the hidden context.

中：Observation 不直接包含当前 wind probability、wind context、goal reward preference 或 dynamic edge probability。只有发生 goal transition 时才会得到 goal identity；由于目标坐标固定，这属于结构化事件信息，而不是直接读取 hidden context。

### Multi-group tile coding / 多组 Tile Coding

EN: All control methods use the same linear function approximation. For each action, the representation contains:

- 8 tilings of absolute position $(x,y)$ at resolution 4;
- 8 tilings of relative position $(g_x^*-x,g_y^*-y)$ at resolution 4;
- 4 joint tilings of absolute and relative position at resolution 3;
- one local-geometry feature indexed by wall mask and action;
- one action-specific bias feature.

The nominal active count is therefore

$$
8+8+4+1+1=22
$$

features per state–action pair, stored in an IHT of size 8192.

中：所有控制算法使用完全相同的线性函数逼近。对每个 action，representation 包含：8 个绝对位置 tilings、8 个相对位置 tilings、4 个绝对与相对位置联合 tilings、1 个由 wall mask 和 action 索引的 local-geometry feature，以及 1 个 action bias。因此每个 state-action pair 名义上激活 22 个稀疏 features，存储在大小为 8192 的 IHT 中。

EN: Absolute features support place-specific values, relative features share navigation structure, joint features retain context-sensitive interactions, and the wall mask prevents excessive aliasing across locally different geometries. This representation passed the stationary competence tests before any non-stationary claim was evaluated.

中：Absolute features 表示位置特定价值，relative features 共享导航规律，joint features 保留位置与相对目标关系的交互，而 wall mask 减少局部几何不同却被错误 alias 的情况。在检验任何 non-stationary 主张之前，这套 representation 已先通过 stationary competence tests。

### Average-reward control / Average-reward 控制

For differential Q-learning, the real-transition TD error is

$$
\delta_t=r_{t+1}-\bar R_t+\max_a Q(s_{t+1},a)-Q(s_t,a_t),
$$

and the reward-rate estimate is updated by

$$
\bar R_{t+1}=\bar R_t+\beta\delta_t.
$$

EN: The behavior policy is $ε$-greedy with $ε=0.05$. For the main Q-learning and Dyna agents, the effective real-update step size is 0.2 divided across the 22 active features; planning uses one quarter of that step size. The reward-rate step size is 0.005.

中：行为策略是 $ε$-greedy，$ε=0.05$。主实验中的 Q-learning 与 Dyna 使用 effective real-update step size 0.2，再除以 22 个 active features；planning step size 是 real step size 的四分之一。Reward-rate step size 为 0.005。

### Why not use the old environment? / 为什么没有继续使用老环境？

EN: The old $5\times5$ environment independently cycled wind direction, global reward multipliers, target coordinates, and complete obstacle maps. Those switches were useful stress tests but did not always change the optimal policy, and multiple mechanisms could be confounded. The new environment creates decision-critical, oracle-validated interventions: local reward propagation, one critical edge, continuous wind probability, and recurring factor combinations.

中：老的 $5\times5$ 环境独立循环 wind direction、全局 reward multipliers、目标坐标和整套 obstacle maps。这些开关适合作为 stress test，但不一定改变 optimal policy，而且多个机制容易混杂。新环境改用 decision-critical、经过 oracle 验证的干预：局部 reward propagation、单条关键 edge、连续 wind probability，以及 recurring factor combinations。

---

## 3. Experimental Design and Results

## 3. 实验设计与实验结果

### Four diagnostic continual changes

### 四种诊断性持续变化

#### A. Local reward revaluation / 局部奖励重估

EN: From step 0 to 10,000, Goal A pays 6 and Goal B pays 4. At step 10,000, A drops to 1 while B remains 4. From 10,000 to 10,500, the agent is placed near A on every interaction, so it can observe the new reward but cannot re-explore the full decision region. At 10,500 it returns to the shared state $(0,3)$.

中：0–10,000 step 中 A=6、B=4。10,000 step 时 A 突降为 1，B 保持 4。10,000–10,500 step 中，agent 每一步都被放置在 A 附近，因此能够观察新 reward，却不能重新探索整个决策区域。10,500 step 时回到共享状态 $(0,3)$。

EN: This asks whether local model evidence can alter a distant choice before the agent physically revisits all intervening states. The 500-step local window is a state-distribution intervention, not an episode, so dynamic-regret diagnostics are omitted inside that window.

中：该实验检验局部模型证据能否在 agent 尚未重新访问所有中间 states 时修改远端选择。500-step local window 是 state-distribution intervention，不是 episode，因此该窗口内不计算公共 choice 的 dynamic regret。

#### B. Abrupt corridor blockage / 突发通道阻塞

EN: At step 10,000, the upper edge between $(2,1)$ and $(3,1)$ changes from block probability 0 to 1. Rewards, wind, and the rest of the topology stay fixed. The old oracle policy loses 0.6924 average reward in the new context, confirming that the change is decision-critical.

中：10,000 step 时，上通道 $(2,1)$ 与 $(3,1)$ 之间的 edge block probability 从 0 变为 1。奖励、风和其他拓扑完全不变。旧 oracle policy 在新 context 中损失 0.6924 average reward，说明这确实是 decision-critical change。

#### C. Smooth wind drift / 平滑风力漂移

EN: Wind direction is down. Its probability follows a triangular schedule with period 4,000:

$$
p_t=0.8\left(1-\left|2\frac{t\bmod 4000}{4000}-1\right|\right).
$$

It repeatedly moves $0\rightarrow0.8\rightarrow0$ for five cycles. This tests continuous parameter tracking rather than one-shot recovery.

中：Wind direction 固定向下，其概率按照上式做周期为 4,000 的三角形变化，在 20,000 steps 内重复五次 $0\rightarrow0.8\rightarrow0$。它检验持续参数跟踪，而不是单次 change 后的恢复。

#### D. Recurring + novel composition / 复现与未见组合

| Steps | Goal rewards | Wind | Role |
|---|---|---|---|
| 0–4k | A=6, B=3 | right, $p=0.45$ | first context |
| 4k–8k | A=3, B=6 | left, $p=0.45$ | second context |
| 8k–12k | A=6, B=3 | right, $p=0.45$ | recurrence |
| 12k–16k | A=6, B=3 | left, $p=0.45$ | unseen whole combination |
| 16k–20k | A=3, B=6 | left, $p=0.45$ | recurrence |

EN: The held-out A-high/left-wind context combines a reward factor and a wind factor that were each seen previously, but not together. The current version does not change obstacles in this scenario, so it is a controlled two-factor composition test rather than a full context-generalization benchmark.

中：A-high/left-wind 这个 held-out context 由两个分别见过、但从未共同出现的 reward 与 wind factors 组成。当前版本没有在该场景中改变 obstacles，因此它是受控的双因子组合测试，而不是完整的 context-generalization benchmark。

### Protocol

### 实验协议

EN:

- 4 scenarios × 10 methods × 20 paired seeds = 800 final runs;
- 20,000 real steps per run = 16 million real environment steps;
- every Dyna method has a budget of up to 5 planning backups per real step;
- exact policy diagnostics every 100 real steps, producing 200 points per run;
- one cross-scenario pilot selects EMA decay 0.97 and factor learning rate 0.05;
- no final-scenario retuning;
- all context switches preserve the learner state.

中：正式实验为 4 个场景 × 10 种方法 × 20 个 paired seeds，共 800 runs；每个 run 20,000 real steps，总计 1600 万真实环境交互。每种 Dyna 方法每个 real step 的 planning budget 最多为 5 次 backups。每 100 steps 做一次只读 exact-policy diagnostic，因此每个 run 有 200 个点。Pilot 统一选择 EMA decay 0.97 和 factor learning rate 0.05，final scenarios 不再单独调参；所有 context switches 都保留 learner state。

#### All evaluated methods / 所有方法

1. Q-learning;
2. SARSA($\lambda$);
3. Latest-outcome Dyna;
4. Empirical Dyna without forgetting;
5. EMA Dyna;
6. Prioritized EMA;
7. Factored Dyna with uniform planning;
8. CAFD-Lite;
9. CAFD-Surprise;
10. perfect-current-model Oracle Dyna.

### Metrics

### 指标

#### Exact dynamic regret / 精确 Dynamic Regret

At diagnostic time $t$, freeze the current context and the learner's current $ε$-greedy policy. Exact finite-state Markov-chain evaluation gives policy gain $g_t^{\pi_t}$, while average-reward value iteration gives oracle gain $g_t^*$. Then

$$
\mathcal{R}_t=g_t^*-g_t^{\pi_t}.
$$

The aggregate uses

$$
\widetilde{\mathcal{R}}_t=
\frac{g_t^*-g_t^{\pi_t}}{\max\left(0.1,|g_t^*|\right)}.
$$

EN: This diagnostic is read-only: it does not run evaluation episodes, consume extra environment interactions, or update the agent. It separates policy quality from changes in the absolute reward scale or intrinsic difficulty of the current context.

中：该 diagnostic 是只读计算：不运行 evaluation episodes、不消耗额外 environment interactions，也不更新 agent。它能够把“当前 policy 的质量变化”与“当前 context 的 reward scale 或任务难度变化”区分开。

#### World-model error / World-model Error

For every one of the 210 state–action pairs, aggregate the true and predicted joint outcomes into next-state marginals and expected rewards. Define

$$
E_T(s,a)=\frac12\sum_{s'}\left|P(s'\mid s,a)-\hat P(s'\mid s,a)\right|,
$$

$$
E_R(s,a)=\min\left(1,\frac{|\hat{\mathbb E}[r\mid s,a]-\mathbb E[r\mid s,a]|}{6}\right),
$$

and

$$
E_M=\frac1{210}\sum_{s,a}\left(0.8E_T(s,a)+0.2E_R(s,a)\right).
$$

EN: An unseen key in an unstructured model receives error 1. Model-free methods have no model error and are therefore omitted rather than assigned zero. This is a global prediction metric; it is not identical to control-relevant error and does not compare the full reward/next-state joint distribution.

中：非结构化模型中未见过的 key 记为 error 1。Model-free 方法没有 world model，因此该值是缺失而不是 0。这个指标衡量全局预测误差，不完全等价于 control-relevant error，也没有直接比较完整 reward/next-state joint distribution。

#### Online reward / 在线奖励

Cumulative stream average reward is

$$
\bar r_t^{\mathrm{stream}}=\frac1t\sum_{i=1}^t r_i.
$$

Trailing-500 reward is

$$
\bar r_t^{(500)}=\frac1{500}\sum_{i=t-499}^t r_i.
$$

EN: Stream average measures total deployment utility but reacts slowly after a change. Trailing-500 reward shows immediate loss and recovery, but its absolute oscillations also reflect context difficulty. Dynamic regret is therefore the primary adaptation metric, with online reward as an intuitive complementary measure.

中：Stream average 衡量整个部署过程的总效用，但在 change 后响应很慢。Trailing-500 reward 能显示即时损失和恢复，但其绝对振荡也会反映 context 自身难度。因此 dynamic regret 是主要适应指标，online reward 是更直观的补充指标。

### Performance across all four changes

### 四类变化下的性能

![Selected dynamic-regret comparison](poster_assets/dynamic_regret_selected.png)

#### Why these five curves? / 为什么只画这五条？

| Poster method | Scientific role |
|---|---|
| Q-learning | model-free reference |
| EMA Dyna | strongest compact unstructured adaptive-model reference |
| Factored Dyna | isolates factorization |
| CAFD-Lite | factorization plus mixed prioritized planning |
| Oracle Dyna | perfect-current-model reference, not a learned competitor |

EN: These methods form an interpretable mechanism ladder. Adding more curves would repeat conclusions already documented in the full table while hiding the important separation between unstructured and factored models.

中：这五种方法形成可解释的机制阶梯。继续添加曲线只会重复完整结果表中已经记录的结论，并掩盖“非结构化模型与因子化模型之间的关键分界”。

#### What each panel demonstrates / 每个 panel 说明什么

- **Local reward revaluation.** EN: Factorization rapidly learns the two reward heads; CAFD propagates the revision to upstream choices. 中：Factorization 快速学习独立 reward heads，CAFD 将修正传播到上游 choice states。
- **Abrupt blockage.** EN: Updating one edge factor reconstructs affected transitions without relearning unrelated state–action outcomes. 中：更新一条 edge factor 就能重构相关 transitions，而不必重新学习无关 state-action outcomes。
- **Smooth wind drift.** EN: The largest and clearest separation occurs here: a global wind factor tracks drift, while unstructured models remain stale across much of the state space. 中：这里的差距最大、最清楚：global wind factor 能跟踪 drift，而非结构化模型在大量状态上长期陈旧。
- **Recurring/composition.** EN: Factored methods reacquire recurring contexts faster and handle the held-out A-high/left-wind combination better. The experiment shows factor recombination, but not a learned context bank. 中：Factored methods 更快恢复 recurring contexts，也更好处理 held-out A-high/left-wind 组合；它证明 factor recombination，但没有证明已经学习了 context bank。

#### Full normalized-regret results / 完整 normalized-regret 结果

Lower is better; values are mean ± 95% confidence interval.

| Method | LoCA | Obstacle | Wind | Recurring |
|---|---:|---:|---:|---:|
| Q-learning | 0.2990 ± 0.0525 | 0.2987 ± 0.0893 | 0.6735 ± 0.1681 | 0.4691 ± 0.0248 |
| SARSA($\lambda$) | 0.5248 ± 0.1082 | 0.6461 ± 0.1770 | 0.7597 ± 0.1542 | 0.4826 ± 0.0228 |
| Latest Dyna | 0.2986 ± 0.1290 | 0.3136 ± 0.1497 | 0.6460 ± 0.1754 | 0.4335 ± 0.0303 |
| Empirical Dyna | 0.2440 ± 0.0775 | 0.3861 ± 0.0756 | 0.6439 ± 0.1895 | 0.3906 ± 0.0391 |
| EMA Dyna | 0.2179 ± 0.0800 | 0.2461 ± 0.0894 | 0.6661 ± 0.1864 | 0.3684 ± 0.0347 |
| Prioritized EMA | 0.3049 ± 0.1485 | 0.3493 ± 0.1704 | 0.6123 ± 0.1846 | 0.3883 ± 0.0438 |
| Factored Dyna | 0.1044 ± 0.0030 | 0.1216 ± 0.0056 | 0.2635 ± 0.0143 | 0.2684 ± 0.0122 |
| **CAFD-Lite** | **0.0933 ± 0.0031** | **0.1118 ± 0.0037** | 0.2332 ± 0.0091 | **0.2618 ± 0.0082** |
| CAFD-Surprise | 0.0934 ± 0.0036 | 0.1390 ± 0.0055 | **0.2287 ± 0.0070** | 0.2766 ± 0.0109 |
| Oracle Dyna | 0.1008 ± 0.0043 | 0.1050 ± 0.0063 | 0.2370 ± 0.0112 | 0.2503 ± 0.0077 |

### The gain is real online reward, not only a diagnostic

### 优势确实转化为在线奖励

![Stream average reward table](poster_assets/stream_average_reward_table.png)

EN: CAFD-Lite minus Q-learning stream-average-reward differences are +0.0867, +0.0863, +0.1722, and +0.0717 for LoCA, obstacle, wind, and recurring scenarios. The paired-bootstrap 95% confidence interval excludes zero in every case. Relative to Q-learning's mean reward, these are approximately 19.9%, 19.4%, 93.8%, and 19.0% improvements.

中：CAFD-Lite 相对 Q-learning 的 stream-average-reward 差值在 LoCA、obstacle、wind、recurring 中分别为 +0.0867、+0.0863、+0.1722 和 +0.0717。四个 paired-bootstrap 95% CI 都排除 0。相对于 Q-learning 的均值，提升约为 19.9%、19.4%、93.8% 和 19.0%。Poster 使用三线表直接展示绝对均值和 95% CI，每列真实最高的 learned-method mean 加粗。

EN: Unstructured EMA Dyna improves reward only by +0.0218, +0.0209, −0.0174, and +0.0224 relative to Q-learning, with all four confidence intervals including zero. The result is therefore not “all MBRL beats model-free RL”; it is “structured MBRL produces a robust advantage in these diagnostics.”

中：非结构化 EMA Dyna 相对 Q-learning 的 reward 差值只有 +0.0218、+0.0209、−0.0174 和 +0.0224，四个 CI 都包含 0。因此，结论不是“所有 MBRL 都胜过 model-free RL”，而是“结构化 MBRL 在这些 diagnostics 中产生稳定优势”。

### Mechanism evidence

### 机制证据

![Selected world-model tracking curves](poster_assets/model_tracking_selected.png)

EN: These time-series panels show world-model error throughout learning. EMA Dyna remains around 0.16–0.39 late in training and repeatedly becomes stale under changing wind, whereas the factored model is near zero in abrupt scenarios and around 0.04 under wind and recurring changes. Factored Dyna and CAFD-Lite use the same learned model—their difference is planning—so their model-error curves are mathematically identical and are shown as one line.

中：这些 time-series panels 展示整个学习过程中的 world-model error。EMA Dyna 在训练后期仍约为 0.16–0.39，并在变化 wind 下反复陈旧；factored model 在 abrupt scenarios 中接近 0，在 wind 与 recurring 中约为 0.04。Factored Dyna 与 CAFD-Lite 使用完全相同的 learned model，区别只在 planning，所以两者的 model-error curves 数学上相同，图中合并为一条线。

EN: This figure is direct evidence that structure improves predictive adaptation. The regret curves alone show better control; the model-error dynamics show that factored predictions remain coherent across states and context changes. If space permits, `world_model_evidence.png` is a useful supplementary inset because its right panel compares the learned wind factor directly with ground truth.

中：这张图直接证明结构化表示改善了预测适应。Regret curves 只说明控制更好；model-error dynamics 说明因子化预测能够跨 states 和 context changes 保持一致。如果空间允许，`world_model_evidence.png` 可作为补充小图，因为其右 panel 直接比较 learned wind factor 与 ground truth。

#### Statistical interpretation / 统计解释

EN: Main curves show the mean and 95% confidence interval over 20 seeds. Method differences use paired seed-level comparisons and 10,000 bootstrap resamples. Pairing matters because methods face the same environment seed. We do not infer significance from overlap of marginal error bands.

中：主曲线显示 20 seeds 的均值和 95% CI。方法差异使用 seed-level paired comparisons，并进行 10,000 次 bootstrap。由于不同方法面对相同 environment seed，paired design 非常重要。我们不通过两条 marginal error bands 是否重叠来判断显著性。

---

## 4. Proposed Algorithm: CAFD-Lite

## 4. 新算法：CAFD-Lite

### Change-Aware Factored Dyna

### 变化感知因子化 Dyna

#### Model representation / 模型表示

CAFD-Lite represents the world model as

$$
M_t=\left\{M_{\mathrm{stable}},p_t(w),p_t(\mathrm{block}\mid e),r_t^A,r_t^B\right\}.
$$

EN:

- $M_{\mathrm{stable}}$ contains known coordinate-action mechanics, boundaries, and fixed walls.
- $p_t(w)$ is a global categorical distribution over none, up, right, down, and left wind.
- $p_t(\mathrm{block}\mid e)$ contains local upper- and lower-corridor edge probabilities.
- $r_t^A$ and $r_t^B$ are separate goal-reward heads.

中：$M_{\mathrm{stable}}$ 包含已知坐标动作规律、边界和固定墙体；$p_t(w)$ 是 none/up/right/down/left 的全局 categorical wind distribution；$p_t(\mathrm{block}\mid e)$ 表示上下 corridor edges 的局部阻塞概率；$r_t^A$ 和 $r_t^B$ 是两个独立目标奖励 heads。

EN: This structural prior must be stated clearly. CAFD does not discover grid topology from pixels. It knows stable mechanics and learns only the dynamic factors online. The scientific claim is about factor reuse under a suitable inductive bias.

中：必须明确披露这个 structural prior。CAFD 并没有从 pixels 中发现 grid topology；它已知稳定运动机制，只在线学习动态因素。因此我们的科学主张是“在合适 inductive bias 下的 factor reuse”，而不是端到端自动因果发现。

#### Factor updates / 因子更新

For a scalar factor $θ$ with observation $y_t$, fixed-rate CAFD uses

$$
\theta_{t+1}=\theta_t+\eta(y_t-\theta_t),\qquad \eta=0.05.
$$

EN: A goal transition updates only its corresponding reward head. An attempted critical-edge crossing updates only that edge's block probability. On a legal non-goal transition, the residual between intended and realized displacement updates the categorical wind distribution. Every update leaves unrelated factors unchanged.

中：Goal transition 只更新对应 reward head；尝试穿越关键 edge 时只更新该 edge 的 block probability；合法非目标 transition 中，intended displacement 与 realized displacement 的 residual 用于更新 categorical wind distribution。每次更新都不会改动无关 factors。

#### Reconstructing transitions / 重构 Transition

EN: For any state–action pair, the model combines stable motion, edge probability, wind distribution, goal reward, and the restart distribution to reconstruct a complete outcome distribution. A single wind observation therefore changes predicted transitions at every state where wind can act.

中：对于任意 state-action pair，模型组合 stable motion、edge probability、wind distribution、goal reward 和 restart distribution，重构完整 outcome distribution。因此一次 wind observation 会修改所有可能受 wind 影响的 states 的预测。

#### Change-aware planning / 变化感知 Planning

The priority of an affected key is based on its expected differential Bellman error:

$$
p(s,a)=\left|
\sum_{r,s'}\hat P(r,s'\mid s,a)
\left(r-\bar R+\max_{a'}Q(s',a')\right)-Q(s,a)
\right|.
$$

EN: A changed global factor revises predictions for all affected keys. To cap queue overhead, up to four affected keys seed the priority queue on each real update; a predecessor index then propagates high-priority value changes backward. CAFD-Lite uses a fixed budget of five planning backups after every real transition:

$$
1\text{ prioritized expected backup}+4\text{ uniform model samples}.
$$

中：一个改变的 global factor 会修改所有受影响 keys 的预测。为限制 queue 开销，每次 real update 最多抽取四个受影响 keys 初始化 priority queue；predecessor index 随后把高优先级 value changes 向前驱状态传播。CAFD-Lite 在每个 real transition 后使用五次 planning backups：1 次 prioritized expected backup 加 4 次 uniform model samples。

EN: We chose the mixed strategy because using all five backups for priority caused function-approximation interference in development tests. The comparison with Factored Dyna is fair in planning-update count: both use five backups; only their allocation differs.

中：选择 mixed strategy 是因为开发实验中五次全部使用 priority 会加剧 function-approximation interference。与 Factored Dyna 的比较在 planning-update count 上公平：两者都是五次，只是分配方式不同。

#### CAFD-Surprise extension / CAFD-Surprise 扩展

Prediction surprise is

$$
u_t=-\log \hat p(r_t,s_{t+1}\mid s_t,a_t),
$$

and the adaptive learning rate is

$$
\eta_t=0.01+(0.30-0.01)\sigma\left(1.5(u_t-2.5)\right).
$$

EN: CAFD-Surprise obtains the lowest numerical regret under wind drift, but its paired difference from fixed-rate CAFD-Lite is not significant. It is significantly worse under abrupt obstacle and recurring/compositional changes. We therefore present it as an exploratory negative result, not the main algorithm.

中：CAFD-Surprise 在 wind drift 中得到数值上最低的 regret，但相对 fixed-rate CAFD-Lite 的 paired difference 不显著；它在 abrupt obstacle 与 recurring/composition 中显著更差。因此我们把它作为探索性的负结果，而不是主算法。

#### Mechanism ablations / 机制消融

Treatment minus control normalized-regret differences:

| Comparison | LoCA | Obstacle | Wind | Recurring |
|---|---:|---:|---:|---:|
| CAFD-Lite − Factored Dyna | −0.0111 | −0.0097 | −0.0304 | −0.0066 |
| Bootstrap CI excludes zero? | yes | yes | yes | no |
| Surprise − CAFD-Lite | +0.0002 | +0.0271 | −0.0044 | +0.0147 |
| Bootstrap CI excludes zero? | no | yes, worse | no | yes, worse |

EN: Factorization creates the large gap from EMA Dyna; mixed priority creates a smaller additional gain in three scenarios. Priority applied to the unstructured EMA model is not significantly better in any scenario. This supports the interaction hypothesis: planning allocation is useful when a model revision generalizes coherently.

中：Factorization 产生了相对 EMA Dyna 的主要差距；mixed priority 在三个场景中提供较小的追加收益。把 priority 用在非结构化 EMA model 上，在任何场景都没有显著改善。这支持“机制交互”假设：只有当模型修正能够一致泛化时，planning allocation 才稳定有用。

#### Likely question: “Why can CAFD beat Oracle Dyna?” / 可能追问：“为什么 CAFD 有时比 Oracle Dyna 还好？”

EN: Oracle Dyna has a perfect current transition model, but it still uses the same finite planning budget, approximate linear value function, stochastic planning distribution, and differential bootstrapping update. A perfect model is not a perfect controller. Small reversals between CAFD and Oracle do not imply that the learned model is more accurate than truth.

中：Oracle Dyna 拥有完美的当前 transition model，但它仍使用相同的有限 planning budget、近似线性 value function、随机 planning distribution 和 differential bootstrapping update。Perfect model 不等于 perfect controller。CAFD 与 Oracle 的小幅反转不表示 learned model 比真实模型更准确。

---

## 5. Conclusions

## 5. 结论

### Evidence-aligned conclusions / 与证据逐条对应的结论

#### Conclusion 1: Factorization is the main gain / 结论一：Factorization 是主要增益来源

EN: CAFD-Lite reduces mean normalized dynamic regret relative to Q-learning by 68.8%, 62.6%, 65.4%, and 44.2% in LoCA, obstacle, wind, and recurring scenarios. Factored Dyna without priority already achieves most of this improvement. This conclusion is supported by all four panels of `dynamic_regret_selected.png` and the error trajectories in `model_tracking_selected.png`.

中：CAFD-Lite 相对 Q-learning 在 LoCA、obstacle、wind、recurring 场景中分别降低 68.8%、62.6%、65.4% 和 44.2% mean normalized dynamic regret。即使没有 priority，Factored Dyna 已经获得大部分改善。证据来自 `dynamic_regret_selected.png` 的四个 panels 和 `model_tracking_selected.png` 的 error trajectories。

#### Conclusion 2: The model tracks interpretable mechanisms / 结论二：模型确实跟踪了可解释机制

EN: Factored models reduce tail world-model error from approximately 0.16–0.39 for EMA Dyna to near zero in abrupt scenarios and around 0.04 under wind and recurring changes. The learned wind factor follows all five drift cycles. This supports the operational “understanding” claim: parameter tracking plus systematic prediction generalization.

中：Factored models 将 EMA Dyna 约 0.16–0.39 的 tail model error 降至 abrupt scenarios 中接近 0、wind 与 recurring 中约 0.04。学到的 wind factor 能跟踪全部五个 drift cycles。这支持我们对“理解”的操作性定义：参数跟踪与系统性预测泛化。

#### Conclusion 3: Lower regret produces higher deployment reward / 结论三：更低 Regret 转化为更高部署收益

EN: CAFD-Lite improves stream average reward over Q-learning in every scenario, with paired differences +0.0867, +0.0863, +0.1722, and +0.0717. Every paired-bootstrap confidence interval excludes zero. The three-line table reports absolute reward means and confidence intervals; paired statistics establish significance.

中：CAFD-Lite 在所有场景中都提高了相对 Q-learning 的 stream average reward，paired differences 分别是 +0.0867、+0.0863、+0.1722 和 +0.0717，所有 paired-bootstrap CI 都排除 0。三线表报告绝对 reward 均值与置信区间，paired statistics 负责建立显著性。

#### Conclusion 4: More sophistication is not automatically better / 结论四：更复杂不一定更好

EN: Recency alone is inconsistent, priority alone is not robust on an unstructured model, and surprise adaptation has two significant failure cases. The reliable contribution is the structural decomposition, followed by modest mixed-priority improvement.

中：单独使用 recency 并不稳定；在非结构化模型上单独加入 priority 也不稳健；surprise adaptation 有两个显著失败场景。可靠贡献是结构分解，其次才是 mixed priority 的适度增益。

### Baselines to show on the poster

### Poster 应展示哪些 Baselines

**Recommended main-figure set / 推荐主图集合**

1. **Q-learning:** necessary model-free anchor;
2. **EMA Dyna:** compact representative of unstructured adaptive MBRL;
3. **Factored Dyna:** factorization-only ablation;
4. **CAFD-Lite:** proposed method;
5. **Oracle Dyna:** perfect-current-model reference.

EN: Remove SARSA($\lambda$), Latest Dyna, Empirical Dyna, Prioritized EMA, and CAFD-Surprise from the poster curves. This is scientifically defensible because the full experiment still reports them, and each retained method answers a unique causal comparison. If space is extremely limited, remove Oracle from the time-series legend and show it as a thin dashed reference or mention it only in the caption; do not remove EMA or Factored Dyna, because those two establish where the gain comes from.

中：建议从 poster 曲线中删除 SARSA($\lambda$)、Latest Dyna、Empirical Dyna、Prioritized EMA 和 CAFD-Surprise。这在科学上是合理的，因为完整实验仍报告它们，而保留的每种方法都对应一个唯一的因果比较。如果空间极端有限，可以把 Oracle 从普通 legend 中移出，仅作为细虚线或 caption 参考；不要删 EMA 或 Factored Dyna，因为这两条决定了“增益究竟来自哪里”。

### Recommended visual hierarchy / 推荐视觉层级

EN:

1. **Largest result figure:** selected five-method dynamic-regret panels;
2. **Mechanism figure:** unstructured-versus-factored model-error dynamics;
3. **Compact evidence table:** stream average reward;
4. **Environment figure:** next to the environment text;
5. Keep formulas to dynamic regret and the model factorization only if the physical poster is crowded; move the full model-error equation to a QR-linked appendix.

中：最大结果图应是五方法 dynamic-regret panels；第二重要的是 unstructured-versus-factored model-error dynamics；stream-average-reward 使用紧凑三线表；environment figure 与环境文字并排。如果实体 poster 非常拥挤，只保留 dynamic-regret 与 model-factorization 两个公式，把完整 model-error 公式放到二维码附录。

### Scope and limitations / 适用范围与局限

EN:

- Results are matched by environment steps and planning updates, not by wall-clock time. CAFD takes roughly 8–10 times the wall-clock time of Q-learning in this implementation.
- Stable action mechanics and fixed walls are structural prior knowledge.
- The formal wind test uses one triangular period of 4,000 steps; unseen timescales and random-walk drift remain future tests.
- Recurring composition uses one stream and two factors; no explicit context bank is implemented.
- The setting uses handcrafted state variables and linear tile coding, not pixels or a neural world model.
- The claim is therefore about the value of factorized inductive bias in a controlled continual diagnostic, not universal superiority of MBRL.

中：结果在 environment steps 和 planning updates 上匹配，但没有按 wall-clock 匹配；当前实现中 CAFD 耗时约为 Q-learning 的 8–10 倍。Stable action mechanics 与 fixed walls 是结构先验。正式 wind test 只使用周期 4,000 的 triangular schedule；未见 timescale 与 random-walk drift 仍待测试。Recurring composition 只涉及一个 stream 和两个 factors，没有显式 context bank。任务使用手工 state variables 与线性 tile coding，而不是 pixels 或 neural world model。因此主张是“factorized inductive bias 在受控 continual diagnostic 中有价值”，不是 MBRL 的普适优越性。

### Final answer if asked “What did you learn?” / 被问“最终学到了什么？”时的回答

EN: We learned that having a model is not enough. A model helps continual RL when its representation separates stable structure from changing mechanisms, so local evidence produces coherent global prediction changes. In our experiments, this factorization—not generic replay or forgetting—accounts for most of the lower regret and higher online reward. Prioritized planning helps further when it operates on those coherent revisions.

中：我们的核心发现是：仅仅“有一个 model”并不够。只有当模型表示把稳定结构与变化机制分离，使局部证据能够产生一致的全局预测修正时，world model 才真正帮助 continual RL。在实验中，主要的 lower regret 与 higher online reward 来自这种 factorization，而不是泛化意义上的 replay 或 forgetting；当 prioritized planning 作用于这些一致模型修正时，它还能提供进一步收益。

---

## Poster assets and full evidence / Poster 素材与完整证据

- Environment diagram: [grid_world.png](poster_assets/grid_world.png)
- Editable vector environment diagram: [grid_world.svg](poster_assets/grid_world.svg)
- Selected regret curves: [dynamic_regret_selected.png](poster_assets/dynamic_regret_selected.png)
- Stream average-reward table: [stream_average_reward_table.png](poster_assets/stream_average_reward_table.png)
- Selected model-error dynamics: [model_tracking_selected.png](poster_assets/model_tracking_selected.png)
- Compact model-summary alternative: [world_model_evidence.png](poster_assets/world_model_evidence.png)
- Full rolling-reward curves: [rolling_average_reward_curves.png](phase6_9_summary/rolling_average_reward_curves.png)
- Full numerical summary: [aggregate_summary.csv](phase6_9_summary/aggregate_summary.csv)
- Paired mechanism ablations: [paired_mechanism_ablations.csv](phase6_9_summary/paired_mechanism_ablations.csv)
- Detailed research report: [PHASE6_9_CONTINUAL_MBRL_REPORT.md](PHASE6_9_CONTINUAL_MBRL_REPORT.md)
