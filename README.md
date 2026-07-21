# stream-rl-grid

一个用于持续强化学习实验的 non-episodic、infinite-horizon Windy Grid World。每个转移只使用一次，不使用 replay buffer、batch 或 episode return；智能体在结构化非平稳环境中进行 continuing average-reward control。

当前模型自由主流程提供：

- 6 种 Differential Sarsa 算法：`tidbd`、`sarsa`、`true_online_sarsa`、`adaptive_epsilon_sarsa`、`expected_sarsa`、`expected_sarsa_tidbd`；
- 2 种线性动作价值表示：精确离散表 `tabular-one-hot` 和稀疏因子化表示 `sparse-factorized`；
- 固定或 TD-error 自适应的 ε-greedy 策略；
- Tkinter GUI、精确续训 checkpoint、固定步数运行、时间戳曲线与性能指标导出；
- continuing-task 指标和按环境事件计算的 adaptation delay。

项目思路参考：

- *Streaming Deep Reinforcement Learning Finally Works* 中的逐样本即时更新与资格迹思想；
- *TIDBD: Adapting Temporal-difference Step-sizes Through Stochastic Meta-descent* 的逐特征步长更新；
- RLSS Lecture 02 的 on-policy control 与资格迹更新；
- RLSS Lecture 03 的 continuing average-reward / differential TD 更新。

## 快速开始

Python 需要带 Tk 支持。安装依赖后，在仓库根目录运行 GUI：

```powershell
python run_gui.py
```

精确运行 50,000 个真实环境步，并在结束时自动保存曲线和指标：

```powershell
python run_gui.py --steps 50000
```

无 GUI 运行固定步数：

```powershell
python -m stream_rl_grid.cli --profile combined --steps 50000
```

默认算法为 `tidbd`，默认价值表示为 `tabular-one-hot`。

## Continuing 环境

智能体观测为：

$$
S_t=(current_x,current_y,goal_x,goal_y,previous\_action)
$$

风阶段、奖励阶段、上下文地图编号和全局时钟均为隐藏变量。动作集合为 `up / right / down / left / stay`，其中 `stay` 仍受风影响。

环境遵循以下规则：

- 主动动作和风位移逐格执行；任一步碰到边界或障碍物，整个转移取消，智能体回到动作前位置并获得碰撞惩罚；
- `invalid_action` 只表示智能体直接尝试越界或进入障碍物；仅由风造成的碰撞计入 `collision`，但不计入 `invalid_action`；
- 到达目标后获得目标奖励，并立即随机传送到一个合法非目标格继续训练；GUI 中绿色起始方格会同步到这次新位置；
- 环境始终返回 `terminated=False, truncated=False`；到达目标或环境变化都不会清空资格迹；
- 地图切换不能立即把智能体当前格变成有效障碍物；若新地图在该格有障碍物，它会在智能体离开后激活；
- 地图生成器保证所有合法格连通，GUI 中破坏连通性的障碍物编辑会被拒绝。

`w_strength` 是发生一次额外单位风位移的概率，例如 `0.3` 表示每个转移有 30% 概率产生所选方向的一格风位移。

## 算法

六种算法共用同一个训练器、策略接口、价值表示工厂和 checkpoint 路径。采样式 Sarsa 算法的 differential TD error 为：

$$
\delta_t=R_{t+1}-\bar R_t+Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
$$

平均奖励估计更新为：

$$
\bar R_{t+1}=\bar R_t+\eta\delta_t
$$

这里是 continuing 任务，因此等价地使用 $\gamma=1$，不在目标到达处重置学习状态。

### `sarsa`

固定步长 Differential Sarsa(λ) baseline，采用 replacing traces：

$$
z_t\leftarrow\lambda z_{t-1},\qquad z_t[active]\leftarrow1
$$

$$
w\leftarrow w+\alpha\delta_tz_t
$$

适合用作固定步长基线。GUI 中的 `Initial effective step size` 决定初始有效步长，`Epsilon` 为固定探索率。

### `expected_sarsa`

固定步长 Differential Expected Sarsa(λ)，使用与 `sarsa` 相同的 replacing traces，但将采样得到的下一动作价值替换为当前 ε-greedy 策略下的期望：

$$
\delta_t=R_{t+1}-\bar R_t+
\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)-Q(S_t,A_t)
$$

行为动作 $A_{t+1}$ 仍由 ε-greedy 策略采样一次并用于下一真实环境步，但不参与当前 expected target。并列最优动作的贪婪概率沿用公共策略接口均分。到达目标后的随机传送观测直接作为 $S_{t+1}$，不重置 trace 或平均奖励。

### `tidbd`

Differential Sarsa(λ) + TIDBD。它为每个价值参数维护 $\beta_i=\log\alpha_i$ 和元迹 $H_i$，在线调整逐参数步长：

$$
\beta_i\leftarrow\operatorname{clip}(\beta_i+\theta\delta_tx_iH_i,\beta_{min},\beta_{max}),
\qquad \alpha_i=e^{\beta_i}
$$

其余 TD、replacing trace 和平均奖励更新与 `sarsa` 共用相同语义。实现不叠加 ObGD；`beta` 仅使用数值边界，并检查 NaN/Inf。GUI 中 `Theta` 控制元步长，`Beta min/max` 控制逐参数步长范围。

### `expected_sarsa_tidbd`

Differential Expected Sarsa(λ) + TIDBD。Expected target、行为策略和 continuing 语义与 `expected_sarsa` 相同，权重与元迹更新复用 `tidbd` 的逐参数步长机制。该版本用于测试 Expected target 与 TIDBD 的组合效果；`expected_sarsa` 则提供固定步长对照。

### `true_online_sarsa`

固定步长、differential continuing 形式的标准 True Online Sarsa(λ)，采用 Dutch trace：

$$
e_t\leftarrow\lambda e_{t-1}+(1-\alpha\lambda e_{t-1}^{\mathsf T}x_t)x_t
$$

$$
w\leftarrow w+\alpha(\delta_t+Q_t-Q_{old})e_t-\alpha(Q_t-Q_{old})x_t,
\qquad Q_{old}\leftarrow Q_{t+1}
$$

`q_old` 在连续数据流中不会因到达目标而重置，并随 checkpoint 保存和恢复。

### `adaptive_epsilon_sarsa`

固定步长 Differential Sarsa(λ)，但 ε-greedy 的探索率根据 TD error 异常程度在线变化：

$$
u_t=(1-\kappa)u_{t-1}+\kappa|\delta_t|
$$

$$
\epsilon_t=\operatorname{clip}\left(
\epsilon_{min}+c\max(0,u_t-u_{ref}),
\epsilon_{min},\epsilon_{max}
\right)
$$

默认值为 `κ=0.01`、`epsilon_min=0.02`、`epsilon_max=0.30`、`c=0.10`、`u_ref=1.0`。初始化时 $u_0=u_{ref}$、$\epsilon_0=\epsilon_{min}$。本步更新出的 ε 只影响下一个尚未选择的动作，不会重新采样已经用于 Sarsa 更新的 $A_{t+1}$。

环境稳定时 ε 接近下限；换季、目标规律变化或上下文切换使 TD error 增大时，探索率会暂时升高，并在重新适应后下降。选择该算法时，普通 `Epsilon` 设置不参与行为策略，实际 ε 由上述参数决定。

## ε-greedy 策略与 GUI 策略图

除 `adaptive_epsilon_sarsa` 外，其余算法使用固定 `Epsilon`。对动作价值并列最大的动作，贪婪概率会在这些动作间均分；每个动作同时获得 $\epsilon/5$ 的探索概率。

GUI 显示的是冻结的 `(height, width, 5)` ε-greedy 概率矩阵，而不是在线循环刚刚采样的单个动作。由于真实观测包含 `previous_action`，二维地图会按照各 `previous_action` 条件状态的历史访问次数加权混合策略；完全未访问的位置显示均匀策略。

## 动作价值函数表示

所有算法都使用线性动作价值函数：

$$
Q(s,a)=w^{\mathsf T}\phi(s,a)
$$

算法和表示可独立组合，六种算法都支持下面两种表示。

### `tabular-one-hot`（默认）

对完整的 `(x, y, goal_x, goal_y, previous_action, candidate_action)` 分配唯一、无碰撞的索引。每个状态—动作对只激活一个表项，不进行散列或跨状态泛化。

`previous_action` 有 6 个取值（五个动作加初始 `NO_ACTION`），参数量为：

$$
(width\times height)^2\times6\times5
$$

优点是表达精确、容易解释；缺点是参数量随地图面积的平方增长，未访问组合之间不能共享经验。

### `sparse-factorized`

每个状态—动作对激活 7 个互不重叠的二值特征组：

1. 候选动作偏置；
2. 智能体位置 × 候选动作；
3. 目标位置 × 候选动作；
4. 相对位移 `(goal_x-x, goal_y-y)` × 候选动作；
5. 相对方向 × 候选动作；
6. Manhattan 距离 × 候选动作；
7. 上一动作 × 候选动作。

该表示在相似状态间共享参数，参数量为：

$$
5+2(width\times height\times5)+(2width-1)(2height-1)\times5+9\times5+(width+height-1)\times5+6\times5
$$

例如 8×8 地图中，`sparse-factorized` 有 1,920 个参数，而 `tabular-one-hot` 有 122,880 个参数。因子化表示引入了泛化，也可能带来特征干扰，因此应通过多随机种子和非平稳事件恢复指标与表格表示对比。

为了让不同表示的初始总体更新尺度可比，代码将 `Initial effective step size` 除以名义激活特征数：表格表示除以 1，因子化表示除以 7。

## Structured non-stationarity

可选 profile：

- `stationary`：固定风、目标、奖励和地图；
- `seasonal_wind`：风方向和奖励倍率按 `wind_period` 循环；
- `moving_goal`：目标每 `target_move_interval` 步沿蛇形路径往返，自动跳过非法轨迹点；
- `hidden_context`：障碍物地图每 `context_switch_interval` 步切换，但上下文编号不提供给智能体；
- `combined`：同时启用 seasonal wind/reward、moving goal 和 hidden context；
- `customize`：用于手工配置环境；即使填写了上述周期，也不会自动触发 season、goal 或 context 调度。

同一步发生的多个变化会合并成一个 compound environment event。Adaptation delay 只跟踪 `season:*`、`goal_moved` 和 `context:*`，手动编辑事件不纳入该指标。

## GUI 使用

```powershell
python run_gui.py
python run_gui.py --steps 50000
```

`--steps 0`（默认）表示持续运行直到手动停止；正整数表示精确执行该数量的真实环境步，不会因 GUI 批量刷新而越过目标步数。加载 checkpoint 后，固定步数表示在已保存的 step 基础上再运行指定数量的步。

GUI 主要功能：

- 在 `Agent` 页选择 Algorithm、Value representation、固定 ε、λ、有效初始步长、TIDBD 和 adaptive ε 参数；
- 在 `Training` 页设置指标窗口、GUI 更新间隔、自动 checkpoint 周期和 adaptation recovery 参数；
- 生成地图、预览上下文地图、移动障碍物，并在暂停时应用环境修改；
- 开始、暂停/继续、请求 checkpoint、加载 checkpoint、停止但不保存；
- 实时显示网格、当前位置/随机重启起点、目标、策略概率、汇总指标；
- 实时绘制 `Average reward`（rolling reward 与 $\bar R$）和 `Adaptation diagnostics`（mean $|\delta|$、mean α、ε，以及自适应算法的平滑 $|\delta|$）；
- 点击 `Save curves with timestamp` 保存曲线和完整性能结果。

GUI 的 `ui_update_steps` 只控制可视化刷新频率；设置为 1 才能逐真实步观察。它不改变训练更新频率。

固定步数 GUI 运行完成后会先渲染最终 snapshot，再自动执行与保存按钮相同的时间戳导出。

## 性能指标

设指标窗口长度为 $W$（默认 1,000），保存时输出以下 continuing-task 指标：

- **Rolling Average Reward**：$\hat R_t^{(W)}=\frac{1}{W}\sum_{i=t-W+1}^{t}R_i$；数据不足 $W$ 时使用当前已有样本；
- **Goal reaching rate**：最近 $W$ 步的目标到达次数，并同时报告 `goals_per_1000_steps`；
- **Mean inter-goal time**：终点落在最近 $W$ 步内的相邻目标到达间隔的均值；少于两次目标到达时为 `NaN`，曲线会留空；
- **Collision rate**：最近 $W$ 步中全部碰撞比例；
- **Invalid-action rate**：最近 $W$ 步中主动越界或主动进入障碍物的比例，不含纯风碰撞；
- **Average-reward estimation error**：窗口填满后计算 $e_t=\bar R_t-\hat R_t^{(W)}$，曲线绘制 $|e_t|$，汇总结果还包括 mean bias、MAE 和 RMSE；
- **Adaptation delay**：环境事件后，性能恢复到事件前基线容差范围并持续满足条件所需的步数。

### Adaptation delay 定义

事件发生时，最近 $W$ 步的 rolling reward 作为基线 $B$。默认恢复阈值为：

$$
B-(1-\rho)\max(|B|,B_{floor})
$$

默认 `ρ=0.9`、`B_floor=1.0`。事件后的 `adaptation_recovery_window=100` 步滑动均值达到阈值，并连续满足 `adaptation_sustain_steps=100` 次检查后，记录真实 delay。

每个事件具有一种状态：

- `recovered`：已满足恢复条件，`delay` 为真实恢复步数；
- `censored`：尚未恢复就发生了下一个环境事件，当前事件无法继续独立归因；
- `pending`：运行结束或保存时仍在等待恢复，后续精确续训可继续跟踪；
- `unavailable`：事件发生前还没有积累满 $W$ 步，无法建立基线。

只有 `recovered` 事件参与 adaptation delay 的 mean/median。保存图中 unresolved 状态会画在 y=0；这不代表它们的恢复时间为 0。

若 `Mean inter-goal time` 或 `Adaptation delay by environment event` 为空，应先检查导出的 JSON/CSV：前者通常表示目标到达不足两次，后者通常表示没有匹配的自动环境事件。尤其 `customize` profile 不会自动产生这些调度事件。

## 时间戳结果导出

点击 `Save curves with timestamp`，或等待固定步数 GUI 运行自动完成，会在以下目录生成同一时间戳的一组文件：

```text
runs/<run-id>/figures/
├── learning_curves_<timestamp>.png
├── performance_metrics_<timestamp>.png
├── performance_metrics_<timestamp>.csv
├── adaptation_events_<timestamp>.csv
└── performance_summary_<timestamp>.json
```

- `learning_curves` 保存 GUI 中的 Average reward 和 Adaptation diagnostics；
- `performance_metrics.png` 保存 Rolling reward、窗口目标数、Mean inter-goal time、Invalid-action rate、Average-reward estimation error 和按事件的 Adaptation delay；
- `performance_metrics.csv` 保存所有采样曲线；
- `adaptation_events.csv` 保存事件类型、状态、基线、恢复阈值、结束步和 delay；
- `performance_summary.json` 保存当前汇总指标、配置关联信息和事件明细。

PNG 文件元数据和图内右下角都记录带时区的保存时间。上述额外性能曲线不在 GUI 中实时绘制，只在保存时生成，以免增加界面刷新负担。

## 无图形界面运行

固定步数运行并在结束时保存 exact-continuation checkpoint：

```powershell
python -m stream_rl_grid.cli --profile combined --steps 50000
```

持续运行，按 `Ctrl+C` 后保存 checkpoint：

```powershell
python -m stream_rl_grid.cli --profile combined --steps 0
```

选择算法和价值表示：

```powershell
python -m stream_rl_grid.cli `
  --profile combined `
  --algorithm expected_sarsa_tidbd `
  --representation sparse-factorized `
  --steps 50000
```

固定步长 Expected Sarsa 对照只需改为：

```powershell
python -m stream_rl_grid.cli `
  --profile combined `
  --algorithm expected_sarsa `
  --representation sparse-factorized `
  --steps 50000
```

使用 TD-error 自适应探索：

```powershell
python -m stream_rl_grid.cli `
  --profile combined `
  --algorithm adaptive_epsilon_sarsa `
  --epsilon-kappa 0.01 `
  --epsilon-min 0.02 `
  --epsilon-max 0.30 `
  --epsilon-scale 0.10 `
  --epsilon-u-ref 1.0 `
  --steps 50000
```

精确续训；`--steps` 表示额外运行的步数：

```powershell
python -m stream_rl_grid.cli `
  --resume checkpoints/<run-id>/step-000000050000.pkl `
  --steps 50000
```

`--fixed-alpha` 是兼容旧命令的快捷参数，会强制选择 `sarsa`。CLI 当前直接暴露 profile、地图尺寸、随机种子、算法、表示和 adaptive ε 参数；λ、固定步长、TIDBD 元步长及指标参数可在 GUI 中设置，或通过 `AppConfig` 编程配置。

## 多随机种子 benchmark

当前 benchmark 专门比较 `tidbd` 与固定步长 `sarsa`，不会自动包含其余四种算法：

```powershell
python -m stream_rl_grid.benchmark `
  --profiles stationary seasonal_wind moving_goal hidden_context combined `
  --representation sparse-factorized `
  --steps 50000 `
  --seeds 0 1 2 3 4
```

输出目录包含 `summary.csv` 和 `learning_curves.png`。曲线为各随机种子的窗口平均奖励均值，并绘制 95% 正态近似置信区间。比较算法时不应只看单次 reward 曲线，还应结合 goal rate、collision/invalid-action rate、TD error、adaptation delay、方差和计算成本。

## Checkpoint 与精确续训

模型自由 checkpoint 保存的不只是权重，还包括：

- 算法、价值表示元数据、权重、资格迹和平均奖励估计；
- `tidbd` 与 `expected_sarsa_tidbd` 的 `beta/H`，True Online Sarsa 的 `q_old`，adaptive ε 的 $u_t$ 与当前 ε；
- 当前观测和已经选好的下一动作；
- 环境位置、随机重启起点、目标、上下文地图、风/奖励/地图调度相位及延迟障碍物；
- 环境、agent、Python 和 NumPy 随机数状态；
- 滑动窗口、性能曲线、adaptation event/pending 状态和配置。

保存使用临时文件加原子替换，避免中断时留下半个 checkpoint。算法、网格尺寸、动作集合、观测字段或价值表示不兼容时会拒绝加载；不同价值表示不能交叉加载，旧 tile-coding checkpoint 也不兼容。

## 测试

项目测试覆盖 continuing 目标传送、碰撞回退、`stay` 风效应、地图切换延迟障碍物、六种算法、两种价值表示、checkpoint 精确续训、固定步数 GUI、时间戳导出和性能指标：

```powershell
E:\anaconda3\python.exe -m unittest discover -s tests -v
```
