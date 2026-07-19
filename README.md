# stream-rl-grid

一个面向持续学习实验的 Windy Grid World：每个真实转移即时更新、没有 batch、没有 episode 终止。智能体可选择双组 tile coding 或 D=55 显式线性特征，并统一采用 ε-greedy behavior policy。项目提供：

- Differential Q-learning；
- Watkins's Differential Q(λ)；
- Differential SARSA(λ)；
- Differential Dyna-Q；
- Differential Dyna-Q(λ)；
- Differential SARSA(λ) + TIDBD。

项目参考了：

- `Streaming Deep Reinforcement Learning Finally Works` 中的逐样本即时更新与资格迹思想；
- `TIDBD: Adapting Temporal-difference Step-sizes Through Stochastic Meta-descent` 的逐特征步长更新；
- RLSS Lecture 02 的线性函数逼近和 tile coding；
- RLSS Lecture 03 的 continuing average-reward / differential TD 更新。

## 算法

动作值函数是线性的：

$$Q(s, a) = w^T x(s, a)$$


Differential SARSA 的 TD error 不包含折扣因子：

$$delta = reward - R_bar + Q(next_state, next_action) - Q(state, action)$$

平均奖励估计为：

$$ R_bar <- R_bar + eta * delta $$

Differential Q-learning、Q(λ)、Dyna-Q 和 Dyna-Q(λ) 使用 off-policy greedy target：

$$delta = reward - R_bar + max_a Q(next_state, a) - Q(state, action)$$

资格迹采用 replacing traces：

$$
z <- lambda * z
z[active_features] <- 1
$$

Q(λ) 和 Dyna-Q(λ) 采用 Watkins trace cutting：如果 behavior policy 选出的下一动作不是 greedy action，则在本次更新后清空资格迹。Dyna 的模型是从原始 `(observation, action)` 到最新 `(reward, next_observation)` 的表；每个真实步之后执行 `planning_steps` 次模型更新，平均奖励率只由真实转移更新。Dyna-Q(λ) 的资格迹只沿连续的真实经验流传播，随机抽取且彼此不连续的 planning samples 仍使用 one-step Q-learning。

TIDBD 为每个权重维护 $`$beta_i = log(alpha_i)$`$ 和元迹 $H_i$：

$$
beta_i <- beta_i + theta * delta * x_i * H_i
alpha_i <- exp(beta_i)
w_i <- w_i + alpha_i * delta * z_i
H_i <- H_i * max(0, 1 - alpha_i * x_i * z_i) + alpha_i * delta * z_i
$$

实现不叠加 ObGD，以免改变 TIDBD 实验含义；只设置宽松的 `beta` 数值边界并检测 NaN/Inf。

## 状态与函数逼近

智能体可观察：

$$
(current_x, current_y, goal_x, goal_y, previous_action)
$$

它看不到风阶段、奖励阶段、地图模式编号或全局时钟。

双组 tile coding 分别编码：

1. 绝对位置 `(x, y, previous_action, candidate_action)`；
2. 相对目标位置 `(goal_x-x, goal_y-y, previous_action, candidate_action)`。

另有一个 categorical bias feature。默认每组 8 个 tilings，因此正常情况下每次有 17 个激活特征。

## Continuing 环境规则

- 动作包括上、右、下、左、原地停留；
- `stay` 仍然受到风的影响；
- 主动动作和风的位移逐格执行；
- 任一步碰到边界或障碍物，整个转移取消，智能体留在动作前的位置并得到碰撞惩罚；
- 到达目标后得到目标奖励；默认将智能体立即随机传送到合法非目标格；
- `After reaching target` 可切换为移动目标模式：目标随机迁移到另一个非障碍格，智能体留在旧目标格继续行动；
- 环境始终返回 `terminated=False, truncated=False`；
- 到达目标、风季节变化、目标移动和地图切换均不清空资格迹；
- 地图切换时，如果智能体当前格在新地图中是障碍物，该障碍物暂不激活；智能体离开后立即激活。

## Structured non-stationarity

环境不再使用互斥的 profile。GUI 和 `EnvironmentConfig` 提供四个完全独立、
可以任意组合的开关：

- `wind_changes`：风向按 `wind_period` 在上、右、下、左之间循环；
- `goal_moves`：目标按 `target_move_interval` 沿路径移动；
- `obstacle_switches`：障碍物地图按 `context_switch_interval` 切换；
- `reward_changes`：奖励倍率按独立的 `reward_period` 循环。

四项默认全部关闭。未开启 `wind_changes` 时仍可配置固定风向；开启后固定风向
设置被周期风向取代。风阶段与奖励阶段彼此独立，不再隐式绑定。

地图生成器保证所有合法格连通。面板中可以先点击一个障碍物，再点击一个空格来移动障碍物；破坏连通性的修改会被拒绝。

## 启动图形面板

Python 需要带 Tk 支持。安装依赖后，在仓库根目录执行：

```powershell
python run_gui.py
```

面板支持：

- 环境、奖励、调度周期和算法超参数设置；
- 地图生成、上下文地图预览和障碍物手动移动；
- 开始、暂停、继续、手动保存、停止并保存；
- checkpoint 加载并精确续训；
- 网格、平均奖励、目标到达率、碰撞率、TD error、步长和 Dyna planning 状态实时显示。

训练在后台线程中执行，GUI 不参与智能体观测。

## 无图形界面运行

运行固定步数：

```powershell
python -m stream_rl_grid.cli --steps 50000
```

无限运行，人工按 `Ctrl+C` 停止并自动保存：

```powershell
python -m stream_rl_grid.cli --steps 0
```

精确续训：

```powershell
python -m stream_rl_grid.cli --resume checkpoints/<run-id>/step-000000050000.pkl --steps 0
```

固定步长基线：

```powershell
python -m stream_rl_grid.cli --fixed-alpha --steps 50000
```

选择其他算法：

```powershell
python -m stream_rl_grid.cli --algorithm q_learning --steps 50000
python -m stream_rl_grid.cli --algorithm q_lambda --steps 50000
python -m stream_rl_grid.cli --algorithm sarsa --steps 50000
python -m stream_rl_grid.cli --algorithm dyna_q --planning-steps 5 --steps 50000
python -m stream_rl_grid.cli --algorithm dyna_q_lambda --planning-steps 5 --steps 50000
```

自由组合非平稳因素，例如只改变转移机制：

```powershell
python -m stream_rl_grid.cli --wind-changes --goal-moves --obstacle-switches --steps 50000
```

只改变奖励，或同时打开全部变化：

```powershell
python -m stream_rl_grid.cli --reward-changes --reward-period 2000 --steps 50000
python -m stream_rl_grid.cli --wind-changes --goal-moves --obstacle-switches --reward-changes --steps 50000
```

切换到“目标迁移、智能体不瞬移”的到达目标逻辑：

```powershell
python -m stream_rl_grid.cli --goal-reached-behavior relocate_target --steps 50000
```

## 多随机种子验证

比较 TIDBD 与固定步长 Differential Sarsa：

```powershell
python -m stream_rl_grid.benchmark --steps 50000 --seeds 0 1 2 3 4
```

输出包括逐运行 CSV、均值学习曲线和 95% 正态近似置信区间。主要指标是滑动窗口平均奖励，不使用 episode return。

## Checkpoint 内容

checkpoint 不只保存 `w`，还保存：

- `w, R_bar`，以及所选算法需要的 `z`、`beta/H` 或 Dyna model；
- 当前观测和已经选好的下一动作；
- 环境位置、目标、地图、风/奖励/地图调度相位；
- 延迟激活障碍物；
- IHT 字典与碰撞计数；
- Python 和 NumPy 随机数状态；
- 滑动指标、曲线、配置和格式版本。

保存采用临时文件 + 原子替换，避免中断时留下半个 checkpoint。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## Algorithm and environment configuration

Algorithms share the interface in `stream_rl_grid/algo/base.py` and are registered by
their implementations. `AgentConfig.algorithm` accepts `"q_learning"`, `"q_lambda"`,
`"sarsa"`, `"dyna_q"`, or `"tidbd"`; the GUI exposes the same selector.
The policy shown by the GUI is a frozen `(height, width, 5)` epsilon-greedy probability
matrix built from the current learned parameters; it is separate from the action sampled
by the online behavior loop.

Default maps can be authored directly in `EnvironmentConfig` with
`obstacle_coordinates`, `start_position`, and `goal_position`. Wind is probabilistic:
`w_strength=0.3` means a 30% chance of one additional cell of displacement in the selected
wind direction on each transition.

## Selectable feature representations

`AgentConfig.feature_representation` selects `"tile_coding"` (the default),
`"handcrafted_lfa"`, or `"handcrafted_lfa_nuisance"`. The GUI exposes the same
choices and hides tile-only
parameters when the D=55 representation is selected. Headless runs can select it with:

```powershell
python -m stream_rl_grid.cli --features handcrafted_lfa --algorithm sarsa --steps 50000
```

The hand-crafted representation is the collaborator branch's exact unit-norm linear
encoding. For each candidate action it uses an independent eleven-weight block:

```text
[1, x, y, x^2, y^2, xy, dx, dy, dx^2, dy^2, dxdy]
```

Here `x` and `y` are scaled to `[-1, 1]`, while `dx` and `dy` are the agent-to-goal
offsets divided by the corresponding grid span. The eleven values are L2-normalized,
then placed in the selected action's block, giving `11 * 5 = 55` weights. The
`previous_action` observation component is intentionally ignored. Because this feature
vector has unit norm, `effective_initial_step` is used directly; tile coding continues
to divide it by its nominal active-feature count.

The D=71 option is the collaborator's controlled nuisance-feature extension. At each
stream time, a dedicated RNG independently samples one of 16 nuisance categories. Its
one-hot feature is appended to the D=55 vector and the combined vector is divided by
`sqrt(2)`, preserving unit norm. The same sampled category is used for every candidate
action at that time. Its RNG is checkpointed and is independent of environment,
epsilon-greedy, and Dyna-planning randomness.

```powershell
python -m stream_rl_grid.cli --features handcrafted_lfa_nuisance --algorithm tidbd --steps 50000
```

## Phase-one D=55 sweep (945 runs)

The phase-one comparison fixes the `handcrafted_lfa` D=55 representation for every
algorithm, including TIDBD. It expands 63 parameter configurations over three
non-stationary settings and five paired seeds, for exactly 945 independent 60,000-step
runs. From the repository root, start the complete resumable sweep once with:

```powershell
python run_phase1_sweep.py
```

Results are written under `experiment_results/phase1_d55`. Four workers are used by
default when the machine has enough logical CPUs; override this with `--workers N`.
Every active run overwrites `progress.pkl` every 5,000 steps. Re-running the same
command skips completed jobs, truncates any CSV rows beyond the last safe checkpoint,
and resumes unfinished jobs exactly.

Each run writes:

- `metrics.csv`: trailing reward, exact cumulative reward AUC, exact stream-average
  reward, interval reward, TD error, goal/collision rates, alpha diagnostics, and
  environment events;
- `events.csv`: exact pre-change baseline, fixed-window post-change reward/AUC, and
  recovery time for every external change;
- `summary.json`: final metrics, event aggregates, parameters, and run status;
- `config.json`: the complete serialized run configuration.

The common schedule has a 6,000-step period per factor, with staggered first changes:
wind at 5,500, goal at 7,000, obstacles at 8,500, and reward at 10,000. Thus no two
external factors change on the same step. Environment maps, goal paths, and event
timings are stored in `experiment_manifest.json` and reused across algorithms.

After all runs finish, aggregate the CSV files, select the best configuration of each
algorithm by mean exact stream-average reward, and generate figures with colored
vertical change lines using:

```powershell
python plot_phase1_results.py
```

This creates the flat `phase1_summary/` directory at the repository root. It contains
`aggregate_summary.csv`, `selected_configs.csv`, three learning-curve figures, and
three adaptation-metric figures side by side. Use `--output PATH` to choose another
summary directory. Plotting is intentionally separate from the GUI and from training.

测试覆盖各算法的 bootstrap target、Watkins trace cutting、Dyna planning、continuing 目标传送、碰撞回退、`stay` 的风效应、地图切换延迟障碍物、TIDBD 数值有限性，以及 checkpoint 后的精确续训。
