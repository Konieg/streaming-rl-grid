# Phase 6–9 统计结论

所有 runs 都是 continuing average-reward streams；竖直 context 线不会终止环境或重置 learner。每 100 real steps 的 exact policy diagnostic 是只读计算。

## 各场景最低 normalized dynamic regret

- **loca_reward**：CAFD-Lite，`0.0933 ± 0.0031`。
- **obstacle_abrupt**：CAFD-Lite，`0.1118 ± 0.0037`。
- **wind_drift**：CAFD-Surprise，`0.2287 ± 0.0070`。
- **recurring_composition**：CAFD-Lite，`0.2618 ± 0.0082`。

## 相对 Q-learning 的机制结论

- loca_reward / EMA Dyna：regret difference `-0.0811 ± 0.0943`。
- loca_reward / Prioritized EMA：regret difference `0.0059 ± 0.1567`。
- loca_reward / Factored Dyna：regret difference `-0.1946 ± 0.0523`（CI 排除 0）。
- loca_reward / CAFD-Lite：regret difference `-0.2058 ± 0.0526`（CI 排除 0）。
- loca_reward / CAFD-Surprise：regret difference `-0.2056 ± 0.0514`（CI 排除 0）。
- obstacle_abrupt / EMA Dyna：regret difference `-0.0527 ± 0.1298`。
- obstacle_abrupt / Prioritized EMA：regret difference `0.0506 ± 0.1982`。
- obstacle_abrupt / Factored Dyna：regret difference `-0.1772 ± 0.0874`（CI 排除 0）。
- obstacle_abrupt / CAFD-Lite：regret difference `-0.1869 ± 0.0882`（CI 排除 0）。
- obstacle_abrupt / CAFD-Surprise：regret difference `-0.1598 ± 0.0884`（CI 排除 0）。
- wind_drift / EMA Dyna：regret difference `-0.0074 ± 0.3007`。
- wind_drift / Prioritized EMA：regret difference `-0.0613 ± 0.2700`。
- wind_drift / Factored Dyna：regret difference `-0.4100 ± 0.1716`（CI 排除 0）。
- wind_drift / CAFD-Lite：regret difference `-0.4404 ± 0.1742`（CI 排除 0）。
- wind_drift / CAFD-Surprise：regret difference `-0.4448 ± 0.1680`（CI 排除 0）。
- recurring_composition / EMA Dyna：regret difference `-0.1007 ± 0.0448`（CI 排除 0）。
- recurring_composition / Prioritized EMA：regret difference `-0.0808 ± 0.0438`（CI 排除 0）。
- recurring_composition / Factored Dyna：regret difference `-0.2007 ± 0.0297`（CI 排除 0）。
- recurring_composition / CAFD-Lite：regret difference `-0.2073 ± 0.0269`（CI 排除 0）。
- recurring_composition / CAFD-Surprise：regret difference `-0.1925 ± 0.0311`（CI 排除 0）。

负 difference 表示比 Q-learning 更低的 dynamic regret。
Oracle Dyna 只作为 perfect-current-model upper bound，不计作 learned method。

## 配对机制消融

- loca_reward / recency_minus_empirical：paired difference `-0.0260 ± 0.0072`（CI 排除 0）。
- loca_reward / priority_minus_uniform_ema：paired difference `0.0870 ± 0.1703`。
- loca_reward / priority_minus_uniform_factored：paired difference `-0.0111 ± 0.0034`（CI 排除 0）。
- loca_reward / surprise_minus_fixed_cafd：paired difference `0.0002 ± 0.0036`。
- obstacle_abrupt / recency_minus_empirical：paired difference `-0.1400 ± 0.0200`（CI 排除 0）。
- obstacle_abrupt / priority_minus_uniform_ema：paired difference `0.1032 ± 0.1952`。
- obstacle_abrupt / priority_minus_uniform_factored：paired difference `-0.0097 ± 0.0053`（CI 排除 0）。
- obstacle_abrupt / surprise_minus_fixed_cafd：paired difference `0.0271 ± 0.0062`（CI 排除 0）。
- wind_drift / recency_minus_empirical：paired difference `0.0222 ± 0.0189`（CI 排除 0）。
- wind_drift / priority_minus_uniform_ema：paired difference `-0.0539 ± 0.2518`。
- wind_drift / priority_minus_uniform_factored：paired difference `-0.0304 ± 0.0161`（CI 排除 0）。
- wind_drift / surprise_minus_fixed_cafd：paired difference `-0.0044 ± 0.0117`。
- recurring_composition / recency_minus_empirical：paired difference `-0.0223 ± 0.0288`。
- recurring_composition / priority_minus_uniform_ema：paired difference `0.0199 ± 0.0545`。
- recurring_composition / priority_minus_uniform_factored：paired difference `-0.0066 ± 0.0172`。
- recurring_composition / surprise_minus_fixed_cafd：paired difference `0.0147 ± 0.0103`（CI 排除 0）。

这里同样是 treatment − control；负值表示加入该机制后 regret 更低。
`preferred_goal_fraction` 是辅助指标，定义为当前奖励更高目标的到达比例；它不参与 dynamic regret 或任何显著性主结论。
