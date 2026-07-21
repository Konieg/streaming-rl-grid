# Approved Poster Outline

## Slide 1: Learning Reusable Environment Mechanisms for Continual Reinforcement Learning with Model-Based Methods

- **Header:** center the exact title, team-member line, and keyword line within the existing blue-gradient header while preserving all three institutional logos.
- **Motivation:** ask whether MBRL can offer continual adaptation through mechanism learning—not only sample efficiency. Operationalize “understanding” as parameter tracking, systematic generalization, and compositional reuse.
- **Environment:** explain the continuing $7\times7$ two-goal grid, no episodic termination or learner reset, hidden context, reward semantics, and shared 22-active-feature tile coding.
- **Method:** introduce CAFD-Lite as a factored model of stable mechanics plus changing wind, corridor, and goal-reward factors. Show the observation-to-factor-update-to-mixed-planning flow and the 1-prioritized-plus-4-uniform planning rule.
- **Evidence:** show four-scenario dynamic regret, stream average reward, and world-model error. Emphasize that factorization creates the main gap and prioritized planning adds a smaller benefit.
- **Conclusion:** report 44–69% lower regret versus Q-learning, +0.072 to +0.172 stream average reward, and the limitation that an unstructured model alone is insufficient.
- **Layout role:** single-page portrait scientific poster; left column establishes question, task, and method; right column carries the main evidence and supported conclusions.

### Required images

- Template background and institutional branding; **strict layout/style input**; preserve dimensions, gradient bands, logos, logo placement, colors, and whitespace structure. Add content without overwriting the source file.

  ![Poster template](../poster_assets/poster_template.pptx)

- Environment illustration; **strict content input**; preserve topology, labels, rewards, factor legend, and continuing-task note.

  ![Continuing two-goal grid](../poster_assets/grid_world.png)

- Primary control evidence; **strict data input**; preserve data, axes, labels, line colors, legends, confidence bands, event markers, and values.

  ![Selected dynamic-regret comparison](../poster_assets/dynamic_regret_selected.png)

- Online-reward evidence; **strict data input**; preserve all numbers, uncertainty intervals, labels, bolding, and footnote.

  ![Stream average reward table](../poster_assets/stream_average_reward_table.png)

- Model-fidelity evidence; **strict data input**; preserve data, axes, labels, line colors, confidence bands, event markers, and values.

  ![Selected model-tracking comparison](../poster_assets/model_tracking_selected.png)

### Style-only references

- `../poster_assets/examples/253208540294-poster.pdf`
- `../poster_assets/examples/Feature-wise Step-Size Adaptation for Continual Reinforcement Learning in Non-Stationary Gridworlds.pdf`
- `../poster_assets/examples/poster_siirlve.pdf`

### Exact header copy

1. Learning Reusable Environment Mechanisms for Continual Reinforcement Learning with Model-Based Methods
2. Team Members: Ziheng Wang, Yidan Wu, Fandi Gou, Cunyang Xu
3. Keywords: Continual RL; Model-Based Reinforcement Learning
