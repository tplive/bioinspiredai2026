
#set page(paper: "a4")
#set page(numbering: "1 / 1", number-align: center + bottom)

#include "title.typ"

#outline(title: [Table of Contents])

#set heading(numbering: "1.1.a")
#set par(justify: true)

= Introduction
This report evaluates a Deep Q-Network (DQN) agent on FrozenLake-v1 for three assignment settings: 4x4 deterministic, 4x4 slippery, and 8x8 slippery. The objective is to meet target mean rewards and analyze policy behavior for the two 4x4 cases.

#set page(columns: 2)
= Method
The implementation in `assignment9.py` uses Stable-Baselines3 DQN with `MlpPolicy`.

Experimental protocol:
- One model is trained per environment and seed.
- Seeds: 0, 1, 2.
- Evaluation: mean reward over 100 deterministic episodes (`evaluate_policy`).
- Hyperparameters are environment-specific (timesteps, exploration fraction, replay buffer, and target update interval), as defined in the experiment matrix.

= Results
Numerical results are taken from `results/run_2026-03-29_12-05-53.log`.

#figure(
	table(
		columns: 7,
		align: center,
		[Environment], [Target], [Seed 0], [Seed 1], [Seed 2], [Mean], [Pass],
		[4x4, deterministic], [1.00], [0.000], [0.000], [0.000], [0.000], [No],
		[4x4, slippery], [0.70], [0.740], [0.780], [0.730], [0.750], [Yes],
		[8x8, slippery], [0.50], [0.530], [0.500], [0.430], [0.487], [No],
	),
	caption: [Mean reward per environment across three seeds.]
)

The 4x4 slippery setting reaches the required threshold. The deterministic 4x4 and 8x8 slippery settings do not meet their targets with the current setup.

= Policy Comparison for 4x4
This section addresses the required states:
- State 1: one step right of the start.
- State 6: second row, third column, with holes on both left and right.

Expected optimal behavior:
- Deterministic (`is_slippery = false`):
	- State 1: Right is optimal (short safe path progression).
	- State 6: Down is optimal (safe and goal-directed).
- Stochastic (`is_slippery = true`):
	- State 1: Right or Down are both reasonable safe actions.
	- State 6: Left or Right are optimal in expectation. With slippery dynamics, intended Left/Right reaches a hole with probability 1/3 and slips to safe Up/Down with probability 2/3, while intended Up/Down reaches a hole with probability 2/3.

Interpretation using achieved rewards:
- 4x4 deterministic mean reward is 0.000 for all seeds, so the learned policy is clearly not optimal.
- 4x4 slippery mean reward is 0.750, which indicates a strong and reasonable learned policy under stochastic transitions.

Overall, the stochastic 4x4 agent appears near-optimal, while the deterministic 4x4 agent fails to learn a successful path with the current hyperparameters/training setup.

#bibliography("references.bib", title: [Bibliography])


