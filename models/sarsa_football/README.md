# SARSA 2v2 Grid Football

## The System:

- 2v2 football on a grid
- agents use on-policy SARSA (tabular)
- observation space is encoded as discrete state features
- two fixed goals, one per side

## Run

```bash
solara run app.py
```

## State features

5-element tuple used as Q-table key:

- row bin (0-3, absolute)
- col bin (0-3, team-relative — 0 = near own goal, 3 = near opponent goal, same encoding for both teams)
- ball distance bin (0 = close ≤2, 1 = mid ≤5, 2 = far, 3 = unknown)
- has ball (0/1)
- opponent nearby within 2 cells (0/1)

## Actions

- move in 8 directions
- pass
- tackle (50% success rate, only strips ball from carrier)

## Learning loop

- epsilon-greedy action selection, decayed each step (min 0.05)
- SARSA update: Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
- each agent has its own Q-table, teams do not share weights
- Q-table pre-seeded with a small directional bias toward the opponent goal when carrying

## Reward shaping

- +10 / -10 on goal scored / conceded (fires once per agent per goal event)
- carrier: progress reward scaled by distance to opponent goal, -0.3 anti-loop penalty
- off-ball teammate: supporting lane reward + -0.15 anti-loop
- no possession: pressure reward scaled by distance to ball

## Why?

- first RL building block toward Rocket League bot work
- I wanted a small environment to iterate quickly
- tabular SARSA keeps debugging and failure analysis simple
- discrete grid makes policy errors easy to inspect visually

## Metrics

- win rate: team A vs team B over episodes
- goals for / goals against per team
- steps per goal (how quickly scoring happens)
- score progression over steps
- V(s) heatmaps over training windows

heatmap interpretation:
- each team's heatmap is projected from its own agents' Q-tables
- not a shared value function, separate per-team spatial projections
- colBin is team-relative so both heatmaps read left-to-right as "own half → opponent half"

## Mesa features used

- OrthogonalMooreGrid for 2D movement and neighborhood checks
- CellAgent / FixedAgent for players and goal markers
- Ball as a plain Agent tracking a coordinate position
- Scenario class for tunable model parameters
- DataCollector for score, carrier, and belief tracking
- SolaraViz + SpaceRenderer for live simulation rendering
- matplotlib components for score plots and V(s) heatmaps

## What I learned

- state design decides whether SARSA can learn direction at all (without it, team B just loops around its own goal)
- team-relative column encoding was necessary; without it both teams learned contradictory Q-values for the same states
- reward shaping can create loops if progress is not asymmetric
- per-team visual diagnostics are essential for debugging policy drift
- tiny environment changes can flip behavior completely
- update order and exploration schedule strongly affect stability

## Mesa limitations discovered

- no built-in RL training loop abstractions for SARSA/Q-learning workflows
- still need manual feature engineering and reward design
- custom logging/debugging pipelines are required for policy analysis
- interactive viz is great for intuition but not enough for large experiments
- reproducibility needs extra handling (seeds, config snapshots, run tracking)

## Observed behaviour

- initially a lot of episodes end OOB
- goal scoring with ~40% success on spawning with possession after 1k steps
- passing in early steps results in agents not scoring
- scoring from center positions is more likely
- ~2500 steps of training required for scoring to stabilise
- outcome is very sensitive to epsilon decay rate and reward weights (hyperparameters)