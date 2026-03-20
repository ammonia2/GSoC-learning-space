# Wolf-Sheep Needs Variant

A custom variant of Mesa's Wolf-Sheep model where sheep behavior is driven by two internal needs: fear and hunger, instead of purely random movement.

## Why?

The needs-based extension to Wolf-Sheep was suggested as a starting point in the [GSoC 2026 Project Ideas](https://github.com/mesa/mesa/wiki/GSoC-2026-Project-Ideas) for the Behavioral Framework project. Predator-prey environments are also my current domain in Multi-Agent RL research, so this was a natural first model.

## What Changed

Wolves were left unchanged. As pure predators, hunger is already implicit in their energy system and there's no meaningful second drive to add. Sheep got two explicit needs:

- `fear`: spikes based on wolf density within a sensing radius, decays each step using an exponential smoothing factor
- `hunger`: derived from current energy, normalized to [0, 1]

Each step, sheep score neighboring cells by safety and food availability, weighted by whichever need is currently dominant. Feeding only happens if hunger exceeds fear.

## Metrics

Beyond the standard population plots, three mortality diagnostics were added:

- `SheepStarved` (cumulative)
- `SheepEaten` (cumulative)  
- `StarvationRate = starved / (starved + eaten)`

The starvation rate tells whether the fear mechanic is making sheep too cautious to eat, or whether the tradeoff is ecologically reasonable.

## Mesa Features Used

As this is an extension to the existing Wolf-Sheep model, only features like DataCollector, Model attributes and Agent constructors were modified. Nothing additional used in the basic remodeling.

- CellAgent and FixedAgent from mesa.discrete_space
- OrthogonalVonNeumannGrid. Learned about the spatial structure, how neighborhood queries work
- DataCollector. For custom metrics like SheepStarted and Eaten
- Solara viz layer. app.py modifications for sliders and additional plots
## What I Learned

Needs-based behavior did stabilize the population compared to the baseline, which was expected. What wasn't expected was how sensitive the stability was to hyperparameters like fear decay and sensing radius. Small changes could tip the model back into extinction, which made it clear that behavioral parameters are as important as ecological ones.

Coming from MARL, Mesa's cell-based movement feels more constrained. There's no reward signal, no policy to optimize. Instead you hand-code the decision logic, which forces you to be explicit about what you think the agent actually "wants." That explicitness is both Mesa's strength and its current limitation: you can build rich behavioral models, but the framework gives you no structure for expressing drives, rewards, or competing goals. You wire it all yourself.

## What Would've Been Easier With Better Mesa Support

- **A structured state and reward model**: fear decay, hunger normalization, and action scoring were all hand-coded with no reusable abstraction. The `StateAgent` proposed in [PR #2547](https://github.com/mesa/mesa/pull/2547) would have cleaned this up significantly.
- **Parameter learning**: tuning fear decay, sensing radius, and scoring weights was manual trial and error. A built-in GA or grid search utility would help — or at minimum a cleaner `batch_run` interface for parameter sweeps.
- **Distance-aware neighborhood queries**: `get_neighborhood(radius=3)` returns a flat list with no distance information. Fear magnitude should realistically scale with how close the wolf is, but there's no API for that without manually computing distances.
- **Internal state introspection**: tracking `fear` and `hunger` over time required manually adding DataCollector entries for every internal variable. There's no built-in way to observe agent-level state evolution.

## Run
```bash
cd GSoC-learning-space/models/wolf_sheep_needs
solara run app.py
```