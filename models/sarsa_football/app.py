"""Grid Football visualization."""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np
import solara
from matplotlib.figure import Figure
from agents import FootballAgent, Goal
from model import Football, FootballScenario
from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.utils import update_counter


def footballPortrayal(agent):
    """Map model agents to matplotlib visual styles for rendering."""
    if isinstance(agent, FootballAgent):
        color = "red" if agent.team == "A" else "blue"
        hasBall = getattr(agent.model, "ball_carrier", None) is agent
        size = 150 if hasBall else 60
        marker = "*" if hasBall else "o"
        return AgentPortrayalStyle(color=color, size=size, marker=marker, zorder=3)
    if isinstance(agent, Goal):
        color = "lime" if agent.team == "A" else "orange"
        return AgentPortrayalStyle(color=color, size=200, marker="s", zorder=1)
    return None


def postProcess(ax):
    """Apply chart annotations and legend after each render pass."""
    m = model
    ax.set_title(f"Grid Football  |  A: {m.score['A']}  —  B: {m.score['B']}", fontsize=11)
    ax.set_xlim(-1, m.width)
    ax.set_ylim(-1, m.height)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    handles = [
        mpatches.Patch(color="red", label="Team A"),
        mpatches.Patch(color="blue", label="Team B"),
        mpatches.Patch(color="lime", label="Goal A"),
        mpatches.Patch(color="orange", label="Goal B"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)


def _stateForCell(model, team: str, coord: tuple[int, int]) -> tuple:
    """Build a state tuple matching encode_state for a hypothetical agent at a given cell."""
    row, col = coord
    height = model.height
    width = model.width
    ballPos = model.ball.position

    rowBin = min(3, row * 4 // height)

    # team-relative column: mirrors encode_state so heatmap aligns with learned Q-values
    relCol = col if team == "A" else (width - 1) - col
    colBin = min(3, relCol * 4 // width)

    if ballPos is None:
        ballDistBin = 3
    else:
        dist = abs(row - ballPos[0]) + abs(col - ballPos[1])
        ballDistBin = 0 if dist <= 2 else 1 if dist <= 5 else 2

    carrier = model.ball_carrier
    hasBall = 0
    if carrier is not None and carrier.cell is not None:
        hasBall = int(carrier.team == team and carrier.cell.coordinate == coord)

    opponents = [a for a in model.agents if isinstance(a, FootballAgent) and a.team != team and a.cell is not None]
    oppNearby = 0
    if opponents:
        minOppDist = min(abs(row - a.cell.coordinate[0]) + abs(col - a.cell.coordinate[1]) for a in opponents)
        oppNearby = int(minOppDist <= 2)

    return (rowBin, colBin, ballDistBin, hasBall, oppNearby)


def _stateDistance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Return Hamming distance between two discrete state tuples."""
    return sum(int(x != y) for x, y in zip(a, b))


def _estimateVForState(refAgent: FootballAgent, state: tuple[int, int, int, int, int]) -> float:
    """Estimate V(s) from agent Q-table, falling back to nearest learned state."""
    qValues = refAgent.q_table.get(state)
    if qValues is not None:
        return float(max(qValues.values()))

    if not refAgent.q_table:
        return 0.0

    nearestState = min(refAgent.q_table.keys(), key=lambda s: _stateDistance(s, state))
    return float(max(refAgent.q_table[nearestState].values()))


def _buildVGrids(model) -> dict[str, np.ndarray]:
    """Compute V(s) grid arrays for both teams from current Q-tables."""
    teamAgents = {
        "A": next((a for a in model.agents if isinstance(a, FootballAgent) and a.team == "A"), None),
        "B": next((a for a in model.agents if isinstance(a, FootballAgent) and a.team == "B"), None),
    }
    gridByTeam: dict[str, np.ndarray] = {}
    for team in ["A", "B"]:
        refAgent = teamAgents[team]
        gridValues = np.zeros((model.height, model.width), dtype=float)
        if refAgent is not None:
            for r in range(model.height):
                for c in range(model.width):
                    state = _stateForCell(model, team, (r, c))
                    gridValues[r, c] = _estimateVForState(refAgent, state)
        gridByTeam[team] = gridValues
    return gridByTeam


@solara.component
def VGridComponent(model):
    """Render per-cell V(s)=max_a Q(s,a) heatmaps for Team A and Team B."""
    tick = update_counter.get()

    gridByTeam = _buildVGrids(model)
    gridA = gridByTeam["A"].copy()
    gridB = gridByTeam["B"].copy()

    globalVmin = float(min(np.min(gridA), np.min(gridB)))
    globalVmax = float(max(np.max(gridA), np.max(gridB)))
    if abs(globalVmax - globalVmin) < 1e-9:
        globalVmin -= 1.0
        globalVmax += 1.0

    vmin, vmax = globalVmin, globalVmax

    def makeFigure():
        f = Figure(figsize=(10, 4), dpi=100, constrained_layout=True)
        axs = f.subplots(1, 2)
        for idx, (team, grid) in enumerate(zip(["A", "B"], [gridA, gridB])):
            im = axs[idx].imshow(grid, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
            f.colorbar(im, ax=axs[idx], fraction=0.046, pad=0.04)
            axs[idx].set_title(f"Team {team} V(s) Heatmap")
            axs[idx].set_xlabel("col")
            axs[idx].set_ylabel("row")
        return f

    fig = solara.use_memo(makeFigure, dependencies=[tick])
    if fig is not None:
        solara.FigureMatplotlib(fig, format="png", bbox_inches="tight", dependencies=[tick])


model = Football(scenario=FootballScenario())

renderer = SpaceRenderer(model, backend="matplotlib")
renderer.setup_agents(footballPortrayal)
renderer.draw_structure()
renderer.draw_agents()
renderer.post_process = postProcess

scorePlot = make_plot_component({"score_A": "tab:red", "score_B": "tab:blue"})

modelParams = {
    "width": Slider("Grid Width", 10, 10, 30),
    "height": Slider("Grid Height", 10, 10, 30),
    "goal_width": Slider("Goal Width", 4, 2, 8),
    "opponent_threat_dist": Slider("Threat Distance", 3, 1, 10),
}

page = SolaraViz(
    model,
    renderer,
    components=[scorePlot, VGridComponent],
    model_params=modelParams,
    name="Grid Football",
)
page  # noqa