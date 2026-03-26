from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.experimental.scenarios import Scenario
from agents import Ball, FootballAgent, Goal


class FootballScenario(Scenario):
    """Scenario parameters for Grid Football."""

    width: int = 10
    height: int = 10
    goal_width: int = 4
    opponent_threat_dist: int = 5
    belief_update_freq: int = 2
    intention_timeout: int = 10


class Football(Model):
    """2v2 Grid Football with SARSA agents."""

    description = "A SARSA 2v2 Football model on a grid space."

    def __init__(self, scenario: FootballScenario = None):
        """Initialize the football environment, entities, and data collection."""
        if scenario is None:
            scenario = FootballScenario()
        super().__init__(scenario=scenario)

        self.height = scenario.height
        self.width = scenario.width
        self.goal_width = scenario.goal_width
        self.ball_carrier = None
        self.score = {"A": 0, "B": 0}
        self.last_goal = None
        self.last_goal_step = None
        self.last_touch_team = None

        self.grid = OrthogonalMooreGrid(
            [self.height, self.width],
            capacity=1,
            random=self.random,
        )
        self.goals: dict[str, list[Goal]] = {"A": [], "B": []}
        self.players_by_team: dict[str, list[FootballAgent]] = {"A": [], "B": []}

        self._create_goals()
        self._create_players()

        center = (self.height // 2, self.width // 2)
        self.ball = Ball(self, position=center)
        self._reset_kickoff(kickoff_team="A")

        model_reporters = {
            "score_A": lambda m: m.score.get("A", 0),
            "score_B": lambda m: m.score.get("B", 0),
            "ball_position": lambda m: m.ball.position,
            "ball_carrier_team": lambda m: (
                m.ball_carrier.team if m.ball_carrier is not None else None
            ),
        }
        agent_reporters = {
            "team": lambda a: a.team if isinstance(a, FootballAgent) else None,
            "has_ball": lambda a: a is getattr(a.model, "ball_carrier", None),
            "belief_ball_position": lambda a: (
                a.beliefs.get("ball_position") if isinstance(a, FootballAgent) else None
            ),
        }
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
        )
        self.datacollector.collect(self)

    def _starting_slots(self) -> dict[str, list[tuple[int, int]]]:
        """Return deterministic starting positions for both teams."""
        mid = self.height // 2
        rows = [max(0, mid - 1), min(self.height - 1, mid + 1)]
        left_col = max(1, self.width // 4)
        right_col = min(self.width - 2, self.width - 1 - self.width // 4)
        return {
            "A": [(rows[0], left_col), (rows[1], left_col)],
            "B": [(rows[0], right_col), (rows[1], right_col)],
        }

    def _goal_rows(self) -> list[int]:
        """Compute contiguous row indices occupied by each goal mouth."""
        span = max(1, min(self.goal_width, self.height))
        start = (self.height - span) // 2
        return list(range(start, start + span))

    def _create_goals(self) -> None:
        """Create fixed goal agents for both teams at field boundaries."""
        for row in self._goal_rows():
            self.goals["A"].append(Goal(self, team="A", cell=self.grid[row, 0]))
            self.goals["B"].append(Goal(self, team="B", cell=self.grid[row, self.width - 1]))

        for goals in self.goals.values():
            for g in goals:
                g.cell.capacity = 2

    def _create_players(self) -> None:
        """Spawn team agents into deterministic starting slots."""
        slots = self._starting_slots()
        for row, col in slots["A"]:
            self.players_by_team["A"].append(
                FootballAgent(
                    self,
                    team="A",
                    goal_cells=[g.cell for g in self.goals["A"]],
                    cell=self.grid[row, col],
                )
            )
        for row, col in slots["B"]:
            self.players_by_team["B"].append(
                FootballAgent(
                    self,
                    team="B",
                    goal_cells=[g.cell for g in self.goals["B"]],
                    cell=self.grid[row, col],
                )
            )

    def _reset_kickoff(self, kickoff_team: str | None = None) -> None:
        """Reset players and ball to kickoff state after goal or restart."""
        for agent in self.players_by_team["A"] + self.players_by_team["B"]:
            if agent._mesa_cell is not None:
                agent._mesa_cell.remove_agent(agent)
                agent._mesa_cell = None

        slots = self._starting_slots()
        for i, agent in enumerate(self.players_by_team["A"]):
            row, col = slots["A"][i % len(slots["A"])]
            target = self.grid[row, col]
            target.add_agent(agent)
            agent._mesa_cell = target
            agent.intention = None
            agent.commitment_steps = 0
        for i, agent in enumerate(self.players_by_team["B"]):
            row, col = slots["B"][i % len(slots["B"])]
            target = self.grid[row, col]
            target.add_agent(agent)
            agent._mesa_cell = target
            agent.intention = None
            agent.commitment_steps = 0

        center = (self.height // 2, self.width // 2)
        self.ball.place_loose(center)
        self.ball_carrier = None

        if kickoff_team is None:
            kickoff_team = self.random.choice(["A", "B"])
        kicker = self.random.choice(self.players_by_team[kickoff_team])
        self.ball.carrier = kicker
        self.ball.position = kicker.cell.coordinate
        self.ball_carrier = kicker
        self.last_touch_team = kicker.team

    def _goal_cells_set(self) -> set[tuple[int, int]]:
        """Return set of all goal coordinates."""
        coords = set()
        for goals in self.goals.values():
            for g in goals:
                coords.add(g.cell.coordinate)
        return coords

    def _is_out_of_bounds(self, pos: tuple[int, int]) -> bool:
        """Return True when the ball is on boundary outside any goal cell."""
        row, col = pos
        on_boundary = row in {0, self.height - 1} or col in {0, self.width - 1}
        return on_boundary and pos not in self._goal_cells_set()

    def _detect_goal(self) -> tuple[str, tuple[int, int]] | None:
        """Return scoring team if ball is on a goal cell."""
        pos = self.ball.position
        if pos is None:
            return None
        goal_cell = self.grid[pos]
        goal_agents = [a for a in goal_cell.agents if isinstance(a, Goal)]
        if not goal_agents:
            return None
        attacker = self.ball_carrier.team if self.ball_carrier is not None else None
        if attacker is None:
            attacker = "B" if pos[1] == 0 else "A"
        if any(g.allows_score_by(attacker) for g in goal_agents):
            return attacker, pos
        return None

    def register_goal(self, attacker_team: str, pos: tuple[int, int]) -> bool:
        """Validate and record a scoring event for the attacking team."""
        goal_cell = self.grid[pos]
        goal_agents = [a for a in goal_cell.agents if isinstance(a, Goal)]
        if not goal_agents:
            return False
        if not any(g.allows_score_by(attacker_team) for g in goal_agents):
            return False
        self.score[attacker_team] = self.score.get(attacker_team, 0) + 1
        self.last_goal = {"team": attacker_team, "cell": pos}
        self.last_goal_step = int(getattr(self, "steps", 0))
        return True

    def step(self) -> None:
        """Advance one simulation tick for agents, ball, scoring, and resets."""
        self.agents_by_type[FootballAgent].shuffle_do("step")
        self.ball.step()

        if self.ball_carrier is not None:
            self.last_touch_team = self.ball_carrier.team

        scored = self._detect_goal()
        if scored is not None:
            scorer, pos = scored
            if self.register_goal(scorer, pos):
                kickoff_team = "A" if scorer == "B" else "B"
                print(f"[GOAL] step={self.steps} scorer={scorer} pos={pos} score={self.score}\n")
                self._reset_kickoff(kickoff_team=kickoff_team)
        elif self.ball.position is not None and self._is_out_of_bounds(self.ball.position):
            kickoff_team = (
                "B" if self.last_touch_team == "A"
                else "A" if self.last_touch_team == "B"
                else None
            )
            print(f"[OOB] step={self.steps} ball_pos={self.ball.position} last_touch={self.last_touch_team} carrier={self.ball_carrier.team if self.ball_carrier else None}\n")
            self._reset_kickoff(kickoff_team=kickoff_team)

        self.datacollector.collect(self)