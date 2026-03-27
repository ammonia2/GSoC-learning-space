from __future__ import annotations
from mesa import Agent
from mesa.discrete_space import CellAgent, FixedAgent


class FootballAgent(CellAgent):
    """2v2 SARSA football player."""

    # action space
    ACTIONS = [
        "move_up", "move_down", "move_left", "move_right",
        "move_upleft", "move_upright", "move_downleft", "move_downright",
        "pass", "tackle",
    ]
    INITIAL_Q_VALUE = 0.5

    def __init__(self, model, team: str, goal_cells: list, cell=None):
        """Initialize a football agent with team assignment and SARSA parameters."""
        super().__init__(model)
        self.team = team
        self.goal_cells = goal_cells
        self.cell = cell
        self.beliefs = {
            "opponent_positions": [],
            "ball_position": None,
            "teammate_position": None,
        }
        
        self.q_table = self._initialize_q_table()  # state -> action values
        self.epsilon = 0.3  # exploration rate
        self.epsilonMin = 0.05
        self.epsilonDecay = 0.995
        self.alpha = 0.1  # learning rate
        self.gamma = 0.95  # discount factor
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.prevPos = None

    def _default_q_values(self) -> dict[str, float]:
        """Return default action values for a newly-seen state."""
        return {action: self.INITIAL_Q_VALUE for action in self.ACTIONS}

    def _initialize_q_table(self) -> dict:
        """Pre-populate all encoded state combinations.

        For states where the agent has the ball (hasBall=1), there is a small
        directional bias toward the actions that advance the ball into the
        opponent half.  Team A scores at high col → prefer move_up/upright.
        Team B scores at low col → prefer move_down/downleft.
        This gives agents a head-start so they don't need thousands of random
        steps just to discover which direction the opponent goal is.
        """
        q_table = {}
        advance_actions_A = {"move_up", "move_upright", "move_upleft"}
        advance_actions_B = {"move_down", "move_downright", "move_downleft"}
        advance_actions = advance_actions_A if self.team == "A" else advance_actions_B

        for rowBin in range(4):
            for colBin in range(4):
                for ballDistBin in [0, 1, 2, 3]:
                    for hasBall in [0, 1]:
                        for oppNearby in [0, 1]:
                            state = (rowBin, colBin, ballDistBin, hasBall, oppNearby)
                            vals = self._default_q_values()
                            if hasBall == 1:
                                for a in advance_actions:
                                    vals[a] = self.INITIAL_Q_VALUE + 0.1
                            q_table[state] = vals
        return q_table

    def _players(self, same_team: bool | None = None):
        """Return players filtered by team relationship to this agent."""
        return [
            a for a in self.model.agents
            if isinstance(a, FootballAgent)
            and (same_team is None or (a.team == self.team) is same_team)
        ]

    def _ball(self):
        """Return the ball object if present on the model."""
        return getattr(self.model, "ball", None)

    def _carrier(self):
        """Return the current ball carrier, if any."""
        ball = self._ball()
        return ball.carrier if ball is not None else getattr(self.model, "ball_carrier", None)

    def _set_carrier(self, carrier) -> None:
        """Set ball possession to the provided carrier and sync model state."""
        ball = self._ball()
        if ball is not None:
            ball.carrier = carrier
        self.model.ball_carrier = carrier

    def _ball_position(self):
        """Return current ball coordinates, following the carrier when controlled."""
        carrier = self._carrier()
        if carrier is not None and carrier.cell is not None:
            return carrier.cell.coordinate
        ball = self._ball()
        return ball.position if ball is not None else None

    @staticmethod
    def _manhattan(a, b):
        """Compute Manhattan distance between two coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _has_ball(self):
        """Return True when this agent currently possesses the ball."""
        return self._carrier() is self

    def _goal_center(self):
        """Return the center coordinate of this agent's defended goal."""
        if not self.goal_cells:
            return self.cell.coordinate if self.cell else (0, 0)
        xs = [c.coordinate[0] for c in self.goal_cells]
        ys = [c.coordinate[1] for c in self.goal_cells]
        return (sum(xs) // len(xs), sum(ys) // len(ys))

    def _opponent_goal_center(self):
        """Return the center coordinate of the opponent goal."""
        own = self._goal_center()
        return (own[0], self.model.scenario.width - 1 - own[1])

    def _can_enter_cell(self, cell):
        """Allow entry into cells not occupied by another mobile agent."""
        if self.cell is not None and cell == self.cell:
            return True
        # block own goal entry
        if cell in self.goal_cells:
            return False
        fixedCount = sum(1 for a in cell.agents if isinstance(a, FixedAgent))
        mobileOccupants = [a for a in cell.agents if a is not self and not isinstance(a, FixedAgent)]
        effectiveCapacity = (cell.capacity or 1) - fixedCount
        return len(mobileOccupants) < effectiveCapacity

    def encode_state(self) -> tuple:
        """
        Encode beliefs into a small, hashable state tuple for Q-table indexing.

        Returns a 5-element tuple:
          - rowBin:        coarse row position (0-3)
          - colBin:        team-relative column bin — 0 means near own goal,
                           3 means near opponent goal (so both teams learn
                           "advance = higher colBin" without contradictory Q-values)
          - ballDistBin:   distance to ball (0=close ≤2, 1=mid ≤5, 2=far, 3=unknown)
          - hasBall:       whether this agent possesses the ball (0/1)
          - oppNearby:     nearest opponent is within 2 cells (0/1)
        """
        myPos = self.cell.coordinate if self.cell else (0, 0)
        ballPos = self.beliefs["ball_position"]
        oppPositions = self.beliefs["opponent_positions"]
        height = self.model.scenario.height
        width = self.model.scenario.width

        # coarse row bin (absolute — same meaning for both teams)
        rowBin = min(3, myPos[0] * 4 // height)

        # team-relative column bin: flip for team B so both teams encode
        # "distance already covered toward opponent goal" the same way
        col = myPos[1]
        if self.team == "B":
            col = (width - 1) - col
        colBin = min(3, col * 4 // width)

        if ballPos is None:
            ballDistBin = 3
        else:
            dist = self._manhattan(myPos, ballPos)
            ballDistBin = 0 if dist <= 2 else 1 if dist <= 5 else 2

        hasBall = 1 if self._has_ball() else 0

        oppNearby = 0
        if oppPositions:
            oppNearby = 1 if min(self._manhattan(myPos, o) for o in oppPositions) <= 2 else 0

        return (rowBin, colBin, ballDistBin, hasBall, oppNearby)

    def step(self):
        """SARSA step: update beliefs, process reward, select and execute action."""
        self._update_beliefs()
        
        reward = self._compute_reward()

        if self.last_state is not None and self.last_action is not None:
            state = self.encode_state()
            next_action = self.select_action(state)
            self._update_q_table(self.last_state, self.last_action, reward, state, next_action)
            self.last_reward = 0

        state = self.encode_state()
        action = self.select_action(state)
        self.execute_action(action)

        self.last_state = state
        self.last_action = action

        self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)

    def select_action(self, state: tuple) -> str:
        """Epsilon-greedy action selection."""
        if state not in self.q_table:
            self.q_table[state] = self._default_q_values()

        availableActions = [a for a in self.ACTIONS if not (self._has_ball() and a == "tackle")]

        if self.random.random() < self.epsilon:
            return self.random.choice(availableActions)

        qValues = self.q_table[state]
        maxQ = max(qValues[a] for a in availableActions)
        bestActions = [a for a in availableActions if qValues[a] == maxQ]
        return self.random.choice(bestActions)

    def execute_action(self, action: str) -> None:
        """Execute the selected action: move, pass, tackle."""
        if self.cell is None:
            return

        directionMap = {
            "move_right": (1, 0),
            "move_left": (-1, 0),
            "move_up": (0, 1),
            "move_down": (0, -1),
            "move_upright": (1, 1),
            "move_upleft": (-1, 1),
            "move_downright": (1, -1),
            "move_downleft": (-1, -1),
        }

        if action in directionMap:
            dr, dc = directionMap[action]
            row, col = self.cell.coordinate
            newRow, newCol = row + dr, col + dc
            height = self.model.scenario.height
            width = self.model.scenario.width
            if 0 <= newRow < height and 0 <= newCol < width:
                targetCell = self.model.grid[newRow, newCol]
                if self._can_enter_cell(targetCell):
                    try:
                        self.move_to(targetCell)
                    except Exception:
                        pass
                elif self._has_ball():
                    teammate = next(
                        (a for a in self._players(same_team=True) if a is not self and a.cell is not None),
                        None,
                    )
                    if teammate is not None:
                        self._set_carrier(teammate)

        elif action == "pass":
            if self._has_ball():
                teammate = next(
                    (a for a in self._players(same_team=True) if a is not self and a.cell is not None),
                    None,
                )
                if teammate is not None:
                    self._set_carrier(teammate)

        elif action == "tackle":
            nearbyOpponents = [
                opp for opp in self._players(same_team=False)
                if opp.cell is not None and opp.cell in self.cell.connections.values()
            ]
            if nearbyOpponents:
                target = self.random.choice(nearbyOpponents)
                if self.random.random() < 0.5 and target._has_ball():
                    self._set_carrier(self)


    def _update_q_table(self, state: tuple, action: str, reward: float, next_state: tuple, next_action: str) -> None:
        """
        SARSA update rule:
        Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))
        """
        if state not in self.q_table:
            self.q_table[state] = self._default_q_values()
        if next_state not in self.q_table:
            self.q_table[next_state] = self._default_q_values()
        
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q

    def _compute_reward(self) -> float:
        reward = 0.0
        myPos = self.cell.coordinate if self.cell else (0, 0)
        ballPos = self.beliefs["ball_position"]
        carrier = self._carrier()
        maxDist = self.model.scenario.width + self.model.scenario.height

        # Goal reward — consume the flag so it fires exactly once per agent per goal
        currentStep = int(getattr(self.model, "steps", 0))
        rewardedKey = f"_goal_reward_claimed_{self.unique_id}"
        alreadyClaimed = getattr(self.model, rewardedKey, -1)
        if (
            getattr(self.model, "last_goal_step", -1) == currentStep
            and hasattr(self.model, "last_goal")
            and alreadyClaimed != currentStep
        ):
            setattr(self.model, rewardedKey, currentStep)
            scorer = self.model.last_goal.get("team")
            reward += 10.0 if scorer == self.team else -10.0

        if ballPos is not None:
            oppGoal = self._opponent_goal_center()

            if self._has_ball():
                # Carrier: reward closing distance to opponent goal
                currDist = self._manhattan(myPos, oppGoal)
                reward += (maxDist - currDist) / maxDist
                # Anti-loop: penalise revisiting the same cell
                if self.prevPos is not None and myPos == self.prevPos:
                    reward -= 0.3
                else:
                    reward -= 0.05  # small step cost even when moving

            elif carrier is not None and carrier.team == self.team:
                # Teammate support: stay in a useful supporting lane
                distToCarrier = self._manhattan(myPos, carrier.cell.coordinate)
                distToOppGoal = self._manhattan(myPos, oppGoal)
                reward += (maxDist - distToOppGoal) / maxDist * 0.3
                reward += 0.2 if 2 <= distToCarrier <= 5 else 0.0
                # Anti-loop for off-ball agents too
                if self.prevPos is not None and myPos == self.prevPos:
                    reward -= 0.15

            else:
                # No teammate has ball — pressure the ball
                distToBall = self._manhattan(myPos, ballPos)
                reward += (maxDist - distToBall) / maxDist * 0.2
                if self.prevPos is not None and myPos == self.prevPos:
                    reward -= 0.15

        self.prevPos = myPos
        return reward

    def _update_beliefs(self):
        """Refresh local observations of opponents, ball location, and teammate."""
        step_count = int(getattr(self.model, "steps", 0))
        update_freq = max(1, int(self.model.scenario.belief_update_freq))
        if step_count % update_freq != 0 and self.beliefs["ball_position"] is not None:
            return
        opponents = self._players(same_team=False)
        teammates = [a for a in self._players(same_team=True) if a is not self]
        ball_pos = self._ball_position()
        teammate_pos = (
            teammates[0].cell.coordinate
            if teammates and teammates[0].cell is not None else None
        )
        
        self.beliefs.update({
            "opponent_positions": [o.cell.coordinate for o in opponents if o.cell is not None],
            "ball_position": ball_pos,
            "teammate_position": teammate_pos,
        })


class Goal(FixedAgent):
    """Fixed goal marker."""

    def __init__(self, model, team: str, cell):
        """Create a goal marker bound to a team and fixed grid cell."""
        super().__init__(model)
        self.team = team
        self.cell = cell

    def allows_score_by(self, attacker_team: str) -> bool:
        """Return whether the attacker team is allowed to score on this goal."""
        return str(attacker_team) != str(self.team)


class Ball(Agent):
    """Ball tracked as coordinate position."""

    def __init__(self, model, position=None):
        """Initialize the ball with optional loose position and no carrier."""
        super().__init__(model)
        self.carrier = None
        self.position = position

    @property
    def is_loose(self):
        """Return True when the ball is not controlled by any player."""
        return self.carrier is None

    def place_loose(self, position):
        """Drop the ball at a coordinate and clear any carrier."""
        self.carrier = None
        self.position = position

    def step(self):
        """Update ball position to follow its carrier each model tick."""
        if self.carrier is not None and self.carrier.cell is not None:
            self.position = self.carrier.cell.coordinate