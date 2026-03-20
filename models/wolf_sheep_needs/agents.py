from mesa.discrete_space import CellAgent, FixedAgent
import numpy as np

class Animal(CellAgent):
    """The base animal class."""

    def __init__(
        self, model, energy=8, p_reproduce=0.04, energy_from_food=4, cell=None
    ):
        """Initialize an animal.

        Args:
            model: Model instance
            energy: Starting amount of energy
            p_reproduce: Probability of reproduction (asexual)
            energy_from_food: Energy obtained from 1 unit of food
            cell: Cell in which the animal starts
        """
        super().__init__(model)
        self.energy = energy
        self.p_reproduce = p_reproduce
        self.energy_from_food = energy_from_food
        self.cell = cell

    def spawn_offspring(self):
        """Create offspring by splitting energy and creating new instance."""
        self.energy /= 2
        self.__class__(
            model = self.model,
            energy = self.energy,
            p_reproduce = self.p_reproduce,
            energy_from_food = self.energy_from_food,
            cell=self.cell,
        )

    def feed(self):
        """Abstract method to be implemented by subclasses."""

    def step(self):
        """Execute one step of the animal's behavior. Abstract function
        as needs are different for each animal"""

class Sheep(Animal):
    """A sheep that walks around, reproduces (asexually) and gets eaten."""

    def __init__(self, model, energy=8, p_reproduce=0.04, energy_from_food=4,
                 fear_decay=0.8, cell=None):
        super().__init__(model, energy, p_reproduce, energy_from_food, cell)
        
        # decay weights (0 to 1)
        self.fear_decay = fear_decay

        # custom needs (initial values do not matter)
        self.fear = 0
        self.hunger = 1

    def spawn_offspring(self):
        # implementation different due to constructor
        self.energy /= 2
        self.__class__(
            model=self.model,
            energy=self.energy,
            p_reproduce=self.p_reproduce,
            energy_from_food=self.energy_from_food,
            fear_decay=self.fear_decay,
            cell=self.cell,
        )

    def wolf_pressure(self, cell, radius=3):
        """Count the number of wolves in the given radius"""
        wolf_nearby = 0 # no of wolves in proximity radius
        for cell in cell.get_neighborhood(radius=radius):
            for obj in cell.agents:
                if isinstance(obj, Wolf):
                    wolf_nearby += 1
        
        return wolf_nearby

    def has_grass(self, cell):
        """Return 1 if the cell has grass else 0"""
        return any(
            isinstance(obj, GrassPatch) and obj.fully_grown
            for obj in cell.agents
        )

    def feed(self):
        """If possible, eat grass at current location."""
        grass_patch = next(
            obj for obj in self.cell.agents if isinstance(obj, GrassPatch)
        )
        if grass_patch.fully_grown:
            self.energy += self.energy_from_food
            grass_patch.get_eaten()

    def step(self):
        # needs estimation
        fear_signal = 1 - np.exp(-0.35 * self.wolf_pressure(self.cell))
        self.fear = self.fear_decay * self.fear + (1 - self.fear_decay) * fear_signal
        
        # setting max energy as twice energy_from_food
        self.hunger = np.clip(1 - self.energy / (2 * self.energy_from_food), 0, 1)

        self.move()
        if self.hunger > self.fear:
            self.feed()

        self.energy -=1

        # death and reproduction stays the same
        if self.energy < 0:
            self.model.sheep_starved += 1
            self.remove()
        elif self.random.random() < self.p_reproduce:
            self.spawn_offspring()


    def move(self):
        """Move towards a cell where there isn't a wolf, and preferably with grown grass."""
        # needs to be replaced to not move randomly (but according to needs)
        cells_without_wolves = []
        cells_with_grass = []

        for cell in self.cell.neighborhood:
            has_wolf = False
            has_grass = False

            for obj in cell.agents:
                # If there's a wolf, we can early exit
                if isinstance(obj, Wolf):
                    has_wolf = True
                    break
                elif isinstance(obj, GrassPatch) and obj.fully_grown:
                    has_grass = True

            # Prefer cells without wolves
            if not has_wolf:
                cells_without_wolves.append(cell)

                # Among safe cells, pick those with grown grass
                if has_grass:
                    cells_with_grass.append(cell)

        # If all surrounding cells have wolves, stay put
        if len(cells_without_wolves) == 0:
            return
        
        # Move to a cell with grass if available, otherwise move to any safe cell
        target_cells = (
            cells_with_grass if len(cells_with_grass) > 0 else cells_without_wolves
        )

        scores = []
        for cell in target_cells:
            # scoring the qualities of target cells (cell, safety, food)
            safety = 1.0 / (1 + self.wolf_pressure(cell))
            food = 1 if self.has_grass(cell) else 0
            scores.append((cell, safety, food))

        if self.fear > self.hunger:
            # if the need for fear outscores hunger, then cells with safety prioritised
            max_safety = max(s for _, s, _ in scores)
            best_cells = [t for t in scores if t[1] == max_safety]
        else:
            # otherwise both food and hunger are considered
            w_sum = self.hunger + self.fear + 1e-9
            w_food = self.hunger / w_sum
            w_safe = self.fear / w_sum

            best_score = max(w_food * f + w_safe * s for _, s, f in scores)
            best_cells = [t for t in scores if (w_food * t[2] + w_safe * t[1]) == best_score]

        self.cell = self.random.choice([c for c,_, _, in best_cells])


class Wolf(Animal):
    """A wolf that walks around, reproduces (asexually) and eats sheep."""

    def feed(self):
        """If possible, eat a sheep at current location."""
        sheep = [obj for obj in self.cell.agents if isinstance(obj, Sheep)]
        if sheep:  # If there are any sheep present
            sheep_to_eat = self.random.choice(sheep)
            self.energy += self.energy_from_food
            self.model.sheep_eaten += 1
            sheep_to_eat.remove()

    def step(self):
        # Move to random neighboring cell
        self.move()

        self.energy -= 1

        # Try to feed
        self.feed()

        # Handle death and reproduction
        if self.energy < 0:
            self.remove()
        elif self.random.random() < self.p_reproduce:
            self.spawn_offspring()

    def move(self):
        """Move to a neighboring cell, preferably one with sheep."""
        cells_with_sheep = self.cell.neighborhood.select(
            lambda cell: any(isinstance(obj, Sheep) for obj in cell.agents)
        )
        target_cells = (
            cells_with_sheep if len(cells_with_sheep) > 0 else self.cell.neighborhood
        )
        self.cell = target_cells.select_random_cell()


class GrassPatch(FixedAgent):
    """A patch of grass that grows at a fixed rate and can be eaten by sheep."""

    def __init__(self, model, countdown, grass_regrowth_time, cell):
        """Create a new patch of grass.

        Args:
            model: Model instance
            countdown: Time until grass is fully grown again
            grass_regrowth_time: Time needed to regrow after being eaten
            cell: Cell to which this grass patch belongs
        """
        super().__init__(model)
        self.fully_grown = countdown == 0
        self.grass_regrowth_time = grass_regrowth_time
        self.cell = cell

        # Schedule initial growth if not fully grown
        if not self.fully_grown:
            self.model.schedule_event(self.regrow, after=countdown)

    def regrow(self):
        """Regrow the grass."""
        self.fully_grown = True

    def get_eaten(self):
        """Mark grass as eaten and schedule regrowth."""
        self.fully_grown = False
        self.model.schedule_event(self.regrow, after=self.grass_regrowth_time)