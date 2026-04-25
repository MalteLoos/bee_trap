from mesa import DataCollector, Model
from mesa.discrete_space import OrthogonalMooreGrid
from cell_agent import FoodCell
import numpy as np


class ForagingModel(Model):
    """
    ABC discrete
    requirements:
      - Mesa 3.5.1
      - Adjustable grid
      - Single nest at grid center
      - 40–60 creatures (default 50)
      - Food in 10–12 clusters covering 10–15% of environment
      - Food is finite and non-renewable
      - Simulation ends only when all creatures have died
      - Swarm-intelligence mechanism: ABC waggle dance (update/select/remove_waggle)

    default environment:
      - Grid: 60 × 60
      - Nest: center, radius 3
      - Creatures: 50
      - Food coverage: 15%
      - Food clusters: 12
    """

    #construction
    def __init__(
        self,
        #default environment
        width=60,
        height=60,
        num_creatures=50,
        num_food_clusters=12,
        food_coverage=0.15,
        nest_radius=3,
        
        #energy parameters
        energy_max=200.0,
        energy_decay=0.5,        #energy lost passively
        energy_move_cost=1.0,    #extra energy lost when moving
        energy_food_gain=20.0,   #energy restored per food unit

        #temperature parameters
        temp_safe=36.0,          #starting temp and safe threshold
        temp_crit=42.0,          #ending temp and death threshold
        temp_heat_rate=0.5,      #temp rise per step outside nest
        temp_cool_rate=1.0,      #temp drop per step inside nest
        
        #ABC limit: abandon after 'abc_limit'
        abc_limit=20,
        rng=None,
    ):
        super().__init__()
        if rng is not None:
            self.random = rng

        #grid (adjustable size)
        self.width = width
        self.height = height
        self.grid = OrthogonalMooreGrid([width, height], torus=False, capacity=100)

        #physics parameters — read by CreatureAgent in agent.py
        self.energy_max = energy_max
        self.energy_decay = energy_decay
        self.energy_move_cost = energy_move_cost
        self.energy_food_gain = energy_food_gain
        self.temp_safe = temp_safe
        self.temp_crit = temp_crit
        self.temp_heat_rate = temp_heat_rate
        self.temp_cool_rate = temp_cool_rate

        #ABC parameter
        self.abc_limit = abc_limit

        #single nest at center
        self.nest_pos = (width // 2, height // 2)
        self.nest_radius = nest_radius
        self.nest_cells = self._compute_nest_cells()

        #ABC waggle dance board: {(x, y): quality_score}
        #employed bees write here when they return; onlookers read it.
        self.waggle_board = {}

        #build environment and place bees
        self._init_cells(num_food_clusters, food_coverage)
        self._init_creatures(num_creatures)

        self.datacollector = DataCollector(
            model_reporters={
                "alive_creatures": lambda m: m.count_alive(),
                "food_remaining": lambda m: m.count_food(),
            }
        )
        self.datacollector.collect(self)

    #initialization
    def _compute_nest_cells(self):
        """Return the set of (x, y) coordinates that form the nest."""
        cx, cy = self.nest_pos
        r = self.nest_radius
        return {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        }

    def _init_cells(self, num_clusters, food_coverage):
        """
        Place FoodCell agents on every grid cell.

        Food distribution:
          - target cells  = floor(width * height * food_coverage)
          - split evenly across num_clusters
          - each cluster grows outward from a random center (outside the nest)
        """
        total_cells = self.width * self.height
        target = int(total_cells * food_coverage)
        per_cluster = max(1, target // num_clusters)

        #positions outside the nest
        outside = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.nest_cells
        ]

        #pick cluster centres random
        centre_idx = np.random.choice(len(outside), size=num_clusters, replace=False)
        centres = [outside[i] for i in centre_idx]

        #for each centre take the `per_cluster`
        food_set = set()
        for cx, cy in centres:
            candidates = sorted(
                [p for p in outside if p not in food_set],
                key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
            )
            food_set.update(candidates[:per_cluster])

        #create one Food Source per grid cell
        for cell in self.grid:
            x, y = cell.coordinate
            is_nest = (x, y) in self.nest_cells
            food_amount = 10.0 if (x, y) in food_set else 0.0
            agent = FoodCell(self, cell, is_nest=is_nest, food_amount=food_amount)
            self.agents.add(agent)

    def _init_creatures(self, num_creatures):
        """Spawn creatures at random nest cells."""
        from agent import CreatureAgent

        nest_grid_cells = [
            cell for cell in self.grid
            if cell.coordinate in self.nest_cells
        ]
        for _ in range(num_creatures):
            start_cell = self.random.choice(nest_grid_cells)
            creature = CreatureAgent(self, start_cell)
            self.agents.add(creature)

    #ABC waggle dance interface (swarm-intelligence mechanism, §4.1)

    def update_waggle(self, food_pos, quality):
        """
        Employed bee deposits a food source signal at the nest.
        quality  — proportional to food_amount found at food_pos.
        """
        self.waggle_board[food_pos] = quality

    def remove_waggle(self, food_pos):
        """Remove an exhausted or abandoned food source from the board."""
        self.waggle_board.pop(food_pos, None)

    def select_food_source(self):
        """
        Onlooker bee selects a food source via roulette-wheel selection
        proportional to quality.  Returns None if the board is empty.
        """
        if not self.waggle_board:
            return None
        positions = list(self.waggle_board.keys())
        qualities = np.array([self.waggle_board[p] for p in positions], dtype=float)
        total = qualities.sum()
        if total == 0.0:
            return self.random.choice(positions)
        probs = qualities / total
        return positions[np.random.choice(len(positions), p=probs)]

    #queries for DataCollector and step logic
    def is_in_nest(self, pos):
        """Return True if grid coordinate `pos` is inside the nest."""
        return pos in self.nest_cells

    def count_alive(self):
        from agent import CreatureAgent
        return len(self.agents_by_type[CreatureAgent])

    def count_food(self):
        return sum(a.food_amount for a in self.agents_by_type[FoodCell])

    #steps (ends when all agent have died)
    def step(self):
        from agent import CreatureAgent
        self.agents_by_type[CreatureAgent].shuffle().do("step")
        self.datacollector.collect(self)
        if self.count_alive() == 0:
            self.running = False
