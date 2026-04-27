from mesa.agent import Agent

from cell_agent import FoodCell


class CreatureAgent(Agent):
    def __init__(self, model, start_cell):
        super().__init__(model)
        self.cell = start_cell
        self.cell.add_agent(self)

        self.energy = model.energy_max
        self.temperature = model.temp_safe
        self.carrying_food = 0.0
        self.target_food_pos = None
        self.last_food_pos = None
        self.role = "scout"
        self.unsuccessful_steps = 0
        self.max_speed = 3
        self.perception_radius = 1
        self.carry_capacity = 2.0
        self._is_recovering = False

    def remove(self):
        if getattr(self, "cell", None) is not None and self in self.cell.agents:
            self.cell.remove_agent(self)
        super().remove()

    def step(self):
        if not self._is_alive():
            self._die()
            return

        self._update_role()
        moved_steps = self._act()
        self._metabolize(moved_steps)
        self._interact_with_food()

        if not self._is_alive():
            self._die()

    def _act(self):
        if self.carrying_food > 0.0 or self._needs_nest_recovery():
            return self._move_towards(self.model.nest_pos)

        visible_food = self._visible_food_cells()
        if visible_food:
            best_food = max(visible_food, key=lambda cell: cell.food_amount)
            self.target_food_pos = best_food.cell.coordinate
            self.last_food_pos = best_food.cell.coordinate
            self.role = "employed"
            self.unsuccessful_steps = 0
            return self._move_towards(best_food.cell.coordinate)

        if self.role == "employed" and self.target_food_pos is not None:
            return self._move_towards(self.target_food_pos)

        return self._random_walk()

    def _update_role(self):
        if self.model.is_in_nest(self.cell.coordinate):
            self._recover_energy_in_nest()

            if self.carrying_food > 0.0:
                self.model.nest_food_store += self.carrying_food
                self.carrying_food = 0.0
            self._try_pick_waggle_or_scout()

        if self.unsuccessful_steps >= self.model.abc_limit:
            self.target_food_pos = None
            self.role = "scout"
            self.unsuccessful_steps = 0
            if self.model.is_in_nest(self.cell.coordinate):
                self._try_pick_waggle_or_scout()

    def _try_pick_waggle_or_scout(self):
        if self.model.waggle_board:
            chosen = self.model.select_food_source()
            if chosen is not None:
                self.target_food_pos = chosen
                self.role = "employed"
                return
        self.target_food_pos = None
        self.role = "scout"

    def _metabolize(self, moved_steps):
        self.energy -= self.model.energy_decay + moved_steps * self.model.energy_move_cost

        if self.model.is_in_nest(self.cell.coordinate):
            self.temperature = max(
                self.model.temp_safe,
                self.temperature - self.model.temp_cool_rate,
            )
        else:
            self.temperature += self.model.temp_heat_rate

    def _interact_with_food(self):
        if self.carrying_food > 0.0:
            return

        food_cell = self._food_cell_at(self.cell.coordinate)
        if food_cell is None or not food_cell.has_food():
            # Only count as a failure when the bee has actually arrived at its
            # target and found nothing — not during travel steps en-route.
            at_target = (
                self.target_food_pos is None
                or self.cell.coordinate == self.target_food_pos
            )
            if at_target:
                self.unsuccessful_steps += 1
            return

        richness = self._estimate_cluster_richness()
        energy_deficit = max(0.0, self.model.energy_max - self.energy)
        food_for_full_energy = energy_deficit / self.model.energy_food_gain
        bag_space = max(0.0, self.carry_capacity - self.carrying_food)

        requested = food_for_full_energy + bag_space
        if requested <= 0.0:
            self.unsuccessful_steps += 1
            return

        gathered = food_cell.consume_food(requested)
        for_self = min(gathered, food_for_full_energy)
        for_bag = max(0.0, gathered - for_self)

        self.energy = min(
            self.model.energy_max,
            self.energy + for_self * self.model.energy_food_gain,
        )
        self.carrying_food += for_bag
        self.last_food_pos = self.cell.coordinate
        self.target_food_pos = self.cell.coordinate
        self.role = "employed"
        self.unsuccessful_steps = 0

        if richness > 0.0:
            self.model.update_waggle(self.cell.coordinate, richness)
        if not food_cell.has_food():
            self.model.remove_waggle(self.cell.coordinate)
            # Before abandoning, check if a neighbouring cell in the cluster
            # still has food — if so, stay employed and shift to that cell.
            visible_food = self._visible_food_cells()
            if visible_food:
                best = max(visible_food, key=lambda c: c.food_amount)
                self.target_food_pos = best.cell.coordinate
            else:
                self.target_food_pos = None
                self._try_pick_waggle_or_scout()

    def _recover_energy_in_nest(self):
        if self.energy >= self.model.energy_max:
            return

        if self.model.nest_food_store <= 0.0:
            return

        deficit = self.model.energy_max - self.energy
        food_needed = deficit / self.model.energy_food_gain
        consumed_from_nest = min(food_needed, self.model.nest_food_store)
        self.model.nest_food_store -= consumed_from_nest
        self.energy = min(
            self.model.energy_max,
            self.energy + consumed_from_nest * self.model.energy_food_gain,
        )

    def _estimate_cluster_richness(self):
        nearby_food = self._visible_food_cells()
        return sum(cell.food_amount for cell in nearby_food)

    def _visible_food_cells(self):
        cells = [self.cell]
        if self.perception_radius > 0:
            cells.extend(list(self.cell.get_neighborhood(radius=self.perception_radius)))

        visible_food = []
        seen = set()
        for grid_cell in cells:
            coordinate = grid_cell.coordinate
            if coordinate in seen:
                continue
            seen.add(coordinate)
            food_cell = self._food_agent_in_cell(grid_cell)
            if food_cell is not None and food_cell.has_food():
                visible_food.append(food_cell)
        return visible_food

    def _food_cell_at(self, coordinate):
        for grid_cell in self.model.grid:
            if grid_cell.coordinate == coordinate:
                return self._food_agent_in_cell(grid_cell)
        return None

    @staticmethod
    def _food_agent_in_cell(grid_cell):
        for agent in grid_cell.agents:
            if isinstance(agent, FoodCell):
                return agent
        return None

    def _move_towards(self, target_pos):
        moved_steps = 0
        while moved_steps < self.max_speed and self.cell.coordinate != target_pos:
            next_cell = self._best_step_towards(target_pos)
            if next_cell is None:
                break
            self._move_to(next_cell)
            moved_steps += 1
        return moved_steps

    def _best_step_towards(self, target_pos):
        neighbors = list(self.cell.neighborhood)
        if not neighbors:
            return None

        current_distance = self._distance(self.cell.coordinate, target_pos)
        better_neighbors = [
            neighbor
            for neighbor in neighbors
            if self._distance(neighbor.coordinate, target_pos) < current_distance
        ]
        choices = better_neighbors or neighbors
        return min(choices, key=lambda neighbor: self._distance(neighbor.coordinate, target_pos))

    def _random_walk(self):
        moved_steps = 0
        while moved_steps < self.max_speed and self.cell.neighborhood:
            next_cell = self.random.choice(list(self.cell.neighborhood))
            self._move_to(next_cell)
            moved_steps += 1

            if self._visible_food_cells():
                break

        return moved_steps

    def _move_to(self, next_cell):
        self.cell.remove_agent(self)
        next_cell.add_agent(self)
        self.cell = next_cell

    def _needs_nest_recovery(self):
        energy_critical = self.energy <= 0.25 * self.model.energy_max
        temperature_critical = self.temperature >= self.model.temp_safe + 0.75 * (
            self.model.temp_crit - self.model.temp_safe
        )
        if energy_critical or temperature_critical:
            self._is_recovering = True
        elif self._is_recovering and self.temperature <= self.model.temp_safe and self.energy >= self.model.energy_max:
            self._is_recovering = False
        return self._is_recovering

    def _is_alive(self):
        return self.energy > 0.0 and self.temperature < self.model.temp_crit

    def _die(self):
        if self.target_food_pos is not None:
            self.model.remove_waggle(self.target_food_pos)
        self.remove()

    @staticmethod
    def _distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])