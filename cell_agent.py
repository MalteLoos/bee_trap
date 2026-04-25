from mesa.discrete_space import FixedAgent

class FoodCell(FixedAgent):
    
    def __init__(self, model, cell, is_nest=False, food_amount=0.0):
        super().__init__(model)
        self.cell = cell
        self.is_nest = is_nest
        self.food_amount = food_amount

    def has_food(self):
        return self.food_amount > 0.0

    def consume_food(self, amount=1.0):
        consumed = min(self.food_amount, amount)
        self.food_amount -= consumed
        return consumed
