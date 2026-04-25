from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from model import ForagingModel
from cell_agent import FoodCell

def _rgb_to_hex(r, g, b):
    r_int = int(max(0, min(255, r * 255)))
    g_int = int(max(0, min(255, g * 255)))
    b_int = int(max(0, min(255, b * 255)))
    return "#{:02x}{:02x}{:02x}".format(r_int, g_int, b_int)

def agent_portrayal(agent):
    if agent is None:
        return

    if isinstance(agent, CreatureAgent):
        t_frac = min(1.0, (agent.temperature - agent.model.temp_safe) /
                     (agent.model.temp_crit - agent.model.temp_safe))
        color = _rgb_to_hex(1.0, 1.0 - t_frac, 0.0)
        return AgentPortrayalStyle(marker="o", size=80, zorder=2, color=color)

    if isinstance(agent, FoodCell):
        if agent.is_nest:
            color = "#4682b4"  # steelblue
        elif agent.has_food():
            intensity = agent.food_amount / 10.0
            color = _rgb_to_hex(0.0, 0.4 + 0.6 * intensity, 0.0)
        else:
            color = "#e8e8e8"
        return AgentPortrayalStyle(marker="s", size=200, zorder=0, color=color)


model_params = {
    "width": {
        "type": "SliderInt", "value": 60, "label": "Grid width",
        "min": 20, "max": 100, "step": 10,
    },
    "height": {
        "type": "SliderInt", "value": 60, "label": "Grid height",
        "min": 20, "max": 100, "step": 10,
    },
    "num_creatures": {
        "type": "SliderInt", "value": 50, "label": "Creatures (40–60)",
        "min": 40, "max": 60, "step": 1,
    },
    "num_food_clusters": {
        "type": "SliderInt", "value": 12, "label": "Food clusters (10–12)",
        "min": 10, "max": 12, "step": 1,
    },
    "food_coverage": {
        "type": "SliderFloat", "value": 0.15, "label": "Food coverage (10–15%)",
        "min": 0.10, "max": 0.15, "step": 0.01,
    },
}

model = ForagingModel()

survival_plot = make_plot_component(
    {"alive_creatures": "red", "food_remaining": "green"},
    backend="matplotlib",
)

renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.draw_agents()
renderer.render()

page = SolaraViz(
    model,
    renderer,
    components=[survival_plot],
    model_params=model_params,
    name="Group Project — ABC Discrete",
)
page
