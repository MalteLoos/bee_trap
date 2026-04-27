from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from model import ForagingModel
from cell_agent import FoodCell
from agent import CreatureAgent


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


def dual_axis_post_process(ax):
    """Split alive creatures and food remaining onto separate y-axes."""
    line_by_label = {line.get_label(): line for line in ax.lines}
    alive_line = line_by_label.get("alive_creatures")
    food_line = line_by_label.get("food_remaining")
    if alive_line is None or food_line is None:
        return

    alive_x = alive_line.get_xdata()
    alive_y = alive_line.get_ydata()
    alive_color = alive_line.get_color()

    food_x = food_line.get_xdata()
    food_y = food_line.get_ydata()
    food_color = food_line.get_color()

    ax.cla()
    ax_right = ax.twinx()

    left_plot = ax.plot(alive_x, alive_y, color=alive_color, label="alive_creatures")[0]
    right_plot = ax_right.plot(
        food_x,
        food_y,
        color=food_color,
        label="food_remaining",
    )[0]

    ax.set_ylabel("alive_creatures", color=alive_color)
    ax.tick_params(axis="y", labelcolor=alive_color)

    ax_right.set_ylabel("food_remaining", color=food_color)
    ax_right.tick_params(axis="y", labelcolor=food_color)

    ax.grid(True, alpha=0.25)
    ax.legend([left_plot, right_plot], ["alive_creatures", "food_remaining"], loc="best")


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
    post_process=dual_axis_post_process,
)

renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.render()

page = SolaraViz(
    model,
    renderer,
    components=[survival_plot],
    model_params=model_params,
    name="Group Project — ABC Discrete",
)
page
