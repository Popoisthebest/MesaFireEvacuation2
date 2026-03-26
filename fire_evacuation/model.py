import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid
from mesa.time import RandomActivation

from .agent import Door, Fire, FireExit, Furniture, GuideSignal, Human, Smoke, Wall


class EvacuationController:
    """Centralized dynamic guidance system for balancing exits and recommending paths."""

    def __init__(self, model: "FireEvacuation"):
        self.model = model

    def assign_exit(self, agent: Human):
        return agent.attempt_exit_plan()

    def recommend_path(self, agent: Human):
        target = agent.planned_target[1]
        if target:
            return agent.get_path(self.model.graph, target)
        return []

    def rebalance_exits(self):
        if self.model.guidance_mode == "static":
            return

        alive_agents = [
            a
            for a in self.model.schedule.agents
            if isinstance(a, Human) and a.get_status() == Human.Status.ALIVE
        ]
        if not alive_agents:
            return

        for agent in alive_agents:
            if (
                self.model.schedule.steps % self.model.replan_interval == 0
                or not agent.planned_target[1]
            ):
                self.assign_exit(agent)


class FireEvacuation(Model):
    MIN_HEALTH = 0.75
    MAX_HEALTH = 1

    MIN_SPEED = 1
    MAX_SPEED = 2

    MIN_NERVOUSNESS = 1
    MAX_NERVOUSNESS = 10

    MIN_EXPERIENCE = 1
    MAX_EXPERIENCE = 10

    MIN_VISION = 1

    def __init__(
        self,
        floor_plan_file: str,
        human_count: int,
        collaboration_percentage: float,
        fire_probability: float,
        visualise_vision: bool,
        random_spawn: bool,
        save_plots: bool,
        guidance_mode: str = "dynamic",
        replan_interval: int = 3,
        use_prediction: bool = False,
        guidance_profile: str | None = None,
    ):
        with open(os.path.join("fire_evacuation/floorplans/", floor_plan_file), "rt") as f:
            floorplan = np.matrix([line.strip().split() for line in f.readlines()])

        floorplan = np.rot90(floorplan, 3)
        width, height = np.shape(floorplan)

        self.width = width
        self.height = height
        self.human_count = human_count
        self.collaboration_percentage = collaboration_percentage
        self.visualise_vision = visualise_vision
        self.fire_probability = fire_probability
        self.fire_started = False
        self.save_plots = save_plots
        if guidance_profile == "static":
            guidance_mode = "static"
            use_prediction = False
        elif guidance_profile == "dynamic_no_prediction":
            guidance_mode = "dynamic"
            use_prediction = False
        elif guidance_profile == "dynamic_prediction":
            guidance_mode = "dynamic"
            use_prediction = True

        self.guidance_mode = guidance_mode
        self.replan_interval = max(1, replan_interval)
        self.use_prediction = use_prediction

        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(height, width, torus=False)

        self.furniture: dict[Coordinate, Furniture] = {}
        self.fire_exits: dict[Coordinate, FireExit] = {}
        self.doors: dict[Coordinate, Door] = {}

        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        self.risk_map: dict[Coordinate, float] = {}
        self.congestion_map: dict[Coordinate, float] = {}
        self.exit_congestion_map: dict[Coordinate, int] = defaultdict(int)
        self.smoke_level_map: dict[Coordinate, float] = defaultdict(float)
        self.predicted_smoke_map: dict[Coordinate, float] = defaultdict(float)
        self.agent_capacity_per_cell = 3
        self.guide_signals: dict[Coordinate, GuideSignal] = {}

        self.lambda_risk = 4.0
        self.risk_alpha = 3.0
        self.risk_beta = 1.5
        self.risk_gamma = 2.0
        self.risk_delta = 1.0

        for (x, y), value in np.ndenumerate(floorplan):
            pos: Coordinate = (x, y)
            value = str(value)
            floor_object = None
            if value == "W":
                floor_object = Wall(pos, self)
            elif value == "E":
                floor_object = FireExit(pos, self)
                self.fire_exits[pos] = floor_object
                self.doors[pos] = floor_object
            elif value == "F":
                floor_object = Furniture(pos, self)
                self.furniture[pos] = floor_object
            elif value == "D":
                floor_object = Door(pos, self)
                self.doors[pos] = floor_object
            elif value == "S":
                self.spawn_pos_list.append(pos)

            if floor_object:
                self.grid.place_agent(floor_object, pos)
                self.schedule.add(floor_object)
                if isinstance(floor_object, Door) or isinstance(floor_object, FireExit):
                    signal = GuideSignal(pos, self)
                    self.grid.place_agent(signal, pos)
                    self.schedule.add(signal)
                    self.guide_signals[pos] = signal

        self.graph = nx.Graph()
        for agents, x, y in self.grid.coord_iter():
            pos = (x, y)
            if len(agents) == 0 or not any(not agent.traversable for agent in agents):
                neighbors_pos = self.grid.get_neighborhood(pos, moore=True, include_center=True, radius=1)
                for neighbor_pos in neighbors_pos:
                    if self.grid.is_cell_empty(neighbor_pos) or not any(
                        not agent.traversable for agent in self.grid.get_cell_list_contents(neighbor_pos)
                    ):
                        self.graph.add_edge(pos, neighbor_pos, weight=1.0)

        self.controller = EvacuationController(self)

        self.datacollector = DataCollector(
            {
                "Alive": lambda m: self.count_human_status(m, Human.Status.ALIVE),
                "Dead": lambda m: self.count_human_status(m, Human.Status.DEAD),
                "Escaped": lambda m: self.count_human_status(m, Human.Status.ESCAPED),
                "Incapacitated": lambda m: self.count_human_mobility(m, Human.Mobility.INCAPACITATED),
                "Normal": lambda m: self.count_human_mobility(m, Human.Mobility.NORMAL),
                "Panic": lambda m: self.count_human_mobility(m, Human.Mobility.PANIC),
                "Verbal Collaboration": lambda m: self.count_human_collaboration(m, Human.Action.VERBAL_SUPPORT),
                "Physical Collaboration": lambda m: self.count_human_collaboration(m, Human.Action.PHYSICAL_SUPPORT),
                "Morale Collaboration": lambda m: self.count_human_collaboration(m, Human.Action.MORALE_SUPPORT),
                "Avg Evacuation Time": lambda m: self.get_average_evacuation_time(),
                "Survival Rate": lambda m: self.get_survival_rate(),
                "Deaths": lambda m: self.count_human_status(m, Human.Status.DEAD),
                "Avg Smoke Exposure": lambda m: self.get_average_smoke_exposure(),
                "Avg Path Changes": lambda m: self.get_average_path_changes(),
                "Max Exit Congestion": lambda m: max(m.exit_congestion_map.values() or [0]),
            }
        )

        number_collaborators = int(round(self.human_count * (self.collaboration_percentage / 100)))

        for _ in range(0, self.human_count):
            pos = self.grid.find_empty() if self.random_spawn else np.random.choice(self.spawn_pos_list)
            if pos:
                health = np.random.randint(self.MIN_HEALTH * 100, self.MAX_HEALTH * 100) / 100
                speed = np.random.randint(self.MIN_SPEED, self.MAX_SPEED)

                collaborates = number_collaborators > 0
                if collaborates:
                    number_collaborators -= 1

                vision_distribution = [0.0058, 0.0365, 0.0424, 0.9153]
                vision = int(
                    np.random.choice(
                        np.arange(self.MIN_VISION, self.width + 1, (self.width / len(vision_distribution))),
                        p=vision_distribution,
                    )
                )

                nervousness_distribution = [0.025, 0.025, 0.1, 0.1, 0.1, 0.3, 0.2, 0.1, 0.025, 0.025]
                nervousness = int(
                    np.random.choice(range(self.MIN_NERVOUSNESS, self.MAX_NERVOUSNESS + 1), p=nervousness_distribution)
                )

                experience = np.random.randint(self.MIN_EXPERIENCE, self.MAX_EXPERIENCE)
                believes_alarm = np.random.choice([True, False], p=[0.9, 0.1])

                human = Human(
                    pos,
                    health=health,
                    speed=speed,
                    vision=vision,
                    collaborates=collaborates,
                    nervousness=nervousness,
                    experience=experience,
                    believes_alarm=believes_alarm,
                    model=self,
                )
                self.grid.place_agent(human, pos)
                self.schedule.add(human)

        self.update_congestion_map()
        self.update_risk_map()
        self.update_graph_weights()
        self.running = True

    def save_figures(self):
        DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        OUTPUT_DIR = DIR + "/output"

        results = self.datacollector.get_model_vars_dataframe()
        dpi = 100
        fig, axes = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi, nrows=1, ncols=3)

        status_results = results.loc[:, ["Alive", "Dead", "Escaped"]]
        status_plot = status_results.plot(ax=axes[0])
        status_plot.set_title("Human Status")
        status_plot.set_xlabel("Simulation Step")
        status_plot.set_ylabel("Count")

        mobility_results = results.loc[:, ["Incapacitated", "Normal", "Panic"]]
        mobility_plot = mobility_results.plot(ax=axes[1])
        mobility_plot.set_title("Human Mobility")
        mobility_plot.set_xlabel("Simulation Step")
        mobility_plot.set_ylabel("Count")

        collaboration_results = results.loc[:, ["Verbal Collaboration", "Physical Collaboration", "Morale Collaboration"]]
        collaboration_plot = collaboration_results.plot(ax=axes[2])
        collaboration_plot.set_title("Human Collaboration")
        collaboration_plot.set_xlabel("Simulation Step")
        collaboration_plot.set_ylabel("Successful Attempts")
        collaboration_plot.set_ylim(ymin=0)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.suptitle(
            "Percentage Collaborating: " + str(self.collaboration_percentage) + "%, Number of Human Agents: " + str(self.human_count),
            fontsize=16,
        )
        plt.savefig(OUTPUT_DIR + "/model_graphs/" + timestr + ".png")
        plt.close(fig)

    def start_fire(self):
        rand = np.random.random()
        if rand < self.fire_probability:
            fire_furniture: Furniture = np.random.choice(list(self.furniture.values()))
            pos = fire_furniture.pos
            fire = Fire(pos, self)
            self.grid.place_agent(fire, pos)
            self.schedule.add(fire)
            self.fire_started = True
            print(f"Fire started at position {pos}")

    def _iter_agents(self, agent_type):
        return [agent for agent in self.schedule.agents if isinstance(agent, agent_type)]

    def update_congestion_map(self):
        self.congestion_map = {}
        self.exit_congestion_map = defaultdict(int)

        for _, x, y in self.grid.coord_iter():
            pos = (x, y)
            humans = [a for a in self.grid.get_cell_list_contents(pos) if isinstance(a, Human) and a.get_status() == Human.Status.ALIVE]
            self.congestion_map[pos] = min(1.0, len(humans) / self.agent_capacity_per_cell)

        for exit_pos in self.fire_exits.keys():
            neighborhood = self.grid.get_neighborhood(exit_pos, moore=True, include_center=True, radius=1)
            queue_len = 0
            for pos in neighborhood:
                queue_len += len([a for a in self.grid.get_cell_list_contents(pos) if isinstance(a, Human) and a.get_status() == Human.Status.ALIVE])
            self.exit_congestion_map[exit_pos] = queue_len

    def predict_smoke_spread(self, k_steps: int = 2):
        smoke_positions = {
            agent.pos for agent in self._iter_agents(Smoke) if agent.pos is not None
        }
        predicted_levels = defaultdict(float)
        frontier = set(smoke_positions)

        for step in range(1, k_steps + 1):
            new_frontier = set()
            for pos in frontier:
                neighbors = self.grid.get_neighborhood(pos, moore=False, include_center=False, radius=1)
                for npos in neighbors:
                    contents = self.grid.get_cell_list_contents(npos)
                    if any(not a.spreads_smoke for a in contents):
                        continue
                    if npos not in smoke_positions:
                        predicted_levels[npos] += 1 / (step + 1)
                        new_frontier.add(npos)
            smoke_positions.update(new_frontier)
            frontier = new_frontier

        self.predicted_smoke_map = predicted_levels
        return predicted_levels

    def update_risk_map(self):
        self.smoke_level_map = defaultdict(float)
        for smoke in self._iter_agents(Smoke):
            self.smoke_level_map[smoke.pos] += 1.0

        predicted_smoke = self.predict_smoke_spread(2) if self.use_prediction else defaultdict(float)

        risk_map = {}
        for _, x, y in self.grid.coord_iter():
            pos = (x, y)
            neighborhood = self.grid.get_neighborhood(pos, moore=True, include_center=True, radius=2)
            fire_count = len([a for a in self.grid.get_cell_list_contents(neighborhood) if isinstance(a, Fire)])
            smoke_count = self.smoke_level_map[pos]
            congestion = self.congestion_map.get(pos, 0.0)
            visibility_penalty = 1.0 if smoke_count > 0 else 0.0

            fire_risk = min(1.0, fire_count / 6)
            smoke_risk = min(1.0, (smoke_count + predicted_smoke[pos]) / 3)

            risk = (
                self.risk_alpha * fire_risk
                + self.risk_beta * smoke_risk
                + self.risk_gamma * congestion
                + self.risk_delta * visibility_penalty
            )
            risk_map[pos] = risk

        self.risk_map = risk_map

    def get_cell_risk(self, pos: Coordinate):
        return self.risk_map.get(pos, 0.0)

    def update_graph_weights(self):
        for u, v in self.graph.edges():
            self.graph[u][v]["weight"] = 1 + self.lambda_risk * self.get_cell_risk(v)

    def step(self):
        """Advance the model by one step in hazard-first order."""
        if not self.fire_started:
            self.start_fire()

        # 1. update fire
        for fire in list(self._iter_agents(Fire)):
            fire.step()

        # 2. update smoke
        for smoke in list(self._iter_agents(Smoke)):
            smoke.step()

        # 3. update congestion
        self.update_congestion_map()

        # 4. update risk map
        self.update_risk_map()

        # 5. update graph weights
        self.update_graph_weights()

        # 6. update guidance system
        self.controller.rebalance_exits()
        self.update_guide_signals()

        # 7 + 8. agents replan and move
        for human in list(self._iter_agents(Human)):
            human.step()

        self.schedule.steps += 1
        self.schedule.time += 1

        self.datacollector.collect(self)

        if self.count_human_status(self, Human.Status.ALIVE) == 0:
            self.running = False
            if self.save_plots:
                self.save_figures()

    def update_guide_signals(self):
        direction_vectors = {
            (-1, 0): "←",
            (1, 0): "→",
            (0, -1): "↓",
            (0, 1): "↑",
            (-1, -1): "↙",
            (1, 1): "↗",
            (-1, 1): "↖",
            (1, -1): "↘",
            (0, 0): "•",
        }
        for pos, signal in self.guide_signals.items():
            best_exit = min(self.fire_exits.keys(), key=lambda exit_pos: self.get_cell_risk(exit_pos))
            dx = np.sign(best_exit[0] - pos[0])
            dy = np.sign(best_exit[1] - pos[1])
            signal.update_signal(direction_vectors.get((dx, dy), "•"), self.get_cell_risk(pos))

    def get_average_evacuation_time(self):
        escaped = [a for a in self._iter_agents(Human) if a.escaped]
        if not escaped:
            return 0
        return float(np.mean([a.evacuation_time for a in escaped if a.evacuation_time is not None] or [0]))

    def get_survival_rate(self):
        alive_or_escaped = self.count_human_status(self, Human.Status.ALIVE) + self.count_human_status(self, Human.Status.ESCAPED)
        return (alive_or_escaped / self.human_count) * 100 if self.human_count else 0

    def get_average_smoke_exposure(self):
        humans = self._iter_agents(Human)
        if not humans:
            return 0
        return float(np.mean([h.smoke_exposure for h in humans]))

    def get_average_path_changes(self):
        humans = self._iter_agents(Human)
        if not humans:
            return 0
        return float(np.mean([h.path_change_count for h in humans]))

    @staticmethod
    def count_human_collaboration(model, collaboration_type):
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human):
                if collaboration_type == Human.Action.VERBAL_SUPPORT:
                    count += agent.get_verbal_collaboration_count()
                elif collaboration_type == Human.Action.MORALE_SUPPORT:
                    count += agent.get_morale_collaboration_count()
                elif collaboration_type == Human.Action.PHYSICAL_SUPPORT:
                    count += agent.get_physical_collaboration_count()
        return count

    @staticmethod
    def count_human_status(model, status):
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.get_status() == status:
                count += 1
        return count

    @staticmethod
    def count_human_mobility(model, mobility):
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.get_mobility() == mobility:
                count += 1
        return count
