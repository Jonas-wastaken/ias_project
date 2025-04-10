"""This module contains:
- LightAgent class: Represents a traffic light in the traffic grid, determining the flow of traffic."""

import mesa
import random
import numpy as np
from dataclasses import dataclass, field
import polars as pl
import pyoptinterface as poi
from pyoptinterface import highs, gurobi
from networkx import closeness_centrality
from pathlib import Path
from abc import ABC, abstractmethod

from graph import Graph
from car import CarAgent
from data import SimData


class LightAgent(mesa.Agent):
    """Agent, which represents a traffic light in the traffic gird. It has a fixed position on an intersection and decides which cars are allowed to move (only cars from one direction/edge can move at the same time).

    Inherits from mesa.Agent.

    Attributes:
        position (str): The ID of the node, where the agent is currently located.
        waiting_cars (dict): A nested dictionary of cars waiting at the intersection. (outer dict: cars; inner dict: last_intersection, local_waiting_time)
        default_switching_cooldown (int): The default number of steps the agent waits before changing the open lane again.
        current_switching_cooldown (int): The current number of steps the agent has to wait before changing the open lane again.
        neighbor_lights (list): A list of the neighboring lights of the agent.
        open_lane (str): The ID of the edge from where cars are allowed to cross the intersection.
        centrality (float): Closeness centrality measured as the reciprocal of the average shortest path distance to u over all n-1 reachable nodes using networkx.closeness_centrality().


    ## Methods:
        **set_position(self, position: str) -> None**:
            Sets the position of the agent to the given node ID.
        **update_waiting_cars(self) -> None**:
            Updates the details of the cars waiting at the intersection (waiting_cars).
        **change_open_lane (self) -> None**:
            Changes from where cars are allowed to cross the intersection.
        **rotate_in_open_lane_cycle(self) -> None**:
            Rotates the open lane to the next neighbor light in the cycle.
        **optimize_open_lane(self) -> str**:
            Decides which lane should be open based on the number of waiting cars.
        **get_num_connections(self, grid: Graph) -> int**:
            Gets the number of connected intersections.
        **get_avg_distance(self, grid: Graph) -> float**:
            Gets the average distance to connected intersections.
        **get_is_entrypoint(self, grid: Graph) -> bool**:
            Checks if intersection is connected to a border.
        **optimize_open_lane(self) -> str**:
            Decides which lane should be open based on the number of waiting cars.
        **get_num_connections(self, grid: Graph) -> int**:
            Gets the number of connected intersections.
        **get_avg_distance(self, grid: Graph) -> float**:
            Gets the average distance to connected intersections.
        **get_is_entrypoint(self, grid: Graph) -> bool**:
            Checks if intersection is connected to a border.

        **Coming soon**:
        - estimate_coming_cars?
    """

    def __init__(self, model: mesa.Model, **kwargs):
        """Initializes a new LightAgent. The agent is placed by the model on an intersection.

        Args:
            model (mesa.Model): The model instance in which the agent lives.
        """
        super().__init__(model)
        self.position = kwargs.get("position", None)
        self.neighbor_lights = self.get_connected_intersections(grid=self.model.grid)

        self.traffic = TrafficData()
        self.lanes = Lanes().construct(self)

        self.default_switching_cooldown = 5
        self.current_switching_cooldown = self.default_switching_cooldown
        self.open_lane = self.neighbor_lights[
            random.randint(0, len(self.neighbor_lights) - 1)
        ]  # Randomly select a neighbor light as the open lane

    def step(self, optimization_type: str, steps: int) -> None:
        """Actions each LightAgent takes each step.

        - Checks if it is blocked by cooldown
            - If not, it opens the best lane, determined by optimization technique
            - If blocked, cooldown is decremented by 1

        Args:
            optimization_type (str): Type of optimization to use
            steps (int): Internal step counter of TrafficModel instance
        """
        if self.current_switching_cooldown <= 0:
            if optimization_type == "none":
                self.change_open_lane(self.rotate_in_open_lane_cycle())
            elif optimization_type == "simple":
                self.change_open_lane(SimpleOptimizer(light=self).get_optimal_lane())
            elif optimization_type == "advanced":
                self.change_open_lane(
                    AdvancedOptimizer(light=self, mode="base").get_optimal_lane()
                )
            elif optimization_type == "advanced_ml":
                self.change_open_lane(
                    AdvancedOptimizer(light=self, mode="ml").get_optimal_lane()
                )
        else:
            self.current_switching_cooldown -= 1

        for lane in self.neighbor_lights:
            self.traffic.update_data(light=self, steps=steps, lane=lane)

    def set_position(self, position: str) -> None:
        """Sets the position of the agent to the given node ID.

        Args:
            position (str): The ID of the node, where the agent is currently located.
        """
        self.position = position

    def change_open_lane(self, lane: str) -> None:
        """Changes from where cars are allowed to cross the intersection, if the current switching cooldown allows it.

        Args:
            lane (str): The ID of the edge from where cars are allowed to cross the intersection.


        Raises:
            LightCooldown: If the current switching cooldown does not allow changing the open lane.
        """
        if self.current_switching_cooldown > 0:
            raise LightCooldown(
                "The current switching cooldown does not allow changing the open lane."
            )

        if self.open_lane != lane:
            self.open_lane = lane
            self.current_switching_cooldown = self.default_switching_cooldown

    def rotate_in_open_lane_cycle(self) -> str:
        """Rotates the open lane to the next neighbor light in the cycle."""
        current_index = self.neighbor_lights.index(self.open_lane)
        next_index = (current_index + 1) % len(self.neighbor_lights)

        return self.neighbor_lights[next_index]

    def get_num_connections(self, grid: Graph) -> int:
        """Gets the number of connected intersections.

        Args:
            grid (Graph): Graph instance

        Returns:
            int: Number of connected intersections
        """
        num_connections = len(
            [
                node
                for node in grid.neighbors(self.position)
                if node.startswith("intersection")
            ]
        )

        return num_connections

    def get_avg_distance(self, grid: Graph) -> float:
        """Gets the average distance to connected intersections.

        Args:
            grid (Graph): Graph instance

        Returns:
            float: Average distance to connected intersections
        """
        neighbors = [
            node
            for node in grid.neighbors(self.position)
            if node.startswith("intersection")
        ]

        distances = np.array(
            [
                grid.get_edge_data(u=self.position, v=neighbor)["weight"]
                for neighbor in neighbors
            ]
        )

        avg_distance = np.mean(distances)

        return avg_distance

    def is_entrypoint(self, grid: Graph) -> bool:
        """Checks if intersection is connected to a border.

        Args:
            grid (Graph): Graph instance

        Returns:
            bool: True if intersection is connected to a border
        """
        for node in grid.neighbors(self.position):
            if node.startswith("border"):
                return True

        return False

    def get_num_arrivals(self) -> int:
        """Gets the number of cars arriving at the light

        Returns:
            int: Number of cars arriving at the light
        """
        num_arrivals = 0
        for car in self.model.get_agents_by_type("CarAgent"):
            car: CarAgent
            if car.position == self.position and car.wait_times.is_arrival(
                car=car, light=self
            ):
                num_arrivals += 1

        return num_arrivals

    def get_num_cars(self, lane: str) -> int:
        """Gets the number of cars currently at the LightAgent instance

        Args:
            lane (str): Lane of LightAgent

        Returns:
            int: Number of cars currently at the LightAgent instance
        """
        num_cars = 0
        for car in self.model.get_agents_by_type("CarAgent"):
            car: CarAgent
            if (
                car.position == self.position
                and car.check_if_car_at_light()
                and self.model.get_last_intersection_of_car(car.unique_id) == lane
            ):
                num_cars += 1

        return num_cars

    def get_num_cars_per_lane(self) -> dict:
        """Gets the number of cars per lane currently at the LightAgent instance.

        Returns:
            dict: Number of cars per lane
        """
        num_cars_per_lane = {lane: 0 for lane in self.neighbor_lights}
        for car in self.model.get_agents_by_type("CarAgent"):
            car: CarAgent
            if car.position == self.position and car.waiting:
                num_cars_per_lane[
                    self.model.get_last_intersection_of_car(car.unique_id)
                ] += 1

        return num_cars_per_lane

    def get_centrality(self, grid: Graph) -> float:
        """Gets the centrality of the intersection a light is placed on.

        - Measured as the reciprocal of the average shortest path distance to *u* over all *n-1* reachable nodes
        - Uses networkx.closeness_centrality()

        Args:
            grid (Graph): Graph instance

        Returns:
            float: _description_
        """
        centrality = closeness_centrality(G=grid, u=self.position, distance="weight")

        return centrality

    def get_connected_intersections(self, grid: Graph) -> list[str]:
        """Gets IDs of connected intersection nodes.

        Args:
            grid (Graph): Graph instance

        Returns:
            list[str]: List of connected node ids
        """
        connected_intersections = [
            node
            for node in grid.neighbors(self.position)
            if node.startswith("intersection")
        ]

        return connected_intersections


class LightCooldown(Exception):
    """Exception raised when the switching cooldown of a light is preventing an open lane switch."""

    def __init__(self, message: str):
        """Initializes LightCooldown exception.

        Args:
            message (str): The message to be displayed when the exception is raised.
        """
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"


class Optimizer(ABC):
    """Abstract base class for all optimizers"""

    @abstractmethod
    def __init__(self, light: LightAgent):
        """Calculates optimal lane to open

        Args:
            light (LightAgent): LightAgent instance
        """
        pass

    @abstractmethod
    def get_cars_at_light(self) -> dict:
        """Gets number of CarAgents per lane of LightAgent instance

        Returns:
            dict: Number of CarAgents per lane of LightAgent instance
        """
        pass

    @abstractmethod
    def get_dec_vars(self) -> list[str]:
        """Gets all lanes of a LightAgent

        Returns:
            list[str]: List of all connected lanes from this intersection
        """
        pass

    @abstractmethod
    def init_model(self) -> None:
        """Initializes and builds the underlying optimization model

        - Silences model output
        - Adds decision variables
        - Adds constraints
        - Defines objective
        - Calculates optimal decision
        """
        pass

    @abstractmethod
    def get_optimal_lane(self) -> str:
        """Retrieves the result of the optimizer

        Returns:
            str: Optimal lane to open
        """
        pass


class SimpleOptimizer(Optimizer):
    """Simple optimizer, calculating the best lane to open without considering time constraints or future arrivals"""

    def __init__(self, light: LightAgent):
        """Calculates optimal lane to open

        Args:
            light (LightAgent): LightAgent instance
        """
        self.light = light
        self.cars_at_light = self.get_cars_at_light()
        self.init_model()

    def get_cars_at_light(self) -> dict:
        """Gets number of CarAgents per lane of LightAgent instance

        Returns:
            dict: Number of CarAgents per lane of LightAgent instance
        """
        cars_at_light = self.light.model.get_cars_per_lane_of_light(
            self.light.position, 0
        )

        return cars_at_light

    def get_dec_vars(self) -> list[str]:
        """Gets all lanes of a LightAgent

        Returns:
            list[str]: List of all connected lanes from this intersection
        """
        return self.light.neighbor_lights

    def init_model(self) -> None:
        """Initializes and builds the underlying optimization model

        - Silences model output
        - Adds decision variables
            - Binary decision per lane -> 1: open lane 0: don't open lane
        - Adds constraints
            - Only one lane can be open at a time
        - Defines objective
            - Sum of cars able to cross intersection
        - Calculates optimal decision
        """
        self.model = highs.Model()
        self.model.set_model_attribute(poi.ModelAttribute.Silent, True)

        lanes = self.model.add_variables(
            self.get_dec_vars(), domain=poi.VariableDomain.Binary
        )

        self.model.add_linear_constraint(poi.quicksum(lanes), poi.Eq, 1)

        objective = poi.quicksum(
            lanes[lane] * self.cars_at_light[lane] for lane in lanes
        )
        self.model.set_objective(objective, poi.ObjectiveSense.Maximize)

        self.model.optimize()

    def get_optimal_lane(
        self,
    ) -> str:  # TODO: Glaube man kann die Decision eleganter retrieven
        """Retrieves the result of the optimizer

        Returns:
            str: Optimal lane to open
        """
        optimal_value = self.model.get_obj_value()
        optimal_lanes = [
            lane
            for lane in self.cars_at_light
            if optimal_value == self.cars_at_light[lane]
        ]
        if len(optimal_lanes) > 1:
            optimal_lane = random.choice(
                optimal_lanes
            )  # Randomly select one of the optimal lanes, if there are multiple (TODO: chose the one where the cars have waited the longest)
        else:
            optimal_lane = optimal_lanes[0]
        self.light.model.update_lights_decision_log(
            self.light, self.cars_at_light, optimal_lane, self.light.model.steps
        )

        return optimal_lane


class AdvancedOptimizer(Optimizer):
    def __init__(self, light: LightAgent, mode: str = "base"):
        """Calculates optimal lane to open

        Args:
            light (LightAgent): LightAgent instance
        """
        self.light = light
        self.mode = mode
        self.time = range(-1, self.light.default_switching_cooldown + 1)
        self.cars_at_light = self.get_cars_at_light()
        self.init_model()

    def get_cars_at_light(self):
        if self.mode == "base":
            return self._request_cars_at_light()
        elif self.mode == "ml":
            return self._predict_cars_at_light()
        else:
            raise ValueError(
                f"mode must be one of ['base'|'ml']. Got {self.mode} instead"
            )

    def get_dec_vars(self) -> tuple[list[int], list[str]]:
        """Gets all lanes of a LightAgent

        Returns:
            tuple[list[int], list[str]: Tuple of time range and connected lanes
        """
        return self.time, self.light.neighbor_lights

    def init_model(self):
        """Initializes and builds the underlying optimization model

        - Silences model output
        - Adds decision variables
            - Binary decision per lane -> 1: open lane 0: don't open lane
        - Adds constraints
            - Only one lane can be open at a time
            - The LightAgent can only open a new lane after a cooldown
        - Defines objective
            - Sum of cars able to cross intersection
        - Calculates optimal decision
        """
        self.model = gurobi.Model(self._init_env())
        self.model.set_model_attribute(poi.ModelAttribute.Silent, True)

        time, connected_lanes = self.get_dec_vars()
        self.lanes = self.model.add_variables(
            time, connected_lanes, domain=poi.VariableDomain.Binary
        )
        for lane in self.get_dec_vars()[1]:
            self.lanes[-1, lane] = 1 if lane == self.light.open_lane else 0

        for tick in time[1:]:
            self.model.add_linear_constraint(
                poi.quicksum(self.lanes[tick, lane] for lane in connected_lanes),
                poi.Eq,
                1,
            )

        for lane in connected_lanes:
            self.model.add_quadratic_constraint(
                poi.quicksum(
                    (
                        (self.lanes[tick - 1, lane] - self.lanes[tick, lane])
                        * (self.lanes[tick - 1, lane] - self.lanes[tick, lane])
                    )
                    for tick in time[1:]
                ),
                poi.Leq,
                1.0,
            )

        objective = poi.quicksum(
            poi.quicksum(
                self.lanes[tick, lane] * self.cars_at_light[tick][lane]
                for lane in connected_lanes
            )
            for tick in time[1:]
        )
        self.model.set_objective(objective, poi.ObjectiveSense.Maximize)

        self.model.optimize()

    def get_optimal_lane(self):
        for lane in self.light.neighbor_lights:
            if self.model.get_value(self.lanes[0, lane]) > 0.1:
                return lane

    def _init_env(self) -> gurobi.Env:
        """Initializes the Gurobi environment with the provided secrets.

        Returns:
            gurobi.Env: Object holding gurobi environment variables
        """
        secrets = self._load_secrets()
        env = gurobi.Env(empty=True)
        env.set_raw_parameter("WLSACCESSID", secrets.get("WLSACCESSID"))
        env.set_raw_parameter("WLSSECRET", secrets.get("WLSSECRET"))
        env.set_raw_parameter("LICENSEID", secrets.get("LICENSEID"))
        env.set_raw_parameter("OutputFlag", 0)
        env.start()

        return env

    def _load_secrets(self) -> dict:
        """Loads Gurobi secrets from the gurobi.lic file.

        - File must be located in a .secrets folder in the working directory

        Returns:
            dict: Gurobi Secrets
        """
        license_path = Path.joinpath(Path.cwd(), Path(".secrets/"), Path("gurobi.lic"))
        secrets = {}
        with open(license_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                try:
                    key, value = line.split("=", 1)
                    secrets[key.strip()] = value.strip()
                except ValueError:
                    pass

        return secrets

    def _request_cars_at_light(self) -> dict:
        """Gets the number of cars at the LightAgent instance over the given time range

        Returns:
            dict: Dictionary holding the number of cars at the LightAgent instance over the given time range
        """
        cars_at_light = {tick: {} for tick in self.time[1:]}
        for tick in self.time[1:]:
            cars_at_light[tick] = self.light.model.get_cars_per_lane_of_light(
                self.light.position, tick
            )

        return cars_at_light

    def _predict_cars_at_light(self) -> dict:
        """Predicts the number of cars at the LightAgent instance over the given time range

        Returns:
            dict: Dictionary holding the number of cars at the LightAgent instance over the given time range
        """
        cars_at_light = {tick: {} for tick in self.time[1:]}
        for tick in self.time[1:]:
            cars_per_lane = {neighbor: {} for neighbor in self.light.neighbor_lights}
            for neighbor in self.light.neighbor_lights:
                model_time = 200 - (self.light.model.steps % 200)
                centrality = self.light.get_centrality(grid=self.light.model.grid)
                is_entrypoint = self.light.is_entrypoint(grid=self.light.model.grid)
                distance = self.light.lanes.get_distance(lane=neighbor)

                cars_per_lane[neighbor] = self.light.model.regressor.predict(
                    model_time, centrality, is_entrypoint, distance
                )

            cars_at_light[tick] = cars_per_lane

        return cars_at_light


@dataclass
class Lanes:
    data: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={"Lane": pl.String, "Distance": pl.Int16}, strict=False
        )
    )

    def construct(self, light: LightAgent):
        self.data.extend(
            other=pl.DataFrame(
                data=[
                    {
                        "Lane": lane,
                        "Distance": light.model.grid.get_edge_data(
                            light.position, lane
                        )["weight"],
                    }
                    for lane in light.neighbor_lights
                ],
                schema={"Lane": pl.String, "Distance": pl.Int16},
                strict=False,
            )
        )

        return self

    def get_distance(self, lane: str) -> int:
        distance = (
            self.data.filter(pl.col("Lane") == lane).select(pl.col("Distance")).item()
        )

        return distance


@dataclass
class TrafficData(SimData):
    """Holds the number of cars at a LightAgent instance at each step"""

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Constructs the data schema"""
        self.data = pl.DataFrame(
            schema={
                "Step": pl.Int32,
                "Light_ID": pl.Int16,
                "Time": pl.Int16,
                "Lane": pl.String,
                "Num_Cars": pl.Int16,
            },
            strict=False,
        )

    def update_data(self, light: LightAgent, steps: int, lane: str) -> None:
        """Updates the data

        Args:
            light (LightAgent): LightAgent instance
            steps (int): Internal step counter of TrafficModel instance
            lane (str): Lane of intersection
        """
        self.data.vstack(
            other=pl.DataFrame(
                data={
                    "Step": steps,
                    "Light_ID": light.unique_id,
                    "Time": 200 - (steps % 200),
                    "Lane": lane,
                    "Num_Cars": light.get_num_cars(lane),
                },
                schema={
                    "Step": pl.Int32,
                    "Light_ID": pl.Int16,
                    "Time": pl.Int16,
                    "Lane": pl.String,
                    "Num_Cars": pl.Int16,
                },
                strict=False,
            ),
            in_place=True,
        )

    def get_data(self) -> pl.DataFrame:
        """Returns the data

        Returns:
            pl.DataFrame: Data
        """
        return self.data
