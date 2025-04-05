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

        self.arrivals = ArrivalsData()
        self.traffic = TrafficData()

        self.default_switching_cooldown = 5
        self.current_switching_cooldown = self.default_switching_cooldown
        self.open_lane = self.neighbor_lights[
            random.randint(0, len(self.neighbor_lights) - 1)
        ]  # Randomly select a neighbor light as the open lane
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
                self.change_open_lane(self.optimize_open_lane())
            elif optimization_type == "advanced":
                self.change_open_lane(self.optimize_open_lane_with_cooldown())
        else:
            self.current_switching_cooldown -= 1

            self.arrivals.update_data(light=self, steps=steps)
            self.traffic.update_data(light=self, steps=steps)

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

    def optimize_open_lane(self) -> str:
        """Decides which lane should be open based on the number of waiting cars."""
        opt_model = highs.Model()

        opt_model.set_model_attribute(poi.ModelAttribute.Silent, True)

        possible_lanes = self.neighbor_lights
        cars_at_light = self.model.get_cars_per_lane_of_light(self.position, 0)
        lanes = opt_model.add_variables(
            possible_lanes, domain=poi.VariableDomain.Binary
        )

        # Constraints
        opt_model.add_linear_constraint(poi.quicksum(lanes), poi.Eq, 1)

        # Objective
        objective = poi.quicksum(lanes[lane] * cars_at_light[lane] for lane in lanes)
        opt_model.set_objective(objective, poi.ObjectiveSense.Maximize)

        opt_model.optimize()

        # Decide which lane should be open
        optimal_value = opt_model.get_obj_value()
        optimal_lanes = [
            lane for lane in cars_at_light if optimal_value == cars_at_light[lane]
        ]

        if len(optimal_lanes) > 1:
            optimal_lane = random.choice(
                optimal_lanes
            )  # Randomly select one of the optimal lanes, if there are multiple (TODO: chose the one where the cars have waited the longest)
        else:
            optimal_lane = optimal_lanes[0]

        # Log result
        self.model.update_lights_decision_log(
            self, cars_at_light, optimal_lane, self.model.steps
        )

        return optimal_lane

    def optimize_open_lane_with_cooldown(self) -> str:
        """Decides which lane should be open based on the number of waiting cars, taking the light cooldown into account."""
        light_cooldown = self.default_switching_cooldown
        secrets = self._load_gurobi_secrets()
        env = self._initialize_gurobi_env(secrets)
        opt_model = gurobi.Model(env)
        opt_model.set_model_attribute(poi.ModelAttribute.Silent, True)

        possible_lanes = self.neighbor_lights
        time = range(-1, light_cooldown + 1)
        cars_at_light = self._get_cars_at_light_over_time(time)
        current_open_lane = self.open_lane

        lanes = self._initialize_lanes(
            opt_model, time, possible_lanes, current_open_lane
        )
        self._add_constraints(opt_model, lanes, time, possible_lanes)
        self._set_objective(opt_model, lanes, cars_at_light, time, possible_lanes)

        opt_model.optimize()
        return self._get_optimal_lane(opt_model, lanes, possible_lanes)

    def _load_gurobi_secrets(self) -> dict:
        """Loads Gurobi secrets from the license file."""
        license_path = Path.joinpath(Path(".secrets/"), Path("gurobi.lic"))
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

    def _initialize_gurobi_env(self, secrets: dict) -> gurobi.Env:
        """Initializes the Gurobi environment with the provided secrets."""
        env = gurobi.Env(empty=True)
        env.set_raw_parameter("WLSACCESSID", secrets.get("WLSACCESSID"))
        env.set_raw_parameter("WLSSECRET", secrets.get("WLSSECRET"))
        env.set_raw_parameter("LICENSEID", secrets.get("LICENSEID"))
        env.set_raw_parameter("OutputFlag", 0)
        env.start()
        return env

    def _get_cars_at_light_over_time(self, time: range) -> dict:
        """Gets the number of cars at the light over the given time range."""
        cars_at_light = {tick: {} for tick in time[1:]}
        for tick in time[1:]:
            cars_at_light[tick] = self.model.get_cars_per_lane_of_light(
                self.position, tick
            )
        return cars_at_light

    def _initialize_lanes(self, opt_model, time, possible_lanes, current_open_lane):
        """Initializes the lane variables for the optimization model."""
        lanes = opt_model.add_variables(
            time, possible_lanes, domain=poi.VariableDomain.Binary
        )
        for lane in possible_lanes:
            lanes[-1, lane] = 1 if lane == current_open_lane else 0
        return lanes

    def _add_constraints(self, opt_model, lanes, time, possible_lanes):
        """Adds constraints to the optimization model."""
        for tick in time[1:]:
            opt_model.add_linear_constraint(
                poi.quicksum(lanes[tick, lane] for lane in possible_lanes), poi.Eq, 1
            )
        for lane in possible_lanes:
            opt_model.add_quadratic_constraint(
                poi.quicksum(
                    (
                        (lanes[tick - 1, lane] - lanes[tick, lane])
                        * (lanes[tick - 1, lane] - lanes[tick, lane])
                    )
                    for tick in time[1:]
                ),
                poi.Leq,
                1.0,
            )

    def _set_objective(self, opt_model, lanes, cars_at_light, time, possible_lanes):
        """Sets the objective function for the optimization model."""
        objective = poi.quicksum(
            poi.quicksum(
                lanes[tick, lane] * cars_at_light[tick][lane] for lane in possible_lanes
            )
            for tick in time[1:]
        )
        opt_model.set_objective(objective, poi.ObjectiveSense.Maximize)

    def _get_optimal_lane(self, opt_model, lanes, possible_lanes) -> str:
        """Determines the optimal lane from the optimization results."""
        for lane in possible_lanes:
            if opt_model.get_value(lanes[0, lane]) > 0.1:
                return lane

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

    def get_num_cars(self) -> int:
        """Gets the number of cars currently at the light.

        Returns:
            int: Number of cars currently at the light
        """
        num_cars = 0
        for car in self.model.get_agents_by_type("CarAgent"):
            car: CarAgent
            if car.position == self.position:
                num_cars += 1

        return num_cars

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


@dataclass
class ArrivalsData(SimData):
    """Holds the number of cars arriving at a LightAgent instance at each step"""

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Constructs the data schema"""
        self.data = pl.DataFrame(
            schema={
                "Step": pl.Int32,
                "Light_ID": pl.Int16,
                "Time": pl.Int16,
                "Arrivals": pl.Int16,
            },
            strict=False,
        )

    def update_data(self, light: LightAgent, steps: int) -> None:
        """Updates the data

        Args:
            light (LightAgent): LightAgent instance
            steps (int): Internal step counter of TrafficModel instance
        """
        self.data.vstack(
            pl.DataFrame(
                data={
                    "Step": steps,
                    "Light_ID": light.unique_id,
                    "Time": 200 - (steps % 200),
                    "Arrivals": light.get_num_arrivals(),
                },
                schema={
                    "Step": pl.Int32,
                    "Light_ID": pl.Int16,
                    "Time": pl.Int16,
                    "Arrivals": pl.Int16,
                },
            ),
            in_place=True,
        )

    def get_data(self) -> pl.DataFrame:
        """Returns the data

        Returns:
            pl.DataFrame: Data
        """
        return self.data


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
                "Num_Cars": pl.Int16,
            },
            strict=False,
        )

    def update_data(self, light: LightAgent, steps: int) -> None:
        """Updates the data

        Args:
            light (LightAgent): LightAgent instance
            steps (int): Internal step counter of TrafficModel instance
        """
        self.data.vstack(
            other=pl.DataFrame(
                data={
                    "Step": steps,
                    "Light_ID": light.unique_id,
                    "Time": 200 - (steps % 200),
                    "Num_Cars": light.get_num_cars(),
                },
                schema={
                    "Step": pl.Int32,
                    "Light_ID": pl.Int16,
                    "Time": pl.Int16,
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
