"""
light.py

This module defines the `LightAgent` class, representing traffic lights within the
`TrafficModel` simulation. It also includes helper classes for managing traffic
light data (`TrafficData`, `Lanes`), optimization strategies (`Optimizer`,
`SimpleOptimizer`, `AdvancedOptimizer`), and a custom exception (`LightCooldown`).

`LightAgent` manages traffic flow at intersections by controlling which lane
(incoming edge) is currently open. It interacts with `CarAgent` instances,
tracks waiting cars, and uses different optimization strategies to decide when
and which lane to open next, considering factors like waiting cars, cooldown periods,
and potentially predicted arrivals (in advanced modes).

Classes:
    - LightAgent: Represents a traffic light agent in the simulation.
    - LightCooldown: Custom exception for cooldown violations.
    - Optimizer (ABC): Abstract base class for optimization strategies.
    - SimpleOptimizer: Basic optimization based on current waiting cars.
    - AdvancedOptimizer: More complex optimization considering time horizons and
                         optionally using ML predictions.
    - Lanes: Dataclass to store information about connected lanes (distance).
    - TrafficData: Dataclass inheriting from `SimData` to store historical
                   traffic counts per lane for a light.

Dependencies:
    - random: For random choices in initialization and optimization tie-breaking.
    - abc: For defining abstract base classes.
    - dataclasses: For creating data classes.
    - pathlib: For handling file paths (used in AdvancedOptimizer for secrets).
    - mesa: Core agent-based modeling framework.
    - numpy: For numerical operations (e.g., calculating average distance).
    - polars: For efficient data handling in `TrafficData`.
    - pyoptinterface: For interfacing with optimization solvers (HiGHS, Gurobi).
    - networkx: For graph analysis (e.g., centrality).
    - car.CarAgent: Referenced for interactions.
    - data.SimData: Base class for `TrafficData`.
    - graph.Graph: Referenced for grid structure information.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import mesa
import numpy as np
import polars as pl
import pyoptinterface as poi
from networkx import closeness_centrality
from pyoptinterface import gurobi, highs

from car import CarAgent
from data import SimData
from graph import Graph


class LightAgent(mesa.Agent):
    """Represents a traffic light agent located at an intersection in the traffic grid.

    This agent controls the flow of traffic by deciding which incoming lane (edge)
    is allowed to pass cars through the intersection at any given time. It manages
    a cooldown period between lane switches and can employ different strategies
    (none, simple, advanced, advanced_ml) to optimize lane switching based on
    factors like waiting cars and potentially predicted arrivals.

    Inherits from `mesa.Agent`.

    Attributes:
        position (str): The ID of the intersection node where the agent is located.
        neighbor_lights (list[str]): List of IDs of adjacent intersection nodes
                                     representing incoming lanes.
        traffic (TrafficData): Data collector for historical traffic counts per lane.
        lanes (Lanes): Dataclass storing distances for each connected lane.
        default_switching_cooldown (int): The fixed number of steps the agent must
                                          wait after switching the open lane before
                                          it can switch again.
        current_switching_cooldown (int): The remaining number of steps in the
                                          current cooldown period. Decrements each step.
        open_lane (str): The ID of the neighboring intersection node (lane) from
                         which cars are currently allowed to cross.

    ## Methods:
        **step(self, optimization_type: str, steps: int) -> None**:
            Executes the agent's actions for a single simulation step.
        **set_position(self, position: str) -> None**:
            Sets the agent's position attribute.
        **change_open_lane(self, lane: str) -> None**:
            Changes the currently open lane to the specified lane.
        **rotate_in_open_lane_cycle(self) -> str**:
            Determines the next lane to open by simple rotation.
        **get_num_connections(self, grid: Graph) -> int**:
            Calculates the number of intersection nodes connected to this light's position.
        **get_avg_distance(self, grid: Graph) -> float**:
            Calculates the average distance (edge weight) to connected intersection nodes.
        **is_entrypoint(self, grid: Graph) -> bool**:
            Checks if the light's intersection is connected to any border node.
        **get_num_arrivals(self) -> int**:
            Counts the number of cars currently arriving at this light's intersection.
        **get_num_cars(self, lane: str) -> int**:
            Counts the number of cars currently waiting at this light from a specific lane.
        **get_centrality(self, grid: Graph) -> float**:
            Calculates the closeness centrality of the light's intersection node.
        **get_connected_intersections(self, grid: Graph) -> list[str]**:
            Retrieves the IDs of all neighboring nodes that are intersections.
    """

    def __init__(self, model: mesa.Model, **kwargs):
        """Initializes a LightAgent instance.

        Sets up the agent's position, identifies neighboring intersections (lanes),
        initializes data collectors (`TrafficData`, `Lanes`), sets the default
        cooldown, and randomly selects an initial open lane.

        Args:
            model (mesa.Model): The `TrafficModel` instance the agent belongs to.
            **kwargs: Keyword arguments, expected to include 'position' (str).
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
        ]

    def step(self, optimization_type: str, steps: int) -> None:
        """Executes the agent's actions for a single simulation step.

        Checks if the switching cooldown has expired. If so, it determines the
        next lane to open based on the specified `optimization_type` and attempts
        to switch. If the cooldown is active, it decrements the counter. Finally,
        it records the current number of cars in each incoming lane using the
        `TrafficData` collector.

        Args:
            optimization_type (str): The strategy to use for deciding the next
                                     open lane ('none', 'simple', 'advanced', 'advanced_ml').
            steps (int): The current step number of the simulation model.
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
        """Sets the agent's position attribute.

        Args:
            position (str): The ID of the intersection node.
        """
        self.position = position

    def change_open_lane(self, lane: str) -> None:
        """Changes the currently open lane to the specified lane.

        This action is only performed if the `current_switching_cooldown` is zero
        or less. If the specified `lane` is different from the current `open_lane`,
        the `open_lane` attribute is updated, and the `current_switching_cooldown`
        is reset to the `default_switching_cooldown`.

        Args:
            lane (str): The ID of the neighboring intersection node (lane) to open.

        Raises:
            LightCooldown: If `current_switching_cooldown` is greater than 0,
                           indicating the agent cannot switch lanes yet.
        """
        if self.current_switching_cooldown > 0:
            raise LightCooldown(
                "The current switching cooldown does not allow changing the open lane."
            )

        if self.open_lane != lane:
            self.open_lane = lane
            self.current_switching_cooldown = self.default_switching_cooldown

    def rotate_in_open_lane_cycle(self) -> str:
        """Determines the next lane to open by simple rotation.

        Finds the index of the current `open_lane` in the `neighbor_lights` list
        and returns the ID of the next lane in the list (cycling back to the
        start if necessary). This implements the 'none' optimization strategy.

        Returns:
            str: The ID of the next lane in the rotation cycle.
        """
        current_index = self.neighbor_lights.index(self.open_lane)
        next_index = (current_index + 1) % len(self.neighbor_lights)

        return self.neighbor_lights[next_index]

    def get_num_connections(self, grid: Graph) -> int:
        """Calculates the number of intersection nodes connected to this light's position.

        Args:
            grid (Graph): The `Graph` instance representing the traffic network.

        Returns:
            int: The count of neighboring nodes whose IDs start with 'intersection'.
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
        """Calculates the average distance (edge weight) to connected intersection nodes.

        Args:
            grid (Graph): The `Graph` instance representing the traffic network.

        Returns:
            float: The mean weight of edges connecting this light's position to
                   neighboring intersection nodes. Returns NaN if no intersection neighbors.
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
        """Checks if the light's intersection is connected to any border node.

        Args:
            grid (Graph): The `Graph` instance representing the traffic network.

        Returns:
            bool: True if any neighbor node ID starts with 'border', False otherwise.
        """
        for node in grid.neighbors(self.position):
            if node.startswith("border"):
                return True

        return False

    def get_num_arrivals(self) -> int:
        """Counts the number of cars currently arriving at this light's intersection.

        Iterates through all `CarAgent` instances in the model, checking if they
        are at this light's position and if their `WaitTimeData` indicates they
        just arrived in the current step.

        Returns:
            int: The total count of cars arriving at this light in the current step.
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
        """Counts the number of cars currently waiting at this light from a specific lane.

        Iterates through `CarAgent` instances, checking if they are at this light's
        position, are currently waiting (not moving), and arrived from the specified
        `lane` (based on the model's tracking of the last intersection).

        Args:
            lane (str): The ID of the incoming lane (neighboring intersection) to check.

        Returns:
            int: The number of cars waiting at this light that arrived from the specified lane.
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

    def get_centrality(self, grid: Graph) -> float:
        """Calculates the closeness centrality of the light's intersection node.

        Uses `networkx.closeness_centrality` based on edge weights ('distance')
        to measure how central the intersection is within the graph.

        Args:
            grid (Graph): The `Graph` instance representing the traffic network.

        Returns:
            float: The closeness centrality value for the light's position node.
        """
        centrality = closeness_centrality(G=grid, u=self.position, distance="weight")

        return centrality

    def get_connected_intersections(self, grid: Graph) -> list[str]:
        """Retrieves the IDs of all neighboring nodes that are intersections.

        Args:
            grid (Graph): The `Graph` instance representing the traffic network.

        Returns:
            list[str]: A list of node IDs for neighboring intersections.
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
    """Abstract base class for traffic light optimization strategies.

    Defines the interface for different algorithms that decide which incoming
    lane a `LightAgent` should open next. Subclasses must implement the
    abstract methods to provide specific optimization logic.
    """

    @abstractmethod
    def __init__(self, light: LightAgent):
        """Initializes the optimizer instance for a specific LightAgent.

        Subclasses should use this method to store the associated `LightAgent`
        and potentially perform initial setup or data retrieval needed for
        the optimization process.

        Args:
            light (LightAgent): The `LightAgent` instance this optimizer will control.
        """
        pass

    @abstractmethod
    def get_cars_at_light(self) -> dict:
        """Retrieves the number of cars currently waiting at each incoming lane.

        Subclasses must implement this method to gather the necessary car count
        information, which typically forms the basis for the optimization decision.
        This might involve querying the associated `LightAgent` or the `TrafficModel`.

        Returns:
            dict: A dictionary where keys are lane IDs (neighboring intersection IDs)
                  and values are the number of cars waiting in that lane.
        """
        pass

    @abstractmethod
    def get_dec_vars(self) -> list[str]:
        """Identifies the decision variables for the optimization problem.

        Typically, the decision variables correspond to the incoming lanes of the
        `LightAgent`. Subclasses must return these identifiers.

        Returns:
            list[str]: A list of strings, where each string is the ID of an
                       incoming lane (neighboring intersection node).
        """
        pass

    @abstractmethod
    def init_model(self) -> None:
        """Initializes and configures the underlying optimization model.

        Subclasses should implement this method to:
        1. Set up the specific optimization solver (e.g., HiGHS, Gurobi).
        2. Define the decision variables based on `get_dec_vars()`.
        3. Add constraints relevant to the optimization strategy (e.g., only one
           lane open at a time, cooldown constraints).
        4. Define the objective function (e.g., maximize cars passing, minimize wait time).
        5. Optionally, run the optimization process if it's done during initialization.
        """
        pass

    @abstractmethod
    def get_optimal_lane(self) -> str:
        """Determines and returns the optimal lane to open based on the optimization result.

        Subclasses must implement this method to extract the solution from the
        optimization model (run in `init_model` or called here) and return the ID
        of the lane chosen as optimal.

        Returns:
            str: The ID of the incoming lane (neighboring intersection node) that
                 the optimizer determined should be opened next.
        """
        pass


class SimpleOptimizer(Optimizer):
    """Implements a basic optimization strategy for `LightAgent`.

    This optimizer selects the lane with the maximum number of currently waiting
    cars. It does not consider future arrivals, cooldown periods beyond the
    immediate decision, or other complex factors. It uses the HiGHS solver via
    `pyoptinterface`.

    Inherits from `Optimizer`.

    Attributes:
        light (LightAgent): The associated LightAgent instance.
        cars_at_light (dict): A dictionary mapping lane IDs to the number of
                              cars currently waiting in that lane.
        model (highs.Model): The HiGHS optimization model instance.

    ## Methods:
        **get_cars_at_light(self) -> dict**:
            Retrieves the current number of waiting cars for each lane.
        **get_dec_vars(self) -> list[str]**:
            Returns the list of incoming lane IDs for the associated `LightAgent`.
        **init_model(self) -> None**:
            Initializes, builds, and solves the HiGHS optimization model.
        **get_optimal_lane(self) -> str**:
            Extracts the optimal lane from the solved optimization model.

    """

    def __init__(self, light: LightAgent):
        """Initializes the SimpleOptimizer.

        Stores the `LightAgent`, retrieves the current car counts for each lane,
        and immediately initializes and solves the optimization model.

        Args:
            light (LightAgent): The `LightAgent` instance this optimizer will manage.
        """
        self.light = light
        self.cars_at_light = self.get_cars_at_light()
        self.init_model()

    def get_cars_at_light(self) -> dict:
        """Retrieves the current number of waiting cars for each incoming lane.

        Calls the `TrafficModel`'s method to get the count of cars waiting at
        the associated `LightAgent`'s position, specifically for the current time step (tick 0).

        Returns:
            dict: A dictionary where keys are lane IDs (neighboring intersection IDs)
                  and values are the counts of cars currently waiting in that lane.
        """
        cars_at_light = self.light.model.get_cars_per_lane_of_light(
            self.light.position, 0
        )

        return cars_at_light

    def get_dec_vars(self) -> list[str]:
        """Returns the list of incoming lane IDs for the associated `LightAgent`.

        These lane IDs serve as the decision variables for the optimization model.

        Returns:
            list[str]: A list of strings, each representing an incoming lane ID
                       (neighboring intersection node ID).
        """
        return self.light.neighbor_lights

    def init_model(self) -> None:
        """Initializes, builds, and solves the HiGHS optimization model.

        Configures a HiGHS model to find the lane with the maximum waiting cars:
        - Sets the model to run silently.
        - Creates binary decision variables, one for each incoming lane (1 if open, 0 if closed).
        - Adds a constraint ensuring exactly one lane is chosen (sum of variables equals 1).
        - Sets the objective function to maximize the sum of (decision variable * waiting cars)
          for each lane.
        - Solves the optimization problem immediately.
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

    def get_optimal_lane(self) -> str:
        """Extracts the optimal lane from the solved optimization model.

        Identifies the lane(s) corresponding to the maximum objective value
        (maximum waiting cars). If multiple lanes have the same maximum number
        of cars, one is chosen randomly as the optimal lane. Logs the decision.

        Returns:
            str: The ID of the incoming lane chosen as optimal (the one with the
                 most waiting cars, with random tie-breaking).
        """
        optimal_value = self.model.get_obj_value()
        optimal_lanes = [
            lane
            for lane in self.cars_at_light
            if optimal_value == self.cars_at_light[lane]
        ]
        if len(optimal_lanes) > 1:
            optimal_lane = random.choice(optimal_lanes)
        else:
            optimal_lane = optimal_lanes[0]
        self.light.model.update_lights_decision_log(
            self.light, self.cars_at_light, optimal_lane, self.light.model.steps
        )

        return optimal_lane


class AdvancedOptimizer(Optimizer):
    """Implements an advanced optimization strategy for `LightAgent`.

    This optimizer considers a time horizon that includes the light's cooldown
    period. It aims to maximize the number of cars passing through the intersection
    over this horizon. It can operate in two modes:
    - 'base': Estimates future arrivals based on historical averages.
    - 'ml': Predicts future arrivals using a pre-trained machine learning model.

    It uses the Gurobi solver via `pyoptinterface` and requires Gurobi license
    secrets to be configured.

    Inherits from `Optimizer`.

    Attributes:
        light (LightAgent): The associated LightAgent instance.
        mode (str): The operating mode ('base' or 'ml').
        time (range): The time horizon considered, from -1 (previous state) up to
                      the end of the default cooldown period.
        cars_at_light (dict): A nested dictionary mapping time ticks within the
                              horizon to lane IDs and their corresponding car counts
                              (current or estimated/predicted).
        model (gurobi.Model): The Gurobi optimization model instance.
        lanes (poi.VariableMatrix): Decision variables representing whether a lane
                                    is open at a specific time tick.

    ## Methods:
        **get_cars_at_light(self) -> dict**:
            Retrieves/estimates/predicts car counts for each lane over the time horizon.
        **get_dec_vars(self) -> tuple[range, list[str]]**:
            Returns the dimensions for the decision variables: time horizon and lanes.
        **init_model(self) -> None**:
            Initializes, builds, and solves the Gurobi optimization model.
        **get_optimal_lane(self) -> str**:
            Extracts the optimal lane for the current time step (t=0) from the solved model.
        **_init_env(self) -> gurobi.Env**:
            Initializes and configures the Gurobi optimization environment.
        **_load_secrets(self) -> dict**:
            Loads Gurobi license secrets from a 'gurobi.lic' file.
        **_request_cars_at_light(self) -> dict**:
            DEPRECATED: Gets car counts by directly querying the model for future ticks.
        **_get_cars_waiting(self) -> dict**:
            Gets the number of cars currently waiting at each incoming lane.
        **_approx_incoming_cars(self) -> dict**:
            Approximates the number of incoming cars for future steps based on historical averages.
        **_predict_incoming_cars(self) -> dict**:
            Predicts the number of incoming cars for future steps using a pre-trained model.
        
    """

    def __init__(self, light: LightAgent, mode: str = "base"):
        """Initializes the AdvancedOptimizer.

        Stores the `LightAgent` and the operating `mode`. Defines the time horizon
        based on the light's cooldown. Retrieves/estimates/predicts car counts
        for the time horizon and initializes and solves the Gurobi optimization model.

        Args:
            light (LightAgent): The `LightAgent` instance this optimizer will manage.
            mode (str, optional): The strategy for estimating future arrivals.
                                  'base' uses historical averages, 'ml' uses a
                                  pre-trained model. Defaults to "base".

        Raises:
            ValueError: If the provided `mode` is not 'base' or 'ml'.
        """
        self.light = light
        self.mode = mode
        self.time = range(-1, self.light.default_switching_cooldown + 1)
        self.cars_at_light = self.get_cars_at_light()
        self.init_model()

    def get_cars_at_light(self):
        """Retrieves/estimates/predicts car counts for each lane over the time horizon.

        Gets the currently waiting cars for the present time step (tick 0).
        For future time steps within the horizon, it either approximates arrivals
        using historical averages (`_approx_incoming_cars` if mode='base') or
        predicts arrivals using a machine learning model (`_predict_incoming_cars`
        if mode='ml').

        Returns:
            dict: A nested dictionary. Outer keys are time ticks (int) from the
                  horizon (0 to cooldown). Inner keys are lane IDs (str), and
                  values are the number of cars (int) waiting or expected.
        """
        cars_at_light = {tick: {} for tick in self.time[1:]}
        cars_at_light[0] = self._get_cars_waiting()
        if self.mode == "base":
            for tick in self.time[2:]:
                cars_at_light[tick] = self._approx_incoming_cars()
        elif self.mode == "ml":
            for tick in self.time[2:]:
                cars_at_light[tick] = self._predict_incoming_cars()
        else:
            raise ValueError(
                f"mode must be one of ['base'|'ml']. Got {self.mode} instead"
            )

        return cars_at_light

    def get_dec_vars(self) -> tuple[range, list[str]]:
        """Returns the dimensions for the decision variables: time horizon and lanes.

        Identifies the time steps and the incoming lane IDs that form the basis
        for the optimization decision variables (lane open status at each time step).

        Returns:
            tuple[range, list[str]]: A tuple containing the time range object
                                     and a list of incoming lane IDs.
        """
        return self.time, self.light.neighbor_lights

    def init_model(self):
        """Initializes, builds, and solves the Gurobi optimization model.

        Configures a Gurobi model to maximize car throughput over the time horizon:
        - Initializes the Gurobi environment using license secrets.
        - Sets the model to run silently.
        - Creates binary decision variables `lanes[time, lane]` (1 if `lane` is
          open at `time`, 0 otherwise). Sets the state for the previous step (t=-1).
        - Adds linear constraints: Exactly one lane must be open at each future time step.
        - Adds quadratic constraints: Enforces the cooldown by ensuring that for each
          lane, the sum of squared differences between consecutive time steps is at
          most 1 (allowing at most one change from 0 to 1 or 1 to 0 over the horizon).
        - Sets the objective function: Maximize the total sum of (decision variable * car count)
          across all lanes and future time steps in the horizon.
        - Solves the optimization problem immediately.
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
        """Extracts the optimal lane for the current time step (t=0) from the solved model.

        Iterates through the decision variables for the current time step (tick 0)
        and returns the ID of the lane whose variable value is close to 1 (indicating
        it should be opened now).

        Returns:
            str: The ID of the incoming lane chosen as optimal for the current step.
                 Returns None if the optimization fails or no lane is selected (unlikely
                 given constraints).
        """
        for lane in self.light.neighbor_lights:
            if self.model.get_value(self.lanes[0, lane]) > 0.1:
                return lane

    def _init_env(self) -> gurobi.Env:
        """Initializes and configures the Gurobi optimization environment.

        Loads Gurobi license secrets using `_load_secrets` and sets the necessary
        parameters (WLSACCESSID, WLSSECRET, LICENSEID) for the environment.
        Configures the environment to run silently (OutputFlag=0).

        Returns:
            gurobi.Env: The configured Gurobi environment object.
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
        """Loads Gurobi license secrets from a 'gurobi.lic' file.

        Expects the license file to be located at '.secrets/gurobi.lic' relative
        to the current working directory. Parses the file, ignoring comments and
        empty lines, and extracts key-value pairs.

        Returns:
            dict: A dictionary containing the Gurobi license secrets (e.g.,
                  'WLSACCESSID', 'WLSSECRET', 'LICENSEID').
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
        """DEPRECATED: Gets car counts by directly querying the model for future ticks.

        This method is likely deprecated because predicting/approximating future
        arrivals is handled by `_approx_incoming_cars` or `_predict_incoming_cars`.
        Directly querying the model for future states might not be feasible or intended.

        Returns:
            dict: A nested dictionary of car counts per lane per future time tick.
        """
        cars_at_light = {tick: {} for tick in self.time[1:]}
        for tick in self.time[1:]:
            cars_at_light[tick] = self.light.model.get_cars_per_lane_of_light(
                self.light.position, tick
            )

        return cars_at_light

    def _get_cars_waiting(self):
        """Gets the number of cars currently waiting at each incoming lane.

        Queries the associated `LightAgent`'s `TrafficData` collector to get the
        most recent count of waiting cars for each lane.

        Returns:
            dict: A dictionary where keys are lane IDs and values are the current
                  number of waiting cars in that lane.
        """
        waiting_cars = {
            lane: self.light.traffic.get_current_cars(lane)
            for lane in self.light.neighbor_lights
        }

        return waiting_cars

    def _approx_incoming_cars(self):
        """Approximates the number of incoming cars for future steps based on historical averages.

        For each incoming lane, calculates the mean number of cars observed historically
        using the data stored in the `LightAgent`'s `TrafficData` collector.

        Returns:
            dict: A dictionary where keys are lane IDs and values are the average
                  historical number of cars observed in that lane.
        """
        avg_traffic = {
            lane: self.light.traffic.data.filter(pl.col("Lane") == lane)
            .select(pl.col("Num_Cars"))
            .mean()
            .item()
            for lane in self.light.neighbor_lights
        }

        return avg_traffic

    def _predict_incoming_cars(self):
        """Predicts the number of incoming cars for future steps using a pre-trained model.

        For each incoming lane, uses the `TrafficModel`'s `regressor` instance
        to predict the number of cars. The prediction uses features like the
        current time within a cycle, the light's centrality, whether it's an
        entry point, and the distance of the incoming lane.

        Returns:
            dict: A dictionary where keys are lane IDs and values are the predicted
                  number of cars for that lane.
        """
        cars_per_lane = {
            neighbor: self.light.model.regressor.predict(
                200 - (self.light.model.steps % 200),
                self.light.get_centrality(grid=self.light.model.grid),
                self.light.is_entrypoint(grid=self.light.model.grid),
                self.light.lanes.get_distance(lane=neighbor),
            )
            for neighbor in self.light.neighbor_lights
        }

        return cars_per_lane


@dataclass
class Lanes:
    """Stores information about the incoming lanes connected to a `LightAgent`.

    This dataclass holds a Polars DataFrame containing the IDs (neighboring
    intersection node IDs) and distances (edge weights) of the lanes leading
    into the intersection managed by a specific `LightAgent`.

    Attributes:
        data (pl.DataFrame): A Polars DataFrame with columns 'Lane' (str) and
                             'Distance' (Int16), storing the ID and distance
                             for each incoming lane.

    ## Methods:
        **construct(self, light: LightAgent) -> Lanes**:
            Populates the DataFrame with lane data for the given `LightAgent`.
        **get_distance(self, lane: str) -> int**:
            Retrieves the distance (edge weight) for a specific incoming lane.
    """

    data: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={"Lane": pl.String, "Distance": pl.Int16}, strict=False
        )
    )

    def construct(self, light: LightAgent):
        """Populates the internal DataFrame with lane data for the given `LightAgent`.

        Retrieves the neighboring intersection nodes (lanes) and their corresponding
        edge weights (distances) from the model's grid and stores them in the
        `data` DataFrame.

        Args:
            light (LightAgent): The `LightAgent` instance whose incoming lanes
                                are being recorded.

        Returns:
            Lanes: The current `Lanes` instance, now populated with data.
        """
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
        """Retrieves the distance (edge weight) for a specific incoming lane.

        Filters the internal DataFrame to find the entry for the specified `lane`
        and returns its corresponding 'Distance' value.

        Args:
            lane (str): The ID of the incoming lane (neighboring intersection node)
                        to query.

        Returns:
            int: The distance (weight) of the edge leading from the specified
                 `lane` to the light's intersection.

        Raises:
            pl.exceptions.NoRowsReturnedError: If the specified `lane` is not found
                                               in the DataFrame (implicitly handled
                                               by Polars `.item()`, may raise error).
                                               Consider adding error handling if needed.
        """
        distance = (
            self.data.filter(pl.col("Lane") == lane).select(pl.col("Distance")).item()
        )

        return distance


@dataclass
class TrafficData(SimData):
    """Stores historical traffic data for a LightAgent instance.

    Inherits from `SimData` and uses a Polars DataFrame to record the number
    of cars waiting in each incoming lane at each simulation step.

    Attributes:
        data (pl.DataFrame): A Polars DataFrame holding the traffic data.
                             Columns include 'Step', 'Light_ID', 'Time', 'Lane',
                             'Open_Lane', and 'Num_Cars'.

    ## Methods:
        **__post_init__(self) -> None**:
            Initializes the Polars DataFrame with a predefined schema.
        **update_data(self, light: LightAgent, steps: int, lane: str) -> None**:
            Appends a new record of traffic data for a specific lane at the current step.
        **get_current_cars(self, lane: str) -> int**:
            Retrieves the most recently recorded number of cars for a specific lane.
        **get_data(self) -> pl.DataFrame**:
            Returns the entire collected traffic data as a Polars DataFrame.
    """

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Initializes the Polars DataFrame with the predefined schema.

        Sets up the structure for storing traffic data, including step number,
        light ID, time within a cycle, specific lane, the currently open lane,
        and the number of cars in that specific lane.
        """
        self.data = pl.DataFrame(
            schema={
                "Step": pl.Int32,
                "Light_ID": pl.Int16,
                "Time": pl.Int16,
                "Lane": pl.String,
                "Open_Lane": pl.String,
                "Num_Cars": pl.Int16,
            },
            strict=False,
        )

    def update_data(self, light: LightAgent, steps: int, lane: str) -> None:
        """Appends a new record of traffic data for a specific lane at the current step.

        Calculates the time within the 200-step cycle and records the number
        of cars currently waiting in the specified `lane` for the given `light`
        at the current simulation `steps`. Also records which lane is currently open.

        Args:
            light (LightAgent): The LightAgent instance for which data is being recorded.
            steps (int): The current step number in the simulation.
            lane (str): The specific incoming lane (neighbor intersection ID)
                        for which the car count is being recorded.
        """
        self.data.extend(
            other=pl.DataFrame(
                data={
                    "Step": steps,
                    "Light_ID": light.unique_id,
                    "Time": 200 - (steps % 200),
                    "Lane": lane,
                    "Open_Lane": light.open_lane,
                    "Num_Cars": light.get_num_cars(lane),
                },
                schema={
                    "Step": pl.Int32,
                    "Light_ID": pl.Int16,
                    "Time": pl.Int16,
                    "Lane": pl.String,
                    "Open_Lane": pl.String,
                    "Num_Cars": pl.Int16,
                },
                strict=False,
            ),
        )

    def get_current_cars(self, lane: str) -> int:
        """Retrieves the most recently recorded number of cars for a specific lane.

        Filters the data for the specified `lane` and returns the 'Num_Cars'
        value from the latest entry (highest step number).

        Args:
            lane (str): The lane (neighbor intersection ID) to query.

        Returns:
            int: The number of cars recorded for the specified lane in the
                 most recent update. Returns 0 if no data exists for the lane.
        """
        current_cars = (
            self.data.filter(pl.col("Lane") == lane).select("Num_Cars").tail(1).item()
        )

        return current_cars

    def get_data(self) -> pl.DataFrame:
        """Returns the entire collected traffic data as a Polars DataFrame.

        Provides access to the complete historical record of traffic counts
        managed by this instance.

        Returns:
            pl.DataFrame: The DataFrame containing all recorded traffic data.
        """
        return self.data
