"""This module contains:
- LightAgent class: Represents a traffic light in the traffic grid, determining the flow of traffic."""

import mesa
import random
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from networkx import closeness_centrality

from graph import Graph
from car import CarAgent


class LightAgent(mesa.Agent):
    """Agent, which represents a traffic light in the traffic gird. It has a fixed position on an intersection and decides which cars are allowed to move (only cars from one direction/edge can move at the same time).

    Inherits from mesa.Agent.

    Attributes:
        position (str): The ID of the node, where the agent is currently located.
        waiting_cars (dict): A nested dictionary of cars waiting at the intersection. (outer dict: cars; inner dict: last_intersection, global_waiting_time, local_waiting_time)
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
        self.centrality = closeness_centrality(
            G=model.grid, u=self.position, distance="weight"
        )
        self.neighbor_lights = [
            node
            for node in self.model.grid.neighbors(self.position)
            if node.startswith("intersection")
        ]
        self.default_switching_cooldown = 5
        self.current_switching_cooldown = self.default_switching_cooldown
        # self.waiting_cars = {}
        # self.waiting_cars = self.update_waiting_cars()
        self.open_lane = self.neighbor_lights[
            random.randint(0, len(self.neighbor_lights) - 1)
        ]  # Randomly select a neighbor light as the open lane

    def set_position(self, position: str) -> None:
        """Sets the position of the agent to the given node ID.

        Args:
            position (str): The ID of the node, where the agent is currently located.
        """
        self.position = position

    def update_waiting_cars(self) -> None:  # TODO: Fix
        """NOT WORKING: Updates the details of the cars waiting at the intersection (waiting_cars)"""

        # if self.waiting_cars is not None:
        #     # Remove all cars that have moved from the waiting_cars list
        #     for car in self.waiting_cars.keys():
        #         if not car.waiting:
        #             self.waiting_cars.pop(car)

        #     # Add all new cars that are now waiting at the intersection
        #     for car in self.model.get_agents_by_type("CarAgent"):
        #         if (
        #             car.position == self.position
        #             and car not in self.waiting_cars.keys()
        #             and car.waiting
        #         ):
        #             self.waiting_cars[car] = {
        #                 "last_intersection": car.model.get_last_intersection_of_car(
        #                     car.unique_id
        #                 ),
        #                 "local_waiting_time": 0,
        #             }

        #     # Update car attributes in the waiting_cars list
        #     for car in self.waiting_cars.keys():
        #         self.waiting_cars[car]["local_waiting_time"] += 1

        # else:
        #     self.waiting_cars = {}
        #     for car in self.model.get_agents_by_type("CarAgent"):
        #         if car.position == self.position and car.waiting:
        #             self.waiting_cars[car] = {
        #                 "last_intersection": car.model.get_last_intersection_of_car(
        #                     car.unique_id
        #                 ),
        #                 "local_waiting_time": 1,
        #             }

    def change_open_lane(self, lane: str) -> None:
        """Changes from where cars are allowed to cross the intersection, if the current switching cooldown allows it.

        Args:
            lane (str): The ID of the edge from where cars are allowed to cross the intersection.

        Raises:
            LightCooldown: If the current switching cooldown does not allow changing the open lane.
        """
        if self.current_switching_cooldown <= 0:
            self.open_lane = lane
            self.current_switching_cooldown = self.default_switching_cooldown
        else:
            raise LightCooldown(
                "The current switching cooldown does not allow changing the open lane."
            )

    def rotate_in_open_lane_cycle(self) -> None:
        """Rotates the open lane to the next neighbor light in the cycle."""
        current_index = self.neighbor_lights.index(self.open_lane)
        next_index = (current_index + 1) % len(self.neighbor_lights)
        self.change_open_lane(self.neighbor_lights[next_index])

    def optimize_open_lane(self) -> str:
        """Decides which lane should be open based on the number of waiting cars."""
        opt_model = highs.Model()

        opt_model.set_model_attribute(poi.ModelAttribute.Silent, True)

        possible_lanes = self.neighbor_lights
        cars_at_light = self.model.get_cars_per_lane_of_light(self.position)
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

    def get_is_entrypoint(self, grid: Graph) -> bool:
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

    def get_num_cars(self) -> int:
        """Gets the number of cars currently at the light

        Returns:
            int: Number of cars currently at the light
        """
        num_cars = 0
        for car in self.model.get_agents_by_type("CarAgent"):
            car: CarAgent
            if car.position == self.position and car.waiting:
                num_cars += 1

        return num_cars


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
