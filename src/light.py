"""This module contains:
- LightAgent class: Represents a traffic light in the traffic grid, determining the flow of traffic."""

import mesa
import random


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


    ## Methods:
        **set_position(self, position: str) -> None**:
            Sets the position of the agent to the given node ID.
        **update_waiting_cars(self) -> None**:
            Updates the details of the cars waiting at the intersection (waiting_cars).
        **change_open_lane (self) -> None**:
            Changes from where cars are allowed to cross the intersection.
        **rotate_in_open_lane_cycle(self) -> None**:
            Rotates the open lane to the next neighbor light in the cycle.

        **Coming soon**:
        - decide_open_lane?
        - estimate_coming_cars?
    """

    def __init__(self, model: mesa.Model, **kwargs):
        """Initializes a new LightAgent. The agent is placed by the model on an intersection.

        Args:
            model (mesa.Model): The model instance in which the agent lives.
        """
        super().__init__(model)
        self.position = kwargs.get("position", None)
        self.neighbor_lights = [
            node
            for node in list(self.model.grid.neighbors(self.position))
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
