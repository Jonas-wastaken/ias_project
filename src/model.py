"""This module contains:
- TrafficModel class: A Mesa model simulating traffic."""

import mesa
import random
import numpy as np
from car import CarAgent, AgentArrived
from graph import Graph
from light import LightAgent


class TrafficModel(mesa.Model):
    """A Mesa model simulating traffic.

    Attributes:
        grid (Graph): Graph the environment uses.
        agents (AgentSet): Agents in the environment.
        car_paths (dict): A dictionary containing the paths of all agents.
        cars_waiting_times (dict): A dictionary containing the waiting times of all cars at each intersection.
        num_cars_hist (np.array): An array containing the history of the number of cars in the model.

    ## Methods:
        **step(self) -> None**:
            Advances the environment to next state.
        **create_cars(self, num_agents: int) -> None**:
            Function to add agents to the model.
        **remove_random_cars(self, num_agents: int) -> None**:
            Function to randomly remove n agents from the model.
        **create_lights_for_intersections(self) -> None**:
            Function to add a traffic light to each intersection of the model.
        **get_agents_by_type(self, agent_type: str) -> list**:
            Function to get all agents of a certain type.
        **get_agents_by_id(self, agent_id: list) -> list**:
            Function to get all agents by their unique ID.
        **get_last_intersection_of_car(self, car_id: int) -> str**:
            Function to get the last position of a car.
        **update_cars_waiting_times(self) -> None**:
            Function to update the waiting times of all cars at each intersection.
        **update_car_paths(self) -> None**:
            Function to update the paths of all agents.
        **car_respawn(self) -> None**:
            Respawns cars at each step depending on current time and number of cars in the model.
    """

    def __init__(self, num_cars: int, seed: int = None, **kwargs):
        """Initializes a new traffic environment.

        - Spawns a graph representing the grid
        - Initializes TrafficLight Agents at each intersection node
        - Initializes the specified number of CarAgents

        Args:
            num_cars (int): Number of car agents to spawn.
            seed (int, optional): Seed used in model generation. Defaults to None.
            **kwargs: Additional keyword arguments for configuring the graph object.
                - num_intersections (int): Number of intersections in the graph. Defaults to 30.
                - num_borders (int): Number of border nodes in the graph. Defaults to 10.
                - min_distance (int): Minimum distance between nodes. Defaults to 5.
                - max_distance (int): Maximum distance between nodes. Defaults to 15.
        """
        super().__init__(seed=seed)

        self.grid = Graph(
            num_intersections=kwargs.get("num_intersections", 50),
            num_borders=kwargs.get("num_borders", 25),
            min_distance=kwargs.get("min_distance", 10),
            max_distance=kwargs.get("max_distance", 20),
        )

        self.create_lights_for_intersections()

        self.car_paths = {}
        self.cars_waiting_times = {}
        self.num_cars_hist = np.array(num_cars)
        self.create_cars(num_cars)

    def step(self) -> None:
        """Advances the environment to next state.

        - Each CarAgent moves to it's next position.
            - Depending on TrafficLights
            - Updates internal travel_time
            - If a CarAgent reached it's goal, it is removed from the AgentSet at the next step
        - Each LightAgent opens next lane, if cooldown is completed
            - Else the cooldown is updated
        - CarAgents are respawned based on current time and number of cars in the model
        """
        for car in self.get_agents_by_type("CarAgent")[:]:
            car: CarAgent
            try:
                car.move()
                car.travel_time += 1
            except AgentArrived:
                car.remove()

        for light in self.get_agents_by_type("LightAgent"):
            light: LightAgent
            if light.current_switching_cooldown <= 0:
                light.rotate_in_open_lane_cycle()
            light.current_switching_cooldown -= 1

        self.num_cars_hist = np.append(
            self.num_cars_hist, len(self.get_agents_by_type("CarAgent"))
        )
        self.car_respawn()

    def create_cars(self, num_cars: int) -> None:
        """Function to add cars to the model.

        - Updates car_paths and cars_waiting_times attributes

        Args:
            num_cars (int): Number of cars to add.
        """
        CarAgent.create_agents(model=self, n=num_cars)
        self.update_car_paths()
        self.update_cars_waiting_times()

    def remove_random_cars(self, num_cars: int) -> None:
        """Function to randomly remove n cars from the model.

        Args:
            num_cars (int): Number of cars to remove.
        """
        for _ in range(num_cars):
            car: CarAgent = random.choice(self.get_agents_by_type("CarAgent"))
            self.agents.remove(car)

    # def create_lights(self, num_lights: int, position: str) -> None: TODO: Unused? @mxrio
    #     """Function to add traffic lights to the model.

    #     Args:
    #         num_agents (int): Number of agents to add.
    #     """
    #     LightAgent.create_agents(model=self, n=num_lights, position=position)

    def create_lights_for_intersections(self) -> None:
        """Function to add traffic lights to the model.

        - Assigns one LightAgent to each intersection node
        """
        for intersection in self.grid.get_nodes("intersection"):
            LightAgent.create_agents(model=self, n=1, position=intersection)

    def get_agents_by_type(self, agent_type: str) -> list[mesa.Agent]:
        """Function to get all agents of a certain type.

        Args:
            agent_type (str): Type of agents to get. [CarAgent, LightAgent]

        Returns:
            list[mesa.Agent]: A list of agents of the given type.
        """
        if agent_type == "CarAgent":
            return [agent for agent in self.agents if isinstance(agent, CarAgent)]
        elif agent_type == "LightAgent":
            return [agent for agent in self.agents if isinstance(agent, LightAgent)]
        else:
            raise ValueError(f"Agent type {agent_type} not found")

    def get_agents_by_id(self, agent_id: list) -> list[mesa.Agent]:
        """Function to get all agents by their unique ID.

        Args:
            agent_id (list): List of unique agent IDs.

        Returns:
            list[mesa.Agent]: A list of agents with the given unique IDs.
        """
        agents = [agent for agent in self.agents if agent.unique_id in agent_id]

        return agents

    def get_last_intersection_of_car(self, car_id: int) -> str:
        """Function to get the last position of a car.

        Args:
            car_id (int): ID of the car.

        Returns:
            str: The last position of the car.
        """
        car: CarAgent = self.get_agents_by_id([car_id])[0]
        car_full_path: dict = self.car_paths[car_id]
        car_full_path_keys = list(car_full_path.keys())
        current_position_index = car_full_path_keys.index(car.position)

        if current_position_index == 0:
            previous_position: str = car.position
        else:
            previous_position: str = car_full_path_keys[current_position_index - 1]

        # Get the corresponding intersection, if the cars last position was a border node (TODO) @mxrio
        if previous_position.startswith("border"):
            first_intersection = list(
                car.model.car_paths[car.unique_id].keys()
            )[
                1
            ]  # Warum über car den car_path holen -> self.car_paths[car.unique_id] @mxrio
            lane = list(self.grid.neighbors(previous_position))
            lane.remove(first_intersection)
            previous_position = lane[0]

        return previous_position

    def update_cars_waiting_times(self) -> None:
        """Function to update the waiting times of all cars at each intersection."""

        for car in self.get_agents_by_type("CarAgent"):
            car: CarAgent
            if car.unique_id not in list(self.cars_waiting_times.keys()):
                self.cars_waiting_times[car.unique_id] = {
                    intersection: 0
                    for intersection in list(
                        car.model.car_paths[car.unique_id].keys()  # s.o. @mxrio
                    )
                    if intersection.startswith(
                        "intersection"
                    )  # grid.get_nodes verwenden vielleicht? @mxrio
                }

            if car.waiting:
                self.cars_waiting_times[car.unique_id][car.position] += 1

    def update_car_paths(self) -> None:
        """Function to update the paths of all cars."""
        for car in self.get_agents_by_type("CarAgent"):
            car: CarAgent
            if car.unique_id not in list(self.car_paths.keys()):
                self.car_paths[car.unique_id] = car.path.copy()

    def car_respawn(self):
        """Respawns cars at each steps dependant of current time and number of cars in the model.

        - Calculates the current value on a sine function adjusted to the model's internal time
        - Calculates the next value on the function scaled to the number of cars
        - Calculates the number of cars to add with a variance of ~20%
        """
        sine_value = np.sin(2 * np.pi * self.steps / 200)
        next_sine_value = (sine_value + 1) / 2 * 2 * self.num_cars_hist[0]
        diff = next_sine_value - self.num_cars_hist[-1]
        diff_variance = diff * random.uniform(0.8, 1.2)
        cars_to_add = int(diff_variance)

        if cars_to_add > 0:
            self.create_cars(cars_to_add)
