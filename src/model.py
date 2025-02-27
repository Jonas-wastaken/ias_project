"""This module contains:
- TrafficModel class: A Mesa model simulating traffic."""

import mesa
import random
from car import CarAgent, AgentArrived
from graph import Graph
from light import LightAgent


class TrafficModel(mesa.Model):
    """A Mesa model simulating traffic.

    Attributes:
        num_agents (int): Number of agents to spawn.
        seed (int, optional): Seed used in model generation. Defaults to None.
        grid (Graph): Graph the environment uses.
        agents (AgentSet): Agents in the environment.
        agent_paths (dict): A dictionary containing the paths of all agents.
        cars_waiting_times (dict): A dictionary containing the waiting times of all cars at each intersection.

    ## Methods:
        **step(self) -> None**:
            Advances the environment to next state.
        **create_cars(self, num_agents: int) -> None**:
            Function to add agents to the model.
        **remove_random_cars(self, num_agents: int) -> None**:
            Function to randomly remove n agents from the model.
        **create_lights(self, num_lights: int, position: str) -> None**:
            Function to add traffic lights to the model.
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
        **update_agent_paths(self) -> None**:
            Function to update the paths of all agents.
    """

    def __init__(self, num_cars: int, seed: int = None, **kwargs):
        """Initializes a new traffic environment.

        Args:
            num_agents (int): Number of agents to spawn.
            seed (int, optional): Seed used in model generation. Defaults to None.
            **kwargs: Additional keyword arguments for configuring the graph object.
        """
        super().__init__(seed=seed)
        self.kwargs = kwargs
        self.num_cars = num_cars
        self.grid = Graph(
            num_intersections=self.kwargs.get("num_intersections", 30),
            num_borders=self.kwargs.get("num_borders", 10),
            min_distance=self.kwargs.get("min_distance", 5),
            max_distance=self.kwargs.get("max_distance", 15),
        )
        self.create_lights_for_intersections()
        CarAgent.create_agents(model=self, n=num_cars)
        self.num_cars_hist = [len(self.get_agents_by_type("CarAgent"))]
        self.time = 0
        # initialize paths for all agents
        self.agent_paths = {}
        self.update_agent_paths()
        # initialize waiting times for all cars at each intersection
        self.cars_waiting_times = {}
        self.update_cars_waiting_times()

    def step(self) -> None:
        """Advances the environment to next state.

        - Each CarAgent moves to it's next position. If a CarAgent reached it's goal, it is removed from the AgentSet at the next step
        """
        for car in self.get_agents_by_type("CarAgent")[:]:
            try:
                car.move()
                car.travel_time += 1
            except AgentArrived:
                car.remove()

        self.num_cars_hist.append(len(self.get_agents_by_type("CarAgent")))
        self.agent_respawn()

        for light in self.get_agents_by_type("LightAgent"):
            # Decide if the light should change the open lane (if the cooldown is over)
            if light.current_switching_cooldown <= 0:
                light.rotate_in_open_lane_cycle()
            light.current_switching_cooldown -= 1

        if self.time < 200:
            self.time += 1
        elif self.time == 200:
            self.time = 0

    def create_cars(self, num_cars: int) -> None:
        """Function to add agents to the model.

        Args:
            num_agents (int): Number of agents to add.
        """
        CarAgent.create_agents(model=self, n=num_cars)
        self.update_agent_paths()
        self.update_cars_waiting_times()

    def remove_random_cars(self, num_cars: int) -> None:
        """Function to randomly remove n agents from the model.

        Args:
            num_agents (int): Number of agents to remove.
        """
        for _ in range(num_cars):
            agent = random.choice(self.get_agents_by_type("CarAgent"))
            self.agents.remove(agent)

    def create_lights(self, num_lights: int, position: str) -> None:
        """Function to add traffic lights to the model.

        Args:
            num_agents (int): Number of agents to add.
        """
        LightAgent.create_agents(model=self, n=num_lights, position=position)

    def create_lights_for_intersections(self) -> None:
        """Function to add traffic lights to the model.

        Args:
            tbd
        """
        for intersection in self.grid.get_nodes("intersection"):
            LightAgent.create_agents(model=self, n=1, position=intersection)

    def get_agents_by_type(self, agent_type: str) -> list:
        """Function to get all agents of a certain type.

        Args:
            agent_type (str): Type of agents to get.

        Returns:
            list: A list of agents of the given type.
        """
        if agent_type == "CarAgent":
            return [agent for agent in self.agents if isinstance(agent, CarAgent)]
        elif agent_type == "LightAgent":
            return [agent for agent in self.agents if isinstance(agent, LightAgent)]
        else:
            raise ValueError(f"Agent type {agent_type} not found")

    def get_agents_by_id(self, agent_id: list) -> list:
        """Function to get all agents by their unique ID.

        Args:
            agent_id (list): List of unique agent IDs.

        Returns:
            list: A list of agents with the given unique IDs.
        """
        return [agent for agent in self.agents if agent.unique_id in agent_id]

    def get_last_intersection_of_car(self, agent_unique_id: int) -> str:
        """Function to get the last position of a car.

        Args:
            car_id (int): ID of the car.

        Returns:
            str: The last position of the car.
        """
        car = self.get_agents_by_id([agent_unique_id])[0]
        car_full_path = self.agent_paths[agent_unique_id]
        car_full_path_keys = list(car_full_path.keys())
        current_position_index = car_full_path_keys.index(car.position)

        if current_position_index == 0:
            previous_position = car.position
        else:
            previous_position = car_full_path_keys[current_position_index - 1]

        # Get the corresponding intersection, if the cars last position was a border node (TODO)
        if previous_position.startswith("border"):
            first_intersection = list(car.model.agent_paths[car.unique_id].keys())[1]
            lane = list(self.grid.neighbors(previous_position))
            lane.remove(first_intersection)
            previous_position = lane[0]

        return previous_position

    def update_cars_waiting_times(self) -> None:
        """Function to update the waiting times of all cars at each intersection."""

        for car in self.get_agents_by_type("CarAgent"):
            if car.unique_id not in list(self.cars_waiting_times.keys()):
                self.cars_waiting_times[car.unique_id] = {
                    intersection: 0
                    for intersection in list(
                        car.model.agent_paths[car.unique_id].keys()
                    )
                    if intersection.startswith("intersection")
                }

            if car.waiting:
                self.cars_waiting_times[car.unique_id][car.position] += 1

    def update_agent_paths(self) -> None:
        """Function to update the paths of all agents."""
        for car in self.get_agents_by_type("CarAgent"):
            if car.unique_id not in list(self.agent_paths.keys()):
                self.agent_paths[car.unique_id] = car.path.copy()

    def agent_respawn(self):
        self.steps
        if self.num_cars_hist[-1] < self.num_cars:
            direction = 1
        elif self.num_cars_hist[-1] > self.num_cars:
            direction = -1
        else:
            direction = random.choice([-1, 1])
        current = (
            self.num_cars_hist[-1] + direction + random.randint(-1, 1)
        )  # Introduce variance
        if current > self.num_cars * 2:
            current = self.num_cars * 2
        elif current < 0:
            current = 0
        # Normalize the value to be between 0 and 1
        min_val = 0
        max_val = self.num_cars * 2
        normalized_value = (current - min_val) / (max_val - min_val)
        if random.random() < normalized_value:
            self.create_cars(1)
