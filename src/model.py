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
        **create_agents(self, num_agents: int) -> None**:
            Function to add agents to the model.
        **remove_agents(self, num_agents: int) -> None**:
            Function to randomly remove n agents from the model.
        **create_light_agents(self) -> None**:
            Function to add traffic lights to the model.
        **get_agents_by_type(self, agent_type: str) -> list**:
            Function to get all agents of a certain type.
        **get_agents_by_id(self, agent_id: list) -> list**:
            Function to get all agents by their unique ID.
        **get_last_intersection_of_car(self, car_id: int) -> str**:
            Function to get the last position of a car.
        **update_cars_waiting_times(self) -> None**:
            Function to update the waiting times of all cars at each intersection.
    """

    def __init__(self, num_agents: int, seed: int = None, **kwargs):
        """Initializes a new traffic environment.

        Args:
            num_agents (int): Number of agents to spawn.
            seed (int, optional): Seed used in model generation. Defaults to None.
            **kwargs: Additional keyword arguments for configuring the graph object.
        """
        super().__init__(seed=seed)
        self.kwargs = kwargs
        self.num_agents = num_agents
        self.grid = Graph(
            num_intersections=self.kwargs.get("num_intersections", 10),
            num_borders=self.kwargs.get("num_borders", 3),
            min_distance=self.kwargs.get("min_distance", 2),
            max_distance=self.kwargs.get("max_distance", 10),
        )
        self.create_light_agents()
        CarAgent.create_agents(model=self, n=num_agents)
        self.agent_paths = {
            agent.unique_id: agent.path.copy()
            for agent in self.get_agents_by_type("CarAgent")
        }
        self.cars_waiting_times = {}
        for car in self.get_agents_by_type("CarAgent"):
            self.cars_waiting_times[car.unique_id] = {intersection : 0 for intersection in list(car.model.agent_paths[car.unique_id].keys()) if intersection.startswith("intersection")}

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

        self.update_cars_waiting_times()

        for light in self.get_agents_by_type("LightAgent"):
            # light.update_waiting_cars()

            # Decide if the light should change the open lane (if the cooldown is over)
            if light.current_switching_cooldown <= 0:
                light.rotate_in_open_lane_cycle()
            light.current_switching_cooldown -= 1

    def create_agents(self, num_agents: int) -> None:  # TODO: rename function
        """Function to add agents to the model.

        Args:
            num_agents (int): Number of agents to add.
        """
        CarAgent.create_agents(model=self, n=num_agents)

    def remove_agents(self, num_agents: int) -> None:  # TODO: rename function
        """Function to randomly remove n agents from the model.

        Args:
            num_agents (int): Number of agents to remove.
        """
        for i in range(num_agents):
            agent = random.choice(self.get_agents_by_type("CarAgent"))
            self.agents.remove(agent)

    def create_light_agents(self) -> None:
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
            if car.waiting:
                self.cars_waiting_times[car.unique_id][car.position] += 1
