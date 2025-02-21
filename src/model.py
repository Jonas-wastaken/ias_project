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
            min_distance=self.kwargs.get("min_distance", 1),
            max_distance=self.kwargs.get("max_distance", 10),
        )
        self.create_light_agents()
        CarAgent.create_agents(model=self, n=num_agents)
        self.agent_paths = {agent.unique_id: agent.path.copy() for agent in self.get_agents_by_type("CarAgent")}

    def step(self) -> None:
        """Advances the environment to next state.

        - Each CarAgent moves to it's next position. If a CarAgent reached it's goal, it is removed from the AgentSet at the next step
        """
        for car in self.get_agents_by_type("CarAgent")[:]:
            try:
                car.move()
            except AgentArrived:
                car.remove()

        for light in self.get_agents_by_type("LightAgent"):
            light.update_waiting_cars()
            
            # Decide if the light should change the open lane (if the cooldown is over)
            

            light.current_switching_cooldown -= 1

    def create_agents(self, num_agents: int) -> None:       # TODO: rename function
        """Function to add agents to the model.

        Args:
            num_agents (int): Number of agents to add.
        """
        CarAgent.create_agents(model=self, n=num_agents)

    def remove_agents(self, num_agents: int) -> None:           # TODO: rename function
        """Function to randomly remove n agents from the model.

        Args:
            num_agents (int): Number of agents to remove.
        """
        for i in range(num_agents):
            agent = random.choice(self.agents)
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


    

