"""This module contains:
- TrafficModel class: A Mesa model simulating traffic."""

import mesa
from car import CarAgent, AgentArrived
from graph import Graph


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
    """

    def __init__(self, num_agents: int, seed: int = None):
        """Initializes a new traffic environment.

        Args:
            num_agents (int): Number of agents to spawn.
            seed (int, optional): Seed used in model generation. Defaults to None.
        """
        super().__init__(seed=seed)
        self.num_agents = num_agents
        self.grid = Graph()
        agents = CarAgent.create_agents(model=self, n=num_agents)

    def step(self) -> None:
        """Advances the environment to next state.

        - Each CarAgent moves to it's next position. If a CarAgent reached it's goal, it is removed from the AgentSet at the next step
        """
        for agent in self.agents[:]:
            try:
                agent.move()
            except AgentArrived:
                self.agents.remove(agent)
