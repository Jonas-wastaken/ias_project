"""This module contains:
- TrafficModel class: A Mesa model simulating traffic."""

import mesa
from src.car import CarAgent, AgentArrived
from src.graph import Graph


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
        _agents = CarAgent.create_agents(model=self, n=num_agents)
        self.agent_paths = {agent.unique_id: agent.path.copy() for agent in _agents}

    def step(self) -> None:
        """Advances the environment to next state.

        - Each CarAgent moves to it's next position. If a CarAgent reached it's goal, it is removed from the AgentSet at the next step
        """
        for agent in self.agents[:]:
            try:
                agent.move()
            except AgentArrived:
                self.agents.remove(agent)
