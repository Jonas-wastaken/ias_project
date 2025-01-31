import mesa
from car import CarAgent, AgentArrived
from graph import Graph


class TrafficModel(mesa.Model):
    def __init__(self, num_agents: int, seed: int = None):
        """Mesa Model providing the environment.

        Args:
            num_agents (int): Number of agents to spawn.
            seed (int, optional): Seed used in model generation. Defaults to None.
        """
        super().__init__(seed=seed)
        self.num_agents = num_agents
        self.grid = Graph()

        # Create agents
        agents = CarAgent.create_agents(model=self, n=num_agents)

    def step(self) -> None:
        """Actions the agents do each step."""
        for agent in self.agents[:]:
            try:
                agent.move()
            except AgentArrived:
                self.agents.remove(agent)
