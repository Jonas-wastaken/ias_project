import mesa
from car import CarAgent
from graph import Graph


class TrafficModel(mesa.Model):
    def __init__(self, num_agents, seed=None):
        super().__init__(seed=seed)
        self.num_agents = num_agents
        self.grid = Graph()

        # Create agents
        agents = CarAgent.create_agents(model=self, n=num_agents)

        for agent in agents:
            self.grid.place_agent(agent_id=agent.unique_id)

    def step(self):
        pass
