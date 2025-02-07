"""This module contains:
- CarAgent class: Represents a car navigating an undirected graph.
- AgentArrived exception: Raised when an agent has reached it's goal."""

import mesa
import random
from networkx import dijkstra_path


class CarAgent(mesa.Agent):
    """Agent, which represents a car navigating an undirected graph. It receives it's starting position from the graph instance and then computes a random node as it's goal. Path computation uses Dijkstra's algorithm.

    Inherits from mesa.Agent.

    Attributes:
        start (str): The ID of the node, where the agent starts.
        goal (str): The ID of the node, which is the agent's goal.
        path (list[str]): The path the agent takes to reach it's goal.
        position (str): The ID of the node, where the agent is currently located.

    ## Methods:
        **compute_goal(self) -> str**:
            Assigns a random border node, which is not the starting node, as the goal of the agent.
        **move(self) -> None**:
            Moves the agent to it's next step on the path and sends updated position to the grid.
    """

    def __init__(self, model: mesa.Model):
        """Initializes a new CarAgent. The agent is placed on a random border node, and computes a random goal and the best path there.

        Args:
            model (mesa.Model): The model instance in which the agent lives.
        """
        super().__init__(model)
        self.start = self.model.grid.place_agent(agent_id=self.unique_id)
        self.goal = self.compute_goal()
        self.path = dijkstra_path(
            self.model.grid, self.start, self.goal, weight="weight"
        )
        self.position = self.start

    def compute_goal(self) -> str:
        """Assigns a random border node, which is not the starting node, as the goal of the agent.

        Returns:
            str: ID of the node, which is the agent's goal.
        """
        borders = [node for node in self.model.grid if node.find("border") == 0]
        borders.remove(self.start)
        assigned_goal = borders[random.randint(0, (len(borders) - 1))]

        return assigned_goal

    def move(self) -> None:
        """Moves the agent to it's next step on the path and sends updated position to the grid.

        Raises:
            AgentArrived: If the agent has reached it's goal, the exception is raised.
        """
        if self.position == self.goal:
            raise AgentArrived(
                message=f"The Agent {self.unique_id} arrived at it's goal"
            )
        else:
            next_step = self.path[self.path.index(self.position) + 1]
            self.model.grid.move_agent(self, next_step)
            self.position = next_step


class AgentArrived(Exception):
    """Exception raised when an agent has reached it's goal."""

    def __init__(self, message: str):
        """Initializes AgentArrived exception.

        Args:
            message (str): The message to be displayed when the exception is raised.
        """
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"
