"""This module contains:
- Graph class: Represents an undirected graph with intersection and border nodes. It creates edges between them with random weights and stores agent positions."""

import networkx as nx
import random
import pickle


class Graph(nx.Graph):
    """A class to represent an undirected graph.

    The class inherits from the networkx.Graph class.
    The class creates nodes labeled as 'intersection' and 'border', and adds edges between them with random weights.
    Self-loops are removed from the graph.

    Attributes:
        agent_positions (dict): A dictionary to keep track of the agents' positions.

    ## Methods:
        **add_intersections(self, num_intersections: int) -> None**:
            Add intersection nodes to the graph.
        **add_borders(self, num_borders: int) -> None**:
            Add border nodes to the graph.
        **connect_intersections(self, num_intersections: int, weight_range: tuple[int, int]) -> None**:
            Add edges between intersection nodes with random weights.
        **connect_borders(self, num_intersections: int, num_borders: int, weight_range: tuple[int, int]) -> None**:
            Add edges between border and intersection nodes with random weights.
        **place_agent(self, agent_id: int) -> str**:
            Place an agent on a random border node and store position internally.
        **move_agent(self, agent_id: int, new_position: str) -> None**:
            Move an agent to it's next position.
        **save(self, filename: str = "graph.pickle") -> None**:
            Save class instance to a pickle file.
    """

    def __init__(
        self,
        num_intersections: int = 10,
        num_borders: int = 3,
        weight_range: tuple[int, int] = (1, 10),
    ):
        """Initializes a new Graph instance with intersection and border nodes and edges between them.

        Args:
            num_intersections (int, optional): The number of intersection nodes to create. Defaults to 10.
            num_borders (int, optional): The number of border nodes to create. Defaults to 3.
            weight_range (tuple[int, int], optional): A tuple specifying the range of weights for the edges. Defaults to (1, 10).
        """

        super().__init__()
        self.add_intersections(num_intersections)
        self.add_borders(num_borders)
        self.connect_intersections(num_intersections, weight_range)
        self.connect_borders(num_intersections, num_borders, weight_range)
        super().remove_edges_from(nx.selfloop_edges(self))

        self.agent_positions = {}

    def add_intersections(self, num_intersections: int) -> None:
        """Add intersection nodes to the graph.

        Args:
            num_intersections (int): The number of intersection nodes to add.
        """
        super().add_nodes_from(
            [
                (f"intersection_{i}", {"type": "intersection"})
                for i in range(num_intersections)
            ]
        )

    def add_borders(self, num_borders: int) -> None:
        """Add border nodes to the graph.

        Args:
            num_borders (int): The number of border nodes to add.
        """
        super().add_nodes_from(
            [(f"border_{i}", {"type": "border"}) for i in range(num_borders)]
        )

    def connect_intersections(
        self, num_intersections: int, weight_range: tuple[int, int]
    ) -> None:
        """Add edges between intersection nodes with random weights.

        Args:
            num_intersections (int): The number of intersection nodes in the graph.
            weight_range (tuple[int, int]): A tuple specifying the range of weights for the edges.
        """
        for i in range(num_intersections):
            num_edges = random.randint(1, 4)
            for _ in range(num_edges):
                super().add_edge(
                    f"intersection_{i}",
                    f"intersection_{random.randint(0, (num_intersections - 1))}",
                    weight=random.randint(weight_range[0], weight_range[1]),
                )

    def connect_borders(
        self, num_intersections: int, num_borders: int, weight_range: tuple[int, int]
    ) -> None:
        """Add edges between border and intersection nodes with random weights.

        Args:
            num_intersections (int): The number of intersection nodes in the graph.
            num_borders (int): The number of border nodes in the graph.
            weight_range (tuple[int, int]): A tuple specifying the range of weights for the edges.
        """
        for i in range(num_borders):
            super().add_edge(
                f"border_{i}",
                f"intersection_{random.randint(0, (num_intersections - 1))}",
                weight=random.randint(weight_range[0], weight_range[1]),
            )

    def place_agent(self, agent_id: int) -> str:
        """Places an agent on a random border node and stores position internally.

        Args:
            agent_id (id): ID of agent being placed

        Returns:
            str: ID of assigned node the agent spawns on
        """
        borders = [node for node in self.nodes if node.find("border") == 0]
        assigned_start = borders[random.randint(0, (len(borders) - 1))]

        if assigned_start not in self.agent_positions:
            self.agent_positions[assigned_start] = []
        self.agent_positions[assigned_start].append(agent_id)

        return assigned_start

    def move_agent(self, agent_id: int, new_position: str) -> None:
        """Moves an agent to it's next position

        Args:
            agent_id (int): ID of agent being placed.
            new_position (str): ID of the node the agent moves to.
        """
        self.agent_positions[agent_id] = new_position

    def save(self, filename: str = "graph.pickle") -> None:
        """Save class instance to a pickle file.

        Args:
            filename (str, optional): The name of the file to save the class instance to.
        """

        pickle.dump(self, open(filename, "wb"))
