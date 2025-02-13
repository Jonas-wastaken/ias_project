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
        num_intersections (int): The number of intersection nodes.
        num_borders (int): The number of border nodes.
        min_distance (int): Minimum distance between two connected nodes.
        max_distance (int): Maximum distance between two connected nodes.
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
        num_intersections: int,
        num_borders: int,
        min_distance: int,
        max_distance: int,
    ):
        """Initializes a new Graph instance with intersection and border nodes and edges between them.

        Args:
            num_intersections (int): The number of intersection nodes to create.
            num_borders (int): The number of border nodes to create.
            weight_range (tuple[int, int]): A tuple specifying the range of weights for the edges.
        """

        super().__init__()
        self.num_intersections = num_intersections
        self.num_borders = num_borders
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.add_intersections()
        self.add_borders()
        self.connect_intersections()
        self.connect_borders()

        self.agent_positions = {}

    def add_intersections(self) -> None:
        """Add intersection nodes to the graph."""
        super().add_nodes_from(
            [
                (f"intersection_{i}", {"type": "intersection"})
                for i in range(self.num_intersections)
            ]
        )

    def add_borders(self) -> None:
        """Add border nodes to the graph."""
        super().add_nodes_from(
            [(f"border_{i}", {"type": "border"}) for i in range(self.num_borders)]
        )

    def connect_intersections(self) -> None:
        """Connects each intersection node to min. 2 and max. 4 other intersection nodes.

        - Initializes a list with all nodes of type intersection.
        - Initializes a dictionary to keep track of connections for each intersection node.
        - For each intersection node:
            - Ensures it is connected to at least 2 and at most 4 other intersection nodes.
            - Selects available nodes that are not already fully connected.
            - Randomly selects a number of nodes to connect to, ensuring it does not exceed the limits.
            - Adds the selected nodes to the connections of the current node and vice versa.
        - Adds weighted edges between connected nodes with weights randomly chosen from the specified range."""
        intersections = [node for node in self.nodes if node.startswith("intersection")]
        connections = {node: set() for node in intersections}

        for node in intersections:
            while len(connections[node]) < 2 or len(connections[node]) > 4:
                available_nodes = [
                    x for x in intersections if x != node and len(connections[x]) < 4
                ]
                if not available_nodes:
                    break

                num_to_connect = min(4 - len(connections[node]), random.randint(1, 4))
                num_to_connect = min(num_to_connect, len(available_nodes))
                if num_to_connect <= 0:
                    break

                selected_nodes = random.sample(available_nodes, num_to_connect)

                for target_node in selected_nodes:
                    connections[node].add(target_node)
                    connections[target_node].add(node)

        for node, target_nodes in connections.items():
            edges = [
                (
                    node,
                    target_node,
                    random.randint(self.min_distance, self.max_distance),
                )
                for target_node in target_nodes
            ]
            super().add_weighted_edges_from(edges)

    def connect_borders(self) -> None:
        """Add edges between border and intersection nodes with random weights.

        - Initializes a list with all nodes of type border.
        - Initializes a list with all nodes of type intersection.
        - Iterates through borders, adding an edge between the each border and a random intersection."""
        borders = [node for node in self.nodes if node.startswith("border")]
        intersections = [node for node in self.nodes if node.startswith("intersection")]

        while borders:
            border = borders.pop()
            super().add_edge(
                border,
                random.choice(intersections),
                weight=random.randint(self.min_distance, self.max_distance),
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
