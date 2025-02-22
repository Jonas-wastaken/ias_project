"""This module contains:
- Graph class: Represents an undirected graph with intersection and border nodes. It creates edges between them with random weights and stores agent positions."""

import networkx as nx
import random
import pickle


class Graph(nx.Graph):
    """A class to represent an undirected graph.

    The class inherits from the networkx.Graph class.
    The class creates nodes labeled as 'intersection' and 'border', and adds edges between them with random weights.

    Attributes:
        min_distance (int): Minimum distance between two connected nodes.
        max_distance (int): Maximum distance between two connected nodes.
        agent_positions (dict): A dictionary to keep track of the agents' positions.

    ## Methods:
        **add_intersections(self, num_intersections: int) -> None**:
            Add intersection nodes to the graph.
        **remove_intersections(self, num_intersections: int) -> None**:
            Remove the last n intersection nodes from the graph.
        **connect_intersections(self, new_intersections: list) -> None**:
            Connects each intersection node to min. 2 and max. 4 other intersection nodes.
        **add_borders(self, num_borders: int) -> None**:
            Add border nodes to the graph.
        **remove_borders(self, num_borders: int) -> None**:
            Remove the last n border nodes from the graph.
        **connect_borders(self) -> None**:
            Connects each border between two connected intersections.
        **change_weights(self, min_distance: int, max_distance: int) -> None**:
            Change the weights of the edges in the graph.
        **place_agent(self, agent_id: int) -> str**:
            Place an agent on a random border node and store position internally.
        **move_agent(self, agent_id: int, new_position: str) -> None**:
            Move an agent to its next position.
        **save(self, filename: str = "graph.pickle") -> None**:
            Save class instance to a pickle file.
        **load(cls, filename: str = "graph.pickle") -> Graph**:
            Load a class instance from a pickle file.
        **get_nodes(self, type: str = None) -> list**:
            Get all nodes of a specific type.
        **get_connections(self, \*\*kwargs) -> dict**:
            Get all connections between nodes.
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
            min_distance (int): The minimum distance between nodes.
            max_distance (int): The maximum distance between nodes.
        """

        super().__init__()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.add_intersections(num_intersections)
        self.add_borders(num_borders)

    def add_intersections(self, num_intersections: int) -> None:
        """Add intersection nodes to the graph.

        Args:
            num_intersections (int): The number of intersection nodes to create.
        """
        index = len(self.get_nodes("intersection"))

        new_intersections = [
            (f"intersection_{i}", {"type": "intersection"})
            for i in range(index, index + num_intersections)
        ]

        super().add_nodes_from(new_intersections)

        self.connect_intersections([node[0] for node in new_intersections])

    def remove_intersections(self, num_intersections: int) -> None:
        """Removes the last n intersection nodes from the graph.

        Args:
            num_intersections (int): The number of intersection nodes to remove.
        """
        intersections_to_remove = self.get_nodes("intersection")[-num_intersections:]

        super().remove_nodes_from(intersections_to_remove)

        self.connect_intersections(self.get_nodes("intersection"))
        self.connect_borders()

    def connect_intersections(self, new_intersections: list) -> None:
        """Connects each intersection node to min. 2 and max. 4 other intersection nodes.

        - Initializes a list with all nodes of type intersection.
        - Gets established connections between intersection nodes.
        - For each new intersection node:
            - Ensures it is connected to at least 2 and at most 4 other intersection nodes.
            - Selects available nodes that are not already fully connected.
            - Randomly selects a number of nodes to connect to, ensuring it does not exceed the limits.
            - Adds the selected nodes to the connections of the current node and vice versa.
        - Adds weighted edges between connected nodes with weights randomly chosen from the specified range.

        Args:
            new_intersections (list): A list of new intersection nodes to connect to other intersection nodes
        """
        intersections = self.get_nodes("intersection")
        intersection_connections = {
            key: [v for v in value if v.startswith("intersection")]
            for key, value in self.get_connections(type="intersection").items()
        }

        for node in new_intersections:
            while len(intersection_connections[node]) < 2:
                available_nodes = [
                    x
                    for x in intersections
                    if x != node and len(intersection_connections[x]) < 4
                ]
                if not available_nodes:
                    for u, connected_intersections in intersection_connections.items():
                        if len(connected_intersections) > 2:
                            for v in connected_intersections:
                                if len(intersection_connections[v]) > 2:
                                    intersection_connections[u].remove(v)
                                    intersection_connections[v].remove(u)
                                    available_nodes = [u, v]
                                    break
                            else:
                                continue
                            break
                    else:
                        continue
                    break

                num_to_connect = min(
                    4 - len(intersection_connections[node]), random.randint(2, 4)
                )
                num_to_connect = min(num_to_connect, len(available_nodes))
                if num_to_connect <= 0:
                    break

                selected_nodes = random.sample(available_nodes, num_to_connect)

                for target_node in selected_nodes:
                    intersection_connections[node].append(target_node)
                    intersection_connections[target_node].append(node)

        for node, target_nodes in intersection_connections.items():
            edges = [
                (
                    node,
                    target_node,
                    random.randint(self.min_distance, self.max_distance),
                )
                for target_node in target_nodes
            ]
            super().add_weighted_edges_from(edges)

    def add_borders(self, num_borders: int) -> None:
        """Add border nodes to the graph.

        Args:
            num_borders (int): The number of border nodes to create.
        """
        index = len(self.get_nodes("border"))

        new_borders = [
            (f"border_{i}", {"type": "border"})
            for i in range(index, index + num_borders)
        ]

        super().add_nodes_from(new_borders)

        self.connect_borders()

    def remove_borders(self, num_borders: int) -> None:
        """Removes the last n border nodes from the graph.

        Args:
            num_borders (int): The number of border nodes to remove.
        """

        super().remove_nodes_from(self.get_nodes("border")[-num_borders:])

    def connect_borders(self) -> None:
        """Connects each border between two connected intersections.

        - Gets a random intersection
        - Gets the second intersection randomly from first intersection's connections
        """
        intersections = self.get_nodes("intersection")

        free_borders = [
            border[0]
            for border in dict(self.degree(self.get_nodes("border"))).items()
            if border[1] < 2
        ]

        while free_borders:
            border = free_borders.pop()
            if self.degree(border) == 0:
                intersection_1 = random.choice(intersections)
            elif self.degree(border) == 1:
                intersection_1 = list(self.neighbors(border))[0]
            else:
                continue
            intersection_2 = random.choice(
                [
                    node
                    for node in list(self.neighbors(intersection_1))
                    if node.startswith("intersection")
                ]
            )
            total_weight = super().get_edge_data(intersection_1, intersection_2)[
                "weight"
            ]
            weight_1 = random.randint(int(self.min_distance / 2), total_weight)
            weight_2 = total_weight - weight_1

            super().add_edge(
                u_of_edge=border,
                v_of_edge=intersection_1,
                weight=weight_1,
            )

            super().add_edge(
                u_of_edge=border,
                v_of_edge=intersection_2,
                weight=weight_2,
            )

    def change_weights(self, min_distance: int, max_distance: int) -> None:
        """Change the weights of the edges in the graph.

        Args:
            min_distance (int): The new minimum distance between two connected nodes.
            max_distance (int): The new maximum distance between two connected nodes.
        """
        [
            nx.set_edge_attributes(
                self,
                {
                    (edge[0], edge[1]): {
                        "weight": random.randint(min_distance, max_distance)
                    }
                },
            )
            for edge in list(self.edges)
            if edge[0].startswith("intersection") and edge[1].startswith("intersection")
        ]

        border_connections = self.get_connections(filter_by="border")
        for key in border_connections.keys():
            intersection_1 = border_connections[key][0]
            intersection_2 = border_connections[key][1]
            total_weight = self.get_edge_data(intersection_1, intersection_2)["weight"]
            weight_1 = (
                random.randint(min_distance, total_weight - 1)
                if min_distance != total_weight
                else total_weight - 1
            )
            weight_2 = total_weight - weight_1

            nx.set_edge_attributes(
                self,
                {(key, intersection_1): {"weight": weight_1}},
            )

            nx.set_edge_attributes(
                self,
                {(key, intersection_2): {"weight": weight_2}},
            )

    def place_agent(self, agent_id: int) -> str:
        """Places an agent on a random border node and stores position internally.

        Args:
            agent_id (id): ID of agent being placed

        Returns:
            str: ID of assigned node the agent spawns on
        """
        borders = self.get_nodes(type="border")
        assigned_start = borders[random.randint(0, (len(borders) - 1))]

        return assigned_start

    def save(self, filename: str = "graph.pickle") -> None:
        """Save class instance to a pickle file.

        Args:
            filename (str, optional): The name of the file to save the class instance to.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename: str = "graph.pickle") -> "Graph":
        """Load a class instance from a pickle file.

        Args:
            filename (str, optional): The name of the file to load the class instance from.

        Returns:
            Graph: A new instance of the Graph class loaded from
            the specified pickle file.
        """

        with open(filename, "rb") as file:
            return pickle.load(file)

    def get_nodes(self, type: str = None) -> list:
        """Get all nodes of a specific type.

        Args:
            type (str, optional): The type of nodes to get.

        Returns:
            list: A list of nodes of the specified type.
        """
        if type:
            return [node for node in self.nodes if node.startswith(type)]
        return self.nodes

    def get_connections(self, **kwargs) -> dict:
        """Get all connections between nodes.

        Args:
            \*\*kwargs
                filter_by (str, optional): A string to filter nodes by type and optionally by ID, formatted as "type_id".
                                        If only type is provided, all nodes of that type are considered. Defaults to None.
                weights (bool, optional): Specifies whether weights should be returned.

        Returns:
            dict: A dictionary with nodes as keys and a list of their connected nodes as values.
                  If weights is True, the list will contain tuples with the connected node and the edge weight.
        """
        filter_by = kwargs.get("filter_by", None)
        weights = kwargs.get("weights", False)

        if filter_by:
            try:
                type, id = filter_by.split("_")
            except ValueError:
                type, id = filter_by, None
            if id:
                nodes = [f"{type}_{id}"]
            else:
                nodes = self.get_nodes(type)
        else:
            nodes = self.nodes

        if weights:
            return {
                node: [x[1:] for x in list(self.edges(node, data="weight"))]
                for node in nodes
            }
        else:
            return {node: [x[1] for x in list(self.edges(node))] for node in nodes}
