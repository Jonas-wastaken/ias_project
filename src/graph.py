"""
graph.py

This module provides the Graph class, an extension of `networkx.Graph`, designed
to represent a traffic network. It facilitates the creation and management of
a graph structure composed of 'intersection' and 'border' nodes connected by
weighted edges (roads).

The Graph class includes functionalities for:
- Adding and removing intersection and border nodes.
- Establishing connections between nodes with randomized weights within specified bounds.
- Modifying edge weights.
- Placing agents (e.g., cars) onto random border nodes.
- Retrieving nodes based on type and querying node connections.

Classes:
    - Graph: Represents the traffic grid as an undirected graph, extending networkx.Graph.

Dependencies:
    - networkx: Core graph manipulation library.
    - random: For randomization in node connections and edge weights.
    - pickle: For object serialization (saving/loading the graph).
"""

import pickle
import random

import networkx as nx


class Graph(nx.Graph):
    """Represents the traffic network as an undirected graph.

    This class extends `networkx.Graph` to model a traffic grid. It manages
    'intersection' and 'border' nodes, connecting them with weighted edges
    representing roads. It provides methods for building, modifying, and
    querying the graph structure, as well as saving/loading capabilities.

    Attributes:
        min_distance (int): The minimum weight (distance) assigned to edges
                            connecting intersection nodes. Must be >= 2.
        max_distance (int): The maximum weight (distance) assigned to edges
                            connecting intersection nodes.

    Methods:
        add_intersections(self, num_intersections: int) -> None:
            Add intersection nodes and connect them.
        connect_intersections(self, new_intersections: list) -> None:
            Establish connections between intersection nodes.
        add_borders(self, num_borders: int) -> None:
            Add border nodes and connect them.
        connect_borders(self) -> None:
            Connect border nodes between intersection pairs.
        save(self, filename: str = "graph.pickle") -> None:
            Save the graph instance to a file.
        load(cls, filename: str = "graph.pickle") -> "Graph":
            Load a graph instance from a file.
        get_nodes(self, type: str = None) -> list:
            Retrieve node IDs, optionally filtered by type.
        get_connections(self, **kwargs) -> dict:
            Retrieve connections for specified nodes.
    """

    def __init__(
        self,
        num_intersections: int,
        num_borders: int,
        min_distance: int,
        max_distance: int,
    ):
        """Initializes the Graph with intersections, borders, and connections.

        Creates the specified number of intersection and border nodes, then
        connects them according to the class logic. Ensures `min_distance` is
        at least 2 to allow for border node insertion between intersections.

        Args:
            num_intersections (int): The number of intersection nodes to create.
            num_borders (int): The number of border nodes to create.
            min_distance (int): The minimum distance for edges between intersections.
                                If less than 2, it will be set to 2.
            max_distance (int): The maximum distance for edges between intersections.
        """
        super().__init__()
        self.min_distance = min_distance if min_distance >= 2 else 2
        self.max_distance = max_distance
        self.add_intersections(num_intersections)
        self.add_borders(num_borders)

    def add_intersections(self, num_intersections: int) -> None:
        """Adds a specified number of intersection nodes to the graph.

        New intersection nodes are named sequentially (e.g., 'intersection_0',
        'intersection_1', ...). After adding, it attempts to connect the new
        intersections to existing ones.

        Args:
            num_intersections (int): The quantity of intersection nodes to add.
        """
        index = len(self.get_nodes("intersection"))

        new_intersections = [
            (f"intersection_{i}", {"type": "intersection"})
            for i in range(index, index + num_intersections)
        ]

        super().add_nodes_from(new_intersections)

        self.connect_intersections([node[0] for node in new_intersections])

    def connect_intersections(self, new_intersections: list) -> None:
        """Connects intersection nodes, ensuring connectivity constraints are met.

        This method iterates through a list of specified intersection nodes (typically
        newly added ones) and attempts to establish connections with other intersection
        nodes within the graph. The primary goal is to ensure that every intersection
        node adheres to the connectivity constraints: having a minimum of 2 and a
        maximum of 4 connections to other intersections.

        Connection Logic:
        1.  Retrieves all current intersection nodes and their existing connections.
        2.  For each node in `new_intersections`:
            a.  It enters a loop that continues as long as the node has fewer than 2
                intersection connections.
            b.  Identifies potential `available_nodes` to connect to. These are
                intersection nodes other than the current node that currently have
                fewer than 4 connections.
            c.  **Constraint Handling:** If no `available_nodes` are found (meaning all
                other intersections already have 4 connections), the method attempts
                to free up connection slots. It searches for an existing edge (u, v)
                where both nodes `u` and `v` have more than 2 connections. If found,
                this edge is removed (conceptually, within the temporary connection
                tracking), making `u` and `v` available for new connections.
                *Note: If no such edge can be found to break, the connection process
                for the current node might halt before reaching the minimum of 2.*
            d.  Calculates the number of new connections to add (`num_to_connect`),
                considering the node's remaining capacity (up to 4), a random target
                between 2 and 4, and the actual number of `available_nodes`.
            e.  Randomly selects `num_to_connect` nodes from the `available_nodes`.
            f.  Updates the temporary connection tracking by adding bidirectional links
                between the current node and the selected target nodes.
        3.  After processing all nodes in `new_intersections`, the method iterates
            through the final connection map.
        4.  For each node and its target connections, it generates weighted edges.
            The edge weight is calculated using a formula that biases towards the
            `self.min_distance`:
            `weight = int((max_dist - min_dist) * (random.random()**2) + min_dist)`
        5.  These weighted edges are then added to the graph using the parent class's
            `add_weighted_edges_from` method, permanently modifying the graph structure.

        Args:
            new_intersections (list[str]): A list of intersection node IDs (strings)
                                           for which connections need to be established
                                           or verified according to the constraints.

        Notes:
            - This method directly modifies the graph by adding edges.
            - It relies on `self.get_nodes`, `self.get_connections`,
              `self.min_distance`, and `self.max_distance`.
            - The process prioritizes satisfying the minimum connection constraint (2)
              and respects the maximum constraint (4).
            - The edge weight calculation introduces randomness but favors shorter
              distances due to squaring the random factor.
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
                    int(
                        (self.max_distance - self.min_distance) * (random.random() ** 2)
                        + self.min_distance
                    ),
                )
                for target_node in target_nodes
            ]
            super().add_weighted_edges_from(edges)

    def add_borders(self, num_borders: int) -> None:
        """Adds a specified number of border nodes to the graph.

        New border nodes are named sequentially (e.g., 'border_0', 'border_1', ...).
        After adding, it attempts to connect these border nodes between existing
        connected intersection pairs.

        Args:
            num_borders (int): The quantity of border nodes to add.
        """
        index = len(self.get_nodes("border"))

        new_borders = [
            (f"border_{i}", {"type": "border"})
            for i in range(index, index + num_borders)
        ]

        super().add_nodes_from(new_borders)

        self.connect_borders()

    def connect_borders(self) -> None:
        """Connects border nodes with fewer than two connections into the graph.

        This method iterates through all border nodes currently having 0 or 1
        connections ('free' borders). For each free border, it aims to establish
        exactly two connections, linking it to a pair of intersection nodes that
        are already directly connected to each other.

        Connection Logic:
        1.  Identifies all border nodes with `self.degree(border) < 2`.
        2.  For each such `border`:
            a.  Determines the first connection point (`intersection_1`):
                - If `degree == 0`, a random intersection node is chosen.
                - If `degree == 1`, the existing neighbor (which must be an
                  intersection based on graph structure rules) is used.
            b.  Selects a second connection point (`intersection_2`) by randomly
                choosing an intersection node that is a direct neighbor of
                `intersection_1`. This ensures the border is placed logically
                between two connected intersections.
            c.  Retrieves the weight (`total_weight`) of the existing direct edge
                between `intersection_1` and `intersection_2`.
            d.  Calculates two new weights (`weight_1`, `weight_2`) such that
                `weight_1 + weight_2 = total_weight`, and both `weight_1` and
                `weight_2` are at least 1. This is done by choosing `weight_1`
                randomly between 1 and `total_weight - 1`.
            e.  Adds two new edges to the graph:
                - (`border`, `intersection_1`) with weight `weight_1`.
                - (`border`, `intersection_2`) with weight `weight_2`.
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
            weight_1 = random.randint(1, total_weight - 1)
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

    def save(self, filename: str = "graph.pickle") -> None:
        """Saves the current Graph instance to a pickle file.

        Serializes the Graph object, including its nodes, edges, attributes,
        and internal state (`min_distance`, `max_distance`), into the specified file.

        Args:
            filename (str, optional): The path and name of the file to save the
                                      graph to. Defaults to "graph.pickle".
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
            file.close()

    @classmethod
    def load(cls, filename: str = "graph.pickle") -> "Graph":
        """Loads a Graph instance from a pickle file.

        Deserializes a Graph object from the specified file, restoring its
        structure and attributes.

        Args:
            filename (str, optional): The path and name of the pickle file to
                                      load the graph from. Defaults to "graph.pickle".

        Returns:
            Graph: A new instance of the Graph class loaded from the file.
        """
        with open(filename, "rb") as file:
            return pickle.load(file)

    def get_nodes(self, type: str = None) -> list:
        """Retrieves a list of node IDs, optionally filtered by type.

        Args:
            type (str, optional): The type of nodes to retrieve ('intersection'
                                  or 'border'). If None, returns all node IDs.
                                  Defaults to None.

        Returns:
            list: A list of node IDs (strings) matching the specified type, or
                  all node IDs if no type is provided.
        """
        if type:
            return [node for node in self.nodes if node.startswith(type)]
        return self.nodes

    def get_connections(self, **kwargs) -> dict:
        """Retrieves the connections for specified nodes.

        Returns a dictionary where keys are node IDs and values are lists of
        their neighbors. Can filter by node type/ID and optionally include edge weights.

        Args:
            **kwargs:
                filter_by (str, optional): A node ID (e.g., 'intersection_3') or
                                           node type (e.g., 'border') to filter the
                                           results. If None, returns connections for
                                           all nodes. Defaults to None.
                weights (bool, optional): If True, the neighbor list will contain
                                          tuples of (neighbor_id, weight).
                                          If False, it contains only neighbor IDs.
                                          Defaults to False.

        Returns:
            dict: A dictionary mapping node IDs to their connections. The format
                  of the connection list depends on the `weights` argument.
        """
        filter_by = kwargs.get("filter_by", None)
        weights = kwargs.get("weights", False)

        if filter_by:
            try:
                node_type, node_id = filter_by.split("_")
            except ValueError:
                node_type, node_id = filter_by, None
            if node_id:
                nodes = [f"{node_type}_{node_id}"]
            else:
                nodes = self.get_nodes(node_type)
        else:
            nodes = self.nodes

        if weights:
            return {
                node: [x[1:] for x in list(self.edges(node, data="weight"))]
                for node in nodes
            }
        else:
            return {node: [x[1] for x in list(self.edges(node))] for node in nodes}
