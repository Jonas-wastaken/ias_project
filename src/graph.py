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
        remove_intersections(self, num_intersections: int) -> None:
            Remove intersection nodes and update connections.
        connect_intersections(self, new_intersections: list) -> None:
            Establish connections between intersection nodes.
        add_borders(self, num_borders: int) -> None:
            Add border nodes and connect them.
        remove_borders(self, num_borders: int) -> None:
            Remove border nodes.
        connect_borders(self) -> None:
            Connect border nodes between intersection pairs.
        change_weights(self, min_distance: int, max_distance: int) -> None:
            Update edge weights based on new distance bounds.
        place_agent(self, agent_id: int) -> str:
            Select a random border node for agent placement.
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

    def remove_intersections(self, num_intersections: int) -> None:
        """Removes the last 'n' intersection nodes added to the graph.

        Removes the specified number of intersection nodes, starting from the
        highest index. It then re-evaluates and potentially adjusts connections
        for remaining intersections and borders.

        Args:
            num_intersections (int): The number of intersection nodes to remove.
        """
        intersections_to_remove = self.get_nodes("intersection")[-num_intersections:]

        super().remove_nodes_from(intersections_to_remove)

        self.connect_intersections(self.get_nodes("intersection"))
        self.connect_borders()

    def connect_intersections(self, new_intersections: list) -> None:
        """Connects intersection nodes, ensuring connectivity constraints.

        Connects each intersection node (especially newly added ones) to a
        minimum of 2 and a maximum of 4 other intersection nodes. It prioritizes
        connecting nodes with fewer than 2 connections and avoids exceeding 4
        connections per node. Edge weights are assigned randomly within the
        `min_distance` and `max_distance` bounds, favoring smaller distances.

        Args:
            new_intersections (list): A list of intersection node IDs (strings)
                                      that need connection establishment or verification.
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

    def remove_borders(self, num_borders: int) -> None:
        """Removes the last 'n' border nodes added to the graph.

        Args:
            num_borders (int): The number of border nodes to remove.
        """

        super().remove_nodes_from(self.get_nodes("border")[-num_borders:])

    def connect_borders(self) -> None:
        """Connects unconnected border nodes between pairs of connected intersections.

        Iterates through border nodes with fewer than two connections. For each,
        it randomly selects a pair of connected intersection nodes and inserts
        the border node between them, splitting the original edge weight.
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

    def change_weights(self, min_distance: int, max_distance: int) -> None:
        """Updates the weights of all edges based on new distance bounds.

        Reassigns random weights to all edges connecting intersection pairs
        within the new `min_distance` and `max_distance`. It then recalculates
        the weights for edges involving border nodes to maintain consistency
        with the total distance between the intersections they connect.

        Args:
            min_distance (int): The new minimum distance for intersection edges.
            max_distance (int): The new maximum distance for intersection edges.
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
        """Selects a random border node as a starting position for an agent.

        Does not modify the graph state but returns the ID of a randomly chosen
        border node.

        Args:
            agent_id (int): The ID of the agent being placed (currently unused
                            in the selection logic but kept for potential future use).

        Returns:
            str: The ID of the randomly selected border node (e.g., 'border_5').
        """
        borders = self.get_nodes(type="border")
        assigned_start = borders[random.randint(0, (len(borders) - 1))]

        return assigned_start

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
