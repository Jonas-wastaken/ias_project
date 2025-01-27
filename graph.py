import networkx as nx
import random
import pickle


class Graph(nx.Graph):
    def __init__(
        self,
        num_intersections: int = 10,
        num_borders: int = 3,
        weight_range: tuple[int, int] = (1, 10),
    ):
        """A class to represent a graph.

        The class inherits from the networkx.Graph class and adds methods to save the graph to a file and generate a plot.
        Creates nodes labeled as 'intersection' and 'border', and adds edges between them with random weights.
        Self-loops are removed from the graph.

        Args:
            num_intersections (int, optional): The number of intersection nodes to create. Defaults to 10.
            num_borders (int, optional): The number of border nodes to create. Defaults to 3.
            weight_range (tuple, optional): A tuple specifying the range of weights for the edges. Defaults to (1, 10).
        """

        super().__init__()
        self.add_intersections(num_intersections)
        self.add_borders(num_borders)
        self.add_edges_intersections(num_intersections, weight_range)
        self.add_edges_borders(num_intersections, num_borders, weight_range)
        super().remove_edges_from(nx.selfloop_edges(self))

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

    def add_edges_intersections(
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

    def add_edges_borders(
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

    def save(self, filename: str) -> None:
        """Save class instance to a pickle file.

        Args:
            filename (str): The name of the file to save the class instance to.
        """

        pickle.dump(self, open(filename, "wb"))
