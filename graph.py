import networkx as nx
import random
import pickle
import matplotlib.pyplot as plt


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

        # Add nodes of type intersection
        super().add_nodes_from(
            [
                (f"intersection_{i}", {"type": "intersection"})
                for i in range(num_intersections)
            ]
        )

        # Add nodes of type border
        super().add_nodes_from(
            [(f"border_{i}", {"type": "border"}) for i in range(num_borders)]
        )

        # Add edges between intersections
        for i in range(num_intersections):
            for j in range(random.randint(2, 4)):
                super().add_edge(
                    f"intersection_{i}",
                    f"intersection_{random.randint(0, (num_intersections - 1))}",
                    weight=random.randint(weight_range[0], weight_range[1]),
                )

        # Add edges between intersections and borders
        for i in range(num_borders):
            super().add_edge(
                f"border_{i}",
                f"intersection_{random.randint(0, (num_intersections - 1))}",
                weight=random.randint(weight_range[0], weight_range[1]),
            )

        super().remove_edges_from(nx.selfloop_edges(self))

    def save(self, filename: str):
        """Save class instance to a pickle file.

        Args:
            filename (str): The name of the file to save the class instance to.
        """

        pickle.dump(self, open(filename, "wb"))

    def generate_plot(self) -> plt:
        """Generate a plot of the graph.

        Returns:
            plt: A matplotlib.pyplot object with the graph plotted.
        """

        pos = nx.spring_layout(self, seed=42)

        node_color = [
            "red" if self.nodes[n]["type"] == "border" else "blue" for n in self.nodes
        ]
        nx.draw(self, pos, with_labels=True, node_color=node_color)
        nx.draw_networkx_edge_labels(
            self, pos, edge_labels=nx.get_edge_attributes(self, "weight")
        )

        return plt
