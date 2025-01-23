import networkx as nx
import random
import pickle
import matplotlib.pyplot as plt


class Graph(nx.Graph):
    def __init__(
        self, num_intersections=10, num_borders=3, weight_range=(1, 10), file_path=None
    ):
        """
        A class to represent a graph.

        The class inherits from the networkx.Graph class and adds methods to save the graph to a file and generate a plot.

        Parameters:
            - num_intersections (int): The number of intersection nodes to create.
            - num_borders (int): The number of border nodes to create.
            - weight_range (tuple): A tuple specifying the range of weights for the edges.
            - file_path (str): Path to a file to load the graph from. If provided, other parameters are ignored.

        The method creates nodes labeled as 'intersection' and 'border', and adds edges between them with random weights.
        Self-loops are removed from the graph.
        """

        super().__init__()

        super().add_nodes_from(
            [
                (f"intersection_{i}", {"type": "intersection"})
                for i in range(num_intersections)
            ]
        )

        super().add_nodes_from(
            [(f"border_{i}", {"type": "border"}) for i in range(num_borders)]
        )

        for i in range(num_intersections):
            for j in range(random.randint(2, 4)):
                super().add_edge(
                    f"intersection_{i}",
                    f"intersection_{random.randint(0, (num_intersections - 1))}",
                    weight=random.randint(weight_range[0], weight_range[1]),
                )

        for i in range(num_borders):
            super().add_edge(
                f"border_{i}",
                f"intersection_{random.randint(0, (num_intersections - 1))}",
                weight=random.randint(weight_range[0], weight_range[1]),
            )

        super().remove_edges_from(nx.selfloop_edges(self))

    def save(self, filename):
        """
        Save the graph to a file.

        Parameters:
        filename (str): The name of the file to save the graph to.
        """

        pickle.dump(self, open(filename, "wb"))

    def generate_plot(self):
        """
        Visualize the graph.

        Returns:
        plt: A matplotlib.pyplot object.
        """

        pos = nx.spring_layout(self, seed=42)

        # Change color of nodes based on type
        node_color = [
            "red" if self.nodes[n]["type"] == "border" else "blue" for n in self.nodes
        ]
        nx.draw(self, pos, with_labels=True, node_color=node_color)
        nx.draw_networkx_edge_labels(
            self, pos, edge_labels=nx.get_edge_attributes(self, "weight")
        )

        return plt
