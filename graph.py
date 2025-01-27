import networkx as nx
import random
import pickle
import plotly.graph_objects as go


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

    def generate_plot(self) -> go.Figure:
        """Generate a plot of the graph.

        Returns:
            plt: A matplotlib.pyplot object with the graph plotted.
        """

        # Create a list to store the positions of the nodes
        pos = nx.spring_layout(self)

        # Assign the positions to the nodes
        for node in self.nodes():
            self.nodes[node]["pos"] = pos[node]

        edge_x = []
        edge_y = []
        for edge in self.edges():
            x0, y0 = self.nodes[edge[0]]["pos"]
            x1, y1 = self.nodes[edge[1]]["pos"]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        for node in self.nodes():
            x, y = self.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        node_color = []
        for node in self.nodes(data=True):
            if node[1]["type"] == "intersection":
                node_color.append("blue")
            else:
                node_color.append("red")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                color=node_color,
                size=10,
                line_width=2,
            ),
        )

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(self.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append("# of connections: " + str(len(adjacencies[1])))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="<br>Network graph made with Python", font=dict(size=16)
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        return fig
