import networkx as nx
import plotly.graph_objects as go
from model import TrafficModel


class TrafficGraph(go.Figure):
    """A plotly figure of the traffic grid.

    Places nodes and edges of a traffic grid on a plotly figure. Nodes are colored according to their type. Figure displays information about the nodes on hover.

    Inherits from go.Figure.

    Attributes:
        model (TrafficModel): The traffic model to visualize.

    ## Methods
        **get_coords_edges(self) -> tuple**:
            Get the x and y coordinates of the edges.
        **get_coords_nodes(self) -> tuple**:
            Get the x and y coordinates of the nodes.
        **create_trace_edges(self, edge_x: list, edge_y: list) -> go.Scatter**:
            Create a plotly trace for the edges.
        **create_trace_nodes(self, node_x: list, node_y: list, node_color: list) -> go.Scatter**:
            Create a plotly trace for the nodes.
        **create_node_color(self) -> list**:
            Create a list of colors for the nodes.
        **create_node_text(self) -> list**:
            Create a list of text for the nodes.
    """

    def __init__(self, model: TrafficModel):
        """Create a plotly figure of the traffic grid.

        Args:
            model (TrafficModel): The traffic model to visualize.
        """
        super().__init__()
        self._model = model
        pos = nx.spring_layout(self._model.grid, seed=42)  # Create layout for the graph
        nx.set_node_attributes(
            self._model.grid, pos, "pos"
        )  # Assign positions to nodes
        edge_x, edge_y = self.get_coords_edges()
        edge_trace = self.create_trace_edges(edge_x, edge_y)
        node_x, node_y = self.get_coords_nodes()
        node_color = self.create_node_color()
        node_trace = self.create_trace_nodes(node_x, node_y, node_color)
        node_trace.text = self.create_node_text()

        # Create the figure
        self.add_traces([edge_trace, node_trace])
        self.update_layout(
            title=dict(text="<br>Traffic Grid", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

    def get_coords_edges(self) -> tuple:
        """Get the x and y coordinates of the edges.

        Returns:
            tuple: Lists of x and y coordinates of the edges
        """
        edge_x = []
        edge_y = []
        for edge in self._model.grid.edges():
            x0, y0 = self._model.grid.nodes[edge[0]]["pos"]
            x1, y1 = self._model.grid.nodes[edge[1]]["pos"]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        return edge_x, edge_y

    def get_coords_nodes(self) -> tuple:
        """Get the x and y coordinates of the nodes.

        Returns:
            tuple: Lists of x and y coordinates of the nodes
        """
        node_x = []
        node_y = []
        for node in self._model.grid.nodes():
            x, y = self._model.grid.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        return node_x, node_y

    def create_trace_edges(self, edge_x: list, edge_y: list) -> go.Scatter:
        """Create a plotly trace for the edges.

        Args:
            edge_x (list): X coordinates of the edges
            edge_y (list): Y coordinates of the edges

        Returns:
            go.Scatter: Plotly trace for the edges
        """
        return go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

    def create_trace_nodes(
        self, node_x: list, node_y: list, node_color: list
    ) -> go.Scatter:
        """Create a plotly trace for the nodes.

        Args:
            node_x (list): X coordinates of the nodes
            node_y (list): Y coordinates of the nodes
            node_color (list): Colors of the nodes

        Returns:
            go.Scatter: Plotly trace for the nodes
        """
        return go.Scatter(
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

    def create_node_color(self) -> list:
        """Create a list of colors for the nodes.

        Returns:
            list: List of colors for the nodes
        """
        return [
            "blue" if node[1]["type"] == "intersection" else "red"
            for node in self._model.grid.nodes(data=True)
        ]

    def create_node_text(self) -> list:
        """Create a list of text for the nodes.

        Returns:
            list: List of text for the nodes
        """
        node_adjacencies = []
        node_text = []
        for node, adjacency_dict in enumerate(self._model.grid.adjacency()):
            node_adjacencies.append(len(adjacency_dict[1]))
            node_text.append("# of connections: " + str(len(adjacency_dict[1])))

        return node_text
