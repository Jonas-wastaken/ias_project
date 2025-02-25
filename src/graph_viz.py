"""This module contains:
- TrafficGraph class: A plotly figure of the traffic grid."""

import networkx as nx
import plotly.graph_objects as go
from model import TrafficModel
import numpy as np


class TrafficGraph(go.Figure):
    """A plotly figure of the traffic grid.

    Places nodes and edges of a traffic grid on a plotly figure. Nodes are colored according to their type. Figure displays information about the nodes on hover.

    Inherits from go.Figure.

    Attributes:
        model (TrafficModel): The traffic model to visualize.

    ## Methods
        **get_coords_edges(self) -> tuple[np.array, np.array]**:
            Get the x and y coordinates of the edges.
        **get_coords_nodes(self) -> tuple[np.array, np.array]**:
            Get the x and y coordinates of the nodes.
        **create_trace_edges(self, edge_x: np.array, edge_y: np.array) -> go.Scatter**:
            Create a plotly trace for the edges.
        **create_trace_nodes(self, node_x: np.array, node_y: np.array, node_color: list) -> go.Scatter**:
            Create a plotly trace for the nodes.
        **create_node_color(self) -> list**:
            Create a list of colors for the nodes.
        **create_node_text(self) -> list**:
            Create a list of text for the nodes.
        **get_coords_cars(self) -> tuple[np.array, np.array]**:
            Get the x and y coordinates of the cars.
        ***create_trace_cars(self, car_x: list, car_y: list) -> go.Scatter***:
            Create a plotly trace for the nodes.
    """

    def __init__(self, model: TrafficModel):
        """Create a plotly figure of the traffic grid.

        Args:
            model (TrafficModel): The traffic model to visualize.
        """
        super().__init__()
        self._model = model
        pos = nx.kamada_kawai_layout(self._model.grid)
        nx.set_node_attributes(
            self._model.grid, pos, "pos"
        )  # Assign positions to nodes
        edge_x, edge_y = self.get_coords_edges()
        edge_trace = self.create_trace_edges(edge_x, edge_y)
        node_x, node_y = self.get_coords_nodes()
        node_color = self.create_node_color()
        node_trace = self.create_trace_nodes(node_x, node_y, node_color)
        node_trace.text = self.create_node_text()
        car_x, car_y = self.get_coords_cars()
        car_trace = self.create_trace_cars(car_x, car_y)

        # Create the figure
        self.add_traces([edge_trace, node_trace, car_trace])
        self.update_layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

    def get_coords_edges(self) -> tuple[np.array, np.array]:
        """Get the x and y coordinates of the edges.

        Returns:
            tuple[np.array, np.array]: Arrays of x and y coordinates of the edges
        """
        edge_x = np.empty(shape=(0,), dtype=np.float16)
        edge_y = np.empty(shape=(0,), dtype=np.float16)

        for edge in self._model.grid.edges():
            x0, y0 = self._model.grid.nodes[edge[0]]["pos"]
            x1, y1 = self._model.grid.nodes[edge[1]]["pos"]
            edge_x = np.append(arr=edge_x, values=x0)
            edge_x = np.append(arr=edge_x, values=x1)
            edge_x = np.append(arr=edge_x, values=None)
            edge_y = np.append(arr=edge_y, values=y0)
            edge_y = np.append(arr=edge_y, values=y1)
            edge_y = np.append(arr=edge_y, values=None)

        return edge_x, edge_y

    def get_coords_nodes(self) -> tuple[np.array, np.array]:
        """Get the x and y coordinates of the nodes.

        Returns:
            tuple[np.array, np.array]: Arrays of x and y coordinates of the nodes
        """
        node_x = np.empty(shape=(0,), dtype=np.float16)
        node_y = np.empty(shape=(0,), dtype=np.float16)

        for node in self._model.grid.nodes(data="pos"):
            x, y = node[1]
            node_x = np.append(arr=node_x, values=x)
            node_y = np.append(arr=node_y, values=y)

        return node_x, node_y

    def create_trace_edges(self, edge_x: np.array, edge_y: np.array) -> go.Scatter:
        """Create a plotly trace for the edges.

        Args:
            edge_x (np.array): X coordinates of the edges
            edge_y (np.array): Y coordinates of the edges

        Returns:
            go.Scatter: Plotly trace for the edges
        """
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )
        return edge_trace

    def create_trace_nodes(
        self, node_x: np.array, node_y: np.array, node_color: list
    ) -> go.Scatter:
        """Create a plotly trace for the nodes.

        Args:
            node_x (np.array): X coordinates of the nodes
            node_y (np.array): Y coordinates of the nodes
            node_color (list): Colors of the nodes

        Returns:
            go.Scatter: Plotly trace for the nodes
        """
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                color=node_color,
                size=10,
                line_width=1,
            ),
        )

        return node_trace

    def create_node_color(self) -> list:
        """Create a list of colors for the nodes based on type.

        Returns:
            list: List of colors for the nodes
        """
        node_color = [
            "#C140FF" if node[1]["type"] == "intersection" else "#7FF9E2"
            for node in self._model.grid.nodes(data=True)
        ]

        return node_color

    def create_node_text(self) -> list:
        """Create a list of text for the nodes.

        - Node ID
        - Number of connected intersections (if type is intersection)
        - Number of connected borders (if type is intersection)

        Returns:
            list: List of text for the nodes
        """
        node_text = [
            f"{str(node[0]).title().replace('_', ' ')}<br>Connected Routes: {len([node for node in list(self._model.grid.neighbors(node[0])) if node.startswith('intersection')])}<br>Connected Borders: {len([node for node in list(self._model.grid.neighbors(node[0])) if node.startswith('border')])}"
            if node[1]["type"] == "intersection"
            else f"{str(node[0]).title().replace('_', ' ')}"
            for node in self._model.grid.nodes(data=True)
        ]

        return node_text

    def get_coords_cars(self) -> tuple[np.array, np.array]:
        """Get the x and y coordinates of the cars.

        This method performs the following steps:
        - Retrieves the current position, next position, distance to the next position, and edge weight for each car agent.
        - Computes the path vectors:
            - Calculates the vector between the current and next position.
            - Determines the steps as the vector length divided by the edge weight.
            - Computes the current position on the path by calculating the total steps taken on the current path.

        If an IndexError occurs (e.g., the car has reached its goal), it retrieves the goal position coordinates.

        Returns:
            tuple[np.array, np.array]: Two numpy arrays containing the x and y coordinates of the cars, respectively.
        """

        car_x = np.empty(shape=(0,), dtype=np.float16)
        car_y = np.empty(shape=(0,), dtype=np.float16)

        for car in self._model.get_agents_by_type("CarAgent"):
            try:
                current_pos = car.position
                next_pos = list(car.path.keys())[1]
                distance = car.path[current_pos]
                edge_weight = self._model.grid.get_edge_data(current_pos, next_pos)[
                    "weight"
                ]

                current_pos_coords = np.array(
                    self._model.grid.nodes[current_pos]["pos"]
                )
                next_pos_coords = np.array(self._model.grid.nodes[next_pos]["pos"])

                vector = next_pos_coords - current_pos_coords
                steps = vector / edge_weight
                current_position = current_pos_coords + steps * (edge_weight - distance)

                car_x = np.append(arr=car_x, values=current_position[0])
                car_y = np.append(arr=car_y, values=current_position[1])
            except IndexError:
                x, y = self._model.grid.nodes[car.goal]["pos"]
                car_x = np.append(arr=car_x, values=x)
                car_y = np.append(arr=car_y, values=y)

        return car_x, car_y

    def create_trace_cars(self, car_x: list, car_y: list) -> go.Scatter:
        """Create a plotly trace for the nodes.

        Args:
            car_x (list): X coordinates of the cars
            car_y (list): Y coordinates of the cars
            node_color (list): Colors of the nodes

        Returns:
            go.Scatter: Plotly trace for the cars
        """
        car_trace = go.Scatter(
            x=car_x,
            y=car_y,
            mode="markers",
            hoverinfo=None,
            marker=dict(
                color="red",
                size=5,
                line_width=1,
            ),
        )

        return car_trace
