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
        ***get_coords_lights(self) -> tuple[np.array, np.array]***:
            Get the x and y coordinates of the lights.
        ***create_trace_lights(self, light_x: np.array, light_y: np.array) -> go.Scatter***:
            Create a plotly trace for the open lane of each light.
        ***refresh(self) -> None***:
            Function to refresh the graph.
    """

    def __init__(self, model: TrafficModel, height: int, width: int):
        """Create a plotly figure of the traffic grid.

        Args:
            model (TrafficModel): The traffic model to visualize.
            height (int): Height of the plot in pixels
            width (int): Width of the plot in pixels
        """
        super().__init__()
        self._model = model
        pos = nx.spring_layout(self._model.grid, seed=42)
        nx.set_node_attributes(
            self._model.grid, pos, "pos"
        )  # Assign positions to nodes
        edge_x, edge_y = self.get_coords_edges()
        edge_trace = self.create_trace_edges(edge_x, edge_y)
        node_x, node_y = self.get_coords_nodes()
        node_color = self.create_node_color()
        node_trace = self.create_trace_nodes(node_x, node_y, node_color)
        node_trace.text = self.create_node_text()
        border_x, border_y = self.get_coords_borders()
        border_trace = self.create_trace_borders(border_x, border_y)
        light_x, light_y = self.get_coords_lights()
        arrows = self.create_trace_lights(light_x, light_y)
        car_x, car_y = self.get_coords_cars()
        car_trace = self.create_trace_cars(car_x, car_y)

        # Create the figure
        self.add_traces([edge_trace, node_trace, border_trace, car_trace])
        self.update_layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=arrows,
            height=height,
            width=width,
        )

    def get_coords_nodes(self) -> tuple[np.array, np.array]:
        """Get the x and y coordinates of the nodes.

        Returns:
            tuple[np.array, np.array]: Arrays of x and y coordinates of the nodes
        """
        node_x = np.empty(shape=(0,), dtype=np.float16)
        node_y = np.empty(shape=(0,), dtype=np.float16)

        for node in self._model.grid.nodes(data="pos"):
            if node[0].startswith("intersection"):
                x, y = node[1]
                node_x = np.append(arr=node_x, values=x)
                node_y = np.append(arr=node_y, values=y)

        return node_x, node_y

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
                color="#C140FF",
                size=12.5,
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

    def get_coords_edges(self) -> tuple[np.array, np.array]:
        """Get the x and y coordinates of the edges.

        Returns:
            tuple[np.array, np.array]: Arrays of x and y coordinates of the edges
        """
        edge_x = np.empty(shape=(0,), dtype=np.float16)
        edge_y = np.empty(shape=(0,), dtype=np.float16)

        for edge in self._model.grid.edges():
            if edge[0].startswith("i") and edge[1].startswith("i"):
                x0, y0 = self._model.grid.nodes[edge[0]]["pos"]
                x1, y1 = self._model.grid.nodes[edge[1]]["pos"]
                edge_x = np.append(arr=edge_x, values=x0)
                edge_x = np.append(arr=edge_x, values=x1)
                edge_x = np.append(arr=edge_x, values=None)
                edge_y = np.append(arr=edge_y, values=y0)
                edge_y = np.append(arr=edge_y, values=y1)
                edge_y = np.append(arr=edge_y, values=None)

        return edge_x, edge_y

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
            line=dict(width=0.75, color="#888"),
            hoverinfo="none",
            mode="lines",
        )
        return edge_trace

    def get_coords_border(self, border: str) -> tuple[float, float]:
        intersection_u, intersection_v = self._model.grid.neighbors(border)

        distance_intersection_u = self._model.grid.get_edge_data(
            u=intersection_u, v=border
        )["weight"]

        distance_intersections = self._model.grid.get_edge_data(
            u=intersection_u, v=intersection_v
        )["weight"]

        intersection_u_coords = np.array(self._model.grid.nodes[intersection_u]["pos"])
        intersection_v_coords = np.array(self._model.grid.nodes[intersection_v]["pos"])

        vector = intersection_v_coords - intersection_u_coords
        steps = vector / distance_intersections

        border_coords = intersection_u_coords + steps * (
            distance_intersections - distance_intersection_u
        )

        return border_coords[0], border_coords[1]

    def get_coords_borders(self) -> tuple[np.array, np.array]:
        border_x = np.empty(shape=(0,), dtype=np.float16)
        border_y = np.empty(shape=(0,), dtype=np.float16)

        for border in self._model.grid.get_nodes(type="border"):
            border_coords = self.get_coords_border(border)
            border_x = np.append(arr=border_x, values=border_coords[0])
            border_y = np.append(arr=border_y, values=border_coords[1])

        return border_x, border_y

    def create_trace_borders(self, border_x: list, border_y: list) -> go.Scatter:
        """Create a plotly trace for the nodes.

        Args:
            border_x (list): X coordinates of the borders
            border_y (list): Y coordinates of the borders

        Returns:
            go.Scatter: Plotly trace for the cars
        """
        car_trace = go.Scatter(
            x=border_x,
            y=border_y,
            mode="markers",
            hoverinfo=None,
            marker=dict(
                color="#7FF9E2",
                size=10,
                line_width=1,
            ),
        )

        return car_trace

    def get_coords_cars(self) -> tuple[np.array, np.array]:
        """Get the x and y coordinates of the cars.

        This method performs the following steps:
        - Retrieves the current position, next position, distance to the next position, and edge weight for each car agent.
        - Computes the path vectors:
            - Calculates the vector between the current and next position.
            - Determines the steps as the vector length divided by the edge weight.
            - Computes the current position on the path by calculating the total steps taken on the current path.

        - Handles custom computation of position of border nodes

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

                if car.position.startswith("i") and list(car.path.keys())[1].startswith(
                    "i"
                ):
                    current_pos_coords = np.array(
                        self._model.grid.nodes[current_pos]["pos"]
                    )
                    next_pos_coords = np.array(self._model.grid.nodes[next_pos]["pos"])
                elif car.position.startswith("b") and list(car.path.keys())[
                    1
                ].startswith("i"):
                    current_pos_coords = np.array(self.get_coords_border(current_pos))
                    next_pos_coords = np.array(self._model.grid.nodes[next_pos]["pos"])
                elif car.position.startswith("i") and list(car.path.keys())[
                    1
                ].startswith("b"):
                    current_pos_coords = np.array(
                        self._model.grid.nodes[current_pos]["pos"]
                    )
                    next_pos_coords = self.get_coords_border(car.goal)

                vector = next_pos_coords - current_pos_coords
                steps = vector / edge_weight
                current_position = current_pos_coords + steps * (edge_weight - distance)

                car_x = np.append(arr=car_x, values=current_position[0])
                car_y = np.append(arr=car_y, values=current_position[1])
            except IndexError:
                x, y = self.get_coords_border(car.goal)
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
                color="#FF8300",
                size=7.5,
                line_width=1,
            ),
        )

        return car_trace

    def get_coords_lights(self) -> tuple[np.array, np.array]:
        """Get the x and y coordinates of the lights.

        Returns:
            tuple[np.array, np.array]: Arrays of x and y coordinates of the lights
        """
        light_x = np.empty(shape=(0,), dtype=np.float16)
        light_y = np.empty(shape=(0,), dtype=np.float16)

        for light in self._model.get_agents_by_type("LightAgent"):
            lane_origin_coords = np.array(
                self._model.grid.nodes[light.open_lane]["pos"]
            )
            pos_coords = np.array(self._model.grid.nodes[light.position]["pos"])
            vector = lane_origin_coords - pos_coords

            vector_length = np.linalg.norm(vector)

            x_0, y_0 = pos_coords + (vector / vector_length) * 0.05
            x_1, y_1 = pos_coords

            light_x = np.append(arr=light_x, values=x_0)
            light_x = np.append(arr=light_x, values=x_1)
            light_y = np.append(arr=light_y, values=y_0)
            light_y = np.append(arr=light_y, values=y_1)

        return light_x, light_y

    def create_trace_lights(self, light_x: np.array, light_y: np.array) -> go.Scatter:
        """Create a plotly trace for the open lane of each light.

        Args:
            light_x (np.array): X coordinates of the lane
            light_y (np.array): Y coordinates of the lane

        Returns:
            go.Scatter: Plotly trace for the lanes
        """

        arrows = [
            (
                go.layout.Annotation(
                    x=light_x[i + 1],
                    y=light_y[i + 1],
                    ax=light_x[i],
                    ay=light_y[i],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.25,
                    arrowwidth=1.25,
                    arrowcolor="green",
                )
            )
            for i in range(0, len(light_x) - 1, 2)
        ]

        return arrows

    def refresh(self, height: int, width: int) -> None:
        """Function to refresh the graph.

        Args:
            height (int): Height of the plot in pixels
            width (int): Width of the plot in pixels
        """
        self.__init__(model=self._model, height=height, width=width)
