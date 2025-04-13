"""
graph_viz.py

This module provides the TrafficGraph class, which is a subclass of plotly.graph_objects.Figure.
It visualizes the traffic grid of a TrafficModel, including nodes (intersections, borders),
edges (roads), cars, and traffic lights. The visualization uses the NetworkX library for
graph layout and Plotly for rendering.

Classes:
    - TrafficGraph: A Plotly figure representing the traffic grid.

Dependencies:
    - networkx: For graph manipulation and layout.
    - numpy: For numerical operations, especially array handling.
    - plotly.graph_objects: For creating the visualization.
    - car.CarAgent: For representing car agents in the simulation.
    - light.LightAgent: For representing traffic light agents.
    - model.TrafficModel: The traffic simulation model.

Usage:
    Create an instance of TrafficGraph, passing in a TrafficModel instance, height, and width.

    Example:
    ```python
    from model import TrafficModel
    from graph_viz import TrafficGraph

    model = TrafficModel(num_intersections=5, num_cars=10)
    graph = TrafficGraph(model, height=600, width=800)
    graph.show()
    ```

"""

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from car import CarAgent
from light import LightAgent
from model import TrafficModel


class TrafficGraph(go.Figure):
    """A Plotly figure of the traffic grid.

    Visualizes the traffic grid, including nodes (intersections, borders), edges (roads), cars, and traffic lights.  Node colors indicate their type, and hover information is displayed.

    Inherits from go.Figure.

    Attributes:
        _model (TrafficModel): The traffic model to visualize.

    Methods:
        get_coords_edges(self) -> tuple[np.array, np.array]:
            Get the x and y coordinates of the edges.
        get_coords_nodes(self) -> tuple[np.array, np.array]:
            Get the x and y coordinates of the intersection nodes.
        create_trace_edges(self, edge_x: np.array, edge_y: np.array) -> go.Scatter:
            Create a Plotly trace for the edges.
        create_trace_nodes(self, node_x: np.array, node_y: np.array) -> go.Scatter:
            Create a Plotly trace for the intersection nodes.
        create_node_text(self) -> list:
            Create a list of text for the nodes.
        get_coords_cars(self) -> tuple[np.array, np.array]:
            Get the x and y coordinates of the cars.
        create_trace_cars(self, car_x: list, car_y: list) -> go.Scatter:
            Create a Plotly trace for the cars.
        get_coords_lights(self) -> tuple[np.array, np.array]:
            Get the x and y coordinates of the lights.
        create_trace_lights(self, light_x: np.array, light_y: np.array) -> list[go.layout.Annotation]:
            Create a list of Plotly annotations for the open lane of each light, represented as arrows.
        refresh(self, height: int, width: int) -> None:
            Refresh the graph with updated data, maintaining the given height and width.
        get_coords_borders(self) -> tuple[np.array, np.array]:
            Get the x and y coordinates of the border nodes.
        create_trace_borders(self, border_x: np.array, border_y: np.array) -> go.Scatter:
            Create a Plotly trace for the border nodes.
    """

    def __init__(self, model: TrafficModel, height: int, width: int):
        """Initialize the TrafficGraph visualization.

        Sets up a Plotly figure to visualize the state of a `TrafficModel`.
        It begins by calling the superclass initializer.
        It then calculates the positions of the nodes (intersections) in the
        traffic grid using the `networkx.spring_layout` algorithm, storing
        these positions as node attributes.

        Subsequently, it computes the coordinates for various graphical elements:
        - Edges (roads) connecting the nodes.
        - Nodes (intersections) themselves.
        - Borders (representing entry/exit points).
        - Traffic lights (visualized as arrows indicating allowed directions).
        - Cars currently present in the model.

        For each set of coordinates, it creates a corresponding Plotly trace
        (e.g., `edge_trace`, `node_trace`, `car_trace`). Hover text is specifically
        generated for the nodes to display relevant information.

        Finally, all generated traces (edges, nodes, borders, cars) are added
        to the Plotly figure. The layout of the figure is configured to hide the
        legend, enable the 'closest' hover mode, set appropriate margins, hide
        axis grids and labels, add annotations for traffic lights (arrows), and
        set the overall height and width of the plot.

        Args:
            model (TrafficModel): The traffic model instance containing the grid
                graph, car positions, and traffic light states to visualize.
            height (int): The desired height of the Plotly figure in pixels.
            width (int): The desired width of the Plotly figure in pixels.
        """
        super().__init__()
        self._model = model
        pos = nx.spring_layout(self._model.grid, seed=42)
        nx.set_node_attributes(self._model.grid, pos, "pos")
        edge_x, edge_y = self.get_coords_edges()
        edge_trace = self.create_trace_edges(edge_x, edge_y)
        node_x, node_y = self.get_coords_nodes()
        node_trace = self.create_trace_nodes(node_x, node_y)
        node_trace.text = self.create_node_text()
        border_x, border_y = self.get_coords_borders()
        border_trace = self.create_trace_borders(border_x, border_y)
        light_x, light_y = self.get_coords_lights()
        arrows = self.create_trace_lights(light_x, light_y)
        car_x, car_y = self.get_coords_cars()
        car_trace = self.create_trace_cars(car_x, car_y)

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
        """Get the x and y coordinates of the intersection nodes.

        Extracts the x and y coordinates from the intersection nodes in the traffic grid.

        Returns:
            tuple[np.array, np.array]: Arrays of x and y coordinates of the intersection nodes.
        """
        node_x = np.empty(shape=(0,), dtype=np.float16)
        node_y = np.empty(shape=(0,), dtype=np.float16)

        for node in self._model.grid.nodes(data="pos"):
            if node[0].startswith("intersection"):
                x, y = node[1]
                node_x = np.append(arr=node_x, values=x)
                node_y = np.append(arr=node_y, values=y)

        return node_x, node_y

    def create_trace_nodes(self, node_x: np.array, node_y: np.array) -> go.Scatter:
        """Create a plotly trace for the intersection nodes.

        Generates a scatter plot trace for the intersection nodes, using the provided
        x and y coordinates.

        Args:
            node_x (np.array): X coordinates of the nodes.
            node_y (np.array): Y coordinates of the nodes.

        Returns:
            go.Scatter: Plotly trace for the intersection nodes.
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

        Extracts the x and y coordinates from the edges (roads) in the traffic grid.

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

        Generates a scatter plot trace for the edges (roads), using the provided
        x and y coordinates.

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
        """Calculates the coordinates of a border node based on its connected intersection.

        This method calculates the position of a border node by interpolating between
        the coordinates of its neighboring intersections, based on the edge weights.

        Args:
            border (str): The ID of the border node.

        Returns:
            tuple[float, float]: The x and y coordinates of the border node.
        """
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
        """Get the x and y coordinates of the border nodes.

        Extracts the x and y coordinates from the border nodes in the traffic grid.

        Returns:
            tuple[np.array, np.array]: Arrays of x and y coordinates of the border nodes.
        """
        border_x = np.empty(shape=(0,), dtype=np.float16)
        border_y = np.empty(shape=(0,), dtype=np.float16)

        for border in self._model.grid.get_nodes(type="border"):
            border_coords = self.get_coords_border(border)
            border_x = np.append(arr=border_x, values=border_coords[0])
            border_y = np.append(arr=border_y, values=border_coords[1])

        return border_x, border_y

    def create_trace_borders(self, border_x: list, border_y: list) -> go.Scatter:
        """Create a plotly trace for the border nodes.

        Generates a scatter plot trace for the border nodes, using the provided
        x and y coordinates.

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
        """Calculates and returns the precise x and y coordinates of all car agents.

        This method iterates through each `CarAgent` currently active in the simulation model.
        For each car, it determines its exact position along the path segment it is currently traversing.
        The calculation involves:
        1. Identifying the car's current node (`current_pos`) and the next node in its path (`next_pos`).
        2. Retrieving the remaining distance (`distance`) the car needs to travel to reach `next_pos`.
        3. Getting the total weight (length) of the edge (`edge_weight`) connecting `current_pos` and `next_pos`.
        4. Determining the coordinates of `current_pos` and `next_pos`. This handles different node types:
            - If both are intersections ('i'), it uses the node positions from the model's grid graph.
            - If moving from a border ('b') to an intersection ('i'), it uses the calculated border coordinates and the intersection node position.
            - If moving from an intersection ('i') to a border ('b'), it uses the intersection node position and the calculated border coordinates (using the car's final goal).
        5. Calculating the direction vector between `current_pos` and `next_pos`.
        6. Determining the incremental step vector based on the edge weight.
        7. Calculating the car's current interpolated position along the edge by starting at `current_pos` and moving along the direction vector proportionally to the distance already covered (`edge_weight - distance`).
        8. Appending the calculated x and y coordinates to respective numpy arrays.

        An `IndexError` might occur if a car's path has fewer than two nodes (e.g., the car has just spawned or reached its destination). In this case, the coordinates are assumed to be those of the car's final goal (border node).

        Returns
            tuple[np.array, np.array]: A tuple containing two numpy arrays.
                           The first array holds the x-coordinates of all cars.
                           The second array holds the y-coordinates of all cars.
                           Both arrays use `np.float16` for memory efficiency.
        """

        car_x = np.empty(shape=(0,), dtype=np.float16)
        car_y = np.empty(shape=(0,), dtype=np.float16)

        for car in self._model._agents_by_type[CarAgent]:
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
        """Create a plotly trace for the cars.

        Generates a scatter plot trace for the cars, using the provided
        x and y coordinates.

        Args:
            car_x (list): X coordinates of the cars
            car_y (list): Y coordinates of the cars

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
        """Calculate coordinates for visualizing traffic light states as arrows.

        This method iterates through all `LightAgent` instances in the model.
        For each traffic light, it determines the coordinates needed to draw an arrow
        indicating the currently open lane (the direction traffic is allowed to flow).

        The arrow starts slightly offset from the intersection center along the open lane's
        direction and points back towards the intersection center.

        Calculation Steps:
        1. Get the coordinates of the intersection where the light is located (`pos_coords`).
        2. Get the coordinates of the neighboring intersection connected by the open lane (`lane_origin_coords`).
        3. Calculate the direction vector from the light's intersection towards the open lane's origin.
        4. Normalize the direction vector.
        5. Calculate the arrow's start point (`x_0`, `y_0`) by moving a small distance (0.05 units)
           from the intersection center along the normalized direction vector. This is the arrowhead position.
        6. The arrow's end point (`x_1`, `y_1`) is the exact coordinate of the intersection center. This is the arrow tail position.
        7. Append the start and end x-coordinates (`x_0`, `x_1`) to `light_x` and the start and end
           y-coordinates (`y_0`, `y_1`) to `light_y`.

        Returns:
            tuple[np.array, np.array]: A tuple containing two numpy arrays:
                - `light_x`: An array of x-coordinates. Each pair of consecutive values
                             represents the start (arrowhead) and end (tail) x-coordinate
                             of an arrow for one traffic light.
                - `light_y`: An array of y-coordinates, structured similarly to `light_x`,
                             representing the start and end y-coordinates of the arrows.
        """
        light_x = np.empty(shape=(0,), dtype=np.float16)
        light_y = np.empty(shape=(0,), dtype=np.float16)

        for light in self._model._agents_by_type[LightAgent]:
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

    def create_trace_lights(
        self, light_x: np.array, light_y: np.array
    ) -> list[go.layout.Annotation]:
        """Create a list of Plotly annotations for the open lane of each light, represented as arrows.

        Generates a list of annotations representing arrows that indicate the open lane
        of each traffic light.

        Args:
            light_x (np.array): X coordinates of the lane
            light_y (np.array): Y coordinates of the lane

        Returns:
            list[go.layout.Annotation]: A list of Plotly annotation objects representing the traffic lights.
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
        """Refresh the graph with updated data, maintaining the given height and width.

        Reinitializes the TrafficGraph with the current state of the traffic model,
        effectively refreshing the visualization.

        Args:
            height (int): Height of the plot in pixels
            width (int): Width of the plot in pixels
        """
        self.__init__(model=self._model, height=height, width=width)
