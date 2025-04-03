"""This module contains:
- CarAgent class: Represents a car navigating an undirected graph.
- AgentArrived exception: Raised when an car has reached it's goal."""

import mesa
import random
from pathlib import Path
from dataclasses import dataclass, field
import polars as pl
from networkx import dijkstra_path

from data import SimData


class CarAgent(mesa.Agent):
    """Agent, which represents a car navigating an undirected graph. It receives it's starting position from the graph instance and then computes a random node as it's goal. Path computation uses Dijkstra's algorithm.

    Inherits from mesa.Agent.

    Attributes:
        start (str): The ID of the node, where the car starts.
        goal (str): The ID of the node, which is the car's goal.
        path (dict): A dictionary containing the steps in the car's path as keys and the distance to the next step as values.
        position (str): The ID of the node, where the car is currently located.
        waiting (bool): A flag indicating whether the car is waiting at a traffic light.
        global_waiting_time (int): The total time the car has spent waiting at traffic lights.
        travel_time (int): The total time the car has spent traveling.

    ## Methods:
        **compute_goal(self) -> str**:
            Assigns a random border node, which is not the starting node, as the goal of the car.
        **compute_path(self) -> dict**:
            Computes the path the car takes to reach it's goal using Dijkstra's algorithm.
        **move(self) -> None**:
            Moves the car to it's next step on the path.
        **set_wait_status(self, status: bool) -> None**:
            Sets the waiting status of the car.
        **check_lights(self) -> None**:
            Checks if the car is standing at a light and if it is allowed to drive. Sets the waiting status accordingly.
    """

    def __init__(self, model: mesa.Model):
        """Initializes a new CarAgent. The car is placed on a random border node, and computes a random goal and the best path there.

        Args:
            model (mesa.Model): The model instance in which the car lives.
        """
        super().__init__(model)
        self.start = self.model.grid.place_agent(agent_id=self.unique_id)
        self.position = self.start
        self.goal = self.compute_goal()
        self.path = self.compute_path()
        self.waiting = False
        self.global_waiting_time = 0  # TODO: löschen?
        self.travel_time = 0

        self.wait_times = WaitTimes()

    def compute_goal(self) -> str:
        """Assigns a random border node, which is not the starting node, as the goal of the car.

        Returns:
            str: ID of the node, which is the car's goal.
        """
        borders = [node for node in self.model.grid if node.find("border") == 0]
        borders.remove(self.start)
        assigned_goal = borders[random.randint(0, (len(borders) - 1))]

        return assigned_goal

    def compute_path(self) -> dict:
        """Computes the path the car takes to reach it's goal using Dijkstra's algorithm.

        Returns:
            dict: A dictionary containing the steps in the car's path as keys and the distance to the next step as values.
        """
        steps = dijkstra_path(
            self.model.grid, self.position, self.goal, weight="weight"
        )

        path = {}

        for step in steps:
            try:
                path[step] = self.model.grid.get_edge_data(
                    step, steps[steps.index(step) + 1]
                )["weight"]
            except IndexError:
                path[step] = None

        return path

    def move(self) -> None:
        """Tries to move the car to it's next step on the path and sends updated position to the grid.

        The car can only move if it is not waiting at a light (using check_lights function).
        The car can only move if the distance to the next step in it's path is 0. Otherwise, the distance is decremented by 1.

        Raises:
            AgentArrived: If the car has reached it's goal, this exception is raised.
        """
        self.check_lights()
        if not self.waiting:
            if self.position == self.goal:
                raise AgentArrived(
                    message=f"Agent {self.unique_id} arrived at it's goal"
                )
            else:
                if self.path.get(self.position) == 1:
                    steps = list(self.path.keys())
                    next_index = steps.index(self.position) + 1
                    self.path.pop(self.position)
                    self.position = steps[next_index]
                else:
                    self.path[self.position] = self.path.get(self.position) - 1
        else:
            self.global_waiting_time += 1

        self.travel_time += 1

    def check_lights(self) -> None:
        """Checks if the car is standing at a light and if it is allowed to drive. Sets the waiting status accordingly."""

        # Check if the car is standing at a light
        current_intersection = list(
            self.model.get_agents_by_id([self.unique_id])[0].path
        )[0]
        current_distance = list(
            self.model.get_agents_by_id([self.unique_id])[0].path.values()
        )[0]

        if self.position.startswith("intersection"):
            at_light = (
                self.model.car_paths[self.unique_id][current_intersection]
                == current_distance
            )
        else:
            at_light = False

        if at_light:
            current_light = [
                light
                for light in self.model.get_agents_by_type("LightAgent")
                if light.position == current_intersection
            ][0]

            if (
                self.model.get_last_intersection_of_car(self.unique_id)
                != current_light.open_lane
            ):
                self.waiting = True
            else:
                self.waiting = False


class AgentArrived(Exception):
    """Exception raised when an car has reached it's goal."""

    def __init__(self, message: str):
        """Initializes AgentArrived exception.

        Args:
            message (str): The message to be displayed when the exception is raised.
        """
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"


@dataclass
class WaitTimes(SimData):
    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Constructs the data schema"""
        self.data = pl.DataFrame(
            schema={
                "Car_ID": pl.Int32,
                "Light_ID": pl.Int16,
                "Wait_Time": pl.Int16,
            },
            strict=False,
        )

    def update_data(
        self, car: CarAgent, waiting: bool, light_intersection_mapping: pl.DataFrame
    ) -> None:
        """Updates the data

        Args:
            car (CarAgent): CarAgent instance
            waiting (bool): Indicates whether the CarAgent instance is currently waiting at a light
            light_intersection_mapping (pl.DataFrame): Mapping table for LightAgents and their corresponding intersections
        """
        light_id = (
            light_intersection_mapping.filter(pl.col("Intersection") == car.position)
            .select(pl.col("Light_ID"))
            .item()
        )
        if not waiting:
            self.data = self.data.with_columns(
                pl.when(
                    (pl.col("Car_ID") == car.unique_id)
                    & (pl.col("Light_ID") == light_id)
                )
                .then(0)
                .otherwise(pl.col("Wait_Time"))
                .alias("Wait_Time")
            )
        elif waiting:
            self.data = self.data.with_columns(
                pl.when(
                    (pl.col("Car_ID") == car.unique_id)
                    & (pl.col("Light_ID") == light_id)
                )
                .then(pl.col("Wait_Time") + 1)
                .otherwise(pl.col("Wait_Time"))
                .alias("Wait_Time")
            )

    def save_data(self, path: Path) -> None:
        """Writes the data to a parquet file.

        Args:
            path (Path): Folder path to save the file in
        """
        self.data.write_parquet(file=Path.joinpath(path, "wait_times.parquet"))

    def init_wait_times(
        self, car: CarAgent, light_intersection_mapping: pl.DataFrame
    ) -> None:
        """Adds blank entries for each step the CarAgent instance takes through the grid

        Args:
            car (CarAgent): CarAgent instance
            light_intersection_mapping (pl.DataFrame): Mapping table for LightAgents and their corresponding intersections
        """
        for hop in car.path.keys():
            self.data.vstack(
                other=pl.DataFrame(
                    data={
                        "Car_ID": car.unique_id,
                        "Light_ID": light_intersection_mapping.filter(
                            pl.col("Intersection") == hop
                        ).select("Light_ID"),
                        "Wait_Time": None,
                    },
                    schema={
                        "Car_ID": pl.Int32,
                        "Light_ID": pl.Int16,
                        "Wait_Time": pl.Int16,
                    },
                    strict=False,
                ),
                in_place=True,
            )

    def is_arrival(self, car: CarAgent, light) -> bool:
        if (
            self.data.filter(
                (pl.col("Car_ID") == car.unique_id)
                & (pl.col("Light_ID") == light.unique_id)
            )
            .select(pl.col("Wait_Time"))
            .item()
            == 0
        ):
            return True
        else:
            return False
