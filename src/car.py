"""
car.py

This module defines the `CarAgent` class, representing vehicles navigating the
traffic simulation grid. It also includes the `AgentArrived` exception, raised
when a car reaches its destination, and the `WaitTimes` dataclass for tracking
how long cars wait at traffic lights.

Classes:
    - CarAgent: Represents a car agent in the simulation, handling movement,
                pathfinding, and interaction with traffic lights.
    - AgentArrived: Custom exception raised when a `CarAgent` reaches its goal node.
    - WaitTimes: Dataclass inheriting from `SimData` to store and manage wait
                 time data for individual cars at intersections.

Dependencies:
    - random: For random goal selection (`random.randint`).
    - dataclasses: For creating the `WaitTimes` data class (`dataclass`, `field`).
    - mesa: Core agent-based modeling framework (`mesa.Agent`, `mesa.Model`).
    - polars: For efficient data handling in `WaitTimes` (`pl.DataFrame`, etc.).
    - networkx: For pathfinding using Dijkstra's algorithm (`dijkstra_path`).
    - data.SimData: Base class for `WaitTimes`.
"""

import random
from dataclasses import dataclass, field

import mesa
from networkx import dijkstra_path
import polars as pl

from data import SimData


class CarAgent(mesa.Agent):
    """Represents a car agent navigating the traffic grid graph.

    Each car is initialized at a random border node and assigned another random
    border node as its destination. It calculates the shortest path using
    Dijkstra's algorithm and attempts to follow this path step by step.
    Cars interact with `LightAgent` instances at intersections, potentially
    waiting if the required lane is not open. The agent tracks its travel time
    and wait times at lights.

    Inherits from `mesa.Agent`.

    Attributes:
        start (str): The ID of the border node where the agent was initialized.
        goal (str): The ID of the border node assigned as the agent's destination.
        path (dict): A dictionary representing the remaining path. Keys are node IDs
                     along the path, and values are the remaining distance (steps)
                     to reach the next node in the path. The value is `None` for
                     the final goal node.
        position (str): The ID of the node where the agent is currently located.
                        Can be an intersection or border node.
        previous_position (str | None): The ID of the node the agent was at in the
                                        previous step. Initially None.
        waiting (bool): Flag indicating if the agent is currently stopped at a
                        traffic light (True) or allowed to move (False).
        travel_time (int): The total number of simulation steps the agent has existed.
        wait_times (WaitTimes): A `WaitTimes` instance tracking the agent's wait
                                duration at each intersection it encounters.

    Methods:
        step(self) -> None:
            Executes the agent's actions for one simulation step.
        compute_goal(self) -> str:
            Selects a random border node (different from the start) as the goal.
        compute_path(self) -> dict:
            Calculates the shortest path from the current position to the goal.
        move(self) -> None:
            Attempts to advance the agent along its path for one step.
        check_if_car_at_light(self) -> bool:
            Determines if the car is currently considered "at" an intersection light.
        check_lights(self) -> None:
            Checks the status of the traffic light if the car is at an intersection
            and updates the `waiting` status accordingly.
    """

    def __init__(self, model: mesa.Model):
        """Initializes a new CarAgent instance.

        Assigns a unique ID, places the agent on a random border node using the
        model's grid, computes a random goal node (another border node), and
        calculates the shortest path using Dijkstra's algorithm.
        Initializes travel time, waiting status, and the `WaitTimes` data collector.

        Args:
            model (mesa.Model): The `TrafficModel` instance the agent belongs to.
        """
        super().__init__(model)
        self.start = self.model.grid.place_agent(agent_id=self.unique_id)
        self.previous_position = None
        self.position = self.start
        self.goal = self.compute_goal()
        self.path = self.compute_path()
        self.waiting = False
        self.travel_time = 0

        self.wait_times = WaitTimes()

    def step(self) -> None:
        """Executes the agent's actions for a single simulation step.

        Attempts to move the agent along its path using the `move()` method.
        If the agent is determined to be at a traffic light (`check_if_car_at_light`),
        it updates its wait time statistics in its `wait_times` collector.
        If the `move()` method raises an `AgentArrived` exception (meaning the
        agent reached its goal), the agent's collected wait times are added to
        the model's global collection, and the agent is removed from the simulation.
        """
        try:
            self.move()
            if self.check_if_car_at_light():
                self.wait_times.update_data(
                    car=self,
                    waiting=self.waiting,
                    light_intersection_mapping=self.model.light_intersection_mapping.data,
                )
        except AgentArrived:
            self.model.global_wait_times.update_data(self.wait_times.get_data())
            self.remove()

    def compute_goal(self) -> str:
        """Selects a random border node as the agent's destination.

        Retrieves all border nodes from the model's grid, removes the agent's
        starting node from the list, and randomly selects one of the remaining
        border nodes as the goal.

        Returns:
            str: The ID of the randomly selected goal node.
        """
        borders = [node for node in self.model.grid if node.find("border") == 0]
        borders.remove(self.start)
        assigned_goal = borders[random.randint(0, (len(borders) - 1))]

        return assigned_goal

    def compute_path(self) -> dict:
        """Computes the shortest path from the agent's current position to its goal.

        Uses `networkx.dijkstra_path` with edge weights ('weight') to find the
        sequence of nodes representing the shortest path. Converts this sequence
        into a dictionary where keys are the nodes in the path and values are the
        distances (edge weights) to the *next* node in the path. The value for
        the final goal node is `None`.

        Returns:
            dict: The calculated path dictionary {node_id: distance_to_next_node}.
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
        """Attempts to move the agent one step along its calculated path.

        First, calls `check_lights()` to update the `waiting` status based on the
        traffic light at the current intersection (if applicable).
        If the agent is not `waiting`:
            - If the current `position` is the `goal`, raises `AgentArrived`.
            - If the distance to the next node in `path` is 1, updates the agent's
              `position` to the next node and removes the current node from `path`.
            - Otherwise (distance > 1), decrements the distance to the next node
              in the `path` dictionary by 1.
        Increments the agent's `travel_time` by 1, regardless of whether it moved.

        Raises:
            AgentArrived: If the agent's current `position` matches its `goal`.
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
                    self.previous_position = self.path.pop(self.position)
                    self.position = steps[next_index]
                else:
                    self.path[self.position] = self.path.get(self.position) - 1

        self.travel_time += 1

    def check_if_car_at_light(self) -> bool:
        """Determines if the car is currently considered "at" an intersection light.

        A car is considered "at" a light if its current `position` is an
        intersection node and the remaining distance to the next node in its
        original path (`self.model.car_paths`) matches the current remaining
        distance in its active path (`self.path`). This signifies it has just
        arrived at the intersection node within its path segment.

        Returns:
            bool: True if the car is positioned at an intersection node and has
                  just completed the travel segment leading to it, False otherwise.
        """
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

        return at_light

    def check_lights(self) -> None:
        """Checks traffic light status if the car is at an intersection.

        Determines if the car is currently "at" an intersection light using
        `check_if_car_at_light()`.
        If it is at a light:
            - Identifies the `LightAgent` at the current intersection.
            - Compares the `open_lane` of the `LightAgent` with the car's
              previous position (the lane it arrived from, obtained via
              `model.get_last_intersection_of_car`).
            - Sets the car's `waiting` status to `True` if the lanes do not match
              (light is red for this car), and `False` otherwise (light is green).
        If the car is not currently at a light, this method has no effect.
        """
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
    """Manages and stores wait time data for a single `CarAgent` at intersections.

    This dataclass inherits from `SimData` and uses a Polars DataFrame to track
    how many simulation steps a specific car spends waiting at each traffic light
    (intersection) it encounters along its path.

    Attributes:
        data (pl.DataFrame): A Polars DataFrame holding the wait time records.
                             Columns: 'Car_ID' (Int32), 'Light_ID' (Int16),
                             'Wait_Time' (Int16).

    ## Methods:
        **__post_init__()**:
            Initializes the Polars DataFrame with the wait time schema.
        **update_data(car, waiting, light_intersection_mapping)**:
            Updates the wait time for the car at its current intersection.
        **get_data()**:
            Returns the collected wait time data for the car.
        **init_wait_times(car, light_intersection_mapping)**:
            Initializes wait time records for all intersections in the car's path.
        **is_arrival(car, light)**:
            Checks if the car has just arrived at the specified light in the current step.
    """

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Initializes the Polars DataFrame with the wait time schema.

        Sets up the structure for storing wait time data, including the car's ID,
        the ID of the light at the intersection, and the accumulated wait time
        at that light.
        """
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
        """Updates the wait time for the car at its current intersection.

        Identifies the `Light_ID` corresponding to the car's current intersection
        position using the provided mapping.
        If the car is `waiting` (light is red), increments the 'Wait_Time' for the
        corresponding Car_ID and Light_ID entry in the DataFrame.
        If the car is not `waiting` (light is green or car is moving past), resets
        the 'Wait_Time' for that entry to 0. Assumes an entry for the car/light
        pair already exists (potentially created by `init_wait_times`).

        Args:
            car (CarAgent): The `CarAgent` instance whose wait time is being updated.
            waiting (bool): The current waiting status of the `car`.
            light_intersection_mapping (pl.DataFrame): A DataFrame mapping
                                                       'Intersection' IDs to 'Light_ID's.
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

    def get_data(self) -> pl.DataFrame:
        """Returns the collected wait time data for the car.

        Returns:
            pl.DataFrame: A Polars DataFrame containing all recorded wait times
                          (Car_ID, Light_ID, Wait_Time) for this car instance.
        """
        return self.data

    def init_wait_times(
        self, car: CarAgent, light_intersection_mapping: pl.DataFrame
    ) -> None:
        """Initializes wait time records for all intersections in the car's path.

        Iterates through the nodes (hops) in the car's planned path. For each hop
        that corresponds to a known light intersection (found via the mapping),
        it adds a new row to the `data` DataFrame with the car's ID, the light's ID,
        and an initial 'Wait_Time' of `None` (or potentially 0 depending on Polars handling).
        This pre-populates entries so `update_data` can increment/reset them.

        Args:
            car (CarAgent): The `CarAgent` instance for which to initialize records.
            light_intersection_mapping (pl.DataFrame): A DataFrame mapping
                                                       'Intersection' IDs to 'Light_ID's.
        """
        for hop in car.path.keys():
            self.data.extend(
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
            )

    def is_arrival(self, car: CarAgent, light) -> bool:
        """Checks if the car has just arrived at the specified light in the current step.

        Determines arrival by checking if the recorded 'Wait_Time' for the given
        `car` and `light` is currently 0. This implies the car was not waiting in
        the previous step(s) at this light and has just encountered it (or passed through).

        Args:
            car (CarAgent): The `CarAgent` instance to check.
            light (LightAgent): The `LightAgent` instance representing the intersection
                                to check arrival at.

        Returns:
            bool: True if the car's wait time for this light is currently 0,
                  False otherwise.
        """
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
