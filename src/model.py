"""This module contains:
- TrafficModel class: A Mesa model simulating traffic."""

import datetime
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import mesa
import numpy as np
import polars as pl

from car import AgentArrived, CarAgent
from graph import Graph
from light import LightAgent


class TrafficModel(mesa.Model):
    """A Mesa model simulating traffic.

    Attributes:
        grid (Graph): Graph the environment uses.
        agents (AgentSet): Agents in the environment.
        car_paths (dict): A dictionary containing the paths of all agents.
        cars_waiting_times (dict): A dictionary containing the waiting times of all cars at each intersection.
        num_cars_hist (np.array): An array containing the history of the number of cars in the model.
        light_data (pl.DataFrame): A Polars DataFrame containing information about the LightAgents
        sim_data (pl.DataFrame): A Polars DataFrame containing information about a simulation run.
        lights_decision_log (dict): A dictionary containing the decisions of all lights over time.

    ## Methods:
        **step(self) -> None**:
            Advances the environment to next state.
        **create_cars(self, num_agents: int) -> None**:
            Function to add agents to the model.
        **remove_random_cars(self, num_agents: int) -> None**:
            Function to randomly remove n agents from the model.
        **create_lights_for_intersections(self) -> None**:
            Function to add a traffic light to each intersection of the model.
        **get_agents_by_type(self, agent_type: str) -> list**:
            Function to get all agents of a certain type.
        **get_agents_by_id(self, agent_id: list) -> list**:
            Function to get all agents by their unique ID.
        **get_last_intersection_of_car(self, car_id: int) -> str**:
            Function to get the last position of a car.
        **update_cars_waiting_times(self) -> None**:
            Function to update the waiting times of all cars at each intersection.
        **update_car_paths(self) -> None**:
            Function to update the paths of all agents.
        **car_respawn(self) -> None**:
            Respawns cars at each step depending on current time and number of cars in the model.
        **save_sim_data(self) -> Path**:
            Function to save the sim_data to a parquet file
        **get_cars_per_lane_of_light(self, light_position: str) -> dict**:
            Function to get the number of cars per lane of a light at the defined time.
    """

    def __init__(
        self, num_cars: int, sim_mode: bool = False, seed: int = None, optimization_type: str = "advanced", **kwargs
    ):
        """Initializes a new traffic environment.

        - Spawns a graph representing the grid
        - Initializes TrafficLight Agents at each intersection node
        - Initializes the specified number of CarAgents

        Args:
            num_cars (int): Number of car agents to spawn.
            sim_mode (bool): If True, the model is in simulation mode. Defaults to False.
            seed (int, optional): Seed used in model generation. Defaults to None.
            optimization (str): Optimization technique used for the lights (none, simple, advanced). Defaults to "advanced".
                - none: No optimization, lights are opend in a fixed cycle.
                - simple: Lights are opened based on the curret number of cars waiting at each lane, not taking the switching cooldown into account.
                - advanced: Lights are opened based on the curret and future number of cars waiting at each lane and taking the switching cooldown into account.
            **kwargs: Additional keyword arguments for configuring the graph object.
                - num_intersections (int): Number of intersections in the graph. Defaults to 50.
                - num_borders (int): Number of border nodes in the graph. Defaults to 25.
                - min_distance (int): Minimum distance between nodes. Defaults to 10.
                - max_distance (int): Maximum distance between nodes. Defaults to 20.
        """
        if optimization_type not in ["none", "simple", "advanced"]:
            raise ValueError(
                f"Optimization type '{optimization_type}' not supported. Supported optimizations are: none, simple, advanced."
            )
        
        super().__init__(seed=seed)

        self.grid = Graph(
            num_intersections=kwargs.get("num_intersections", 15),
            num_borders=kwargs.get("num_borders", 5),
            min_distance=kwargs.get("min_distance", 10),
            max_distance=kwargs.get("max_distance", 20),
        )

        self.arrivals = self.ArrivalsData()
        self.traffic = self.TrafficData()
        self.wait_times = self.WaitTimes()
        self.light_intersection_mapping = self.LightIntersectionMapping()
        self.light_data = self.LightData()
        self.n_cars = self.NumCars()
        self.connections = self.Connections()

        self.create_lights()
        self.optimization_type = optimization_type
        self.create_lights_for_intersections()
        self.car_paths = {}
        self.update_car_paths()
        self.lights_decision_log = {}
        self.create_cars(num_cars)

    def step(self) -> None:
        """Advances the environment to next state.

        - Calls Agents step functions
        - Updates simulation data
        - CarAgents are respawned based on current time and number of cars in the model
        """
        self._car_step()
        self._light_step()
        self.n_cars.update_data(
            steps=self.steps, n_cars=len(self.get_agents_by_type("CarAgent"))
        )
        self.car_respawn()

    def _car_step(self) -> None:
        """Actions each CarAgent takes each step.

        - CarAgent moves to next position
            - If CarAgent is between intersections, distance is decremented by one
            - If CarAgent is at an intersection, it changes it's position to the intersection
                - Only if it's lane is open
            - Increments *travel_time* by 1
            - If CarAgent reaches it's goal, it is removed from model
        """
        for car in self.get_agents_by_type("CarAgent")[:]:
            car: CarAgent
            try:
                car.move()
                car.travel_time += 1
                if (
                    car.position.startswith("intersection")
                    and car.path[car.position] == 1
                ):
                    self.wait_times.update_data(
                        car=car,
                        waiting=car.waiting,
                        light_intersection_mapping=self.light_intersection_mapping.data,
                    )
            except AgentArrived:
                car.remove()

    def _light_step(self) -> None:
        """Actions each LightAgent takes each step.

        - Checks if it is blocked by cooldown
            - If not, it opens the best lane, determined by optimization technique
            - If blocked, cooldown is decremented by 1
        """
        for light in self.get_agents_by_type("LightAgent"):
            light: LightAgent
            if light.current_switching_cooldown <= 0:
                if self.optimization_type == "none":
                    light.rotate_in_open_lane_cycle()
                elif self.optimization_type == "simple":
                    light.change_open_lane(light.optimize_open_lane())
                elif self.optimization_type == "advanced":
                    light.change_open_lane(light.optimize_open_lane_with_cooldown())
            else:
                light.current_switching_cooldown -= 1

            self.arrivals.update_data(light=light, steps=self.steps)
            self.traffic.update_data(light=light, steps=self.steps)

    def create_cars(self, num_cars: int) -> None:
        """Function to add cars to the model.

        - Updates car_paths and cars_waiting_times attributes

        Args:
            num_cars (int): Number of cars to add.
        """
        new_cars = CarAgent.create_agents(model=self, n=num_cars)
        self.update_car_paths()

        for car in new_cars:
            self.wait_times.init_wait_times(
                car=car, light_intersection_mapping=self.light_intersection_mapping.data
            )

    def remove_random_cars(self, num_cars: int) -> None:
        """Function to randomly remove n cars from the model.

        Args:
            num_cars (int): Number of cars to remove.
        """
        for _ in range(num_cars):
            car: CarAgent = random.choice(self.get_agents_by_type("CarAgent"))
            self.agents.remove(car)

    def create_lights(self) -> None:
        """Function to add traffic lights to the model.

        - Assigns one LightAgent to each intersection node
        """
        for intersection in self.grid.get_nodes("intersection"):
            light = LightAgent.create_agents(model=self, n=1, position=intersection)
            self.light_intersection_mapping.update_data(light=light[0])
            self.light_data.update_data(light=light[0], grid=self.grid)

            for item in self.grid.get_connections(
                filter_by=intersection, weights=True
            ).values():
                for connection in item:
                    self.connections.update_data(
                        intersection_u=intersection,
                        intersection_v=connection[0],
                        distance=connection[1],
                    )

    def get_agents_by_type(self, agent_type: str) -> list[mesa.Agent]:
        """Function to get all agents of a certain type.

        Args:
            agent_type (str): Type of agents to get. [CarAgent, LightAgent]

        Returns:
            list[mesa.Agent]: A list of agents of the given type.
        """
        if agent_type == "CarAgent":
            return [agent for agent in self.agents if isinstance(agent, CarAgent)]
        elif agent_type == "LightAgent":
            return [agent for agent in self.agents if isinstance(agent, LightAgent)]
        else:
            raise ValueError(f"Agent type {agent_type} not found")

    def get_agents_by_id(self, agent_id: list) -> list[mesa.Agent]:
        """Function to get all agents by their unique ID.

        Args:
            agent_id (list): List of unique agent IDs.

        Returns:
            list[mesa.Agent]: A list of agents with the given unique IDs.
        """
        agents = [agent for agent in self.agents if agent.unique_id in agent_id]

        return agents

    def get_last_intersection_of_car(self, car_id: int) -> str:
        """Function to get the last position of a car.

        Args:
            car_id (int): ID of the car.

        Returns:
            str: The last position of the car.
        """
        car: CarAgent = self.get_agents_by_id([car_id])[0]
        car_full_path: dict = self.car_paths[car_id]
        car_full_path_keys = list(car_full_path.keys())
        current_position_index = car_full_path_keys.index(car.position)

        if current_position_index == 0:
            previous_position: str = car.position
        else:
            previous_position: str = car_full_path_keys[current_position_index - 1]

        # Get the corresponding intersection, if the cars last position was a border node (TODO) @mxrio
        if previous_position.startswith("border"):
            first_intersection = list(
                car.model.car_paths[car.unique_id].keys()
            )[
                1
            ]  # Warum über car den car_path holen -> self.car_paths[car.unique_id] @mxrio
            lane = list(self.grid.neighbors(previous_position))
            lane.remove(first_intersection)
            previous_position = lane[0]

        return previous_position

    # def update_cars_waiting_times(self) -> None:
    #     """Function to update the waiting times of all cars at each intersection."""

    #     for car in self.get_agents_by_type("CarAgent"):
    #         car: CarAgent
    #         if car.unique_id not in list(self.cars_waiting_times.keys()):
    #             self.cars_waiting_times[car.unique_id] = {
    #                 intersection: 0
    #                 for intersection in list(
    #                     car.model.car_paths[car.unique_id].keys()  # s.o. @mxrio
    #                 )
    #                 if intersection.startswith(
    #                     "intersection"
    #                 )  # grid.get_nodes verwenden vielleicht? @mxrio
    #             }

    #         if car.waiting:
    #             self.cars_waiting_times[car.unique_id][car.position] += 1

    def update_car_paths(self) -> None:
        """Function to update the paths of all cars."""
        for car in self.get_agents_by_type("CarAgent"):
            car: CarAgent
            if car.unique_id not in list(self.car_paths.keys()):
                self.car_paths[car.unique_id] = car.path.copy()

    def car_respawn(self):
        """Respawns cars at each steps dependant of current time and number of cars in the model.

        - Calculates the current value on a sine function adjusted to the model's internal time
        - Calculates the next value on the function scaled to the number of cars
        - Calculates the number of cars to add with a variance of ~20%
        """
        sine_value = np.sin(2 * np.pi * self.steps / 200)
        next_sine_value = (
            (sine_value + 1)
            / 2
            * 2
            * self.n_cars.data.row(0)[self.n_cars.data.columns.index("Num_Cars")]
        )
        diff = (
            next_sine_value
            - self.n_cars.data.row(-1)[self.n_cars.data.columns.index("Num_Cars")]
        )
        diff_variance = diff * random.uniform(0.8, 1.2)
        cars_to_add = int(diff_variance)

        if cars_to_add > 0:
            self.create_cars(cars_to_add)

    def get_cars_per_lane_of_light(self, light_position: str, tick: int) -> dict:
        """Function to get the number of cars per lane of a light at the defined time.

        Args:
            light_position (str): The position of the light.
            tick (int): The time when the cars will arrive at the light (0 is the current tick).

        Returns:
            dict: A dictionary containing the number of cars per lane of the light.
        """
        if tick < 0:
            raise ValueError("Tick must be greater than or equal to 0")

        cars_per_lane = {
            lane: 0
            for lane in self.grid.neighbors(light_position)
            if lane.startswith("intersection")
        }

        if tick == 0:
            for car in self.get_agents_by_type("CarAgent"):
                if car.position == light_position and car.waiting:
                    cars_per_lane[self.get_last_intersection_of_car(car.unique_id)] += 1
        else:
            for car in self.get_agents_by_type("CarAgent"):
                if list(car.path.keys())[0] == light_position and list(car.path.values())[0] == tick:
                    cars_per_lane[self.get_last_intersection_of_car(car.unique_id)] += 1

        return cars_per_lane
    

    def update_lights_decision_log(
        self,
        light: LightAgent,
        cars_per_lane: dict,
        decision_lane: str,
        model_step: int,
    ) -> None:
        """Function to update the decision log of all lights.
        The dict looks like this: {light.unique_id: {step:{decision_lane:intersection_3, intersection_1:cars_at_lane_1, intersection_2:cars_at_lane_2, intersection_3:cars_at_lane_3}}}

        """
        if light.unique_id not in list(self.lights_decision_log.keys()):
            self.lights_decision_log[light.unique_id] = {}
            self.lights_decision_log[light.unique_id][model_step] = {
                "decision_lane": decision_lane
            }
            self.lights_decision_log[light.unique_id][model_step].update(cars_per_lane)

        else:
            self.lights_decision_log[light.unique_id][model_step] = {
                "decision_lane": decision_lane
            }
            self.lights_decision_log[light.unique_id][model_step].update(cars_per_lane)

    @dataclass
    class DataPath:
        path: Path = field(
            default_factory=lambda: Path.joinpath(
                Path.cwd(),
                "data",
                datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            )
        )

        def __post_init__(self):
            self.path.mkdir(parents=True, exist_ok=True)

    @dataclass
    class SimData(ABC):
        @abstractmethod
        def __post_init__(self):
            """Constructs the data schema"""
            pass

        @abstractmethod
        def update_data(self) -> None:
            """Updates the data"""
            pass

        @abstractmethod
        def save_data(self, path: Path) -> None:
            """Writes the data to a parquet file.

            Args:
                path (Path): Folder path to save the file in
            """
            pass

    @dataclass
    class ArrivalsData(SimData):
        data: pl.DataFrame = field(default_factory=pl.DataFrame)

        def __post_init__(self):
            """Constructs the data schema"""
            self.data = pl.DataFrame(
                schema={
                    "Index": pl.Int32,
                    "Light_ID": pl.Int16,
                    "Time": pl.Int16,
                    "Arrivals": pl.Int16,
                },
                strict=False,
            )

        def update_data(self, light: LightAgent, steps: int) -> None:
            """Updates the data

            Args:
                light (LightAgent): LightAgent instance
                steps (int): Internal step counter of TrafficModel instance
            """
            self.data.vstack(
                pl.DataFrame(
                    data={
                        "Index": steps,
                        "Light_ID": light.unique_id,
                        "Time": 200 - (steps % 200),
                        "Arrivals": light.get_num_arrivals(),
                    },
                    schema={
                        "Index": pl.Int32,
                        "Light_ID": pl.Int16,
                        "Time": pl.Int16,
                        "Arrivals": pl.Int16,
                    },
                ),
                in_place=True,
            )

        def save_data(self, path: Path) -> None:
            """Writes the data to a parquet file.

            Args:
                path (Path): Folder path to save the file in
            """
            self.data.write_parquet(file=Path.joinpath(path, "arrivals.parquet"))

    @dataclass
    class TrafficData(SimData):
        data: pl.DataFrame = field(default_factory=pl.DataFrame)

        def __post_init__(self):
            """Constructs the data schema"""
            self.data = pl.DataFrame(
                schema={
                    "Index": pl.Int32,
                    "Light_ID": pl.Int16,
                    "Time": pl.Int16,
                    "Num_Cars": pl.Int16,
                },
                strict=False,
            )

        def update_data(self, light: LightAgent, steps: int) -> None:
            """Updates the data

            Args:
                light (LightAgent): LightAgent instance
                steps (int): Internal step counter of TrafficModel instance
            """
            self.data.vstack(
                other=pl.DataFrame(
                    data={
                        "Index": steps,
                        "Light_ID": light.unique_id,
                        "Time": 200 - (steps % 200),
                        "Num_Cars": light.get_num_cars(),
                    },
                    schema={
                        "Index": pl.Int32,
                        "Light_ID": pl.Int16,
                        "Time": pl.Int16,
                        "Num_Cars": pl.Int16,
                    },
                    strict=False,
                ),
                in_place=True,
            )

        def save_data(self, path: Path) -> None:
            """Writes the data to a parquet file.

            Args:
                path (Path): Folder path to save the file in
            """
            self.data.write_parquet(file=Path.joinpath(path, "traffic.parquet"))

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
                light_intersection_mapping.filter(
                    pl.col("Intersection") == car.position
                )
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

        def is_arrival(self, car: CarAgent, light: LightAgent) -> bool:
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

    @dataclass
    class LightIntersectionMapping(SimData):
        data: pl.DataFrame = field(default_factory=pl.DataFrame)

        def __post_init__(self):
            """Constructs the data schema"""
            self.data = pl.DataFrame(
                schema={"Light_ID": pl.Int16, "Intersection": pl.String},
                strict=False,
            )

        def update_data(self, light: LightAgent) -> None:
            """Updates the data

            Args:
                light (LightAgent): LightAgent instance
            """
            self.data = self.data.vstack(
                other=pl.DataFrame(
                    data={"Light_ID": light.unique_id, "Intersection": light.position},
                    schema={"Light_ID": pl.Int16, "Intersection": pl.String},
                    strict=False,
                ),
                in_place=True,
            )

        def save_data(self, path: Path) -> None:
            """Writes the data to a parquet file.

            Args:
                path (Path): Folder path to save the file in
            """
            self.data.write_parquet(
                file=Path.joinpath(path, "light_intersection_mapping.parquet")
            )

    @dataclass
    class LightData(SimData):
        data: pl.DataFrame = field(default_factory=pl.DataFrame)

        def __post_init__(self):
            """Constructs the data schema"""
            self.data = pl.DataFrame(
                schema={
                    "Light_ID": pl.Int16,
                    "Centrality": pl.Float32,
                    "Is_Entrypoint": pl.Boolean,
                },
                strict=False,
                orient="row",
            )

        def update_data(self, light: LightAgent, grid: Graph) -> None:
            """Updates the data

            Args:
                light (LightAgent): LightAgent instance
                grid (Graph): Graph instance the TrafficModel uses
            """
            self.data.vstack(
                other=pl.DataFrame(
                    data=[
                        (
                            light.unique_id,
                            light.get_centrality(grid),
                            light.check_is_entrypoint(grid),
                        )
                    ],
                    schema={
                        "Light_ID": pl.Int16,
                        "Centrality": pl.Float32,
                        "Is_Entrypoint": pl.Boolean,
                    },
                    strict=False,
                    orient="row",
                ),
                in_place=True,
            )

        def save_data(self, path: Path) -> None:
            """Writes the data to a parquet file.

            Args:
                path (Path): Folder path to save the file in
            """
            self.data.write_parquet(file=Path.joinpath(path, "lights.parquet"))

    @dataclass
    class NumCars(SimData):
        data: pl.DataFrame = field(default_factory=pl.DataFrame)

        def __post_init__(self):
            """Constructs the data schema"""
            self.data = pl.DataFrame(
                schema={"Time": pl.Int32, "Num_Cars": pl.Int32}, strict=False
            )

        def update_data(self, steps: int, n_cars: int) -> None:
            """Updates the data

            Args:
                steps (int): Internal step counter of TrafficModel instance
                n_cars (int): Number of CarAgent instances in TrafficModel
            """
            self.data = self.data.vstack(
                other=pl.DataFrame(
                    data={
                        "Time": 200 - (steps % 200),
                        "Num_Cars": n_cars,
                    },
                    schema={"Time": pl.Int32, "Num_Cars": pl.Int32},
                    strict=False,
                ),
                in_place=True,
            )

        def save_data(self, path: Path) -> None:
            """Writes the data to a parquet file.

            Args:
                path (Path): Folder path to save the file in
            """
            self.data.write_parquet(file=Path.joinpath(path, "num_cars.parquet"))

    @dataclass
    class Connections(SimData):
        data: pl.DataFrame = field(default_factory=pl.DataFrame)

        def __post_init__(self):
            """Constructs the data schema"""
            self.data = pl.DataFrame(
                schema={
                    "Intersection_u": pl.String,
                    "Intersection_v": pl.String,
                    "Distance": pl.Int16,
                },
                strict=False,
            )

        def update_data(
            self, intersection_u: str, intersection_v: str, distance: int
        ) -> None:
            """Updates the data

            Args:
                intersection_u (str): Intersection u
                intersection_v (str): Intersection v
                distance (int): Weight of edge between intersection_u and intersection_v
            """

            self.data.vstack(
                other=pl.DataFrame(
                    data={
                        "Intersection_u": intersection_u,
                        "Intersection_v": intersection_v,
                        "Distance": distance,
                    },
                    schema={
                        "Intersection_u": pl.String,
                        "Intersection_v": pl.String,
                        "Distance": pl.Int16,
                    },
                    strict=False,
                ),
                in_place=True,
            )

        def save_data(self, path: Path) -> None:
            """Writes the data to a parquet file.

            Args:
                path (Path): Folder path to save the file in
            """
            self.data.write_parquet(file=Path.joinpath(path, "connections.parquet"))
