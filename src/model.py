"""
model.py

This module defines the `TrafficModel` class, the core Mesa model for simulating
urban traffic flow. It orchestrates the interaction between `CarAgent` and
`LightAgent` instances within a `Graph` representing the road network.

The module also includes several dataclasses inheriting from `SimData` to manage
and store various simulation data points, such as light-intersection mappings,
light metadata, car counts over time, network connections, and global car wait times.

Classes:
    - TrafficModel: The main Mesa model for the traffic simulation.
    - DataCollector: Helper class within `TrafficModel` to aggregate data from agents.
    - LightIntersectionMapping: Dataclass mapping LightAgent IDs to intersection IDs.
    - LightData: Dataclass storing metadata about LightAgents (centrality, entrypoint status).
    - NumCars: Dataclass tracking the total number of cars in the simulation over time.
    - Connections: Dataclass storing details of connections (edges) between intersections.
    - GlobalWaitTimes: Dataclass aggregating wait times for all cars across all lights.

Dependencies:
    - random: For random operations (e.g., car removal, respawn variance).
    - dataclasses: For creating data storage classes.
    - mesa: Core agent-based modeling framework.
    - numpy: For numerical operations (e.g., sine wave in respawn logic).
    - polars: For efficient data handling in storage classes.
    - car.CarAgent: Represents vehicles in the simulation.
    - graph.Graph: Represents the road network.
    - light.LightAgent: Represents traffic lights at intersections.
    - data.SimData: Abstract base class for data storage classes.
    - regressor.Regressor: Used for ML-based traffic prediction (optional).
"""

import random
from dataclasses import dataclass, field

import mesa
import numpy as np
import polars as pl

from car import CarAgent
from data import SimData
from graph import Graph
from light import LightAgent
from regressor import Regressor


class TrafficModel(mesa.Model):
    """A Mesa model simulating urban traffic flow on a graph-based road network.

    This model manages the creation, movement, and interaction of `CarAgent` and
    `LightAgent` instances. It uses a `Graph` object for the network topology
    and includes mechanisms for data collection, car respawning based on a
    sine wave pattern, and different traffic light optimization strategies.

    Attributes:
        grid (Graph): The graph representing the road network (intersections, borders, edges).
        num_cars (int): The initial number of cars to create.
        optimization_type (str): The strategy used by `LightAgent`s ('none', 'simple',
                                 'advanced', 'advanced_ml').
        light_intersection_mapping (LightIntersectionMapping): Dataclass storing the mapping
                                                               between LightAgent IDs and
                                                               intersection node IDs.
        light_data (LightData): Dataclass storing metadata for each `LightAgent`
                                (e.g., centrality, entrypoint status).
        n_cars (NumCars): Dataclass tracking the total number of `CarAgent`s in the
                          simulation over time.
        connections (Connections): Dataclass storing details about the connections
                                   (edges) between intersection nodes.
        regressor (Regressor | None): An instance of the `Regressor` class used for
                                      ML-based predictions if `optimization_type` is
                                      'advanced_ml'. Otherwise, None.
        lights_decision_log (dict): A log storing the decisions made by each `LightAgent`
                                    at each step, including the chosen lane and the car
                                    counts considered. Structure:
                                    {light_id: {step: {'decision_lane': str, lane_id: count,...}}}
        car_paths (dict): A dictionary storing the originally computed, full path for
                          each `CarAgent`. Keys are car IDs, values are path dictionaries
                          {node_id: distance_to_next}. Used for reference.
        global_wait_times (GlobalWaitTimes): Dataclass aggregating the final wait time
                                             data collected from all `CarAgent`s that
                                             reach their destination.
        steps (int): Internal Mesa step counter, incremented automatically.
        schedule (mesa.time.BaseScheduler): The scheduler managing agent activation order.
                                            (Implicitly managed by Mesa).
        _agents_by_type (dict): Internal Mesa dictionary grouping agents by class.

    Methods:
        step():
            Advances the simulation by one time step.
        create_cars(num_cars):
            Creates and adds a specified number of `CarAgent`s.
        remove_random_cars(num_cars):
            Randomly removes a specified number of `CarAgent`s.
        create_lights():
            Creates `LightAgent`s for all intersection nodes in the grid.
        get_agents_by_type(agent_type):
            Returns a Mesa AgentSet of agents of the specified type.
        get_agents_by_id(agent_id):
            Returns a list of agents matching the provided IDs.
        get_last_intersection_of_car(car_id):
            Determines the intersection node a car arrived from.
        update_car_paths():
            Updates the `car_paths` dictionary with paths of newly added cars.
        car_respawn():
            Adds new cars based on a sine wave pattern relative to the current car count.
        get_cars_per_lane_of_light(light_position, tick):
            Counts cars waiting or arriving at a light.
        update_lights_decision_log(light, cars_per_lane, decision_lane, model_step):
            Logs a light's decision.
        DataCollector (nested class):
            Helper to aggregate data from multiple agents.
    """

    def __init__(
        self,
        num_cars: int,
        seed: int = 42,
        optimization_type: str = "advanced_ml",
        **kwargs,
    ):
        """Initializes the TrafficModel simulation environment.

        Sets up the simulation grid (Graph), creates initial `LightAgent`s at
        intersections, initializes data collectors, optionally loads the ML regressor,
        creates the initial set of `CarAgent`s, and computes their initial paths.

        Args:
            num_cars (int): The initial number of `CarAgent`s to create in the simulation.
            seed (int, optional): Seed for the random number generator used by Mesa and
                                  other random processes. Defaults to 42.
            optimization_type (str, optional): The optimization strategy for `LightAgent`s.
                                               Must be one of 'none', 'simple', 'advanced',
                                               or 'advanced_ml'. Defaults to "advanced_ml".
            **kwargs: Additional keyword arguments passed to the `Graph` constructor,
                      controlling network generation (e.g., `num_intersections`,
                      `num_borders`, `min_distance`, `max_distance`).

        Raises:
            ValueError: If `optimization_type` is not one of the supported values.
        """
        super().__init__(seed=seed)

        if optimization_type not in ["none", "simple", "advanced", "advanced_ml"]:
            raise ValueError(
                f"Optimization type '{optimization_type}' not supported. Supported optimizations are: none, simple, advanced."
            )
        else:
            self.optimization_type = optimization_type

        self.grid = Graph(
            num_intersections=kwargs.get("num_intersections", 15),
            num_borders=kwargs.get("num_borders", 5),
            min_distance=kwargs.get("min_distance", 10),
            max_distance=kwargs.get("max_distance", 20),
        )

        self.num_cars = num_cars
        self.light_intersection_mapping = LightIntersectionMapping()
        self.light_data = LightData()
        self.n_cars = NumCars()
        self.connections = Connections()

        self.create_lights()
        if self.optimization_type == "advanced_ml":
            self.regressor = Regressor()
        self.lights_decision_log = {}
        self.create_cars(self.num_cars)
        self.car_paths = {}
        self.update_car_paths()
        self.global_wait_times = GlobalWaitTimes()

    def step(self) -> None:
        """Advances the simulation by one time step.

        Executes the following actions in order:
        1. Records the current number of cars using `n_cars.update_data()`.
        2. Calls the `step()` method for all `LightAgent`s, passing the
           `optimization_type` and current `steps`.
        3. Calls the `step()` method for all `CarAgent`s.
        4. Calls `car_respawn()` to potentially add new cars based on the sine wave logic.
        5. Calls `update_car_paths()` to store the paths of any newly added cars.
        Increments the internal step counter (`self.steps`).
        """
        self.n_cars.update_data(
            steps=self.steps, n_cars=self._agents_by_type[CarAgent].__len__()
        )

        self._agents_by_type[LightAgent].do(
            LightAgent.step, optimization_type=self.optimization_type, steps=self.steps
        )

        self._agents_by_type[CarAgent].do(CarAgent.step)

        self.car_respawn()
        self.update_car_paths()

    def create_cars(self, num_cars: int) -> None:
        """Creates and adds a specified number of `CarAgent`s to the model.

        Uses the `CarAgent.create_agents` class method to instantiate new agents.
        For each newly created car, initializes its wait time tracking structure
        using `car.wait_times.init_wait_times()`.

        Args:
            num_cars (int): The number of new `CarAgent`s to create and add.
        """
        new_cars = CarAgent.create_agents(model=self, n=num_cars)

        for car in new_cars:
            car: CarAgent
            car.wait_times.init_wait_times(
                car=car, light_intersection_mapping=self.light_intersection_mapping.data
            )

    def remove_random_cars(self, num_cars: int) -> None:
        """Randomly selects and removes a specified number of `CarAgent`s from the model.

        Args:
            num_cars (int): The number of `CarAgent`s to randomly remove. If this
                            number is greater than the current number of cars, all
                            cars will be removed.
        """
        for _ in range(num_cars):
            car: CarAgent = random.choice(self._agents_by_type[CarAgent])
            self.agents.remove(car)

    def create_lights(self) -> None:
        """Creates `LightAgent`s for all intersection nodes in the grid.

        Iterates through all nodes identified as 'intersection' in the `grid`.
        For each intersection, creates one `LightAgent`, adds it to the model,
        updates the `light_intersection_mapping` and `light_data` collectors,
        and records the connections (edges) originating from this intersection
        in the `connections` data collector.
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

    def get_agents_by_type(self, agent_type: str):
        """Retrieves all active agents of a specific type from the model.

        Provides convenient access to Mesa's internal `_agents_by_type` dictionary.

        Args:
            agent_type (str): The name of the agent class to retrieve (e.g., "CarAgent",
                              "LightAgent").

        Returns:
            mesa.AgentSet: A set-like object containing all active agents of the
                           specified type currently in the model.

        Raises:
            ValueError: If `agent_type` is not "CarAgent" or "LightAgent".
        """
        if agent_type == "CarAgent":
            return self._agents_by_type[CarAgent]
        elif agent_type == "LightAgent":
            return self._agents_by_type[LightAgent]
        else:
            raise ValueError(f"Agent type {agent_type} not found")

    def get_agents_by_id(self, agent_id: list) -> list[mesa.Agent]:
        """Retrieves specific agents from the model based on their unique IDs.

        Args:
            agent_id (list): A list of unique agent IDs to search for.

        Returns:
            list[mesa.Agent]: A list containing the agent objects corresponding to
                              the provided IDs found in the model. Returns an empty
                              list if no matching agents are found.
        """
        agents = [agent for agent in self.agents if agent.unique_id in agent_id]

        return agents

    def get_last_intersection_of_car(self, car_id: int) -> str:
        """Determines the intersection node a car arrived from in the previous step.

        Uses the car's full original path stored in `car_paths` to find the node
        immediately preceding the car's current `position` in that path. Handles
        the edge case where the car is at the start of its path or if the previous
        node was a border node (returning the connected intersection instead).

        Args:
            car_id (int): The unique ID of the `CarAgent`.

        Returns:
            str: The ID of the intersection node the car came from.
        """
        car: CarAgent = self.get_agents_by_id([car_id])[0]
        car_full_path: dict = self.car_paths[car_id]
        car_full_path_keys = list(car_full_path.keys())
        current_position_index = car_full_path_keys.index(car.position)

        if current_position_index == 0:
            previous_position: str = car.position
        else:
            previous_position: str = car_full_path_keys[current_position_index - 1]

        if previous_position.startswith("border"):
            first_intersection = list(self.car_paths[car.unique_id].keys())[1]
            lane = list(self.grid.neighbors(previous_position))
            lane.remove(first_intersection)
            previous_position = lane[0]

        return previous_position

    def update_car_paths(self) -> None:
        """Stores the initial, full path for any newly added `CarAgent`s.

        Iterates through all current `CarAgent`s. If a car's ID is not already
        a key in the `car_paths` dictionary, it adds the car's ID and a copy
        of its initial `path` attribute to the dictionary. This preserves the
        original path for later reference (e.g., by `get_last_intersection_of_car`).
        """
        for car in self._agents_by_type[CarAgent]:
            car: CarAgent
            if car.unique_id not in list(self.car_paths.keys()):
                self.car_paths[car.unique_id] = car.path.copy()

    def car_respawn(self):
        """Adds new cars to the simulation based on a time-varying pattern.

        Calculates a target number of cars based on a sine wave function that
        cycles every 200 steps and is scaled by the number of cars present at
        the beginning of the cycle. It then determines the difference between this
        target and the current number of cars, adds some random variance (~+/-20%),
        and creates that many new cars if the result is positive.
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
        """Counts cars waiting or arriving at a specific light's incoming lanes.

        Calculates the number of cars associated with each incoming intersection lane
        of the specified `light_position`.
        - If `tick` is 0, it counts cars currently at `light_position` and `waiting`.
        - If `tick` is greater than 0, it counts cars whose *next* node in their path
          is `light_position` and the remaining distance to it equals `tick`.

        Args:
            light_position (str): The node ID of the intersection where the light is located.
            tick (int): The time step relative to the current step to check for cars.
                        0 means currently waiting cars. >0 means cars arriving in `tick` steps.

        Returns:
            dict: A dictionary where keys are the IDs of the incoming intersection lanes
                  and values are the counts of relevant cars for that lane and tick.

        Raises:
            ValueError: If `tick` is less than 0.
        """
        if tick < 0:
            raise ValueError("Tick must be greater than or equal to 0")

        cars_per_lane = {
            lane: 0
            for lane in self.grid.neighbors(light_position)
            if lane.startswith("intersection")
        }

        if tick == 0:
            for car in self._agents_by_type[CarAgent]:
                if car.position == light_position and car.waiting:
                    cars_per_lane[self.get_last_intersection_of_car(car.unique_id)] += 1
        else:
            for car in self._agents_by_type[CarAgent]:
                if (
                    list(car.path.keys())[0] == light_position
                    and list(car.path.values())[0] == tick
                ):
                    cars_per_lane[self.get_last_intersection_of_car(car.unique_id)] += 1

        return cars_per_lane

    def update_lights_decision_log(
        self,
        light: LightAgent,
        cars_per_lane: dict,
        decision_lane: str,
        model_step: int,
    ) -> None:
        """Logs the decision made by a `LightAgent` at a specific step.

        Stores the chosen `decision_lane` and the `cars_per_lane` counts that
        informed the decision in the `lights_decision_log` dictionary, nested
        under the light's unique ID and the current `model_step`.

        Args:
            light (LightAgent): The `LightAgent` that made the decision.
            cars_per_lane (dict): The dictionary of car counts per lane considered
                                  by the optimizer.
            decision_lane (str): The ID of the lane chosen to be opened.
            model_step (int): The simulation step number when the decision was made.
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

    class DataCollector:
        """Helper class to aggregate data from multiple agents into a single DataFrame.

        Designed to collect data stored in attributes that are instances of `SimData`
        (or its subclasses) from a list of agents.
        """

        def __init__(self, agents: list[mesa.Agent], data_name: str) -> None:
            """Initializes the collector and aggregates data from the provided agents.

            Iterates through the `agents`, retrieves the `SimData` object specified
            by `data_name` from each, gets its underlying DataFrame using `.get_data()`,
            and vertically stacks (concatenates) them into a single DataFrame stored
            in the `self.data` attribute.

            Args:
                agents (list[mesa.Agent]): A list or AgentSet of agents to collect data from.
                data_name (str): The name of the attribute on each agent that holds the
                                 `SimData` object (e.g., 'wait_times', 'traffic').

            Raises:
                ValueError: If the `agents` list is empty.
                TypeError: If the specified `data_name` attribute on the agents is not
                           an instance of a `SimData` subclass.
                AttributeError: If the agents do not have an attribute named `data_name`.
            """
            if not agents:
                raise ValueError("AgentSet is empty")

            if not isinstance(getattr(agents[0], data_name), SimData):
                raise TypeError(f"Attribute {data_name} is not a valid data structure.")
            else:
                self.data_instance: SimData = getattr(agents[0], data_name)
                self.data = self.data_instance.get_data()

            for agent in agents[1:]:
                agent_data_instance: SimData = getattr(agent, data_name)
                agent_data = agent_data_instance.get_data()
                self.data.vstack(agent_data, in_place=True)

        def get_data(self) -> pl.DataFrame:
            """Returns the aggregated data collected from all agents.

            Returns:
                pl.DataFrame: A single Polars DataFrame containing the combined data
                              from the specified attribute of all agents provided
                              during initialization.
            """
            return self.data


@dataclass
class LightIntersectionMapping(SimData):
    """Stores a mapping between `LightAgent` unique IDs and their intersection node IDs.

    This dataclass inherits from `SimData` and maintains a Polars DataFrame
    that links each `LightAgent`'s automatically assigned unique ID to the
    string ID of the intersection node where it is located in the `Graph`.
    This mapping is useful for relating light-specific data to network locations.

    Attributes:
        data (pl.DataFrame): A Polars DataFrame with columns 'Light_ID' (Int16)
                             and 'Intersection' (String).
    """

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Initializes the Polars DataFrame with the mapping schema.

        Sets up the structure for storing the mapping between `LightAgent` IDs
        and their corresponding intersection node IDs.
        """
        self.data = pl.DataFrame(
            schema={"Light_ID": pl.Int16, "Intersection": pl.String},
            strict=False,
        )

    def update_data(self, light: LightAgent) -> None:
        """Adds a new mapping entry for a given `LightAgent`.

        Appends a new row to the `data` DataFrame containing the unique ID and
        position (intersection ID) of the provided `LightAgent`.

        Args:
            light (LightAgent): The `LightAgent` instance whose mapping is to be added.
        """
        self.data = self.data.extend(
            other=pl.DataFrame(
                data={"Light_ID": light.unique_id, "Intersection": light.position},
                schema={"Light_ID": pl.Int16, "Intersection": pl.String},
                strict=False,
            ),
        )

    def get_data(self) -> pl.DataFrame:
        """Returns the complete mapping data.

        Provides access to the DataFrame containing all recorded mappings between
        `LightAgent` IDs and intersection IDs.

        Returns:
            pl.DataFrame: The DataFrame holding the light-to-intersection mapping.
        """
        return self.data


@dataclass
class LightData(SimData):
    """Stores metadata associated with each `LightAgent` in the simulation.

    This dataclass inherits from `SimData` and uses a Polars DataFrame to record
    static properties of each traffic light, such as its network
    centrality and whether it serves as an entry point from a border node.

    Attributes:
        data (pl.DataFrame): A Polars DataFrame holding the light metadata.
                             Columns: 'Light_ID' (Int16), 'Centrality' (Float32),
                             'Is_Entrypoint' (Boolean).
    """

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Initializes the Polars DataFrame with the light metadata schema.

        Sets up the structure for storing metadata for each `LightAgent`, including
        its unique ID, calculated closeness centrality within the network graph,
        and a flag indicating if it's directly connected to a border node.
        """
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
        """Adds metadata for a newly created `LightAgent`.

        Calculates the closeness centrality and entrypoint status for the given
        `light` based on the provided `grid`. Appends a new row to the `data`
        DataFrame containing the light's unique ID and these calculated metadata values.

        Args:
            light (LightAgent): The `LightAgent` instance whose metadata is being added.
            grid (Graph): The `Graph` instance representing the road network, used
                          for calculating centrality and checking border connections.
        """
        self.data.vstack(
            other=pl.DataFrame(
                data=[
                    (
                        light.unique_id,
                        light.get_centrality(grid),
                        light.is_entrypoint(grid),
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

    def get_data(self) -> pl.DataFrame:
        """Returns the complete light metadata.

        Provides access to the DataFrame containing all recorded metadata for
        the `LightAgent`s.

        Returns:
            pl.DataFrame: The DataFrame holding the collected light metadata.
        """
        return self.data


@dataclass
class NumCars(SimData):
    """Tracks the total number of `CarAgent`s in the simulation over time.

    Inherits from `SimData` and uses a Polars DataFrame to record the count
    of active `CarAgent`s at each simulation step, along with the time within
    a 200-step cycle.

    Attributes:
        data (pl.DataFrame): A Polars DataFrame holding the car count history.
                             Columns: 'Time' (Int32, representing step within cycle),
                             'Num_Cars' (Int32).
    """

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Initializes the Polars DataFrame with the car count schema.

        Sets up the structure for storing the time (within a 200-step cycle)
        and the corresponding total number of cars present in the model.
        """
        self.data = pl.DataFrame(
            schema={"Time": pl.Int32, "Num_Cars": pl.Int32}, strict=False
        )

    def update_data(self, steps: int, n_cars: int) -> None:
        """Appends a new record of the total car count at the current step.

        Calculates the time within the 200-step cycle based on the provided `steps`.
        Adds a new row to the `data` DataFrame with this calculated time and the
        current total number of cars (`n_cars`).

        Args:
            steps (int): The current step counter from the `TrafficModel` instance.
            n_cars (int): The total number of active `CarAgent` instances in the
                          `TrafficModel` at the current step.
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

    def get_data(self) -> pl.DataFrame:
        """Returns the complete history of car counts over time.

        Provides access to the DataFrame containing all recorded car counts
        and their corresponding time steps within the cycle.

        Returns:
            pl.DataFrame: The DataFrame holding the time series of car counts.
        """
        return self.data


@dataclass
class Connections(SimData):
    """Stores details about the connections (edges) between intersection nodes.

    Inherits from `SimData` and uses a Polars DataFrame to maintain a list
    of all directed edges connecting intersection nodes in the `Graph`, along
    with their associated weights (distances).

    Attributes:
        data (pl.DataFrame): A Polars DataFrame holding the connection data.
                             Columns: 'Intersection_u' (String, source node),
                             'Intersection_v' (String, target node),
                             'Distance' (Int16, edge weight).
    """

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Initializes the Polars DataFrame with the connection schema.

        Sets up the structure for storing directed connections between
        intersection nodes, including the source node ID ('Intersection_u'),
        the target node ID ('Intersection_v'), and the distance ('Distance')
        between them.
        """
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
        """Adds a new connection record (edge) to the DataFrame.

        Appends a row representing a directed edge from `intersection_u` to
        `intersection_v` with the specified `distance` (weight).

        Args:
            intersection_u (str): The ID of the source intersection node.
            intersection_v (str): The ID of the target intersection node.
            distance (int): The weight (distance) of the edge connecting
                            `intersection_u` to `intersection_v`.
        """
        self.data.extend(
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
        )

    def get_data(self) -> pl.DataFrame:
        """Returns the complete list of recorded connections.

        Provides access to the DataFrame containing all stored intersection-to-intersection
        connections and their distances.

        Returns:
            pl.DataFrame: The DataFrame holding the connection data.
        """
        return self.data


@dataclass
class GlobalWaitTimes(SimData):
    """Aggregates wait time data from all `CarAgent` instances that reach their destination.

    Inherits from `SimData` and uses a Polars DataFrame to store the final
    wait time records collected by individual `CarAgent`s upon their arrival.
    This provides a global overview of waiting times across the entire simulation.

    Attributes:
        data (pl.DataFrame): A Polars DataFrame holding the aggregated wait times.
                             Columns: 'Car_ID' (Int32), 'Light_ID' (Int16),
                             'Wait_Time' (Int16).
    """

    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __post_init__(self):
        """Initializes the Polars DataFrame with the global wait time schema.

        Sets up the structure for storing aggregated wait time data, including
        the ID of the car, the ID of the light where waiting occurred, and the
        total time spent waiting at that light by that car during its journey.
        """
        self.data = pl.DataFrame(
            schema={
                "Car_ID": pl.Int32,
                "Light_ID": pl.Int16,
                "Wait_Time": pl.Int16,
            },
            strict=False,
        )

    def update_data(self, wait_times: pl.DataFrame):
        """Appends the wait time data from a completed `CarAgent` journey.

        Takes the final wait time DataFrame collected by a `CarAgent` (typically
        when it raises `AgentArrived`) and vertically stacks (appends) it to the
        global `data` DataFrame.

        Args:
            wait_times (pl.DataFrame): The DataFrame containing the wait time records
                                       from a single `CarAgent` that has finished its trip.
        """
        self.data = self.data.vstack(other=wait_times)

    def get_data(self) -> pl.DataFrame:
        """Returns the aggregated wait time data for all completed car journeys.

        Provides access to the DataFrame containing the combined wait time records
        from all `CarAgent`s that have reached their destination so far.

        Returns:
            pl.DataFrame: The DataFrame holding the globally aggregated wait times.
        """
        return self.data
