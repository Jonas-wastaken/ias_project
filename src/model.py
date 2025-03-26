"""This module contains:
- TrafficModel class: A Mesa model simulating traffic."""

from pathlib import Path
import random
import datetime
import mesa
import numpy as np
import polars as pl
from car import CarAgent, AgentArrived
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
        **def save_sim_data(self) -> Path**:
            Function to save the sim_data to a parquet file
        **get_cars_per_lane_of_light(self, light_position: str) -> dict**:
            Function to get the number of cars per lane of a light.
    """

    def __init__(
        self, num_cars: int, sim_mode: bool = False, seed: int = None, **kwargs
    ):
        """Initializes a new traffic environment.

        - Spawns a graph representing the grid
        - Initializes TrafficLight Agents at each intersection node
        - Initializes the specified number of CarAgents

        Args:
            num_cars (int): Number of car agents to spawn.
            seed (int, optional): Seed used in model generation. Defaults to None.
            **kwargs: Additional keyword arguments for configuring the graph object.
                - num_intersections (int): Number of intersections in the graph. Defaults to 50.
                - num_borders (int): Number of border nodes in the graph. Defaults to 25.
                - min_distance (int): Minimum distance between nodes. Defaults to 10.
                - max_distance (int): Maximum distance between nodes. Defaults to 20.
        """
        super().__init__(seed=seed)

        self.grid = Graph(
            num_intersections=kwargs.get("num_intersections", 15),
            num_borders=kwargs.get("num_borders", 5),
            min_distance=kwargs.get("min_distance", 10),
            max_distance=kwargs.get("max_distance", 20),
        )

        self.create_lights_for_intersections()
        # CarAgent.create_agents(model=self, n=num_cars)
        self.car_paths = {}
        self.update_car_paths()
        self.cars_waiting_times = {}
        self.update_cars_waiting_times()
        self.lights_decision_log = {}
        self.num_cars_hist = np.array([])
        self.create_cars(num_cars)

        self.arrivals_data = pl.DataFrame(
            schema={
                "Index": pl.Int32,
                "Light_ID": pl.Int16,
                "Time": pl.Int16,
                "Arrivals": pl.Int16,
            },
            strict=False,
        )

        self.traffic_data = pl.DataFrame(
            schema={
                "Index": pl.Int32,
                "Light_ID": pl.Int16,
                "Time": pl.Int16,
                "Num_Cars": pl.Int16,
            },
            strict=False,
        )

    def step(self) -> None:
        """Advances the environment to next state.

        - Calls Agents step functions
        - Updates simulation data
        - CarAgents are respawned based on current time and number of cars in the model
        """
        self._car_step()
        self._light_step()

        self.num_cars_hist = np.append(
            self.num_cars_hist, len(self.get_agents_by_type("CarAgent"))
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
                light.change_open_lane(light.optimize_open_lane())
                # light.rotate_in_open_lane_cycle()
            else:
                light.current_switching_cooldown -= 1

            self.arrivals_data = self.update_arrivals_data(
                arrivals_data=self.arrivals_data, light=light
            )
            self.traffic_data = self.update_traffic_data(
                traffic_data=self.traffic_data, light=light
            )

    def create_cars(self, num_cars: int) -> None:
        """Function to add cars to the model.

        - Updates car_paths and cars_waiting_times attributes

        Args:
            num_cars (int): Number of cars to add.
        """
        CarAgent.create_agents(model=self, n=num_cars)
        self.update_car_paths()
        self.update_cars_waiting_times()

    def remove_random_cars(self, num_cars: int) -> None:
        """Function to randomly remove n cars from the model.

        Args:
            num_cars (int): Number of cars to remove.
        """
        for _ in range(num_cars):
            car: CarAgent = random.choice(self.get_agents_by_type("CarAgent"))
            self.agents.remove(car)

    # def create_lights(self, num_lights: int, position: str) -> None: TODO: Unused? @mxrio
    #     """Function to add traffic lights to the model.

    #     Args:
    #         num_agents (int): Number of agents to add.
    #     """
    #     LightAgent.create_agents(model=self, n=num_lights, position=position)

    def create_lights_for_intersections(self) -> None:
        """Function to add traffic lights to the model.

        - Assigns one LightAgent to each intersection node
        """
        for intersection in self.grid.get_nodes("intersection"):
            LightAgent.create_agents(model=self, n=1, position=intersection)

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

    def update_cars_waiting_times(self) -> None:
        """Function to update the waiting times of all cars at each intersection."""

        for car in self.get_agents_by_type("CarAgent"):
            car: CarAgent
            if car.unique_id not in list(self.cars_waiting_times.keys()):
                self.cars_waiting_times[car.unique_id] = {
                    intersection: 0
                    for intersection in list(
                        car.model.car_paths[car.unique_id].keys()  # s.o. @mxrio
                    )
                    if intersection.startswith(
                        "intersection"
                    )  # grid.get_nodes verwenden vielleicht? @mxrio
                }

            if car.waiting:
                self.cars_waiting_times[car.unique_id][car.position] += 1

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
        next_sine_value = (sine_value + 1) / 2 * 2 * self.num_cars_hist[0]
        diff = next_sine_value - self.num_cars_hist[-1]
        diff_variance = diff * random.uniform(0.8, 1.2)
        cars_to_add = int(diff_variance)

        if cars_to_add > 0:
            self.create_cars(cars_to_add)

    def get_cars_per_lane_of_light(self, light_position: str) -> dict:
        """Function to get the number of cars per lane of a light.

        Args:
            light_position (str): The position of the light.

        Returns:
            dict: A dictionary containing the number of cars per lane of the light.
        """
        cars_per_lane = {
            lane: 0
            for lane in self.grid.neighbors(light_position)
            if lane.startswith("intersection")
        }

        for car in self.get_agents_by_type("CarAgent"):
            if car.position == light_position and car.waiting:
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

    def save_sim_data(self) -> Path:
        """Function to save the model data to a parquet file

        Saves:
            - sim_data
            - light_data
            - num_cars_hist

        Returns:
            Path: Path object representing the folder the data is stored in.
        """
        data_path = Path.joinpath(Path.cwd(), "data")
        folder = Path(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        Path.joinpath(data_path, folder).mkdir()

        self.arrivals_data.write_parquet(
            file=Path.joinpath(data_path, folder, "arrivals_data.parquet")
        )

        self.traffic_data.write_parquet(
            file=Path.joinpath(data_path, folder, "traffic_data.parquet")
        )

        self.get_light_data().write_parquet(
            file=Path.joinpath(data_path, folder, "light_data.parquet")
        )

        self.get_connections_data().write_parquet(
            file=Path.joinpath(data_path, folder, "connections.parquet")
        )

        pl.DataFrame(
            data=self.num_cars_hist, schema={"Num_Cars": pl.Int16}
        ).write_parquet(file=Path.joinpath(data_path, folder, "num_cars.parquet"))

        return folder

    def get_light_data(self) -> pl.DataFrame:
        """Gets metadata of LightAgents in the model.

        Returns:
            pl.DataFrame: DataFrame containing metadata about LightAgents
        """
        light_data = pl.DataFrame(
            schema={
                "Light_ID": pl.Int16,
                "Centrality": pl.Float32,
                "Is_Entrypoint": pl.Boolean,
            },
            strict=False,
            orient="row",
        )

        for light in self.get_agents_by_type("LightAgent"):
            light: LightAgent
            light_data.vstack(
                other=pl.DataFrame(
                    data=[
                        (
                            light.unique_id,
                            light.get_centrality(self.grid),
                            light.check_is_entrypoint(self.grid),
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

        return light_data

    def update_arrivals_data(
        self, arrivals_data: pl.DataFrame, light: LightAgent
    ) -> pl.DataFrame:
        """Updates the arrivals data

        Args:
            arrivals_data (pl.DataFrame): Current DataFrame
            light (LightAgent): LightAgent to get and write data from

        Returns:
            pl.DataFrame: Updated DataFrame
        """
        arrivals_data.vstack(
            pl.DataFrame(
                data={
                    "Index": self.steps,
                    "Light_ID": light.unique_id,
                    "Time": 200 - (self.steps % 200),
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

        return arrivals_data

    def get_connections_data(self) -> pl.DataFrame:
        """Constructs a DataFrame describing the connections between LightAgents.

        - Connection is described as Light_u -> Light_v
        - Includes the distance between the nodes the lights are on

        Returns:
            pl.DataFrame: DataFrame containing the data
        """
        connections_data = pl.DataFrame(
            schema={
                "Light_u": pl.Int16,
                "Light_v": pl.Int16,
                "Distance": pl.Int16,
            },
            strict=False,
        )
        for light_u in self.get_agents_by_type("LightAgent"):
            light_u: LightAgent

            for intersection in light_u.get_connected_intersections(self.grid):
                for light_v in self.get_agents_by_type("LightAgent"):
                    light_v: LightAgent

                    if light_v.position == intersection:
                        connections_data.vstack(
                            other=pl.DataFrame(
                                data={
                                    "Light_u": light_u.unique_id,
                                    "Light_v": light_v.unique_id,
                                    "Distance": self.grid.get_edge_data(
                                        u=light_u.position, v=light_v.position
                                    )["weight"],
                                },
                                schema={
                                    "Light_u": pl.Int16,
                                    "Light_v": pl.Int16,
                                    "Distance": pl.Int16,
                                },
                                strict=False,
                            ),
                            in_place=True,
                        )

        return connections_data

    def update_traffic_data(
        self, traffic_data: pl.DataFrame, light: LightAgent
    ) -> pl.DataFrame:
        """Updates the traffic data DataFrame.

        Args:
            traffic_data (pl.DataFrame): Current DataFrame
            light (LightAgent): LightAgent to get and write data from

        Returns:
            pl.DataFrame: Updated DataFrame
        """
        traffic_data.vstack(
            other=pl.DataFrame(
                data={
                    "Index": self.steps,
                    "Light_ID": light.unique_id,
                    "Time": 200 - (self.steps % 200),
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

        return traffic_data
