"""
This Streamlit app visualizes a traffic grid using NetworkX and Plotly.
It allows users to step through the simulation of traffic agents and view their paths.
This module contains:
- App class: Represents the Streamlit application for visualizing a traffic grid.
- GraphContainer class: Holds the visualization of the graph.
- CarPathListContainer class: Container for a horizontally scrollable list of car agent information cards.
- SettingsContainer class: Container for the settings form and reset button.
- EdgesContainer class: Container for displaying the edges of the graph.
"""

import sys
import json
from pathlib import Path
import time
from dataclasses import dataclass
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from model import TrafficModel
from graph_viz import TrafficGraph
from car import CarAgent
from light import LightAgent


class App:
    """A class to represent the Streamlit application for visualizing a traffic grid.

    Attributes:
        screen_height (int): Height of the screen in pixels
        screen_width (int): Width of the screen in pixels

    ## Methods:
        **step(self) -> None**:
            Advances the environment by one step.
    """

    def __init__(self, model: TrafficModel):
        """Loads UI elements into the app

        - Checks if there is an active auto run loop
        """

        self.screen_height = streamlit_js_eval(
            js_expressions="screen.height", key="SCRH"
        )
        self.screen_width = streamlit_js_eval(js_expressions="screen.width", key="SCRW")

        timeout = 0
        while self.screen_height is None or self.screen_width is None and timeout < 30:
            time.sleep(0.1)
            self.screen_height = st.session_state.get("SCRH")
            self.screen_width = st.session_state.get("SCRW")
            timeout += 1

        if self.screen_height is None:
            self.screen_height = 1080

        if self.screen_width is None:
            self.screen_width = 1920

        fig = TrafficGraph(
            model=model,
            height=self.screen_height * 0.65,
            width=self.screen_width * 0.75,
        )

        outer_cols = st.columns([0.75, 0.25], vertical_alignment="top")
        with outer_cols[0]:
            if "graph_container" not in st.session_state:
                st.session_state["graph_container"] = st.empty()

            st.session_state["graph_container"].plotly_chart(
                fig, use_container_width=False, key=f"traffic_plot_{time.time()}"
            )

        with outer_cols[1]:
            inner_cols = st.columns([0.55, 0.25, 0.2])
            with inner_cols[0]:
                with st.popover(
                    label="Settings",
                    help="Adjust the Settings of the Model",
                    use_container_width=True,
                ):
                    SettingsContainer()
            with inner_cols[1]:
                st.session_state["auto_run_steps"] = st.number_input(
                    label="Auto Run Steps",
                    min_value=1,
                    value=st.session_state["auto_run_steps"],
                    label_visibility="collapsed",
                )
            with inner_cols[2]:
                if st.button(
                    label="Run", help="Execute one step", use_container_width=True
                ):
                    st.query_params["run_steps"] = st.session_state["auto_run_steps"]
                    self.step(fig)
            if st.session_state["env_config"]["num_cars"] > 0:
                CarPathContainer()

    def step(self, fig: TrafficGraph) -> None:
        """Advances the environment by one step."""
        while int(st.query_params["run_steps"]) > 0:
            time.sleep(0.1)
            model.step()
            fig.refresh(
                height=self.screen_height * 0.65, width=self.screen_width * 0.75
            )
            st.session_state["graph_container"].plotly_chart(
                fig, use_container_width=False, key=f"traffic_plot_{time.time()}"
            )
            st.query_params["run_steps"] = int(st.query_params["run_steps"]) - 1


class CarPathContainer:
    """Container for a horizontally scrollable list of car agent information cards.

    This class provides functionality to display a list of car agents in a horizontally scrollable manner. It includes methods to scroll the list left and right, render the current path of each car, and display detailed information about each car's path.

    ## Methods:
        **scroll_left(self) -> None**:
            Scrolls the car path list to the left.
        **scroll_right(self) -> None**:
            Scrolls the car path list to the right.
        **render_current_path(self, car: CarAgent) -> None**:
            Renders the current path of a given car agent.
        **render_full_path(self, car: CarAgent) -> None:**
            Renders the full path of a given car agent.
        **get_full_path(self, car_id: int) -> list[tuple[str, int]]**:
            Returns the full path of a car agent as a list of tuples containing node id and distance to the next step.
        **get_current_path(self, car: CarAgent) -> dict**:
            Returns the current path information of a car agent as a dictionary.
    """

    @st.fragment
    def __init__(self):
        """Creates two horizontally scrollable lists of car agent and light agent information cards."""

        car: CarAgent = st.session_state["model"].get_agents_by_type("CarAgent")[
            int(st.query_params["scroll_index_cars"])
        ]

        light: CarAgent = st.session_state["model"].get_agents_by_type("LightAgent")[
            int(st.query_params["scroll_index_lights"])
        ]

        self.render_current_path(car)
        self.render_lights_info(light)

    def scroll_left(self, query_index: str) -> None:
        """Scrolls the car path list to the left.

        - Checks first if there are more items in the list to render
        - Decreases the scroll_index by 1
        """
        if int(st.query_params[query_index]) > 0:
            st.query_params[query_index] = int(st.query_params[query_index]) - 1
            st.rerun(scope="fragment")

    def scroll_right(self, query_index: str) -> None:
        """Scrolls the car path list to the right.

        - Checks first if there are more items in the list to render
        - Increases the scroll_index by 1
        """

        if query_index == "scroll_index_lights":
            if (
                int(st.query_params[query_index])
                < st.session_state["env_config"]["num_intersections"] - 1
            ):
                st.query_params[query_index] = int(st.query_params[query_index]) + 1
                st.rerun(scope="fragment")
        if query_index == "scroll_index_cars":
            if (
                int(st.query_params[query_index])
                < st.session_state["env_config"]["num_cars"] - 1
            ):
                st.query_params[query_index] = int(st.query_params[query_index]) + 1
                st.rerun(scope="fragment")

    def render_current_path(self, car: CarAgent) -> None:
        """Renders a car's path.

        Args:
            car (CarAgent): CarAgent instance
        """
        cols = st.columns([0.3, 0.1, 0.4, 0.1, 0.1], vertical_alignment="center")
        with cols[0]:
            st.subheader(
                f"Car {car.unique_id - st.session_state['env_config']['num_intersections']}"
            )
        # with cols[1]:
        #     num_cars = self.NumCars(st.session_state["model"]).num_cars
        #     st.metric(label="Cars", value=num_cars, label_visibility="hidden")
        # with cols[2]:
        # self.render_full_path(car)
        with cols[3]:
            if st.button("<-", key="cars_left"):
                self.scroll_left("scroll_index_cars")
        with cols[4]:
            if st.button("->", key="cars_right"):
                self.scroll_right("scroll_index_cars")
        st.dataframe(
            self.get_current_path(car),
            use_container_width=True,
            column_config={"value": st.column_config.TextColumn("")},
        )

    def render_full_path(self, car: CarAgent) -> None:
        """Renders the full path of a car agent.

        Args:
            car (CarAgent): CarAgent instance
        """
        with st.popover("Show Full Path"):
            st.markdown("""##### Full Path""")
            st.dataframe(
                self.get_full_path(car_id=car.unique_id),
                use_container_width=True,
                hide_index=True,
                column_config={"0": "", "1": ""},
            )

    def render_lights_info(self, light: LightAgent) -> None:
        """Render the lights info box.

        Args:
            light (LightAgent): LightAgent instance
        """
        cols = st.columns([0.3, 0.1, 0.4, 0.1, 0.1], vertical_alignment="center")
        with cols[0]:
            st.subheader(f"Light {light.unique_id}")
        with cols[3]:
            if st.button("<-", key="lights_left"):
                self.scroll_left("scroll_index_lights")
        with cols[4]:
            if st.button("->", key="lights_right"):
                self.scroll_right("scroll_index_lights")
        st.dataframe(
            self.get_lights_info(light),
            use_container_width=True,
            column_config={"value": st.column_config.TextColumn("")},
        )

    # def get_full_path(self, car_id: int) -> str:
    #     """Creates a List with the full path a car agent takes.

    #     Args:
    #         car_id (int): ID of CarAgent instance

    #     Returns:
    #         list[tuple[str, int]]: List of tuples with node id and distance to next step.
    #     """
    #     path = ""
    #     for hop in st.session_state["model"].car_paths[car_id].items():
    #         path += " "
    #         path += str(hop[0]).title().replace("_", " ")
    #         path += ": "
    #         path += str(hop[1])
    #         path += " ->"
    #         path += "\n"

    #     return path

    def get_current_path(self, car: CarAgent) -> dict:
        """Creates a dict with the current (last) position, next position, distance to next position, waiting status and waiting time of a car agent.

        Args:
            car (CarAgent): CarAgent instance

        Returns:
            dict: Dict holding the path information
        """
        current_position = car.position
        try:
            next_position: str = list(car.path.keys())[1]
        except IndexError:
            next_position = None
        distance = car.path[car.position] if next_position else None
        is_waiting = car.waiting

        path_dict = {
            "Current Position": str(current_position).title().replace("_", " "),
            "Next Position": str(next_position).title().replace("_", " "),
            "Distance": str(distance),
            "Waiting": str(is_waiting),
        }

        return path_dict

    def get_lights_info(self, light: LightAgent):
        """Creates a dict with the position, all lanes, current open lane, switching cooldown and the waiting cars from each lane.

        Args:
            light (LightAgent): LightAgent instance

        Returns:
            dict: Dict holding the status information
        """

        current_position = light.position
        all_lanes = light.neighbor_lights
        open_lane = light.open_lane
        switching_cooldown = light.current_switching_cooldown
        waiting_cars = light.model.get_cars_per_lane_of_light(light.position, 0)

        lights_info = {
            "Position": str(current_position).title().replace("_", " "),
            "Open Lane": str(open_lane),
            "Switching Cooldown": str(switching_cooldown),
        }

        for lane in all_lanes:
            lights_info[f"Waiting Cars {lane}"] = str(waiting_cars[lane])

        return lights_info

    # @dataclass
    # class NumCars:
    #     """Class to hold the current number of cars in the grid"""

    #     num_cars: int

    #     def __init__(self, model: TrafficModel):
    #         """Gets the current number of cars in the grid

    #         Args:
    #             model (TrafficModel): Model
    #         """
    #         self.num_cars = len(model.get_agents_by_type("CarAgent"))


class SettingsContainer:
    """Container for the settings form and reset button.

    This class provides functionality to display a settings form where users can configure the environment parameters.

    ## Methods:
        **render_settings_form(self)-> None**:
            Renders the settings form with input fields for various environment parameters.
        **update_env_config(self, num_cars, num_intersections, num_borders, distance_range, run_steps) -> None**:
            Applies changes to the environment based on user input from the settings form.
        **reset_environment(self) -> None**:
            Resets the environment with user-specified config options.
    """

    def __init__(self):
        """Renders the settings form with input fields for environment parameters.

        - Number of Cars
        - Number of Intersections
        - Number of Borders
        - Distance Range
        - Run Steps
        - Optimization Type
        """
        with st.form(key="Settings"):
            num_cars = self.NumCarsInput().num_cars
            num_intersections = self.NumIntersectionsInput().num_intersections
            num_borders = self.NumBordersInput().num_borders
            distance_range = self.DistanceRangeSlider().distance_range
            optimization_type = self.OptimizationTypeInput().optimization_type

            if st.form_submit_button("Submit"):
                self.update_env_config(
                    num_cars,
                    num_intersections,
                    num_borders,
                    distance_range,
                    optimization_type,
                )

    @dataclass
    class NumCarsInput:
        """Dataclass for the number of cars input field.

        ## Attributes:
            **num_cars (int)**: Number of cars input by the user.
        """

        num_cars: int

        def __init__(self):
            """Initializes the number of cars input field with a default value from session state."""
            self.num_cars = st.number_input(
                label="Number of Cars",
                min_value=0,
                value=st.session_state["model"].num_cars,
                key="num_cars",
            )

    @dataclass
    class NumIntersectionsInput:
        """Dataclass for the number of intersections input field.

        ## Attributes:
            **num_intersections (int)**: Number of intersections input by the user.
        """

        num_intersections: int

        def __init__(self):
            """Initializes the number of intersections input field with a default value from session state."""
            self.num_intersections = st.number_input(
                label="Number of Intersections",
                min_value=1,
                value=st.session_state["env_config"]["num_intersections"],
            )

    @dataclass
    class NumBordersInput:
        """Dataclass for the number of borders input field.

        ## Attributes:
            **num_borders (int)**: Number of borders input by the user.
        """

        num_borders: int

        def __init__(self):
            """Initializes the number of borders input field with a default value from session state."""
            self.num_borders = st.number_input(
                label="Number of Borders",
                min_value=2,
                value=st.session_state["env_config"]["num_borders"],
            )

    @dataclass
    class DistanceRangeSlider:
        """Dataclass for the distance range slider.

        ## Attributes:
            **distance_range (tuple[int, int])**: Distance range input by the user.
        """

        distance_range: tuple[int, int]

        def __init__(self):
            """Initializes the distance range slider with default values from session state."""
            self.distance_range = st.slider(
                label="Distance",
                min_value=2,
                max_value=100,
                value=(
                    st.session_state["env_config"]["min_distance"],
                    st.session_state["env_config"]["max_distance"],
                ),
            )

    @dataclass
    class OptimizationTypeInput:
        """Dataclass for the optimization type input field.

        ## Attributes:
            **optimization_type (str)**: Optimization type input by the user.
        """

        optimization_type: str

        def __init__(self):
            """Initializes the optimization type input field with a default value from session state."""
            self.optimization_type = st.selectbox(
                label="Optimization Type",
                options=["none", "simple", "advanced", "advanced_ml"],
                index=3,
                label_visibility="collapsed",
            )

    def update_env_config(
        self,
        num_cars: int,
        num_intersections: int,
        num_borders: int,
        distance_range: tuple[int, int],
        optimization_type: str,
    ):
        del st.session_state["model"]
        st.session_state["model"] = TrafficModel(
            num_cars=num_cars,
            num_intersections=num_intersections,
            num_borders=num_borders,
            min_distance=distance_range[0],
            max_distance=distance_range[1],
            optimization_type=optimization_type,
        )
        st.query_params["run_steps"] = 0
        st.query_params["scroll_index_cars"] = 0
        st.query_params["scroll_index_lights"] = 0
        st.rerun()


class EdgesContainer:
    """Container for displaying the edges of the graph.

    This class provides functionality to create and display a DataFrame containing information about each edge in the graph.

    ## Methods:
        **create_edges_df(self, model) -> pd.DataFrame**:
            Creates a DataFrame containing information about each edge in the graph.
    """

    def __init__(self, model: TrafficModel):
        """Initializes the EdgesContainer and displays the edges DataFrame.

        Args:
            model (TrafficModel): The traffic model containing the graph.
        """
        st.dataframe(self.create_edges_df(model), hide_index=True)

    def create_edges_df(self, model: TrafficModel) -> pd.DataFrame:
        """Creates a DataFrame containing information about each edge in the graph.

        Args:
            model (TrafficModel): The traffic model containing the graph.

        Returns:
            pd.DataFrame: DataFrame containing information about each edge in the graph.
        """
        edge_df = pd.DataFrame(
            [(u, v, w) for u, v, w in model.grid.edges(data="weight")],
            columns=["U", "V", "Distance"],
        )

        return edge_df


if __name__ == "__main__":
    st.set_page_config(
        page_title="Test",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    if "scroll_index_cars" not in st.query_params:
        st.query_params["scroll_index_cars"] = 0
    if "scroll_index_lights" not in st.query_params:
        st.query_params["scroll_index_lights"] = 0
    if "run_steps" not in st.query_params:
        st.query_params["run_steps"] = 0

    if "model" not in st.session_state:
        st.session_state["model"] = TrafficModel(
            num_cars=15,
            num_intersections=15,
            num_borders=30,
            optimization_type="advanced_ml",
        )
    model: TrafficModel = st.session_state["model"]

    if "auto_run_steps" not in st.session_state:
        st.session_state["auto_run_steps"] = 1

    st.session_state["env_config"] = {
        "num_intersections": len(model.grid.get_nodes("intersection")),
        "num_borders": len(model.grid.get_nodes("border")),
        "min_distance": model.grid.min_distance,
        "max_distance": model.grid.max_distance,
        "num_cars": len(model.get_agents_by_type("CarAgent")),
    }

    App(model)
