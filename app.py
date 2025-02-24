"""
This Streamlit app visualizes a traffic grid using NetworkX and Plotly.
It allows users to step through the simulation of traffic agents and view their paths.
"""

import sys
import os
import time
from dataclasses import dataclass
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from model import TrafficModel
from graph_viz import TrafficGraph
from car import CarAgent


class GraphContainer:
    """Class to hold the visualization of the graph."""

    def __init__(self, model: TrafficModel):
        """Create plot of the model's graph.

        Args:
            model (TrafficModel): Model to plot
        """
        fig = TrafficGraph(model)
        st.plotly_chart(fig, use_container_width=True)


class CarPathListContainer:
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

    def __init__(self):
        """Creates a horizontally scrollable list of car agent information cards.

        - Creates layout with 5 columns
        - Renders navigation buttons in the left- and rightmost column
            - If there are items to the left or right of the list
        - Gets currently visible cars
        - Renders information of visible cars
        """
        cols = st.columns(spec=[0.05, 0.3, 0.3, 0.3, 0.05], vertical_alignment="center")

        if st.session_state["env_config"]["num_cars"] > 3:
            with cols[0]:
                if st.button("<-"):
                    self.scroll_left()
            with cols[-1]:
                if st.button("->"):
                    self.scroll_right()

        visible_cars = st.session_state["model"].get_agents_by_type("CarAgent")[
            int(st.query_params["scroll_index"]) : (
                int(st.query_params["scroll_index"]) + 3
            )
        ]

        for car, i in zip(
            visible_cars,
            range(3),
        ):
            with cols[i + 1]:
                self.render_current_path(car)

    def scroll_left(self) -> None:
        """Scrolls the car path list to the left.

        - Checks first if there are more items in the list to render
        - Decreases the scroll_index by 1
        """
        if int(st.query_params["scroll_index"]) > 0:
            st.query_params["scroll_index"] = int(st.query_params["scroll_index"]) - 1

    def scroll_right(self) -> None:
        """Scrolls the car path list to the right.

        - Checks first if there are more items in the list to render
        - Increases the scroll_index by 1
        """
        if (
            int(st.query_params["scroll_index"])
            < st.session_state["env_config"]["num_cars"] - 3
        ):
            st.query_params["scroll_index"] = int(st.query_params["scroll_index"]) + 1

    def render_current_path(self, car: CarAgent) -> None:
        """Renders a car's path.

        Args:
            car (CarAgent): CarAgent instance
        """
        cols = st.columns(2)
        with cols[0]:
            st.subheader(f"Agent {car.unique_id}")
        with cols[1]:
            self.render_full_path(car)
        st.dataframe(
            self.get_current_path(car),
            use_container_width=True,
            column_config={"value": ""},
        )

    def render_full_path(self, car: CarAgent) -> None:
        """Renders the full path of a car agent.

        Args:
            car (CarAgent): CarAgent instance
        """
        with stylable_container(
            key=f"agent_{car.unique_id}",
            css_styles="""/*css*/
                button {
                    background-color: white;
                    color: black;
                    border: none;
                    white-space: nowrap;
                    margin-top: 0.25rem;
                }
                """,
        ):
            with st.popover("Show Details"):
                st.markdown("""##### Full Path""")
                st.dataframe(
                    self.get_full_path(car_id=car.unique_id),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"0": "", "1": ""},
                )

    def get_full_path(self, car_id: int) -> list[tuple[str, int]]:
        """Creates a List with the full path a car agent takes.

        Args:
            car_id (int): ID of CarAgent instance

        Returns:
            list[tuple[str, int]]: List of tuples with node id and distance to next step.
        """
        path = [
            (node.title().replace("_", " "), distance)
            for node, distance in st.session_state["model"].agent_paths[car_id].items()
        ]

        return path

    def get_current_path(self, car: CarAgent) -> dict:
        """Creates a dict with the current (last) position, next position, distance to next position, waiting status and waiting time of a car agent.

        Args:
            car (CarAgent): CarAgent instance

        Returns:
            dict: Dict holding the path information
        """
        current_position = car.position
        try:
            next_position = list(car.path.keys())[1]
        except IndexError:
            next_position = None
        distance = (
            st.session_state["model"].grid.get_edge_data(
                current_position, next_position
            )["weight"]
            if next_position
            else None
        )
        is_waiting = car.waiting
        global_waiting_time = car.global_waiting_time

        path_dict = {
            "Current Position": current_position.title().replace("_", " "),
            "Next Position": str(next_position).title().replace("_", " "),
            "Distance": distance,
            "Waiting": is_waiting,
            "Waiting Time": global_waiting_time,
        }

        return path_dict


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
        """Initializes the settings container with a reset button and renders the settings form.

        - Creates two columns for layout
        - Adds a subheader for settings
        - Adds a reset button to reset the environment
        - Renders the settings form
        """
        cols = st.columns(2, gap="large", vertical_alignment="center")
        with cols[0]:
            st.subheader("Settings", anchor="left")
        with cols[1]:
            if st.button(
                label="Reset",
                help="Reset the Environment",
                use_container_width=False,
            ):
                # self.reset_environment() TODO: Fix
                pass

        self.render_settings_form()

    def render_settings_form(self) -> None:
        """Renders the settings form with input fields for environment parameters.

        - Number of Cars
        - Number of Intersections
        - Number of Borders
        - Distance Range
        - Run Steps
        """
        with st.form("Settings"):
            num_cars = self.NumCarsInput().num_cars
            num_intersections = self.NumIntersectionsInput().num_intersections
            num_borders = self.NumBordersInput().num_borders
            distance_range = self.DistanceRangeSlider().distance_range
            run_steps = self.RunStepsInput().run_steps
            if st.form_submit_button("Submit"):
                self.update_env_config(
                    num_cars, num_intersections, num_borders, distance_range, run_steps
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
                value=st.session_state["env_config"]["num_cars"],
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
    class RunStepsInput:
        """Dataclass for the run steps input field.

        ## Attributes:
            **run_steps (int)**: Number of run steps input by the user.
        """

        run_steps: int

        def __init__(self):
            """Initializes the run steps input field with a default value from session state."""
            self.run_steps = st.number_input(
                label="Run Steps",
                min_value=1,
                value=st.session_state["env_config"]["auto_run_steps"],
            )

    def update_env_config(
        self, num_cars, num_intersections, num_borders, distance_range, run_steps
    ) -> None:  # TODO: make class
        """Applies changes to the environment based on user input from the settings form.

        Args:
            num_cars (int): Number of cars
            num_intersections (int): Number of intersections
            num_borders (int): Number of borders
            distance_range (tuple[int, int]): Distance range
            run_steps (int): Number of run steps
        """
        # Update the model with new settings for number of agents
        if num_cars > st.session_state["env_config"]["num_cars"]:
            st.session_state["model"].create_agents(
                num_cars - st.session_state["env_config"]["num_cars"]
            )
        elif num_cars < st.session_state["env_config"]["num_cars"]:
            st.session_state["model"].remove_agents(
                st.session_state["env_config"]["num_cars"] - num_cars
            )

        # Update the model with new settings for number of intersections
        if num_intersections > st.session_state["env_config"]["num_intersections"]:
            st.session_state["model"].grid.add_intersections(
                num_intersections - st.session_state["env_config"]["num_intersections"]
            )
        elif num_intersections < st.session_state["env_config"]["num_intersections"]:
            st.session_state["model"].grid.remove_intersections(
                st.session_state["env_config"]["num_intersections"] - num_intersections
            )

        # Update the model with new settings for number of borders
        if num_borders > st.session_state["env_config"]["num_borders"]:
            st.session_state["model"].grid.add_borders(
                num_borders - st.session_state["env_config"]["num_borders"]
            )
        elif num_borders < st.session_state["env_config"]["num_borders"]:
            self.model.grid.remove_borders(
                st.session_state["env_config"]["num_borders"] - num_borders
            )

        # Update the model with new settings for distance range
        if distance_range != (
            st.session_state["env_config"]["min_distance"],
            st.session_state["env_config"]["max_distance"],
        ):
            st.session_state["model"].grid.change_weights(
                min_distance=distance_range[0],
                max_distance=distance_range[1],
            )

        # Update the model with new settings for auto_run_steps
        if run_steps != st.session_state["env_config"]["auto_run_steps"]:
            st.session_state["env_config"]["auto_run_steps"] = run_steps

        # Update the paths for each agent or delete agents if they are not on the grid
        for agent in st.session_state["model"].get_agents_by_type("CarAgent")[:]:
            if (
                agent.position not in st.session_state["model"].grid.nodes
                or agent.goal not in st.session_state["model"].grid.nodes
            ):
                st.session_state["model"].agents.remove(agent)
                continue
            agent.path = agent.compute_path()
            st.session_state["model"].agent_paths[agent.unique_id] = agent.path.copy()

        st.rerun()

    def reset_environment(self) -> None:
        """Resets the environment with user-specified config options."""
        st.session_state["model"] = TrafficModel(
            num_agents=self.num_cars,
            num_intersections=self.num_intersections,
            num_borders=self.num_borders,
            min_distance=self.distance_range[0],
            max_distance=self.distance_range[1],
        )
        st.rerun()


class ConnectionsContainer:
    """Container for displaying the connections of each node in the graph.

    This class provides functionality to create and display a DataFrame containing the connections for each node in the graph.

    ## Methods:
        **create_connections_df(self) -> pd.DataFrame**:
            Creates a DataFrame containing connections for each node in the graph.
    """

    def __init__(self):
        """Initializes the ConnectionsContainer and displays the connections DataFrame."""
        st.dataframe(
            self.create_connections_df(),
            hide_index=True,
        )

    def create_connections_df(self) -> pd.DataFrame:
        """Creates a DataFrame containing connections for each node in the graph.

        Returns:
            pd.DataFrame: DataFrame containing connections for each node in the graph.
        """
        connections_df = pd.DataFrame(
            [
                (
                    node,
                    st.session_state["model"]
                    .grid.get_connections(filter_by=node)
                    .values(),
                )
                for node in st.session_state["model"].grid.get_nodes()
            ],
            columns=["Node", "Connected Nodes"],
        )

        return connections_df


class App:
    """A class to represent the Streamlit application for visualizing a traffic grid.

    The class initializes the Streamlit app, sets up the environment configuration, and provides methods to interact with the traffic model.

    ## Methods:
        **__init__(self)**:
            Initializes the Streamlit app and sets up the environment configuration.
        **create_graph_fig(self) -> TrafficGraph**:
            Creates a figure of the Graph object used as a grid in TrafficModel.
        **create_env_conf_df(self) -> pd.DataFrame**:
            Creates a pandas DataFrame of the environment configuration.
        **create_connections_df(self) -> pd.DataFrame**:
            Creates a DataFrame containing connections for each node in the Graph.
        **create_edges_df(self) -> pd.DataFrame**:
            Creates a DataFrame with information about each edge in the graph.
        **step(self) -> None**:
            Advances the environment by one step.
        **update_env_config(self) -> None**:
            Applies changes to the environment from user options.
        **reset_environment(self) -> None**:
            Resets the environment with user-specified config options.
        **create_full_path_df(self, agent_id: int) -> pd.DataFrame**:
            Creates a DataFrame with the full path a car agent takes.
        **create_current_path_df(self, agent) -> pd.DataFrame**:
            Creates a DataFrame with the current (last) position, next position, and distance to the next position of an agent.
    """

    def __init__(self):
        """Initializes the Streamlit app and sets up the environment configuration."""

        # Initialize the model in session state if it doesn't exist
        self.model: TrafficModel = st.session_state["model"]

        # Environment config
        self.env_config = {
            "num_intersections": len(self.model.grid.get_nodes("intersection")),
            "num_borders": len(self.model.grid.get_nodes("border")),
            "min_distance": min(
                [edge[2] for edge in self.model.grid.edges(data="weight")]
            ),
            "max_distance": max(
                [edge[2] for edge in self.model.grid.edges(data="weight")]
            ),
            "num_cars": len(self.model.get_agents_by_type("CarAgent")),
            "auto_run_steps": 20,
        }

        st.session_state["env_config"] = self.env_config

        # Create two columns for layout
        left_col, right_col = st.columns([0.75, 0.25])

        # Render UI elements
        self.render_left_column(left_col)
        self.render_right_column(right_col)
        if st.session_state["env_config"]["num_cars"] > 0:
            CarPathListContainer()

        # Check for auto run loop
        if int(st.query_params["run_steps"]) > 0:
            time.sleep(0.1)
            st.query_params["run_steps"] = int(st.query_params["run_steps"]) - 1
            if len(self.model.get_agents_by_type("CarAgent")) == 0:
                st.query_params["run_steps"] = 0
            self.step()

    def render_left_column(self, left_col):
        """Renders the left column with the traffic graph visualization."""
        with left_col:
            header_cols = st.columns(
                [0.125, 0.2, 0.15, 0.525],
                gap="small",
                vertical_alignment="center",
            )
            self.render_header_cols(header_cols)
            GraphContainer(self.model)

    def render_header_cols(self, header_cols):
        """Renders the header columns with popovers."""
        with header_cols[0]:
            if st.button(
                label="Step",
                help="Execute one step",
                use_container_width=True,
            ):
                st.query_params["run_steps"] = self.env_config["auto_run_steps"] - 1
                self.step()
        with header_cols[1]:
            with st.popover("Show Connections"):
                ConnectionsContainer()
        with header_cols[2]:
            st.popover(label="Show Edges").dataframe(
                self.create_edges_df(),
                hide_index=True,
            )

    def render_right_column(self, right_col):
        """Renders the right column with UI controls."""
        with right_col:
            ui_cols = st.columns(spec=[0.3, 0.4, 0.3], vertical_alignment="center")
            self.render_ui_controls(ui_cols)
            SettingsContainer()

    def render_ui_controls(self, ui_cols):
        """Renders the UI controls in the right column."""
        with ui_cols[0]:
            pass

    def create_edges_df(self) -> pd.DataFrame:
        """Creates Dataframe with information about each edge in the graph.

        Returns:
            pd.DataFrame: Dataframe with information about each edge in the graph.
        """
        edge_df = pd.DataFrame(
            [(u, v, w) for u, v, w in self.model.grid.edges(data="weight")],
            columns=["U", "V", "Weight"],
        )

        return edge_df

    def step(self) -> None:
        """Advances the environment by one step."""
        self.model.step()
        st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Test",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    if "scroll_index" not in st.query_params:
        st.query_params["scroll_index"] = 0
    if "run_steps" not in st.query_params:
        st.query_params["run_steps"] = 0
    if "model" not in st.session_state:
        st.session_state["model"] = TrafficModel(num_agents=3)
    App()
