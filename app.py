"""
This Streamlit app visualizes a traffic grid using NetworkX and Plotly.
It allows users to step through the simulation of traffic agents and view their paths.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
from model import TrafficModel
from graph_viz import TrafficGraph


class App:
    def __init__(self):
        st.set_page_config(
            page_title=None,
            page_icon=None,
            layout="wide",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        # Initialize the model in session state if it doesn't exist
        if "model" not in st.session_state:
            st.session_state.model = TrafficModel(num_agents=3)
        self.model = st.session_state.model

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
            "num_agents": len(self.model.agents),
            "auto_run_steps": 5,
        }

        # Create two columns for layout
        left_col, right_col = st.columns([0.75, 0.25])

        # Left column for the traffic graph visualization
        with left_col:
            with st.container():
                header_cols = st.columns(
                    [0.125, 0.2, 0.15, 0.525],
                    gap="small",
                    vertical_alignment="center",
                )
                with header_cols[0]:
                    st.popover(label="Settings").dataframe(
                        self.create_env_conf_df(),
                        hide_index=True,
                    )
                with header_cols[1]:
                    st.popover(label="Show Connections").dataframe(
                        self.create_connections_df(),
                        hide_index=True,
                    )
                with header_cols[2]:
                    st.popover(label="Show Edges").dataframe(
                        self.create_edges_df(),
                        hide_index=True,
                    )
            st.plotly_chart(self.create_graph_fig(), use_container_width=True)

        # Right column for the UI controls
        with right_col:
            # Create sub-columns for user controls
            ui_cols = st.columns(spec=[0.3, 0.4, 0.3], vertical_alignment="center")

            # Step button to advance the simulation
            with ui_cols[0]:
                if st.button(
                    label="Step",
                    help="Execute one step",
                    use_container_width=True,
                ):
                    self.step()

            # Options popover to change environment settings
            with ui_cols[1]:
                options_popover = st.popover(
                    label="Options",
                    help="Change the environment settings",
                    use_container_width=True,
                )

                with options_popover:
                    st.markdown("### Options")

                    # Input for number of agents
                    self.num_agents = st.number_input(
                        label="Number of Agents",
                        min_value=1,
                        value=self.env_config["num_agents"],
                    )

                    # Input for number of intersections
                    self.num_intersections = st.number_input(
                        label="Number of Intersections",
                        min_value=1,
                        value=self.env_config["num_intersections"],
                    )

                    # Input for number of borders
                    self.num_borders = st.number_input(
                        label="Number of Borders",
                        min_value=2,
                        value=self.env_config["num_borders"],
                    )

                    # Slider for distance range
                    self.distance_range = st.slider(
                        label="Distance",
                        min_value=1,
                        max_value=100,
                        value=(
                            self.model.grid.min_distance,
                            self.model.grid.max_distance,
                        ),
                    )

                    # Apply button to update the model with new settings
                    if st.button(label="Apply", help="Apply the changes"):
                        self.update_env_config()

            # Reset button to reset the environment
            with ui_cols[2]:
                if st.button(
                    label="Reset",
                    help="Reset the Environment",
                    use_container_width=True,
                ):
                    self.reset_environment()

            # Container to display agent paths
            agent_paths_container = st.container()

            with agent_paths_container:
                # Loop through each agent and display their paths
                for agent in self.model.agents:
                    left_col, right_col = st.columns(2)

                    with left_col:
                        st.subheader(f"Agent {agent.unique_id}")

                    with right_col:
                        with stylable_container(
                            key=f"agent_{agent.unique_id}",
                            css_styles="""
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
                                    self.create_full_path_df(agent_id=agent.unique_id),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                    # Display the agent's current position, distance to next position and next position
                    st.dataframe(
                        self.create_current_path_df(agent),
                        use_container_width=True,
                        hide_index=True,
                    )

    def create_graph_fig(self) -> TrafficGraph:
        """Creates figure of Graph object used as a grid in TrafficModel.

        Returns:
            TrafficGraph: go.Figure object of TrafficModel Graph
        """
        graph_fig = TrafficGraph(self.model)

        return graph_fig

    def create_env_conf_df(self) -> pd.DataFrame:
        """Creates a pandas dataframe of the environment config.

        Returns:
            pd.DataFrame: Dataframe with Settings and Values of environment settings
        """
        env_conf_df = pd.DataFrame(
            self.env_config.items(), columns=["Setting", "Value"]
        )

        return env_conf_df

    def create_connections_df(self) -> pd.DataFrame:
        """Creates Dataframe containing connections for each node in the Graph

        Returns:
            pd.DataFrame: Dataframe containing connections for each node in the Graph
        """
        connections = self.model.grid.get_connections()
        connections_df = pd.DataFrame(
            [(node, connections[node]) for node in connections],
            columns=["Node", "Connected Nodes"],
        )

        return connections_df

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

    def update_env_config(self) -> None:
        """Applies changes to the environment from user options."""
        # Update the model with new settings for number of agents
        if self.num_agents > self.env_config["num_agents"]:
            self.model.create_agents(self.num_agents - self.env_config["num_agents"])
        elif self.num_agents < self.env_config["num_agents"]:
            self.model.remove_agents(self.env_config["num_agents"] - self.num_agents)

        # Update the model with new settings for number of intersections
        if self.num_intersections > self.env_config["num_intersections"]:
            self.model.grid.add_intersections(
                self.num_intersections - self.env_config["num_intersections"]
            )
        elif self.num_intersections < self.env_config["num_intersections"]:
            self.model.grid.remove_intersections(
                self.env_config["num_intersections"] - self.num_intersections
            )

        # Update the model with new settings for number of borders
        if self.num_borders > self.env_config["num_borders"]:
            self.model.grid.add_borders(
                self.num_borders - self.env_config["num_borders"]
            )
        elif self.num_borders < self.env_config["num_borders"]:
            self.model.grid.remove_borders(
                self.env_config["num_borders"] - self.num_borders
            )

        # Update the model with new settings for distance range
        if self.distance_range != (
            self.env_config["min_distance"],
            self.env_config["max_distance"],
        ):
            self.model.grid.change_weights(
                min_distance=self.distance_range[0],
                max_distance=self.distance_range[1],
            )

        # Update the paths for each agent or delete agents if they are not on the grid
        for agent in self.model.agents[:]:
            if (
                agent.position not in self.model.grid.nodes
                or agent.goal not in self.model.grid.nodes
            ):
                self.model.agents.remove(agent)
                continue
            agent.path = agent.compute_path()
            self.model.agent_paths[agent.unique_id] = agent.path.copy()

        st.rerun()

    def reset_environment(self) -> None:
        """Resets the environment with user specified config options."""
        st.session_state.model = TrafficModel(
            num_agents=self.num_agents,
            num_intersections=self.num_intersections,
            num_borders=self.num_borders,
            min_distance=self.distance_range[0],
            max_distance=self.distance_range[1],
        )
        self.model = st.session_state.model
        st.rerun()

    def create_full_path_df(self, agent_id: int) -> pd.DataFrame:
        """Creates a DataFrame with the full path a car agent takes.

        Args:
            agent_id (int): ID of agent

        Returns:
            pd.DataFrame: DataFrame with the full path a car agent takes.
        """
        full_path_df = pd.DataFrame(
            [
                (node.title().replace("_", " "), distance)
                for node, distance in self.model.agent_paths[agent_id].items()
            ],
            columns=["Node", "Distance"],
        )

        return full_path_df

    def create_current_path_df(self, agent) -> pd.DataFrame:
        """Creates a DataFrame with the current (last) position, next position and distance to next position of an agent.

        Args:
            agent (car.CarAgent): CarAgent instance

        Returns:
            pd.DataFrame: DataFrame with the current (last) position, next position and distance to next position of an agent.
        """
        current_position, next_position = list(agent.path.keys())[:2]
        distance = list(agent.path.values())[0]

        current_path_df = pd.DataFrame(
            [
                (
                    current_position.title().replace("_", " "),
                    next_position.title().replace("_", " "),
                    distance,
                )
            ],
            columns=["Current Position", "Next Position", "Distance"],
        )

        return current_path_df


if __name__ == "__main__":
    App()
