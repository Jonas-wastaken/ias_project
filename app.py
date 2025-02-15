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
from src.model import TrafficModel
from src.graph_viz import TrafficGraph

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
model = st.session_state.model

# Graph config
graph_config = {
    "num_intersections": len(model.grid.get_nodes("intersection")),
    "num_borders": len(model.grid.get_nodes("border")),
    "min_distance": model.grid.min_distance,
    "max_distance": model.grid.max_distance,
}

# Create two columns for layout
left_col, right_col = st.columns([0.75, 0.25])

# Left column for the traffic graph visualization
with left_col:
    graph_container = st.container()
    with graph_container:
        fig = TrafficGraph(model)
        st.plotly_chart(fig, use_container_width=True)

        st.json(graph_config)

# Right column for the UI controls
with right_col:
    # Create sub-columns for user controls
    ui_cols = st.columns(spec=[0.3, 0.4, 0.3], vertical_alignment="center")

    # Step button to advance the simulation
    with ui_cols[0]:
        if st.button(label="Step", help="Execute one step", use_container_width=True):
            model.step()

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
            num_agents = st.number_input(
                label="Number of Agents",
                min_value=1,
                value=model.num_agents,
            )

            # Input for number of intersections
            num_intersections = st.number_input(
                label="Number of Intersections",
                min_value=1,
                value=graph_config["num_intersections"],
            )

            # Input for number of borders
            num_borders = st.number_input(
                label="Number of Borders",
                min_value=2,
                value=graph_config["num_borders"],
            )

            # Slider for distance range
            distance_range = st.slider(
                label="Distance",
                min_value=1,
                max_value=100,
                value=(model.grid.min_distance, model.grid.max_distance),
            )

            # Apply button to update the model with new settings
            if st.button(label="Apply", help="Apply the changes"):
                if num_intersections > graph_config["num_intersections"]:
                    model.grid.add_intersections(
                        num_intersections - graph_config["num_intersections"]
                    )
                elif num_intersections < graph_config["num_intersections"]:
                    model.grid.remove_intersections(
                        graph_config["num_intersections"] - num_intersections
                    )

                if num_borders > graph_config["num_borders"]:
                    model.grid.add_borders(num_borders - graph_config["num_borders"])
                elif num_borders < graph_config["num_borders"]:
                    model.grid.remove_borders(graph_config["num_borders"] - num_borders)

                if distance_range != (model.grid.min_distance, model.grid.max_distance):
                    st.session_state.model = TrafficModel(
                        num_agents=num_agents,
                        num_intersections=num_intersections,
                        num_borders=num_borders,
                        min_distance=distance_range[0],
                        max_distance=distance_range[1],
                    )
                    model = st.session_state.model

                for agent in model.agents[:]:
                    if (
                        agent.position not in model.grid.nodes
                        or agent.goal not in model.grid.nodes
                    ):
                        model.agents.remove(agent)
                        continue
                    agent.path = agent.compute_path()
                    model.agent_paths[agent.unique_id] = agent.path.copy()

                st.rerun()

    # Reset button to reset the environment
    with ui_cols[2]:
        if st.button(
            label="Reset", help="Reset the Environment", use_container_width=True
        ):
            st.session_state.model = TrafficModel(
                num_agents=num_agents,
                num_intersections=num_intersections,
                num_borders=num_borders,
                min_distance=distance_range[0],
                max_distance=distance_range[1],
            )
            model = st.session_state.model
            st.rerun()

    # Container to display agent paths
    agent_paths_container = st.container()

    with agent_paths_container:
        # Loop through each agent and display their paths
        for agent in model.agents:
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
                            pd.DataFrame(
                                [
                                    (node.title().replace("_", " "), distance)
                                    for node, distance in model.agent_paths[
                                        agent.unique_id
                                    ].items()
                                ],
                                columns=["Node", "Distance"],
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

            # Display the agent's current position, distance to next position and next position
            current_position, next_position = list(agent.path.keys())[:2]
            distance = list(agent.path.values())[0]
            st.dataframe(
                pd.DataFrame(
                    [
                        (
                            current_position.title().replace("_", " "),
                            next_position.title().replace("_", " "),
                            distance,
                        )
                    ],
                    columns=["Current Position", "Next Position", "Distance"],
                ),
                use_container_width=True,
                hide_index=True,
            )
