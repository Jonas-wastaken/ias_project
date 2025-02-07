"""
This Streamlit app visualizes a traffic grid using NetworkX and Plotly.
It allows users to step through the simulation of traffic agents and view their paths.
"""

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
from model import TrafficModel
from graph_viz import TrafficGraph

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

left_col, right_col = st.columns([0.75, 0.25])

with left_col:
    graph_container = st.container()
    with graph_container:
        fig = TrafficGraph(model)
        st.plotly_chart(fig, use_container_width=False)

with right_col:
    ui_cols = st.columns(spec=2, gap="small")
    with ui_cols[0]:
        if st.button(label="Step", help="Execute one step"):
            model.step()
    with ui_cols[1]:
        if st.button(label="Reset", help="Reset the Environment"):
            st.session_state.model = TrafficModel(num_agents=3)
            model = st.session_state.model
            st.rerun()

    agent_paths_container = st.container()
    with agent_paths_container:
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
                                model.agent_paths[agent.unique_id].items(),
                                columns=["Node", "Distance"],
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
            st.dataframe(
                pd.DataFrame(agent.path.items(), columns=["Node", "Distance"]),
                use_container_width=True,
                hide_index=True,
            )
