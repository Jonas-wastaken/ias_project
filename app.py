"""
This Streamlit app visualizes a traffic grid using NetworkX and Plotly.
It allows users to step through the simulation of traffic agents and view their paths.
"""

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
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

left_col, right_col = st.columns([2, 1])

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
        if st.button(label="Reset", help="Not functional"):
            st.rerun()

    agent_paths_container = st.container()
    with agent_paths_container:
        agent_paths = pd.DataFrame(
            columns=["Agent", "Start", "Goal", "Position", "Path"]
        )
        for agent in model.agents:
            agent_path = pd.DataFrame(
                [
                    {
                        "Agent": agent.unique_id,
                        "Start": agent.start,
                        "Goal": agent.goal,
                        "Position": agent.position,
                        "Path": agent.path,
                    }
                ]
            )
            agent_paths = pd.concat([agent_paths, agent_path], ignore_index=True)
        st.dataframe(agent_paths, use_container_width=True, hide_index=True)
