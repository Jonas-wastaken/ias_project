"""
This Streamlit app visualizes a traffic grid using NetworkX and Plotly.
It allows users to step through the simulation of traffic agents and view their paths.
"""

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from model import TrafficModel

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
        # Create a layout using the spring layout algorithm
        pos = nx.spring_layout(model.grid, seed=42)

        # Assign the positions to the nodes using dictionary comprehension
        nx.set_node_attributes(model.grid, pos, "pos")

        # Create a list to store the x and y coordinates of the edges
        edge_x = []
        edge_y = []
        for edge in model.grid.edges():
            x0, y0 = model.grid.nodes[edge[0]]["pos"]
            x1, y1 = model.grid.nodes[edge[1]]["pos"]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        # Create a trace for the edges
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Create a list to store the x and y coordinates of the nodes
        node_x = []
        node_y = []
        for node in model.grid.nodes():
            x, y = model.grid.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        node_color = []
        for node in model.grid.nodes(data=True):
            if node[1]["type"] == "intersection":
                node_color.append("blue")
            else:
                node_color.append("red")

        # Create a trace for the nodes
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                color=node_color,
                size=10,
                line_width=2,
            ),
        )

        # Add the number of connections to the node text
        node_adjacencies = []
        node_text = []
        for node, adjacency_dict in enumerate(model.grid.adjacency()):
            node_adjacencies.append(len(adjacency_dict[1]))
            node_text.append("# of connections: " + str(len(adjacency_dict[1])))

        # Update the node trace with the node text and color
        node_trace.text = node_text

        # Create the figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text="<br>Traffic Grid", font=dict(size=16)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
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
                        "Path": agent.path[1:-1],
                    }
                ]
            )
            agent_paths = pd.concat([agent_paths, agent_path], ignore_index=True)
        st.dataframe(agent_paths, use_container_width=True, hide_index=True)
