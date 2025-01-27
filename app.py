"""
Interactive network graph visualization using Streamlit and Plotly.
"""

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from graph import Graph

with st.container():
    # Create an instance of the Graph class
    G = Graph()

    # Create a layout using the spring layout algorithm
    pos = nx.spring_layout(G)

    # Assign the positions to the nodes using dictionary comprehension
    nx.set_node_attributes(G, pos, "pos")

    # Create a list to store the x and y coordinates of the edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]["pos"]
        x1, y1 = G.nodes[edge[1]]["pos"]
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
    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)

    node_color = []
    for node in G.nodes(data=True):
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
    for node, adjacency_dict in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacency_dict[1]))
        node_text.append("# of connections: " + str(len(adjacency_dict[1])))

    # Update the node trace with the node text and color
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="<br>Network graph made with Python", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
