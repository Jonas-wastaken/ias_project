import streamlit as st
from graph import Graph

G = Graph()

with st.container():
    st.write("Graph created successfully!")
    st.write("Number of nodes:", G.number_of_nodes())
    st.write("Number of edges:", G.number_of_edges())

with st.container():
    st.pyplot(G.generate_plot())
