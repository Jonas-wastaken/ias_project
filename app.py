import streamlit as st
from graph import Graph

G = Graph()

with st.container():
    st.plotly_chart(G.generate_plot(), use_container_width=True)
