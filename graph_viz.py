import networkx as nx
import plotly.graph_objects as go


class TrafficGraph(go.Figure):
    def __init__(self, model):
        super().__init__()
        self._model = model  # Use a private attribute to store the model
        self.create_graph()

    def create_graph(self):
        # Create a layout using the spring layout algorithm
        pos = nx.spring_layout(self._model.grid, seed=42)

        # Assign the positions to the nodes using dictionary comprehension
        nx.set_node_attributes(self._model.grid, pos, "pos")

        # Create a list to store the x and y coordinates of the edges
        edge_x = []
        edge_y = []
        for edge in self._model.grid.edges():
            x0, y0 = self._model.grid.nodes[edge[0]]["pos"]
            x1, y1 = self._model.grid.nodes[edge[1]]["pos"]
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
        for node in self._model.grid.nodes():
            x, y = self._model.grid.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        node_color = []
        for node in self._model.grid.nodes(data=True):
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
        for node, adjacency_dict in enumerate(self._model.grid.adjacency()):
            node_adjacencies.append(len(adjacency_dict[1]))
            node_text.append("# of connections: " + str(len(adjacency_dict[1])))

        # Update the node trace with the node text and color
        node_trace.text = node_text

        # Create the figure
        self.add_traces([edge_trace, node_trace])
        self.update_layout(
            title=dict(text="<br>Traffic Grid", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
