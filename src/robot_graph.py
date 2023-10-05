import pprint
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Graph(object):
    def __init__(self, n_dependecies):
        self.G = nx.Graph()
        self.n_dependecies = n_dependecies

    def get_node_attr(self, node, attr):
        aux_node = str(node)
        return self.G.nodes[aux_node][attr]

    def add_nodes(self, connections):
        for i in connections:
            node_label = tuple(i)
            attr = self.set_attribute(node_label)
            self.add_one_node(node_label, attr)

    def set_attribute(self, node, angles):
        keyList = np.arange(1, self.n_dependecies + 1, 1)
        dependendies = {keyList[i]: angles[i] for i in range(len(keyList))}
        attrs = {str(node): { "value": node, "is_busy": True, "joint_dependency": dependendies}}
        return attrs

    def add_one_node(self, node, attr):
        self.G.add_node(str(node))
        nx.set_node_attributes(self.G, attr)

    def add_one_edge(self, edge):
        u, v = tuple(edge[0]), tuple(edge[1])  # Convert ndarrays to tuples
        self.G.add_edge(u, v)

    def add_edges(self, edges):
        for i in edges:
            u, v = tuple(i[0]), tuple(i[1])  # Convert ndarrays to tuples
            self.G.add_edge(u, v)

    def has_edge(self, u, v):
        u, v = tuple(u), tuple(v)
        return self.G.has_edge(u, v)

    def save_file(self, name):
        nx.write_gml(self.G, name + ".gml.gz", stringizer = list)

    def search(self, initial_node, goal_node, name):
        if name == "A*":
            path = nx.astar_path(self.G, initial_node, goal_node)
        return path

    def get_nodes(self):
        return self.G.nodes()

    def get_edges(self):
        return self.G.edges()

    def print_graph(self):
        print("Node Attributes:")
        for node, attributes in self.G.nodes(data = True):
            print("Node:", node, "Attributes:", attributes)
        print("Edge Attributes:")
        for u, v, attributes in self.G.edges(data = True):
            print("Edge:", u, "-", v, "Attributes:", attributes)

    def plot_graph(self, trajectory = []):
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot trajectory only if it exists
        if len(trajectory):
            xdata = trajectory[:, 0]
            ydata = trajectory[:, 1]
            zdata = trajectory[:, 2]
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

        # Plot the nodes
        for node, xyz in zip(self.G.nodes(), self.G.nodes()):
            xyz_rounded = tuple(round(coord, 2) for coord in xyz)
            ax.scatter(*xyz_rounded, s=100, ec="w")
            # ax.text(*xyz_rounded, f"Node: {xyz_rounded}", ha="center", va="center")

        # Plot the edges
        for edge in self.G.edges():
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            z = [edge[0][2], edge[1][2]]
            ax.plot(x, y, z, color='grey')
            # Calculate the length of the edge
            length = ((x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2)**0.5
            # Calculate the midpoint of the edge
            midpoint = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2)
            # Add the length as a text annotation at the midpoint
            # ax.text(midpoint[0], midpoint[1], midpoint[2], f"Length: {length:.2f}", ha='center', va='center')

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines on
            ax.grid(True)
            # Set axes labels
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        _format_axes(ax)
        fig.tight_layout()
        plt.show()
