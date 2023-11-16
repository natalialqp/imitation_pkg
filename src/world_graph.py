import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import ast
plt.rcParams.update({'font.size': 20})

class Graph(object):
    def __init__(self):
        self.G = nx.Graph()

    def add_nodes(self, connections):
        for i in connections:
            node_label = tuple(i)  # Convert ndarray to tuple
            self.G.add_node(node_label)

    def add_one_node(self, node):
         # Convert ndarray to tuple
<<<<<<< HEAD
        new_node = self.list_to_string(node)
        self.G.add_node(tuple(new_node))

    def list_to_string(self, vec):
        modified_vector = [0.0 if value in {0, 0., -0., -0.0, -0} else value for value in vec]
        # vector_str = '[' + ', '.join(map(str, modified_vector)) + ']'
        return modified_vector

=======
        new_node = np.array([0. if x == -0.0 else x for x in node])
        self.G.add_node(tuple(new_node))

>>>>>>> 27a3fc396c7d4000f2162cd03163457f79f82175
    def add_one_edge(self, edge):
        u, v = tuple(edge[0]), tuple(edge[1])  # Convert ndarrays to tuples
        self.G.add_edge(u, v)

    def add_edges(self, edges):
        for i in tqdm(edges):
            u, v = tuple(i[0]), tuple(i[1])  # Convert ndarrays to tuples
            self.G.add_edge(u, v)

    def has_edge(self, u, v):
        u, v = tuple(u), tuple(v)
        return self.G.has_edge(u, v)

    def read_object_from_file(self, name):
        aux_graph = nx.Graph()
        aux_graph = nx.read_graphml("./data/objects/" + name + ".net")

        for node in tqdm(aux_graph.nodes()):
            self.G.add_node(ast.literal_eval(node))
        for edges in tqdm(aux_graph.edges()):
            self.G.add_edge(ast.literal_eval(edges[0]), ast.literal_eval(edges[1]))

    def save_object_to_file(self, name):
        nx.write_graphml_lxml(self.G, "./data/objects/" + name + ".net")

    def save_graph_to_file(self, name):
        nx.write_graphml_lxml(self.G, "./data/graphs/" + name + ".net")

    def read_graph_from_file(self, name):
        aux_graph = nx.Graph()
        aux_graph = nx.read_graphml("./data/graphs/" + name + ".net")

        for node in tqdm(aux_graph.nodes()):
            self.G.add_node(ast.literal_eval(node))
        for edges in tqdm(aux_graph.edges()):
            self.G.add_edge(ast.literal_eval(edges[0]), ast.literal_eval(edges[1]))

    def search(self, initial_node, goal_node, name):
        if name == "A*":
            path = nx.astar_path(self.G, initial_node, goal_node)
        return path

    def get_nodes(self):
        return self.G.nodes()

    def get_edges(self):
        return self.G.edges()

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
        for node, xyz in zip(self.get_nodes(), self.get_nodes()):
            xyz_rounded = tuple(round(coord, 3) for coord in xyz)
            ax.scatter(*xyz_rounded, s=100, ec="w")
            # ax.text(*xyz_rounded, f"Node: {xyz_rounded}", ha="center", va="center")

        # Plot the edges
        for edge in self.get_edges():
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            z = [edge[0][2], edge[1][2]]
            ax.plot(x, y, z, color='grey')
            # Calculate the length of the edge
            # length = ((x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2)**0.5
            # Calculate the midpoint of the edge
            # midpoint = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2)
            # Add the length as a text annotation at the midpoint
            # ax.text(midpoint[0], midpoint[1], midpoint[2], f"Length: {length:.2f}", ha='center', va='center')

        def _format_axes(ax):
            """Visualization options for the 3D axes"""
            # Turn gridlines on
            ax.grid(True)
            # Set axes labels
            ax.set_xlabel("\n X [m]", linespacing=3.2)
            ax.set_ylabel("\n Y [m]", linespacing=3.2)
            ax.set_zlabel("\n Z [m]", linespacing=3.2)

        _format_axes(ax)
        fig.tight_layout()
        # plt.savefig("world_small.pdf", format="pdf")
        plt.show()
