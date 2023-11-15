import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import ast
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET

plt.rcParams.update({'font.size': 20})

class Graph(object):
    def __init__(self, n_dependecies):
        self.G = nx.Graph()
        self.n_dependecies = n_dependecies
        self.objects_in_world = {}

    def get_node_attr(self, node, attr):
        # new_node = self.list_to_string(node)
        return self.G.nodes[node][attr]

    def change_node_attr(self, node, attr, value):
        aux_node = str(node)
        self.G.nodes[aux_node][attr] = value

    # def add_nodes(self, connections):
    #     for i in connections:
    #         node_label = tuple(i)
    #         attr = self.set_attribute(node_label)
    #         self.add_one_node(node_label, attr)

    def list_to_string(self, vec):
        modified_vector = [0.0 if value in {0, 0., -0., -0.0, -0} else value for value in vec]
        vector_str = '[' + ', '.join(map(str, modified_vector)) + ']'
        return vector_str

    def set_attribute(self, node, angles):
        # new_node = self.list_to_string(node)
        keyList = np.arange(1, self.n_dependecies, 1)
        dependendies = {"angle_" + str(keyList[i]): round(angles[i], 2) for i in range(len(keyList))}
        # attrs = {new_node: { "value": np.array(node), "occupied": False, "joint_dependency": dependendies}}
        self.add_one_node(node, dependendies)
        # return attrs

    def set_edge_attr(self, u, v, attr):
        nx.set_edge_attributes(self.G, {(u, v):{"occupied": attr}})

    def add_one_node(self, node, attr):
        new_node = self.list_to_string(node)
        self.G.add_node(new_node, value = node, occupied = False, joint_dependency = attr)
        # nx.set_node_attributes(self.G, attr)

    def change_edge_weight(self, u, v, value):
        self.G[u][v]['weight'] = value

    def add_one_edge(self, edge):
        u, v = self.list_to_string(edge[0]), self.list_to_string(edge[1])  # Convert ndarrays to tuples
        self.G.add_edge(u, v, weight = 1)
        self.set_edge_attr(u, v, False)

    def add_edges(self, edges):
        for i in edges:
            self.add_one_edge(i)
            # u, v = tuple(i[0]), tuple(i[1])  # Convert ndarrays to tuples
            # self.G.add_edge(u, v)

    def has_edge(self, u, v):
        u, v = tuple(u), tuple(v)
        return self.G.has_edge(u, v)

    def save_file(self, name):
        nx.write_gml(self.G, name + ".gml.gz", stringizer = list)

    def search(self, initial_node, goal_node, name):
        if name == "A*":
            path = nx.astar_path(self.G, initial_node, goal_node)
        return path

    def shortest_path(self, initial_node, goal_node):
        return nx.shortest_path(self.G, str(initial_node), str(goal_node))

    def number_of_nodes(self):
        return self.G.number_of_nodes()

    def has_path(self, initial_node, goal_node):
        return nx.has_path(self.G, str(initial_node), str(goal_node))

    def has_node(self, node):
        return self.G.has_node(node)

    def get_nodes(self):
        return self.G.nodes()

    def new_object_in_world(self, object_nodes, object_name):
        storage_graph = nx.Graph()
        print("Before Removal:")
        print("Number of nodes:", self.G.number_of_nodes())
        print("Number of edges:", self.G.number_of_edges())
        for node in object_nodes:
            if self.has_node(node):
                node_attr = self.G.nodes[node]
                local_edges = self.G.edges(node, data=True)
                storage_graph.add_node(node, **node_attr)
                storage_graph.add_edges_from(local_edges)
                self.G.remove_node(node)
        self.objects_in_world[object_name] = storage_graph

    def remove_object_from_world(self, name):
        print("After Removal:")
        print("Number of nodes:", self.G.number_of_nodes())
        print("Number of edges:", self.G.number_of_edges())
        self.plot_graph()
        deleted_object = self.objects_in_world[name]
        self.G.add_nodes_from(deleted_object.nodes.items())
        self.G.add_edges_from(deleted_object.edges)
        del self.objects_in_world[name]
        print("After adding again:")
        print("Number of nodes:", self.G.number_of_nodes())
        print("Number of edges:", self.G.number_of_edges())
        self.plot_graph()

    def get_nodes_values(self):
        nodes = self.G.nodes()
        vec_nodes = []
        for node in nodes:
            vec_nodes.append(self.get_node_attr(node, "value"))
        return vec_nodes

    def get_edges(self):
        return self.G.edges()

    def print_graph(self):
        print("List of nodes:")
        for node, attributes in self.G.nodes(data = True):
            print("Node:", node, "Attributes:", attributes)
        print("List of edges:")
        for u, v, attributes in self.G.edges(data = True):
            print("Edge:", u, "-", v, "Attributes:", attributes)

    def save_graph_to_file(self, name):
        output_file = "./data/graphs/" + name + ".xml"

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            file.write('<graph>\n')

            for node, attributes in self.G.nodes(data=True):
                x, y, z = attributes['value']
                value = attributes['value']
                occupied = attributes['occupied']
                joint_dependency = attributes['joint_dependency']
                file.write(f'  <node id="[{x}, {y}, {z}]" value= "{value}" occupied="{occupied}" joint_dependency="{joint_dependency}"/>\n')
            for u, v, attributes in self.G.edges(data = True):
                source = f'{u}'
                target = f'{v}'
                weight = attributes['weight']
                occupied = attributes['occupied']
                file.write(f'  <edge source="{source}" target="{target}" weight="{weight}" occupied="{occupied}"/>\n')
            file.write('</graph>\n')

    def parse_node(self, element):
        node_id = element.get('id')
        value = element.get('value')
        occupied = element.get('occupied')
        joint_dependency = element.get('joint_dependency')
        return node_id, {'value': value, 'occupied': occupied, 'joint_dependency': joint_dependency}

    def parse_edge(self, element):
        source = element.get('source')
        target = element.get('target')
        weight = element.get('weight')
        occupied = element.get('occupied')
        return (source, target), {'weight': float(weight), 'occupied': occupied}

    def read_graph_from_file(self, name):
        file_path = "./data/graphs/" + name + ".xml"
        tree = ET.parse(file_path)
        root = tree.getroot()

        for node_element in root.findall('.//node'):
            node, attributes = self.parse_node(node_element)
            self.G.add_node(node, **attributes)

        for edge_element in root.findall('.//edge'):
            edge, attributes = self.parse_edge(edge_element)
            self.G.add_edge(*edge, **attributes)

    def read_object_from_file(self, name):
        return nx.read_graphml("./data/objects/" + name + ".net")

    def vectorise_string(self, vec):
        aux_data = ''.join([i for i in vec if not (i=='[' or i==']' or i==',')])
        data = np.array(aux_data.split())
        return list(data[0:3].astype(float))

    def plot_graph(self, key, trajectory = []):
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
            xyz_rounded = tuple(round(coord, 3) for coord in self.get_node_attr(xyz, "value"))
            ax.scatter(*xyz_rounded, s=100, ec="w", cmap='Greens')
            # ax.text(*xyz_rounded, f"Node: {xyz_rounded}", ha="center", va="center")

        # Plot the edges
        for edge in self.get_edges():
            first_edge = self.vectorise_string(edge[0])
            second_edge = self.vectorise_string(edge[1])
            x = [first_edge[0], second_edge[0]]
            y = [first_edge[1], second_edge[1]]
            z = [first_edge[2], second_edge[2]]
            ax.plot(x, y, z, color='grey')
            # Calculate the length of the edge
            # length = ((x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2)**0.5
            # Calculate the midpoint of the edge
            # midpoint = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2)
            # Add the length as a text annotation at the midpoint
            # ax.text(midpoint[0], midpoint[1], midpoint[2], f"Length: {length:.2f}", ha='center', va='center')

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines on
            ax.grid(True)
            # Set axes labels
            ax.set_title(key)
            ax.set_xlabel("\n X [m]", linespacing=3.2)
            ax.set_ylabel("\n Y [m]", linespacing=3.2)
            ax.set_zlabel("\n Z [m]", linespacing=3.2)

        _format_axes(ax)
        fig.tight_layout()
        # plt.savefig(key + ".pdf", format="pdf")
        plt.show()
