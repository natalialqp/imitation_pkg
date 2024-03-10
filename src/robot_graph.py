import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
import json
import re
from scipy.optimize import minimize
from math import radians

plt.rcParams.update({"font.size": 20})

class Graph(object):
    def __init__(self, n_dependecies, key):
        self.G = nx.Graph()
        self.n_dependecies = n_dependecies
        self.objects_in_world = {}
        self.key = key
        self.nodes_id = 0

    def get_all_nodes_attr(self, attr):
        nodes = self.get_nodes()
        node_values = [self.get_node_attr(node, attr) for node in nodes]
        return node_values

    def get_node_attr(self, node, attr):
        if not isinstance(node, str):
            node = self.list_to_string(node)
        return self.G.nodes[node][attr]

    def change_node_attr(self, node, attr, value):
        aux_node = str(node)
        self.G.nodes[aux_node][attr] = value

    def list_to_string(self, vec):
        modified_vector = [0.0 if value in {0, 0., -0., -0.0, -0} else value for value in vec]
        vector_str = "[" + ", ".join(map(str, modified_vector)) + "]"
        return vector_str

    def set_attribute(self, node, angles):
        # new_node = self.list_to_string(node)
        if len(angles) > 0:
            keyList = np.arange(1, self.n_dependecies - 1, 1)
        else:
            keyList = []
        dependencies = {"angle_" + str(keyList[i]): round(angles[i], 2) for i in range(len(keyList))}
        self.add_one_node(node, [dependencies])

    def set_edge_attr(self, u, v, attr):
        nx.set_edge_attributes(self.G, {(u, v):{"occupied": attr}})

    def add_one_node(self, node, attr):
        new_node = self.list_to_string(node)
        if not self.has_node(new_node):
            # list_attr = [attr]
            self.G.add_node(new_node, value = node, nodes_id = self.nodes_id, occupied = False, joint_dependency = attr) #list_attr
            self.nodes_id += 1
        else:
            node_attr = self.get_node_attr(new_node, "joint_dependency")
            for a in attr:
                self.update_joint_attr_dict(new_node, a, node_attr)

    def update_joint_attr_dict(self, node, dict, dict_list):
        is_different = all(old_dict[attr] != dict[attr] for old_dict in
                        dict_list for attr in dict) and len(dict) != 0
        if is_different:
            dict_list.append(dict)
            self.change_node_attr(node, "joint_dependency", dict_list)

    def change_edge_weight(self, u, v, value):
        self.G[u][v]["weight"] = value

    def add_one_edge(self, edge):
        u, v = self.list_to_string(edge[0]), self.list_to_string(edge[1])  # Convert ndarrays to tuples
        self.G.add_edge(u, v, weight = 1)
        self.set_edge_attr(u, v, False)

    def add_edges(self, edges):
        for i in edges:
            self.add_one_edge(i)

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
        initial_node = self.list_to_string(initial_node)
        goal_node = self.list_to_string(goal_node)
        return nx.shortest_path(self.G, str(initial_node), str(goal_node))

    def number_of_nodes(self):
        return self.G.number_of_nodes()

    def has_path(self, initial_node, goal_node):
        initial_node = self.list_to_string(initial_node)
        goal_node = self.list_to_string(goal_node)
        return nx.has_path(self.G, initial_node, goal_node)

    def has_node(self, node):
        if type(node) == list:
            node = self.list_to_string(node)
        return self.G.has_node(node)

    def get_nodes(self):
        return self.G.nodes()

    def find_nodes_with_angle_tolerance(self, input_values_list, tolerance = 5):
        result_ids = []
        for node in self.G.nodes(data = True):
            node_id = node[0]
            joint_dependency = node[1].get("joint_dependency")
            if joint_dependency:
                for input_values in input_values_list:
                    meets_tolerance = any(
                        all(
                            abs(angle_dict.get(key, 0) - input_values.get(key, 0)) <= tolerance
                            for key in input_values
                        )
                        for angle_dict in joint_dependency
                    )
                    if meets_tolerance:
                        result_ids.append(node_id)
                        break  # Move to the next node if one dictionary meets the condition
        return result_ids

    def new_object_in_world(self, object_nodes, object_name, end_effector, prev_joint_angles = []):
        storage_graph = nx.Graph()
        if end_effector and prev_joint_angles:
            aux = self.find_nodes_with_angle_tolerance(prev_joint_angles)
            object_nodes.extend(aux)
        future_dependencies = []
        for node in object_nodes:
            if self.has_node(node):
                node_attr = self.G.nodes[node]
                local_edges = self.G.edges(node, data=True)
                storage_graph.add_node(node, **node_attr)
                storage_graph.add_edges_from(local_edges)
                self.G.remove_node(node)
                if end_effector == False:
                    future_dependencies.extend(node_attr["joint_dependency"])
        self.objects_in_world[object_name] = storage_graph
        return future_dependencies

    def remove_object_from_world(self, name):
        self.find_subgraphs()
        # self.plot_graph()
        deleted_object = self.objects_in_world[name]
        self.G.add_nodes_from(deleted_object.nodes.items())
        self.G.add_edges_from(deleted_object.edges)
        del self.objects_in_world[name]
        self.find_subgraphs()
        # self.plot_graph()

    def get_nodes_values(self):
        nodes = self.G.nodes()
        return [self.get_node_attr(node, "value") for node in nodes]

    def get_edges(self):
        return self.G.edges()

    def print_graph(self):
        print("List of nodes:")
        for node, attributes in self.G.nodes(data = True):
            print("Node:", node, "Attributes:", attributes)
        print("List of edges:")
        for u, v, attributes in self.G.edges(data = True):
            print("Edge:", u, "-", v, "Attributes:", attributes)

    def read_last_id_path_from_file(self, robotName):
        try:
            with open(self.key + robotName + "_data.json", "r") as jsonfile:
                lines = jsonfile.readlines()
                if lines:
                    last_line = lines[-1]
                    last_pose = json.loads(last_line)
                    last_id = last_pose.get("id", 0)
                    return last_id
                else:
                    return 0
        except FileNotFoundError:
            return 0
        except json.JSONDecodeError:
            # Handle JSON decoding error if necessary
            return 0

    def save_path_in_library(self, trajectory, dependency, robotName, actionName, babbling_file):
        with open("./data/test_" + robotName + "/paths_lib/" + self.key + "_" + babbling_file + ".json", "a") as jsonfile:
            pose = {"id": actionName, "path": trajectory, "joint_dependency": dependency}
            jsonfile.write(json.dumps(pose))
            jsonfile.write("\n")

    def save_graph_to_file(self, name, robot, babbled_nodes):
        output_file = "./data/test_" + robot + "/graphs/joints_graphs_lib/" + name + "_" + babbled_nodes + ".xml"

        with open(output_file, "w", encoding="utf-8") as file:
            file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            file.write("<graph>\n")

            for node, attributes in self.G.nodes(data=True):
                x, y, z = attributes["value"]
                value = attributes["value"]
                occupied = attributes["occupied"]
                nodes_id = attributes["nodes_id"]
                joint_dependency = attributes["joint_dependency"]
                file.write(f' <node id="[{x}, {y}, {z}]" value= "{value}" nodes_id = "{nodes_id}" occupied="{occupied}" joint_dependency="{joint_dependency}"/>\n')
            for u, v, attributes in self.G.edges(data = True):
                source = f"{u}"
                target = f"{v}"
                weight = attributes["weight"]
                occupied = attributes["occupied"]
                file.write(f' <edge source="{source}" target="{target}" weight="{weight}" occupied="{occupied}"/>\n')
            file.write("</graph>\n")

    def select_joint_dependencies(self, trajectory):
        prev_dict =  {"angle_" + str(key + 1): 0 for key in range(self.n_dependecies - 1)}#change for initial position of the robot joints
        dependencies = []
        for point in trajectory:
            if not isinstance(point, str):
                point = self.list_to_string(point)
            dep_list = self.get_node_attr(point, "joint_dependency")
            least_error_dict = self.find_least_error_dict(prev_dict, dep_list)
            dependencies.append(least_error_dict)
            prev_dict = least_error_dict
        return dependencies

    def calculate_error(self, d_1, d_2):
        # Calculate the sum of squared differences between corresponding values
        d_2 = self.string_to_dict(d_2)
        return sum((d_1[key] - d_2[key]) ** 2 for key in d_1 if key in d_2)

    def find_least_error_dict(self, reference_dict, dict_list):
        # Find the dictionary with the least error
        least_error_dict = min(dict_list, key=lambda d: self.calculate_error(reference_dict, d))
        return least_error_dict

    def parse_node(self, element):
        node_id_cart = element.get("id")
        value = self.vectorise_string(element.get("value"))
        occupied = element.get("occupied")
        node_id = element.get("nodes_id")
        joint_dependency = ast.literal_eval(element.get("joint_dependency"))
        return node_id_cart, {"value": value, "nodes_id": node_id, "occupied": occupied, "joint_dependency": joint_dependency}

    def parse_edge(self, element):
        source = element.get("source")
        target = element.get("target")
        weight = element.get("weight")
        occupied = element.get("occupied")
        return (source, target), {"weight": float(weight), "occupied": occupied}

    def read_graph_from_file(self, name, robot, babbled_nodes):
        file_path = "./data/test_" + robot + "/graphs/joints_graphs_lib/" + name + "_" + babbled_nodes + ".xml"
        tree = ET.parse(file_path)
        root = tree.getroot()
        self.nodes_id = 0

        for node_element in root.findall(".//node"):
            node, attributes = self.parse_node(node_element)
            self.G.add_node(node, **attributes)
            self.nodes_id += 1

        for edge_element in root.findall(".//edge"):
            edge, attributes = self.parse_edge(edge_element)
            self.G.add_edge(*edge, **attributes)

    def read_object_from_file(self, name):
        return nx.read_graphml("./data/objects/" + name + ".net")

    def vectorise_string(self, vec):
        aux_data = "".join([i for i in vec if not (i=="[" or i=="]" or i==",")])
        data = np.array(aux_data.split())
        return list(data[0:3].astype(float))

    def dict_of_dep_to_dict_of_lists(self, dict):
        result_dict = {}
        for entry in dict:
            for key, value in entry.items():
                if key not in result_dict:
                    result_dict[key] = []
                result_dict[key].append(value)
        return result_dict

    def plot_dict_of_dependencies(self, dict_old, dict_new):
        for key in dict_new.keys():
            plt.plot(np.radians(np.array(dict_old[key])), label=f'Old angles - {key}', marker='o')
            plt.plot(np.radians(np.array(dict_new[key])), label=f'New angles - {key}', marker='x')

        # Adding labels and legend
        plt.grid()
        plt.xlabel('Frames')
        plt.ylabel('Angle in Radians')
        plt.legend(loc='upper right')
        plt.title('Old vs new angles after Path Repair')
        plt.show()

    def plot_graph(self, robotName, nodes = '', trajectory = [], candidates = []):
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot trajectory only if it exists
        if len(trajectory):
            xdata = trajectory[:, 0]
            ydata = trajectory[:, 1]
            zdata = trajectory[:, 2]
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="Greens")
                # Plot edges between the points
            for i in range(len(xdata) - 1):
                ax.plot([xdata[i], xdata[i+1]], [ydata[i], ydata[i+1]], [zdata[i], zdata[i+1]], c="blue")
        if len(candidates):
            xdata = candidates[:, 0]
            ydata = candidates[:, 1]
            zdata = candidates[:, 2]
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="Reds")

           # Plot the nodes
        for node, xyz in zip(self.get_nodes(), self.get_nodes()):
            xyz_rounded = tuple(round(coord, 2) for coord in self.get_node_attr(xyz, "value"))
            ax.scatter(*xyz_rounded, s=100, ec="w", cmap="Greens")

        # Plot the edges
        for edge in self.get_edges():
            first_edge = self.vectorise_string(edge[0])
            second_edge = self.vectorise_string(edge[1])
            x = [first_edge[0], second_edge[0]]
            y = [first_edge[1], second_edge[1]]
            z = [first_edge[2], second_edge[2]]
            ax.plot(x, y, z, color="grey")

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines on
            ax.grid(True)
            # Set axes labels
            ax.set_title(self.key)
            ax.set_xlabel("\n X [mm]", linespacing=3.2)
            ax.set_ylabel("\n Y [mm]", linespacing=3.2)
            ax.set_zlabel("\n Z [mm]", linespacing=3.2)

        _format_axes(ax)
        fig.tight_layout()
        plt.savefig("data/test_"+ robotName + "/graphs/graphs_plots/" + self.key + "_" + robotName + "_" + nodes + ".pdf", format="pdf") #_object
        # plt.show()

    def find_neighbors(self, node):
        return self.G.neighbors(self.list_to_string(node))

    def find_closest_point(self, new_point, kdtree):
        distance, index = kdtree.query(new_point)
        return kdtree.data[index]

    def find_trajectory_in_graph(self, tra):
        nodes = self.get_nodes_values()
        kdtree = KDTree(np.array(nodes))
        world_nodes = []
        for node in tra:
            closest_point = self.find_closest_point(node, kdtree)
            if world_nodes:
                if (world_nodes[-1] != closest_point).any():
                    world_nodes.append(closest_point)
            else:
                world_nodes.append(closest_point)
        return world_nodes

    def find_neighbors_candidates(self, nodes):
        return [self.find_neighbors(node) for node in nodes]

    def find_trajectory_shared_nodes(self, tra):
        kdtree = self.get_nodes()
        return [point for point in tra if self.list_to_string(point) not in kdtree]

    def find_subgraphs(self):
        sub_graphs = [self.G.subgraph(c) for c in nx.connected_components(self.G)]
        print(len(sub_graphs))

    def lerp(self, a, b, t):
        interpolated_dict = {key: round((1 - t) * a[key] + t * b[key], 2) for key in a}
        return interpolated_dict

    def string_to_dict(self, dict):
        if type(dict) == str:
            try:
                dict = re.sub(r"\'", '"', dict)
                dict = json.loads(dict)
            except ValueError as e:
                print(f"Error: {e}")
        return dict

    def calculate_average_angle(self, dict_list):
        if not dict_list:
            return {}
        array_of_dicts = np.array([list(d.values()) for d in dict_list])
        avg_array = np.mean(array_of_dicts, axis=0)
        avg_dict = dict(zip(dict_list[0].keys(), avg_array))
        return avg_dict

    def interpolate_joint_angles(self, joint_angles_a, joint_angles_b, t = 0.5):
        joint_angles_a = self.calculate_average_angle(joint_angles_a)
        joint_angles_b = self.calculate_average_angle(joint_angles_b)
        return self.lerp(joint_angles_a, joint_angles_b, t)

    def adding_candidates_to_graph(self, robot, length, candidates):
        nodes = self.get_nodes_values()
        tree = KDTree(np.array(nodes))
        max_dist = 2 * length
        for i, point in enumerate(candidates):
            neighbors_indices = tree.query_ball_point(point, max_dist)
            if len(neighbors_indices) > 1:
                distances, indices = tree.query(point, k = 1)
                initial_angles = self.get_node_attr(tree.data[indices], "joint_dependency")
                angles = []
                for i_a in initial_angles:
                    angles.append(self.gauss_newton_angle_estimation(robot, i_a, point))
                self.add_one_node(point, angles)
                for index in neighbors_indices:
                    node_link = tree.data[index]
                    self.add_one_edge([point, node_link])

    def verify_path_in_graph(self, path):
        missing_nodes = []
        if self.has_node(path[0]) and self.has_node(path[-1]):
            if self.has_path(path[0], path[-1]):
                for point in path:
                    if not self.has_node(point) and point not in missing_nodes:
                        missing_nodes.append(point)
            else:
                print("NO PATH")
                missing_nodes = None
        else:
            print("NO INITAL OR FINAL NODE")
            print(path[0], path[-1])
            missing_nodes = None
        return missing_nodes

    def find_node_dependencies_in_objects(self, node):
        dependencies = []
        for key, value in self.objects_in_world.items():
            if value.has_node(node):
                dependencies.extend(value.nodes[node]["joint_dependency"])
        return dependencies

    def find_blocked_angles(self, missing_nodes):
        blocked_angles = []
        for node in missing_nodes:
            blocked_angles.extend(self.find_node_dependencies_in_objects(node))
        return blocked_angles

    def find_closest_path_in_graph(self, missing_nodes, original_path):
        new_path = []
        i = 0
        print(len(original_path))
        while i < len(original_path):
            node = original_path[i]
            if node not in missing_nodes:
                new_path.append(node)
                prev_node = node
                print("NODE EXISTS ", i, node)
            else:
                next_node = None
                for j in range(i + 1, len(original_path)):
                    if original_path[j] not in missing_nodes:
                        next_node = original_path[j]
                        print("NEXT NODE", j, next_node)
                        break
                    else:
                        print("MISSING NODE", j, node)
                if next_node is not None and self.has_path(prev_node, next_node):
                    sub_path = self.shortest_path(prev_node, next_node)
                    sub_path.pop(0)
                    sub_path = [self.get_node_attr(i, "value") for i in sub_path]
                    aux_path = [item.tolist() if isinstance(item, np.ndarray) else item for item in sub_path]
                    new_path.extend(aux_path)
                    prev_node = new_path[-1]
                    i = j
                else:
                    break
                print("NEW PATH ", i, aux_path)
            i += 1
        return new_path

    def find_new_path(self, missing_nodes, original_path):
        new_path = []
        nodes = self.get_nodes_values()
        kdtree = KDTree(np.array(nodes))
        i = 0
        while i < len(original_path):
            node = original_path[i]
            if node not in missing_nodes:
                new_path.append(node)
                prev_node = node
            else:
                closest_point = self.find_closest_point(node, kdtree)
                if self.has_path(prev_node, closest_point):
                    sub_path = self.shortest_path(prev_node, closest_point)
                    sub_path.pop(0)
                    sub_path = [self.get_node_attr(i, "value") for i in sub_path]
                    aux_path = [item.tolist() if isinstance(item, np.ndarray) else item for item in sub_path]
                    new_path.extend(aux_path)
                    prev_node = new_path[-1]
            i += 1
        return new_path

    def re_path_end_effector(self, missing_nodes, original_path):
        if missing_nodes == []:
            print("Path hasn't changed")
            return original_path
        elif missing_nodes == None:
            print("Impossible to execute the path")
            return None
        else:
            return self.find_new_path(missing_nodes, original_path)

    def gauss_newton_angle_estimation(self, robot, initial_angles, cartesian_point):

        def objective(angles, *args):
            cartesian_point = args[0]
            robot = args[1]
            # Retrieve the forward kinematics function based on the robot instance
            forward_kinematics = getattr(robot, 'forward_kinematics_' + robot.robotName)
            angle_list_len = len(robot.physical_limits_left) + len(robot.physical_limits_right) + len(robot.physical_limits_head)
            angles_zeros = np.zeros((angle_list_len))
            angles = np.deg2rad(np.array(angles))
            if 'Left' in self.key:
                insert_index = 0
            elif 'Right' in self.key:
                insert_index = len(robot.physical_limits_left)
            elif 'Head' in self.key:
                insert_index = len(robot.physical_limits_left) + len(robot.physical_limits_right)
            result_array = np.concatenate([angles_zeros[:insert_index], angles, angles_zeros[insert_index + len(angles):]])
            # Calculate the Euclidean distance between the predicted and observed Cartesian positions
            left_arm, right_arm, head = forward_kinematics(result_array)
            if 'Left' in self.key:
                observed_positions = left_arm[int(self.key[-1])]
            elif 'Right' in self.key:
                observed_positions = right_arm[int(self.key[-1])]
            elif 'Head' in self.key:
                observed_positions = head[int(self.key[-1])]
            distance = np.linalg.norm(observed_positions - cartesian_point)
            return distance

        # Perform optimization to minimize the objective function
        result = minimize(objective, list(initial_angles.values()), args=(cartesian_point, robot), method='L-BFGS-B')

        # The optimized angles
        optimized_angles = {f'angle_{i+1}': round(result.x[i], 2) for i in range(len(result.x))}
        return optimized_angles
