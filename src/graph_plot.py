from cgi import print_arguments
from cmath import sqrt
from tkinter import Grid
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import world_graph
import robot_graph
import itertools
from scipy.spatial import KDTree
import robot
import json
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
import random
from sklearn.metrics import mean_squared_error
import copy
import pose_prediction

plt.rcParams.update({'font.size': 18})

def calculateEdgeLengths(connections):
    edge_lengths = np.zeros((len(connections),))
    for i in range(len(connections)):
        edge_lengths[i] = calculateEdgeLength(connections[i])
    return edge_lengths

def calculateEdgeLength(vertices_pair):
    return np.linalg.norm(vertices_pair[0] - vertices_pair[1])

def calculateMissingNode(vertice_1, vertice_2, vertice_3, distance):
    centroid = (vertice_1 + vertice_2 + vertice_3) / 3.0
    cross_product = np.cross((vertice_1 - vertice_2), (vertice_3 - vertice_2))
    cross_product /= np.linalg.norm(cross_product)
    fourth_point_1 = cross_product * distance + centroid
    fourth_point_2 = -cross_product * distance + centroid
    return fourth_point_1, fourth_point_2

def isInsideWorld(lowEdge, highEdge, point):
    world = np.column_stack((lowEdge, highEdge))
    world = [np.sort(i) for i in world]
    isInside = [True for i in range(len(point)) if world[i][0] <= point[i] <= world[i][1]]
    if len(isInside) == 3:
        return True
    else:
        return False

def findEdgesOptimized(graph, length):
    nodes = graph.get_nodes()
    points = np.array(nodes)
    tree = cKDTree(points)
    max_dist = 2 * length #np.sqrt(3) * length
    edges = []
    for i, point in tqdm(enumerate(points)):
        # Find neighbors within the maximum distance
        neighbors_indices = tree.query_ball_point(point, max_dist)
        for neighbor_index in neighbors_indices:
            if neighbor_index > i:
                if i < len(nodes) and neighbor_index < len(nodes):
                    edges.append((points[i], points[neighbor_index]))
                else:
                    print("Invalid indices:", i, neighbor_index)
    return edges

def find_edges_optimized_robot(graph, length):
    nodes = graph.get_nodes()
    points = [graph.get_node_attr(node, "value") for node in nodes]
    tree = cKDTree(points)
    max_dist = 2 * length #np.sqrt(3) * length
    edges = []
    for i, point in enumerate(points):
        # Find neighbors within the maximum distance
        neighbors_indices = tree.query_ball_point(point, max_dist)
        for neighbor_index in neighbors_indices:
            if neighbor_index > i:
                if i < len(nodes) and neighbor_index < len(nodes):
                    edges.append((points[i], points[neighbor_index]))
                else:
                    print("Invalid indices:", i, neighbor_index)
    return edges

def createWorldCubes(lowEdge, highEdge, length):
    graph_world = world_graph.Graph()
    world = np.column_stack((lowEdge, highEdge))
    world = [np.sort(i) for i in world]
    for x in np.arange(world[0][0], world[0][1], length):
        for y in np.arange(world[1][0], world[1][1], length):
            for z in np.arange(world[2][0], world[2][1], length):
                node = np.around([x, y, z], decimals = 2)  # previously without list
                graph_world.add_one_node(node)
    edges = findEdgesOptimized(graph_world, length)
    graph_world.add_edges(edges)
    return graph_world

def find_closest_point(new_point, kdtree):
    """
    Find the closest point from the KD-tree to the new point.

    Parameters:
        new_point (numpy.ndarray): The new 3D point represented as a NumPy array of shape (3,).
        kdtree (scipy.spatial.KDTree): The KD-tree built from the set of random points.

    Returns:
        numpy.ndarray: The closest point from the set to the new point.
    """
    distance, index = kdtree.query(new_point)
    return kdtree.data[index]

def findRobotWorld(demonstration, length, world):
    kdtree = KDTree(world.get_nodes())
    robotWorld = world_graph.Graph()
    for i in demonstration:
        node = find_closest_point(i, kdtree)
        robotWorld.add_one_node(node)
    edges = findEdgesOptimized(robotWorld, length)
    robotWorld.add_edges(edges)
    return robotWorld

def findJointsGraph(cartesian_points, length, world, robot, joint_angles):
    kdtree = KDTree(world.get_nodes())
    for key in cartesian_points:
        if "0" in key or "1" in key:
            node = cartesian_points[key]
            dependencies = []
        else:
            node = find_closest_point(cartesian_points[key], kdtree)
            dependencies = slice_dict(joint_angles, key.split('_'))
        robot[key].set_attribute(node, dependencies)
        # robot[key].add_one_node(node, att)  # delete the str to use with gml
        edges = find_edges_optimized_robot(robot[key], length)
        robot[key].add_edges(edges)
    return robot

def slice_dict(dict, details):
    sub_list = []
    for i in dict:
        if details[0] in i and len(sub_list) <= int(details[1]) - 2:
            sub_list.append(dict[i])
    return sub_list

def createRobotGraphs(robot):
    joint_dic = {}
    for i in range(len(robot.leftArmAngles)):
        joint_dic["jointLeft_" + str(i)] = robot_graph.Graph(i, "jointLeft_" + str(i))

    for i in range(len(robot.rightArmAngles)):
        joint_dic["jointRight_" + str(i)] = robot_graph.Graph(i, "jointRight_" + str(i))

    for i in range(len(robot.headAngles)):
        joint_dic["jointHead_" + str(i)] = robot_graph.Graph(i, "jointHead_" + str(i))
    return joint_dic

def pathPlanning(demonstration, robotWorld):
    kdtree = KDTree(robotWorld.get_nodes_values())
    path = []
    demonstration = list(demonstration)
    prev_node = tuple(find_closest_point(demonstration.pop(0), kdtree))
    path.append(prev_node)
    for i in demonstration:
        node = tuple(find_closest_point(i, kdtree))
        sub_path = robotWorld.search(prev_node, node, "A*")
        sub_path.pop(0)
        path.extend(sub_path)
        prev_node = node
    return path

def path_planning(demonstration, graph):
    kdtree = KDTree(graph.get_nodes_values())
    path = []
    path_for_error = []
    demonstration = list(demonstration)
    prev_node = find_closest_point(demonstration.pop(0), kdtree)
    path.append(tuple(prev_node))
    for i in demonstration:
        node = find_closest_point(i, kdtree)
        # graph.print_graph()
        if graph.has_path(prev_node, node):
            sub_path = graph.shortest_path(prev_node, node)
            sub_path.pop(0)
            sub_path = [graph.get_node_attr(i, "value") for i in sub_path]
            aux_path = [item.tolist() if isinstance(item, np.ndarray) else item for item in sub_path]
            path.extend(aux_path)
        else:
            path.extend([node])
        prev_node = node
        path_for_error.append(tuple(prev_node))
    MSE, RMSE = error_calculation(demonstration, path_for_error)
    return path, MSE, RMSE

def plot_error(Dict):
    # Extracting data for plotting
    joints = list(Dict.keys())
    values_MSE = [sub_dict["MSE"] for sub_dict in Dict.values()]
    values_RMSE = [sub_dict["RMSE"] for sub_dict in Dict.values()]

    # Plotting
    bar_width = 0.35
    index = range(len(joints))

    fig, ax = plt.subplots()
    bar1 = ax.bar(index, values_MSE, bar_width, label='MSE')
    bar2 = ax.bar([i + bar_width for i in index], values_RMSE, bar_width, label='RMSE')

    # Adding labels and title
    ax.set_xlabel('Joints')
    ax.set_ylabel('Error')
    ax.set_title('Values of MSE and RMSE for each Joint')
    ax.set_xticks([i + bar_width/2 for i in index])
    ax.set_xticklabels(joints,  rotation=0, ha='center')
    ax.legend()
    plt.grid()
    # Show the plot
    plt.show()

def error_calculation(y_true, y_pred):
    RMSE = mean_squared_error(y_true, y_pred, squared = False)
    MSE = mean_squared_error(y_true, y_pred, squared = True)
    return MSE, RMSE

def approximateTrajectory(demonstration, robot_joint_graph):
    kdtree = KDTree(robot_joint_graph.get_nodes_values())
    trajectory = []
    for i in demonstration:
        node = find_closest_point(i, kdtree)
        trajectory.append(node)
    return trajectory

def personalised_random_points(robot_joint_graph):
    amount_points = 20
    kdtree = robot_joint_graph.get_nodes_values()
    df = pd.DataFrame(kdtree)
    zdata = np.linspace(df[2].min(), df[2].max(), amount_points)
    xdata = np.average(df[0]) + (df[0].max() - df[0].min()) * np.sin(25 * zdata) /3
    ydata = np.average(df[1]) + (df[1].max() - df[1].min()) * np.cos(25 * zdata) /3
    vector = np.hstack((xdata, ydata, zdata))
    vector = np.reshape(vector, (amount_points, 3), order='F')
    return vector

def plotPath(key, demonstration, path, candidates = []):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xdem = demonstration[:, 0]
    ydem = demonstration[:, 1]
    zdem = demonstration[:, 2]
    xpath = path[:, 0]
    ypath = path[:, 1]
    zpath = path[:, 2]

    # Scatter plots for the points
    ax.scatter3D(xdem[1:-1], ydem[1:-1], zdem[1:-1], c='blue', marker='o', label='User demonstration')
    ax.scatter3D(xdem[0], ydem[0], zdem[0], c='green', marker='*')
    ax.scatter3D(xdem[-1], ydem[-1], zdem[-1], c='red', marker='*')

    ax.scatter3D(xpath[1:-1], ypath[1:-1], zpath[1:-1], c='pink', marker='o', label='Path planning')
    ax.scatter3D(xpath[0], ypath[0], zpath[0], c='green', marker='*', label='Starting point')
    ax.scatter3D(xpath[-1], ypath[-1], zpath[-1], c='red', marker='*', label='Ending point')

    if len(candidates):
        xdata = candidates[:, 0]
        ydata = candidates[:, 1]
        zdata = candidates[:, 2]
        ax.scatter3D(xdata, ydata, zdata, c='purple', label='Candidates')

    # Plot edges between the points
    for i in range(len(xdem) - 1):
        ax.plot([xdem[i], xdem[i+1]], [ydem[i], ydem[i+1]], [zdem[i], zdem[i+1]], c='grey')
    for i in range(len(xpath) - 1):
        ax.plot([xpath[i], xpath[i+1]], [ypath[i], ypath[i+1]], [zpath[i], zpath[i+1]], c='lightblue')

    ax.set_title(key)
    ax.set_xlabel("\n X [mm]", linespacing=3.2)
    ax.set_ylabel("\n Y [mm]", linespacing=3.2)
    ax.set_zlabel("\n Z [mm]", linespacing=3.2)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend()
    fig.tight_layout()
    # plt.savefig(key + "_world_path.pdf", format="pdf")
    plt.show()

def printRobotGraphs(robot):
    for key in robot:
        print(key, ": ")
        robot[key].print_graph()

def readTxtFile(file_path):
    with open(file_path) as f:
        contents = f.read()
    return json.loads(contents)

def divideFrames(data):
    sub_frame = []
    for key in data:
        frame = [np.array([el]) for el in data[key]]
        sub_frame.append(frame)
        new_data = [np.array([0]) for el in data[key]]
    new_data = np.array(new_data)
    for frame in sub_frame:
        new_data = np.hstack((new_data, np.array(frame)))
    return new_data[:,1:]

def forward_kinematics_n_frames(name, robot, joint_angles):
    left = []
    right = []
    head = []
    for frame in joint_angles:
        frame = np.radians(np.array(frame))
        if name == "qt":
            pos_left, pos_right, pos_head = robot.forward_kinematics_qt(frame)
        elif name == "nao":
            pos_left, pos_right, pos_head = robot.forward_kinematics_nao(frame)
        elif name == "gen3":
            pos_left, pos_right, pos_head = robot.forward_kinematics_kinova(frame)
        left.append(copy.copy(pos_left))
        right.append(copy.copy(pos_right))
        head.append(copy.copy(pos_head))
    return left, right, head

def read_babbling(path_name):
    points = readTxtFile("./data/" + path_name)
    keys = list(points)
    joint_angles = points[keys[0]]
    for count in range(1, len(keys)):
        joint_angles = np.hstack((joint_angles, points[keys[count]]))
    return joint_angles

def learn_environment(robot_graphs, graph_world, cartesian_points, joint_angles_dict):
    for i in range(len(joint_angles_dict)):
        robot_graphs = findJointsGraph(cartesian_points[i], length, graph_world, robot_graphs, joint_angles_dict[i])
    return robot_graphs

def angle_interpolation(joint_angles_list):
    new_joint_angles_list = []
    for count in range(len(joint_angles_list) - 1):
        diff = joint_angles_list[count + 1] - joint_angles_list[count]
        iterations = np.max(np.abs(diff))
        delta = diff / iterations
        actual_angle = joint_angles_list[count]
        for i in range(int(iterations)):
            new_joint_angles_list.append(copy.copy(actual_angle))
            actual_angle += delta
    new_joint_angles_list.append(joint_angles_list[-1])
    return np.array(new_joint_angles_list)

def list_to_string(vec):
        modified_vector = [0.0 if value in {0, 0., -0., -0.0, -0} else value for value in vec]
        vector_str = '[' + ', '.join(map(str, modified_vector)) + ']'
        return vector_str

def find_object_in_world(world, new_object):
    kdtree = cKDTree(world.get_nodes())
    object_nodes = []
    for node in new_object.get_nodes():
        object_nodes.append(list_to_string(find_closest_point(node, kdtree)))
    return object_nodes

def find_trajectory_in_world(world, tra):
    kdtree = cKDTree(world.get_nodes())
    world_nodes = []
    for node in tra:
        closest_point = find_closest_point(node, kdtree)
        if world_nodes:
            if (world_nodes[-1] != closest_point).any():
                world_nodes.append(closest_point)
        else:
            world_nodes.append(closest_point)
    return world_nodes

def read_library_from_file(name):
    try:
        with open(name + "_data.json", "r") as jsonfile:
            data = [json.loads(line) for line in jsonfile.readlines()]
            return data
    except FileNotFoundError:
        print(f"File {name}_data.json not found.")
        return []

def extract_action_from_library(actionName, library):
    robot_pose = {}
    for key in library:
        for item in library[key]:
            if item["id"] == actionName:
                robot_pose[key] = item["path"]
    return robot_pose

def extract_angles_from_library(actionName, library):
    for key in library:
        for item in library[key]:
            if item["id"] == actionName:
                print(key, item["joint_dependency"])

if __name__ == "__main__":

    # flag = "path-planning"
    # flag = "pose-predicition"
    # flag = "explore-world"
    # flag = "object-in-robot-graph"
    flag = "read_library_paths"

    #Read robot configuration from the .yaml filerobot_graphs
    robotName = 'qt'
    file_path = "./robot_configuration_files/" + robotName + ".yaml"
    qt = robot.Robot()
    qt.import_robot(file_path)
    robot_graphs = createRobotGraphs(qt)
    length = 10

    if flag == "explore-world":
        # Define parameter of the world and create one with cubic structure
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        graph_world = createWorldCubes(lowEdge, highEdge, length)
        # graph_world.save_graph_to_file("test")
        # graph_world = world_graph.Graph()
        # graph_world.read_graph_from_file("test")
        # graph_world.plot_graph()
        babbing_path = "self_exploration_qt_100.txt"
        joint_angles = read_babbling(babbing_path)
        new_list = angle_interpolation(joint_angles)
        print("AFTER INTERPOLATION")
        pos_left, pos_right, pos_head = forward_kinematics_n_frames(robotName, qt, new_list)
        print("AFTER FK")
        # s = simulate_position.RobotSimulation(pos_left, pos_right, pos_head)
        # # s.animate()
        cartesian_points = qt.pos_mat_to_robot_mat_dict(pos_left, pos_right, pos_head)
        joint_angles_dict = qt.angular_mat_to_mat_dict(new_list)
        robot_world = learn_environment(robot_graphs, graph_world, cartesian_points, joint_angles_dict)
        print("BEFORE SAVING THE GRAPHS")
        for key in tqdm(robot_world):
            robot_world[key].save_graph_to_file(key)
            # robot_world[key].read_graph_from_file(key)
            # robot_world[key].plot_graph()

    elif flag == "create-object":
        lowEdge = np.array([-60, 100, 140])
        highEdge = np.array([50, 170, 160])
        object = createWorldCubes(lowEdge, highEdge, length)
        object.save_object_to_file("object_1")
        # object.read_object_from_file("object_1")
        object.plot_graph()

    elif flag == "object-in-robot-graph":
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        graph_world = createWorldCubes(lowEdge, highEdge, length)
        object = world_graph.Graph()
        object.read_object_from_file("object_1")
        object_nodes = find_object_in_world(graph_world, object)
        for key in robot_graphs:
            robot_graphs[key].read_graph_from_file(key)
            print(key)
            robot_graphs[key].new_object_in_world(object_nodes, "object_1")
            robot_graphs[key].remove_object_from_world("object_1")

    elif flag == "reproduce-action":
        #reading an action and reproducing
        frames = readTxtFile("./data/angles.txt")
        joint_angles = divideFrames(frames)

    elif flag == "path-planning":
        dict_error = {}
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        graph_world = createWorldCubes(lowEdge, highEdge, length)

        for key in robot_graphs:
            robot_graphs[key].read_graph_from_file(key)
            generated_trajectory = personalised_random_points(robot_graphs[key])
            # robot_graphs[key].plot_graph(generated_trajectory)
            # tra = approximateTrajectory(generated_trajectory, robot_graphs[key])
            # print("GENERATED TRAJECTORY: ", generated_trajectory)
            if robot_graphs[key].number_of_nodes() > 1:
                new_tra_world = find_trajectory_in_world(graph_world, generated_trajectory)
                # print(key)
                candidate_nodes = robot_graphs[key].find_trajectory_shared_nodes(new_tra_world)
                robot_graphs[key].adding_candidates_to_graph(length, candidate_nodes)
                # new_tra = robot_graphs[key].find_trajectory_in_graph(generated_trajectory)
                # neighbors_candidates = robot_graphs[key].find_neighbors_candidates(candidate_nodes)
                tra, MSE, RMSE = path_planning(generated_trajectory, robot_graphs[key]) #generated_trajectory
                plotPath(key, generated_trajectory, np.asarray(tra))
                dict_error[key] = {"MSE": MSE, "RMSE": RMSE}
        plot_error(dict_error)

    elif flag == "pose-predicition":
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        graph_world = createWorldCubes(lowEdge, highEdge, length)

        file_path = "./robot_configuration_files/"+ robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, robotName)
        file_name = "robot_angles_" + robotName
        theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
        pose_predictor.train_pytorch(pose_predictor.robot, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, 1000)
        df = pose_predictor.read_file("combined_actions")
        action = "knife"
        user = 3
        actions = ['teacup', 'teapot', 'spoon', 'ladle', 'shallow_plate',
               'dinner_plate', 'knife', 'fork', 'salt_shaker',
               'sugar_bowl', 'mixer', 'pressure_cooker']
        users = np.arange(1, 21, 1)
        for user in users:
            for action in actions:
                dict_error = {}
                robot_pose = []
                left_side, right_side, head = pose_predictor.read_csv_combined(df, action, user)
                left_side = left_side * 1000
                right_side = right_side * 1000
                head = head * 1000
                angles_left_vec = []
                angles_right_vec = []
                angles_head_vec = []
                cartesian_left_vec = []
                cartesian_right_vec = []
                cartesian_head_vec = []

                for i in range(len(left_side)):
                    angles_left, angles_right, angles_head = pose_predictor.predict_pytorch(left_side[i], right_side[i], head[i])
                    angles_left_vec.append(angles_left)
                    angles_right_vec.append(angles_right)
                    angles_head_vec.append(angles_head)

                    points4, points5, points6 = pose_predictor.robot_embodiment(angles_left, angles_right, angles_head)
                    cartesian_left_vec.append(points4)
                    cartesian_right_vec.append(points5)
                    cartesian_head_vec.append(points6)
                    robot_pose.append((left_side[i], right_side[i], head[i], points4, points5, points6))

                pose_predictor.mat_to_dict_per_joint(cartesian_left_vec, cartesian_right_vec, cartesian_head_vec)

                for key in robot_graphs:
                    robot_graphs[key].read_graph_from_file(key)
                    # if robot_graphs[key].number_of_nodes() > 1:
                    generated_trajectory = pose_predictor.robot.robotDict[key]
                    new_tra_world = find_trajectory_in_world(graph_world, generated_trajectory)
                    #path planning
                    candidate_nodes = robot_graphs[key].find_trajectory_shared_nodes(new_tra_world)
                    robot_graphs[key].adding_candidates_to_graph(length, candidate_nodes)
                    tra, MSE, RMSE = path_planning(generated_trajectory, robot_graphs[key])
                    dep = robot_graphs[key].select_joint_dependencies(tra)
                    robot_graphs[key].save_path_in_library(tra, dep)
                    robot_graphs[key].save_graph_to_file(key)
                        # plotPath(user, generated_trajectory, np.asarray(tra), np.asarray(candidate_nodes)) # key instead of user + action
                        # dict_error[key] = {"MSE": MSE, "RMSE": RMSE}
                # plot_error(dict_error)

    elif flag == "read_library_paths":
        lib_dict = {}
        file_path = "./robot_configuration_files/"+ robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, robotName)
        for key in robot_graphs:
            robot_graphs[key].read_graph_from_file(key)
            lib_dict[key] = read_library_from_file(key)
        dict_pose = extract_action_from_library(10, lib_dict)

        for key in dict_pose:
            print(robot_graphs[key].verify_path_in_graph(dict_pose[key]))
            plotPath(key, np.asarray(dict_pose[key]), np.asarray(dict_pose[key]))