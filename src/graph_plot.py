from tkinter import Grid
import matplotlib.pyplot as plt
import numpy as np
import world_graph
import robot_graph
from scipy.spatial import KDTree
import robot
import json
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
import copy
import pose_prediction
import simulate_position

plt.rcParams.update({'font.size': 18})

class PathPlanning:
    def __init__(self, delta = 10):
        self.robotName = 'qt'
        file_path = "./robot_configuration_files/" + self.robotName + ".yaml"
        self.myRobot = robot.Robot(self.robotName)
        self.myRobot.import_robot(file_path)
        self.robot_graphs = self.createRobotGraphs()
        self.length = delta
        self.find_end_effectors_keys()

    def findEdgesOptimized(self, graph):
        nodes = graph.get_nodes()
        points = np.array(nodes)
        tree = cKDTree(points)
        max_dist = 2 * self.length #np.sqrt(3) * length
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

    def find_edges_optimized_robot(self, graph):
        nodes = graph.get_nodes()
        points = [graph.get_node_attr(node, "value") for node in nodes]
        tree = cKDTree(points)
        max_dist = 2 * self.length #np.sqrt(3) * length
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

    def createWorldCubes(self, lowEdge, highEdge):
        self.graph_world = world_graph.Graph()
        world = np.column_stack((lowEdge, highEdge))
        world = [np.sort(i) for i in world]
        for x in np.arange(world[0][0], world[0][1], self.length):
            for y in np.arange(world[1][0], world[1][1], self.length):
                for z in np.arange(world[2][0], world[2][1], self.length):
                    node = np.around([x, y, z], decimals = 2)  # previously without list
                    self.graph_world.add_one_node(node)
        edges = self.findEdgesOptimized(self.graph_world)
        self.graph_world.add_edges(edges)

    def createObject(self, lowEdge, highEdge):
        new_object = world_graph.Graph()
        world = np.column_stack((lowEdge, highEdge))
        world = [np.sort(i) for i in world]
        for x in np.arange(world[0][0], world[0][1], self.length):
            for y in np.arange(world[1][0], world[1][1], self.length):
                for z in np.arange(world[2][0], world[2][1], self.length):
                    node = np.around([x, y, z], decimals = 2)  # previously without list
                    new_object.add_one_node(node)
        edges = self.findEdgesOptimized(new_object)
        new_object.add_edges(edges)
        return new_object

    def find_closest_point(self, new_point, kdtree):
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

    def findRobotWorld(self, demonstration, world):
        kdtree = KDTree(world.get_nodes())
        robotWorld = world_graph.Graph()
        for i in demonstration:
            node = self.find_closest_point(i, kdtree)
            robotWorld.add_one_node(node)
        edges = self.findEdgesOptimized(robotWorld)
        robotWorld.add_edges(edges)
        return robotWorld

    def findJointsGraph(self, cartesian_points, joint_angles):
        kdtree = KDTree(self.graph_world.get_nodes())
        for key in cartesian_points:
            if "0" in key or "1" in key:
                node = cartesian_points[key]
                dependencies = []
            else:
                node = self.find_closest_point(cartesian_points[key], kdtree)
                dependencies = self.slice_dict(joint_angles, key.split('_'))
            self.robot_graphs[key].set_attribute(node, dependencies)
            # robot[key].add_one_node(node, att)  # delete the str to use with gml
            edges = self.find_edges_optimized_robot(self.robot_graphs[key])
            self.robot_graphs[key].add_edges(edges)
        return self.robot_graphs

    def slice_dict(self, dict, details):
        sub_list = []
        for i in dict:
            if details[0] in i and len(sub_list) <= int(details[1]) - 2:
                sub_list.append(dict[i])
        return sub_list

    def createRobotGraphs(self):
        joint_dic = {}
        for i in range(len(self.myRobot.leftArmAngles)):
            joint_dic["jointLeft_" + str(i)] = robot_graph.Graph(i, "jointLeft_" + str(i))

        for i in range(len(self.myRobot.rightArmAngles)):
            joint_dic["jointRight_" + str(i)] = robot_graph.Graph(i, "jointRight_" + str(i))

        for i in range(len(self.myRobot.headAngles)):
            joint_dic["jointHead_" + str(i)] = robot_graph.Graph(i, "jointHead_" + str(i))
        return joint_dic

    # def path_planning(demonstration, graph):
    #     kdtree = KDTree(graph.get_nodes_values())
    #     path = []
    #     path_for_error = []
    #     demonstration = list(demonstration)
    #     prev_node = find_closest_point(demonstration.pop(0), kdtree)
    #     path.append(tuple(prev_node))
    #     for i in demonstration:
    #         node = find_closest_point(i, kdtree)
    #         if graph.has_path(prev_node, node):
    #             sub_path = graph.shortest_path(prev_node, node)
    #             sub_path.pop(0)
    #             sub_path = [graph.get_node_attr(i, "value") for i in sub_path]
    #             aux_path = [item.tolist() if isinstance(item, np.ndarray) else item for item in sub_path]
    #             path.extend(aux_path)

    #         else:
    #             path.extend([node])
    #             path.append([node])
    #         prev_node = node
    #         path_for_error.append(tuple(prev_node))
    #     MSE, RMSE = error_calculation(demonstration, path_for_error)
    #     return path, MSE, RMSE

    def error_between_points(self, point1, point2, max_distance):
        anomaly_direction = point2 - point1
        anomaly_magnitude = np.linalg.norm(anomaly_direction)
        # Ensure the anomaly is not too close to the previous point
        if anomaly_magnitude > max_distance:
            anomaly_direction /= anomaly_magnitude  # Normalize the direction
            new_point = point1 + max_distance * anomaly_direction
        else:
            new_point = point2
        return new_point

    def path_planning(self, demonstration, graph, max_distance = 30):
        kdtree = KDTree(graph.get_nodes_values())
        path = []
        first_node = demonstration[0]
        prev_node = self.find_closest_point(first_node, kdtree)
        path.append(tuple(prev_node))
        for i in range(1, len(demonstration)):
            candidate_node = self.error_between_points(prev_node, demonstration[i], max_distance)
            node = self.find_closest_point(candidate_node, kdtree)
            path.append(tuple(node))
            prev_node = node
        MSE, RMSE = self.error_calculation(demonstration, path)
        return path, MSE, RMSE

    def path_planning_online(self, demonstration, graph, max_distance = 30):
        kdtree = KDTree(graph.get_nodes_values())
        path = []
        first_node = demonstration[0]
        prev_node = self.find_closest_point(first_node, kdtree)
        path.append(prev_node.tolist())
        for i in range(1, len(demonstration)):
            candidate_node = self.error_between_points(prev_node, demonstration[i], max_distance)
            node = self.find_closest_point(candidate_node, kdtree)
            path.append(node)
            prev_node = node
        path_angles = graph.select_joint_dependencies(path[0])
        return path_angles[-1]

    def plot_error(self, Dict):
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

    def error_calculation(self, y_true, y_pred):
        RMSE = mean_squared_error(y_true, y_pred, squared = False)
        MSE = mean_squared_error(y_true, y_pred, squared = True)
        return MSE, RMSE

    def approximateTrajectory(self, demonstration, robot_joint_graph):
        kdtree = KDTree(robot_joint_graph.get_nodes_values())
        trajectory = []
        for i in demonstration:
            node = self.find_closest_point(i, kdtree)
            trajectory.append(node)
        return trajectory

    def personalised_random_points(self, robot_joint_graph):
        amount_points = 20
        kdtree = robot_joint_graph.get_nodes_values()
        df = pd.DataFrame(kdtree)
        zdata = np.linspace(df[2].min(), df[2].max(), amount_points)
        xdata = np.average(df[0]) + (df[0].max() - df[0].min()) * np.sin(25 * zdata) /3
        ydata = np.average(df[1]) + (df[1].max() - df[1].min()) * np.cos(25 * zdata) /3
        vector = np.hstack((xdata, ydata, zdata))
        vector = np.reshape(vector, (amount_points, 3), order='F')
        return vector

    def plotPath(self, key, demonstration, path, candidates = []):
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

    def printRobotGraphs(self, robot):
        for key in robot:
            print(key, ": ")
            robot[key].print_graph()

    def readTxtFile(self, file_path):
        with open(file_path) as f:
            contents = f.read()
        return json.loads(contents)

    def divideFrames(self, data):
        sub_frame = []
        for key in data:
            frame = [np.array([el]) for el in data[key]]
            sub_frame.append(frame)
            new_data = [np.array([0]) for el in data[key]]
        new_data = np.array(new_data)
        for frame in sub_frame:
            new_data = np.hstack((new_data, np.array(frame)))
        return new_data[:,1:]

    def forward_kinematics_n_frames(self, joint_angles):
        left = []
        right = []
        head = []
        for frame in joint_angles:
            frame = np.radians(np.array(frame))
            if self.robotName == "qt":
                pos_left, pos_right, pos_head = self.myRobot.forward_kinematics_qt(frame)
            elif self.robotName == "nao":
                pos_left, pos_right, pos_head = self.myRobot.forward_kinematics_nao(frame)
            elif self.robotName == "gen3":
                pos_left, pos_right, pos_head = self.myRobot.forward_kinematics_gen3(frame)
            left.append(copy.copy(pos_left))
            right.append(copy.copy(pos_right))
            head.append(copy.copy(pos_head))
        return left, right, head

    def read_babbling(self, path_name):
        points = self.readTxtFile("./data/" + path_name)
        keys = list(points)
        joint_angles = points[keys[0]]
        for count in range(1, len(keys)):
            joint_angles = np.hstack((joint_angles, points[keys[count]]))
        return joint_angles

    def learn_environment(self, cartesian_points, joint_angles_dict):
        for i in range(len(joint_angles_dict)):
            self.robot_graphs = self.findJointsGraph(cartesian_points[i], joint_angles_dict[i])
        return self.robot_graphs

    def angle_interpolation(self, joint_angles_list):
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

    def list_to_string(self, vec):
            modified_vector = [0.0 if value in {0, 0., -0., -0.0, -0} else value for value in vec]
            vector_str = '[' + ', '.join(map(str, modified_vector)) + ']'
            return vector_str

    def find_object_in_world(self, new_object):
        kdtree = cKDTree(self.graph_world.get_nodes())
        object_nodes = []
        object_nodes_str = []
        for node in new_object.get_nodes():
            closest = self.find_closest_point(node, kdtree)
            object_nodes_str.append(self.list_to_string(closest))
            object_nodes.append(closest)
        return object_nodes, object_nodes_str

    def match_object_in_world(self, objectName):
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        self.createWorldCubes(lowEdge, highEdge)
        object = world_graph.Graph()
        object2 = world_graph.Graph()
        object_name = objectName
        object.read_object_from_file(object_name)
        object_nodes, _ = self.find_object_in_world(object)
        object2.add_nodes(object_nodes)
        object2.save_object_to_file(objectName + "_nodes")

    def find_trajectory_in_world(self, tra):
        kdtree = cKDTree(self.graph_world.get_nodes())
        world_nodes = []
        for node in tra:
            closest_point = self.find_closest_point(node, kdtree)
            if world_nodes:
                if (world_nodes[-1] != closest_point).any():
                    world_nodes.append(closest_point)
            else:
                world_nodes.append(closest_point)
        return world_nodes

    def read_library_from_file(self, name):
        try:
            with open("data/graphs/lib/" + name + "_" + self.robotName + "_data.json", "r") as jsonfile:
                data = [json.loads(line) for line in jsonfile.readlines()]
                return data
        except FileNotFoundError:
            print(f"File {name}_data.json not found.")
            return []

    def extract_action_from_library(self, actionName, library):
        robot_pose = {}
        for key in library:
            for item in library[key]:
                if item["id"] == actionName:
                    robot_pose[key] = item["path"]
        return robot_pose

    def extract_angles_from_library(self, actionName, library):
        robot_pose = {}
        for key in library:
            for item in library[key]:
                if item["id"] == actionName:
                    robot_pose[key] = item["joint_dependency"]
        return robot_pose

    def find_end_effectors_keys(self):
        left_numbers = [int(key.split('_')[1]) for key in self.robot_graphs if key.startswith('jointLeft')]
        right_numbers = [int(key.split('_')[1]) for key in self.robot_graphs if key.startswith('jointRight')]
        head_numbers = [int(key.split('_')[1]) for key in self.robot_graphs if key.startswith('jointHead')]

        max_left = 'jointLeft_' + str(max(left_numbers, default=0))
        max_right = 'jointRight_' + str(max(right_numbers, default=0))
        max_head = 'jointHead_' + str(max(head_numbers, default=0))
        self.end_effectors_keys = [max_left, max_right, max_head]

    def find_fk_from_dict(self, dict):
        result_list = []
        for position in range(max(len(joint_data) for joint_data in dict.values())):
            joint_angles = []
            for joint_name, joint_data in dict.items():
                if position < len(joint_data):
                    joint_angles.extend(list(joint_data[position].values()))
            result_list.append(joint_angles)
        pos_left, pos_right, pos_head = self.forward_kinematics_n_frames(result_list)
        s = simulate_position.RobotSimulation(pos_left, pos_right, pos_head)
        s.animate()

    def fill_robot_graphs(self):
        for key in self.robot_graphs:
            self.robot_graphs[key].read_graph_from_file(key, self.robotName)
            # self.robot_graphs[key].plot_graph(self.robotName)

if __name__ == "__main__":

    # flag = "path-planning"
    flag = "pose-predicition"
    # flag = "explore-world"
    # flag = "object-in-robot-graph"
    # flag = "read_library_paths"
    planner = PathPlanning()

    if flag == "explore-world":
        # Define parameter of the world and create one with cubic structure
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        planner.createWorldCubes(lowEdge, highEdge)
        # graph_world.save_graph_to_file("test")
        # graph_world = world_graph.Graph()
        # graph_world.read_graph_from_file("test")
        # graph_world.plot_graph()
        babbing_path = "self_exploration_qt_100.txt"
        joint_angles = planner.read_babbling(babbing_path)
        new_list = planner.angle_interpolation(joint_angles)
        print("AFTER INTERPOLATION")
        pos_left, pos_right, pos_head = planner.forward_kinematics_n_frames(new_list)
        print("AFTER FK")
        # s = simulate_position.RobotSimulation(pos_left, pos_right, pos_head)
        # # s.animate()
        cartesian_points = planner.myRobot.pos_mat_to_robot_mat_dict(pos_left, pos_right, pos_head)
        joint_angles_dict = planner.myRobot.angular_mat_to_mat_dict(new_list)
        robot_world = planner.learn_environment(cartesian_points, joint_angles_dict)
        print("BEFORE SAVING THE GRAPHS")
        for key in tqdm(robot_world):
            robot_world[key].save_graph_to_file(key, planner.robotName)
            # robot_world[key].read_graph_from_file(key, robotName)
            robot_world[key].plot_graph(planner.robotName)

    elif flag == "create-object":
        lowEdge = np.array([-60, 100, 140])
        highEdge = np.array([50, 170, 160])
        object = planner.createObject(lowEdge, highEdge)
        object.save_object_to_file("object_1")
        # object.read_object_from_file("object_1")
        object.plot_graph()

    elif flag == "object-in-robot-graph":
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        object = planner.createObject(lowEdge, highEdge)
        object = world_graph.Graph()
        object.read_object_from_file("object_1")
        _, object_nodes = planner.find_object_in_world(object)
        for key in planner.robot_graphs:
            planner.robot_graphs[key].read_graph_from_file(key, planner.robotName)
            print(key)
            planner.robot_graphs[key].new_object_in_world(object_nodes, "object_1")
            planner.robot_graphs[key].remove_object_from_world("object_1")

    elif flag == "reproduce-action":
        #reading an action and reproducing
        frames = planner.readTxtFile("./data/angles.txt")
        joint_angles = planner.divideFrames(frames)

    elif flag == "path-planning":
        dict_error = {}
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        planner.createWorldCubes(lowEdge, highEdge)

        for key in planner.robot_graphs:
            planner.robot_graphs[key].read_graph_from_file(key, planner.robotName)
            generated_trajectory = planner.personalised_random_points(planner.robot_graphs[key])
            # robot_graphs[key].plot_graph(generated_trajectory)
            # tra = approximateTrajectory(generated_trajectory, robot_graphs[key])
            # print("GENERATED TRAJECTORY: ", generated_trajectory)
            if planner.robot_graphs[key].number_of_nodes() > 1:
                new_tra_world = planner.find_trajectory_in_world(generated_trajectory)
                # print(key)
                candidate_nodes = planner.robot_graphs[key].find_trajectory_shared_nodes(new_tra_world)
                planner.robot_graphs[key].adding_candidates_to_graph(planner.myRobot, planner.length, candidate_nodes)
                # new_tra = robot_graphs[key].find_trajectory_in_graph(generated_trajectory)
                tra, MSE, RMSE = planner.path_planning(generated_trajectory, planner.robot_graphs[key])
                planner.plotPath(key, generated_trajectory, np.asarray(tra))
                dict_error[key] = {"MSE": MSE, "RMSE": RMSE}
        planner.plot_error(dict_error)

    elif flag == "pose-predicition":
        lowEdge = np.array([-450, -450, 100])
        highEdge = np.array([450, 450, 700])
        planner.createWorldCubes(lowEdge, highEdge)

        file_path = "./robot_configuration_files/"+ planner.robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, planner.robotName)
        file_name = "robot_angles_" + planner.robotName
        theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
        pose_predictor.train_pytorch(pose_predictor.robot, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, 1000)
        # df = pose_predictor.read_file("combined_actions")
        # actions = ['teacup', 'teapot', 'spoon', 'ladle', 'shallow_plate', 'dinner_plate', 'knife', 'fork', 'salt_shaker', 'sugar_bowl', 'mixer', 'pressure_cooker']
        actions = ['arm_sides']
        # users = np.arange(1, 21, 1)
        users = [21]
        planner.fill_robot_graphs()

        #for new actions
        df = pose_predictor.read_file("/QT_recordings/human/arms_sides_2")

        for user in tqdm(users):
            for action in actions:
                dict_error = {}
                robot_pose = []
                # left_side, right_side, head, time = pose_predictor.read_csv_combined(df, action, user)
                # left_side = left_side * 1000
                # right_side = right_side * 1000
                # head = head * 1000

                # for new actions
                left_side, right_side, head, time = pose_predictor.read_recorded_action_csv(df, action, user)

                angles_left_vec = []
                angles_right_vec = []
                angles_head_vec = []
                cartesian_left_vec = []
                cartesian_right_vec = []
                cartesian_head_vec = []
                actionName = str(user) + action

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

                for key in planner.robot_graphs:
                    generated_trajectory = pose_predictor.robot.robotDict[key]
                    new_tra_world = planner.find_trajectory_in_world(generated_trajectory)
                    #path planning
                    candidate_nodes = planner.robot_graphs[key].find_trajectory_shared_nodes(new_tra_world)
                    planner.robot_graphs[key].adding_candidates_to_graph(planner.myRobot, planner.length, candidate_nodes)
                    tra, MSE, RMSE = planner.path_planning(generated_trajectory, planner.robot_graphs[key])
                    dep = planner.robot_graphs[key].select_joint_dependencies(tra)
                    planner.robot_graphs[key].save_path_in_library(tra, dep, planner.robotName, actionName)
                    # planner.plotPath(actionName, generated_trajectory, np.asarray(tra)) # key instead of user + action
                    # dict_error[key] = {"MSE": MSE, "RMSE": RMSE}
                # plot_error(dict_error)
        for key in planner.robot_graphs:
            planner.robot_graphs[key].save_graph_to_file(key, planner.robotName)

    elif flag == "read_library_paths":
        lib_dict = {}
        file_path = "./robot_configuration_files/"+ planner.robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, planner.robotName)

        object_name = "object_6"
        object = world_graph.Graph()
        object.read_object_from_file(object_name)
        object_nodes = object.get_nodes_as_string()
        layered_dependencies = []
        for key in planner.robot_graphs:
            planner.robot_graphs[key].read_graph_from_file(key, planner.robotName)
            lib_dict[key] = planner.read_library_from_file(key)
            if key not in planner.end_effectors_keys:
                layered_dependencies.extend(planner.robot_graphs[key].new_object_in_world(object_nodes, object_name, False))
            else:
                planner.robot_graphs[key].new_object_in_world(object_nodes, object_name, True, layered_dependencies)
                layered_dependencies.clear()

        dict_pose = planner.extract_action_from_library("21arm_sides", lib_dict)
        dict_angles = planner.extract_angles_from_library("21arm_sides", lib_dict)
        new_path = {}
        angle_dep_new_path = {}
        angle_dep_old_path = {}
        for key in planner.end_effectors_keys:
            missing_nodes = planner.robot_graphs[key].verify_path_in_graph(dict_pose[key])
            new_path[key] = planner.robot_graphs[key].re_path_end_effector(missing_nodes, dict_pose[key])
            angle_dep_old_path[key] = dict_angles[key]
            if new_path[key]:
                angle_dep_new_path[key] = planner.robot_graphs[key].select_joint_dependencies(new_path[key])
            else:
                angle_dep_new_path[key] = []
            angles_new_path = planner.robot_graphs[key].dict_of_dep_to_dict_of_lists(angle_dep_new_path[key])
            angles_old_path = planner.robot_graphs[key].dict_of_dep_to_dict_of_lists(angle_dep_old_path[key])
            planner.robot_graphs[key].plot_dict_of_dependencies(angles_old_path, angles_new_path)
            planner.plotPath(key, np.asarray(dict_pose[key]), np.asarray(dict_pose[key]))

            # plotPath(key, np.asarray(dict_pose[key]), np.asarray(new_path[key]))
        # find_fk_from_dict(angle_dep_new_path, qt, robotName)
        planner.find_fk_from_dict(angle_dep_old_path)