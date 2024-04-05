import matplotlib.pyplot as plt
import numpy as np
import world_graph
import yaml
from tqdm import tqdm
import pose_prediction
import path_planning

plt.rcParams.update({'font.size': 18})

def read_yaml_file(file_path):
    """
    Reads a YAML file and returns the data as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The data read from the YAML file.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

if __name__ == "__main__":
    """
    Usage example
    Select the corresponding flag to run the desired function.
    """
    config = read_yaml_file("config.yaml")
    robotName = config["robot-name"]
    babblingPoints = config["babbling-points"]
    delta = config["minimum-distance"]

    lowEdge_robot = config["low-edge-" + robotName]
    highEdge_robot = config["high-edge-" + robotName]
    lowEdge_obstacle = config["low-edge-obstacle"]
    highEdge_obstacle = config["high-edge-obstacle"]

    obstacle_name = config["obstacle-name"]
    actions_names = config["actions-names"]
    users_id = config["users-id"]
    function = config["function-name"]
    epochs = config["epochs"]
    iteration_id = config["iteration-id"]
    read_new_action = config["read-new-action"]

    babbing_path = "test_" + robotName + "/self_exploration/self_exploration_" + robotName + "_" + str(babblingPoints) + ".txt"

    planner = path_planning.PathPlanning(robotName, delta)

    if function == "explore-world":
        planner.createWorldCubes(lowEdge_robot, highEdge_robot)
        joint_angles = planner.read_babbling(babbing_path)
        new_list = planner.angle_interpolation(joint_angles)
        #AFTER INTERPOLATION
        pos_left, pos_right, pos_head = planner.forward_kinematics_n_frames(new_list)
        cartesian_points = planner.myRobot.pos_mat_to_robot_mat_dict(pos_left, pos_right, pos_head)
        joint_angles_dict = planner.myRobot.angular_mat_to_mat_dict(new_list)
        #"AFTER FK"
        robot_world = planner.learn_environment(cartesian_points, joint_angles_dict)
        #"BEFORE SAVING THE GRAPHS"
        for key in tqdm(robot_world):
            robot_world[key].save_graph_to_file(key, planner.robotName, str(babblingPoints))
            robot_world[key].plot_graph(planner.robotName, str(babblingPoints))

    elif function == "create-object":
        # Define parameter of the world and create one with cubic structure according to the robot, chech yaml files
        object = planner.createObject(lowEdge_obstacle, highEdge_obstacle)
        object.save_object_to_file(planner.robotName, obstacle_name)
        planner.match_object_in_world(obstacle_name, lowEdge_robot, highEdge_robot)

    elif function == "object-in-robot-graph":
        # Define parameter of the world and create one with cubic structure according to the robot, chech yaml files
        object = planner.createObject(lowEdge_obstacle, highEdge_obstacle)
        object = world_graph.Graph()
        object.read_object_from_file(planner.robotName, obstacle_name)
        _, object_nodes = planner.find_object_in_world(object)
        for key in planner.robot_graphs:
            planner.robot_graphs[key].read_graph_from_file(key, planner.robotName)
            planner.robot_graphs[key].new_object_in_world(object_nodes, obstacle_name)
            planner.robot_graphs[key].remove_object_from_world(obstacle_name)

    elif function == "reproduce-action":
        #reading an action and reproducing
        frames = planner.readTxtFile("./data/angles.txt")
        joint_angles = planner.divideFrames(frames)

    elif function == "path-planning":
        # Define parameter of the world and create one with cubic structure according to the robot, chech yaml files
        planner.createWorldCubes(lowEdge_robot, highEdge_robot)

        file_path = "./robot_configuration_files/"+ planner.robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, planner.robotName)
        file_name = "robot_angles_" + planner.robotName
        theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
        pose_predictor.train_pytorch(pose_predictor.robot, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, epochs)

        df = pose_predictor.read_file("combined_actions")
        planner.fill_robot_graphs(str(babblingPoints))
        for user in tqdm(users_id):
            for action in actions_names:
                dict_error = {}
                robot_pose = []
                left_side, right_side, head, time_ = pose_predictor.read_csv_combined(df, action, user)
                left_side = left_side * 1000
                right_side = right_side * 1000
                head = head * 1000
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
                    #path planning
                    tra, MSE, RMSE = planner.path_planning(generated_trajectory, planner.robot_graphs[key])
                    dep = planner.robot_graphs[key].select_joint_dependencies(tra)
                    if int(key[-1]) > 2:
                        planner.plotPath(actionName + " " + key, generated_trajectory, np.asarray(tra))

    elif function == "pose-predicition":
        # Define parameter of the world and create one with cubic structure according to the robot, chech yaml files
        lowEdge = np.array([-500, -1200, 100])
        highEdge = np.array([1100, 1200, 1700])
        planner.createWorldCubes(lowEdge, highEdge)

        file_path = "./robot_configuration_files/"+ planner.robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, planner.robotName)
        file_name = "robot_angles_" + planner.robotName
        theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
        pose_predictor.train_pytorch(pose_predictor.robot, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, epochs)

        df = pose_predictor.read_file("combined_actions")
        planner.fill_robot_graphs(str(babblingPoints))

        #for new actions
        if read_new_action:
            actions_names = config["new-action-name"]
            users_id = config["users-id-new-action"]
            df = pose_predictor.read_file("/QT_recordings/human/" + actions_names[0])

        for user in tqdm(users_id):
            for action in actions_names:
                dict_error = {}
                robot_pose = []
                if not read_new_action:
                    left_side, right_side, head, time_ = pose_predictor.read_csv_combined(df, action, user)
                    left_side = left_side * 1000
                    right_side = right_side * 1000
                    head = head * 1000

                # for new actions
                else:
                    left_side, right_side, head, time_ = pose_predictor.read_recorded_action_csv(df, action, user)

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
                    planner.robot_graphs[key].save_path_in_library(tra, dep, planner.robotName, actionName, str(babblingPoints))
                    planner.plotPath(actionName, generated_trajectory, np.asarray(tra))

        for key in planner.robot_graphs:
            planner.robot_graphs[key].plot_graph(planner.robotName, str(babblingPoints) + "_" + str(iteration_id))
            planner.robot_graphs[key].save_graph_to_file(key, planner.robotName, str(babblingPoints))

    elif function == "read-library-paths":
        lib_dict = {}
        trajectory_id = config["trajectory-id"]
        file_path = "./robot_configuration_files/"+ planner.robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, planner.robotName)
        object = world_graph.Graph()
        object.read_object_from_file(planner.robotName, obstacle_name)
        object_nodes = object.get_nodes_as_string()
        layered_dependencies = []
        for key in planner.robot_graphs:
            planner.robot_graphs[key].read_graph_from_file(key, planner.robotName, str(babblingPoints))
            lib_dict[key] = planner.read_library_from_file(key, str(babblingPoints))
            if key not in planner.end_effectors_keys:
                layered_dependencies.extend(planner.robot_graphs[key].new_object_in_world(object_nodes, obstacle_name, False))
            else:
                planner.robot_graphs[key].new_object_in_world(object_nodes, obstacle_name, True, layered_dependencies)
                layered_dependencies.clear()
            planner.robot_graphs[key].plot_graph(planner.robotName, str(babblingPoints))
        dict_pose = planner.extract_action_from_library(trajectory_id, lib_dict)
        dict_angles = planner.extract_angles_from_library(trajectory_id, lib_dict)
        new_path = {}
        angle_dep_new_path = {}
        angle_dep_old_path = {}
        for key in planner.end_effectors_keys:
        # for key in planner.robot_graphs:
            # planner.robot_graphs[key].plot_graph(planner.robotName, babbling_points)
            missing_nodes = planner.robot_graphs[key].verify_path_in_graph(dict_pose[key])
            new_path[key] = planner.robot_graphs[key].re_path_end_effector(missing_nodes, dict_pose[key])
            angle_dep_old_path[key] = dict_angles[key]
            if new_path[key]:
                angle_dep_new_path[key] = planner.robot_graphs[key].select_joint_dependencies(new_path[key])
            else:
                angle_dep_new_path[key] = []
            angles_new_path = planner.robot_graphs[key].dict_of_dep_to_dict_of_lists(angle_dep_new_path[key])
            angles_old_path = planner.robot_graphs[key].dict_of_dep_to_dict_of_lists(angle_dep_old_path[key])
            # planner.robot_graphs[key].plot_dict_of_dependencies(angles_old_path, angles_new_path)
            planner.plotPath(key, np.asarray(dict_pose[key]), np.asarray(new_path[key]))
