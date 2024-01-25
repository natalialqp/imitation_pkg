from GMM.GMM_GMR import *
import numpy as np
import pose_prediction
from scipy.signal import savgol_filter
import json
from MultiSignalLSTM import TrajectoryLearningModel, TrajectoryModelWrapper
from sklearn.model_selection import train_test_split
from dummy import eval

def plot_3d_paths(trajectories, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot each trajectory
    for i, trajectory in enumerate(trajectories):
        m = trajectory.shape[1]
        x = trajectory[1, :]
        y = trajectory[2, :]
        z = trajectory[3, :]
        ax.plot(x, y, z, label=f'T{i + 1}')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title)
    ax.legend()
    plt.show()

def extract_action_from_library(actionName, library):
    robot_pose = {}
    for key in library:
        for item in library[key]:
            if item["id"] == actionName:
                robot_pose[key] = item["path"]
    return robot_pose

def extract_angles_from_library(actionName, library):
    robot_pose = {}
    for key in library:
        for item in library[key]:
            if item["id"] == actionName:
                robot_pose[key] = item["joint_dependency"]
    return robot_pose

def extract_action(df, actionName):
    df = df.loc[df['action'] == actionName]
    users = np.arange(1, 21, 1)
    return df

def extract_time(timestamp):
    timestamps_seconds = np.array([float(ts) for ts in timestamp]) / 1e6 # miliseconds, use 1e9 for seconds
    #  Calculate the time difference between consecutive timestamps
    time_diff_seconds = np.diff(timestamps_seconds)
    return np.insert(np.cumsum(time_diff_seconds), 0, 0)

def extract_end_effector_path(vec, time):
    last_vectors = np.array([trajectory[-1] for trajectory in vec])
    return np.insert(last_vectors, 0, time, axis=1).T

def add_time_to_angles(vec, time):
    vec = np.array(vec)
    return np.insert(vec, 0, time, axis=1).T
    # return np.array([trajectory[angle, :] for trajectory in vec])

def extract_vectors(data):
    angles = data[0].keys()
    # Create vectors for each angle
    angle_vectors = []
    for point in data:
        values = [point[i] for i in angles]
        angle_vectors.append(np.deg2rad(values))
    return angle_vectors

def extract_axis(vec, axis):
    time = vec[0, :]
    if axis == "x":
        axis_values = vec[1, :]
    elif axis == "y":
        axis_values = vec[2, :]
    elif axis == "z":
        axis_values = vec[3, :]
    return np.vstack((time, axis_values))

def extract_angle(vec, angle):
    return vec[[0,angle], :]

def extract_angle_vec(mat, angle):
    return [extract_angle(sub_list, angle) for sub_list in mat]

def plot_angles_vs_time(arrays):
    for i in range(1, arrays[0].shape[1]):
        plt.figure()
        plt.title(f"Variable {i} vs Time")
        for j in range(len(arrays)):
            time_values = arrays[j][:, 0]
            variable_values = arrays[j][:, i]
            plt.plot(time_values, variable_values, label=f"Array {j+1}")

        plt.xlabel("Time")
        plt.ylabel(f"Variable {i}")
        plt.legend()
        plt.show()

# def plot_axis_vs_time(vec, axis):
#     variable_to_plot = axis
#     # Extract time and the chosen variable from each array
#     time_values = [array[0, :] for array in vec]
#     variable_values = [array[1, :] for array in vec]

#     # Plot the variable with respect to time for all arrays
#     for i in range(len(vec)):
#         plt.plot(time_values[i], variable_values[i], label=f'{i + 1}')

#     # Add labels and legend
#     plt.xlabel('Time')
#     plt.ylabel(f'{variable_to_plot.capitalize()} Values')
#     plt.legend()
#     plt.show()

def plot_axis_vs_time(vec, axis, target_length=1000, window_length=5, polyorder=3):
    variable_to_plot = axis
    data_as_array = []
    time_as_array = []
    # Extract time and the chosen variable from each array
    time_values = [array[0, :] for array in vec]
    variable_values = [array[1, :] for array in vec]
    # Interpolate each instance to the specified target length
    interp_variable_values = [
        np.interp(np.linspace(0, 1000, target_length), np.linspace(0, 1000, len(time)), variable)
        for time, variable in zip(time_values, variable_values)
    ]

    # Smooth the interpolated signals using the Savitzky-Golay filter
    smoothed_variable_values = [
        savgol_filter(interp_variable, window_length=window_length, polyorder=polyorder)
        for interp_variable in interp_variable_values
    ]

    # Plot the smoothed variable with respect to time for all arrays
    for i, smoothed_variable in enumerate(smoothed_variable_values):
        plt.plot(np.linspace(0, 1000, target_length), smoothed_variable, label=f'{i + 1}')
        data_as_array.extend(smoothed_variable)
        time_as_array.extend(np.linspace(0, 1000, target_length))

    # Calculate and plot the average signal
    avg_signal = np.mean(smoothed_variable_values, axis=0)
    plt.plot(np.linspace(0, 1000, target_length), avg_signal, label='Average', linestyle='--', linewidth=2)

    # Add labels and legend
    plt.xlabel('Normalized Time')
    plt.ylabel(f'{variable_to_plot.capitalize()} Values (Smoothed)')
    plt.legend()
    plt.show()
    return np.vstack([np.array(time_as_array), np.array(data_as_array)])
    # return np.array(time_as_array), np.array(data_as_array)

def read_library_from_file(name, robotName):
    try:
        with open(name + "_" + robotName + "_data.json", "r") as jsonfile:
            data = [json.loads(line) for line in jsonfile.readlines()]
            return data
    except FileNotFoundError:
        print(f"File {name}_data.json not found.")
        return []

def reshape_array(arr2):
    array_shape = arr2.shape
    for i in range(array_shape[0]):
        v = arr2[i].reshape(int(array_shape[1]/100), 100)
        if i == 0:
            arr = v
        else:
            arr = np.dstack([arr, v])
    return arr

def arange_time_array(time, arr, chunk_size=1000):
    time = np.array(time)
    time = time.reshape(int(time.shape[0]/chunk_size), chunk_size)
    add = np.linspace(0, 0.0001, chunk_size)
    add = np.tile(add, (time.shape[0], 1)).T + time.T
    reordered_time = add.T.reshape(-1).tolist()

    arr = np.array(arr)
    arr = arr.reshape(int(arr.shape[0]/chunk_size), chunk_size)
    reordered_list = arr.reshape(-1).tolist()
    return np.vstack((reordered_time, reordered_list))

def plot_time_vs_signal(time, signal, xlabel='Time', ylabel='Signal', title='Time vs Signal'):
    # Plot the time vs signal
    plt.scatter(time, signal)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Show the plot
    plt.show()

def reshape_array(time, arr):
    arr_len = len(arr)
    time = time.reshape(int(arr_len/1000), 1000)[0]
    arr = arr.reshape(int(arr_len/1000), 1000)
    arr = np.vstack((time, arr))
    return arr.T #[:, 0:2]

if __name__ == "__main__":

    gmm = GMM_GMR(4)
    robotName = 'qt'

    #GMM applied on the output of the neural network
    # flag ="gmm-on-nn-output"
    flag = "gmm-on-angles-from-library"
    # flag = "nn-from-library"
    if flag == "gmm-on-nn-output":
        actions = ['teacup', 'teapot', 'spoon', 'ladle', 'shallow_plate', 'dinner_plate', 'knife', 'fork', 'salt_shaker', 'sugar_bowl', 'mixer', 'pressure_cooker']
        file_path = "./robot_configuration_files/"+ robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, robotName)
        file_name = "robot_angles_" + robotName
        theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
        pose_predictor.train_pytorch(pose_predictor.robot, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, 1000)
        df = pose_predictor.read_file("combined_actions")
        users = np.arange(1, 21, 1)
        action = 'shallow_plate'
        left_side_with_time = []
        right_side_with_time = []
        head_with_time = []
        t_x_left = []
        t_y_left = []
        t_z_left = []
        angles_left_with_time = []
        angles_right_with_time = []
        angles_head_with_time = []
        angles_left_dict = {}

        for user in users:
            left_side, right_side, head, timestamps = pose_predictor.read_csv_combined(df, action, user)
            left_side = left_side * 1000
            right_side = right_side * 1000
            head = head * 1000

            angles_left_vec = []
            angles_right_vec = []
            angles_head_vec = []
            cartesian_left_vec = []
            cartesian_right_vec = []
            cartesian_head_vec = []

            cumulative_time = extract_time(timestamps)
            for i in range(len(left_side)):
                angles_left, angles_right, angles_head = pose_predictor.predict_pytorch(left_side[i], right_side[i], head[i])
                angles_left_vec.append(angles_left)
                angles_right_vec.append(angles_right)
                angles_head_vec.append(angles_head)

                points4, points5, points6 = pose_predictor.robot_embodiment(angles_left, angles_right, angles_head)
                cartesian_left_vec.append(points4)
                cartesian_right_vec.append(points5)
                cartesian_head_vec.append(points6)

            if len(cumulative_time) > 1:
                aux_left = extract_end_effector_path(cartesian_left_vec, cumulative_time)
                aux_right = extract_end_effector_path(cartesian_right_vec, cumulative_time)
                aux_head = extract_end_effector_path(cartesian_head_vec, cumulative_time)

                aux_left_angles = add_time_to_angles(angles_left_vec, cumulative_time)
                aux_right_angles = add_time_to_angles(angles_right_vec, cumulative_time)
                aux_head_angles = add_time_to_angles(angles_head_vec, cumulative_time)

                angles_left_with_time.append(aux_left_angles)
                angles_right_with_time.append(aux_right_angles)
                angles_head_with_time.append(aux_head_angles)

                t_x_left.append(extract_axis(aux_left, "x"))
                t_y_left.append(extract_axis(aux_left, "y"))
                t_z_left.append(extract_axis(aux_left, "z"))

                # if user > 9:
                    # lala = extract_axis(aux_left, "x")
                    # print("time" + str(user) + "= np.array(", lala[0], ")")
                    # print("data" + str(user) + "= np.array(", lala[1], ")")

                left_side_with_time.append(aux_left)
                right_side_with_time.append(aux_right)
                head_with_time.append(aux_head)

        # plot_3d_paths(left_side_with_time, "Left EE")
        # plot_3d_paths(right_side_with_time, "Right EE")
        # plot_3d_paths(head_with_time, "Head EE")
        # plot_axis_vs_time(t_x_left, "x")
        # plot_axis_vs_time(t_y_left, "y")
        # plot_axis_vs_time(t_z_left, "z")
        for i in range(1, angles_left_with_time[0].shape[0]):
            smoothed_angle_left = plot_axis_vs_time(extract_angle_vec(angles_left_with_time, i), str(i))
        # plot_angles_vs_time(angles_left_with_time)
        # plot_angles_vs_time(angles_right_with_time)
        # plot_angles_vs_time(angles_head_with_time)
            gmm.fit(smoothed_angle_left)
            timeInput = np.linspace(1, np.max(smoothed_angle_left[0, :]), 1000)
            gmm.predict(timeInput)
            fig = plt.figure()
            fig.suptitle("Axis 1 vs axis 0")
            ax1 = fig.add_subplot(221)
            plt.title("Data")
            gmm.plot(ax=ax1, plotType="Data")
            ax2 = fig.add_subplot(222)
            plt.title("Gaussian States")
            gmm.plot(ax=ax2, plotType="Clusters")
            ax3 = fig.add_subplot(223)
            plt.title("Regression")
            gmm.plot(ax=ax3, plotType="Regression")
            ax4 = fig.add_subplot(224)
            plt.title("Clusters + Regression")
            gmm.plot(ax=ax4, plotType="Clusters")
            gmm.plot(ax=ax4, plotType="Regression")
            predictedMatrix = gmm.getPredictedMatrix()
            # print(predictedMatrix)
            plt.show()

    elif flag=="gmm-on-angles-from-library":
        users = np.arange(1, 21, 1)
        action = 'fork'
        robotName = "qt"
        lib_dict = {}
        t_x_left = []
        t_y_left = []
        t_z_left = []
        left_side_with_time = []
        right_side_with_time = []
        head_with_time = []

        end_effector_dict = ["jointLeft_4", "jointRight_4", "jointHead_3"]
        file_path = "./robot_configuration_files/"+ robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, robotName)
        df = pose_predictor.read_file("combined_actions")

        dict_pose_vec_left = []
        dict_pose_vec_right = []
        dict_pose_vec_head = []
        angles_left_with_time = []
        angles_right_with_time = []
        angles_head_with_time = []

        for key in end_effector_dict:
            lib_dict[key] = read_library_from_file(key, robotName)

        for user in users:
            _, _, _, timestamps = pose_predictor.read_csv_combined(df, action, user)
            cumulative_time = extract_time(timestamps)

            dict_pose = extract_action_from_library(str(user) + action, lib_dict)
            cartesian_left_vec = dict_pose[end_effector_dict[0]]
            cartesian_right_vec = dict_pose[end_effector_dict[1]]
            cartesian_head_vec = dict_pose[end_effector_dict[2]]

            dep_dict = extract_angles_from_library(str(user) + action, lib_dict)
            jointLeft_vectors = extract_vectors(dep_dict[end_effector_dict[0]])
            jointRight_vectors = extract_vectors(dep_dict[end_effector_dict[1]])
            jointHead_vectors = extract_vectors(dep_dict[end_effector_dict[2]])

            if len(cumulative_time) > 1:
                aux_left = add_time_to_angles(cartesian_left_vec, cumulative_time)
                aux_right = add_time_to_angles(cartesian_right_vec, cumulative_time)
                aux_head = add_time_to_angles(cartesian_head_vec, cumulative_time)

                aux_left_angles = add_time_to_angles(jointLeft_vectors, cumulative_time)
                aux_right_angles = add_time_to_angles(jointRight_vectors, cumulative_time)
                aux_head_angles = add_time_to_angles(jointHead_vectors, cumulative_time)

                angles_left_with_time.append(aux_left_angles)
                angles_right_with_time.append(aux_right_angles)
                angles_head_with_time.append(aux_head_angles)

                t_x_left.append(extract_axis(aux_left, "x"))
                t_y_left.append(extract_axis(aux_left, "y"))
                t_z_left.append(extract_axis(aux_left, "z"))
                left_side_with_time.append(aux_left)
                right_side_with_time.append(aux_right)
                head_with_time.append(aux_head)

            dict_pose_vec_left.append(jointLeft_vectors)
            dict_pose_vec_right.append(jointRight_vectors)
            dict_pose_vec_head.append(jointHead_vectors)
        # print("angles_left_with_time: ", angles_left_with_time)
        # plot_angles_vs_time(angles_left_with_time)
        # plot_angles_vs_time(angles_right_with_time)
        # plot_angles_vs_time(angles_head_with_time)

        # plot_3d_paths(left_side_with_time, "Left EE")
        # plot_3d_paths(right_side_with_time, "Right EE")
        # plot_3d_paths(head_with_time, "Head EE")

        for i in range(1, angles_left_with_time[0].shape[0]):
            variable_to_fit = extract_angle_vec(angles_left_with_time, i)
            time, smoothed_angle_left = plot_axis_vs_time(variable_to_fit, str(i))
            # a = reshape_array(time, smoothed_angle_left)
            a = arange_time_array(time, smoothed_angle_left)
            # plot_time_vs_signal(a[0], a[1])
            eval(a.T, robotName, action, "left_" + str(i), 4)

            #gmm from git

            # smoothed_angle_left = a
            # gmm.fit(smoothed_angle_left)
            # timeInput = np.linspace(0, np.max(smoothed_angle_left[0, :]), 1000)
            # gmm.predict(timeInput)
            # fig = plt.figure()
            # fig.suptitle("Axis 1 vs axis 0")
            # ax1 = fig.add_subplot(221)
            # plt.title("Data")
            # gmm.plot(ax=ax1, plotType="Data")
            # ax2 = fig.add_subplot(222)
            # plt.title("Gaussian States")
            # gmm.plot(ax=ax2, plotType="Clusters")
            # ax3 = fig.add_subplot(223)
            # plt.title("Regression")
            # gmm.plot(ax=ax3, plotType="Regression")
            # ax4 = fig.add_subplot(224)
            # plt.title("Clusters + Regression")
            # gmm.plot(ax=ax4, plotType="Clusters")
            # gmm.plot(ax=ax4, plotType="Regression")
            # predictedMatrix = gmm.getPredictedMatrix()
            # # print(predictedMatrix)
            # plt.show()

    elif flag == "nn-from-library":
        users = np.arange(1, 21, 1)
        action = 'spoon'
        robotName = "qt"
        lib_dict = {}
        t_x_left = []
        t_y_left = []
        t_z_left = []
        left_side_with_time = []
        right_side_with_time = []
        head_with_time = []

        end_effector_dict = ["jointLeft_4", "jointRight_4", "jointHead_3"]
        file_path = "./robot_configuration_files/"+ robotName + ".yaml"
        pose_predictor = pose_prediction.Prediction(file_path, robotName)
        df = pose_predictor.read_file("combined_actions")

        dict_pose_vec_left = []
        dict_pose_vec_right = []
        dict_pose_vec_head = []
        angles_left_with_time = []
        angles_right_with_time = []
        angles_head_with_time = []

        for key in end_effector_dict:
            lib_dict[key] = read_library_from_file(key, robotName)

        for user in users:
            _, _, _, timestamps = pose_predictor.read_csv_combined(df, action, user)
            cumulative_time = extract_time(timestamps)

            dict_pose = extract_action_from_library(str(user) + action, lib_dict)
            cartesian_left_vec = dict_pose[end_effector_dict[0]]
            cartesian_right_vec = dict_pose[end_effector_dict[1]]
            cartesian_head_vec = dict_pose[end_effector_dict[2]]

            dep_dict = extract_angles_from_library(str(user) + action, lib_dict)
            jointLeft_vectors = extract_vectors(dep_dict[end_effector_dict[0]])
            jointRight_vectors = extract_vectors(dep_dict[end_effector_dict[1]])
            jointHead_vectors = extract_vectors(dep_dict[end_effector_dict[2]])

            if len(cumulative_time) > 1:
                aux_left = add_time_to_angles(cartesian_left_vec, cumulative_time)
                aux_right = add_time_to_angles(cartesian_right_vec, cumulative_time)
                aux_head = add_time_to_angles(cartesian_head_vec, cumulative_time)

                aux_left_angles = add_time_to_angles(np.rad2deg(jointLeft_vectors), cumulative_time)
                aux_right_angles = add_time_to_angles(np.rad2deg(jointRight_vectors), cumulative_time)
                aux_head_angles = add_time_to_angles(np.rad2deg(jointHead_vectors), cumulative_time)

                angles_left_with_time.append(aux_left_angles)
                angles_right_with_time.append(aux_right_angles)
                angles_head_with_time.append(aux_head_angles)

                t_x_left.append(extract_axis(aux_left, "x"))
                t_y_left.append(extract_axis(aux_left, "y"))
                t_z_left.append(extract_axis(aux_left, "z"))
                left_side_with_time.append(aux_left)
                right_side_with_time.append(aux_right)
                head_with_time.append(aux_head)

            dict_pose_vec_left.append(jointLeft_vectors)
            dict_pose_vec_right.append(jointRight_vectors)
            dict_pose_vec_head.append(jointHead_vectors)
        for i in range(1, angles_left_with_time[0].shape[0]):
            time, smoothed_angle_left = plot_axis_vs_time(extract_angle_vec(angles_left_with_time, i), str(i))
            if i == 1:
                smoothed_angles_left = np.vstack([np.array(time), np.array(smoothed_angle_left)])
            else:
                smoothed_angles_left = np.vstack([smoothed_angles_left, np.array(smoothed_angle_left)])

        data = reshape_array(smoothed_angles_left)
        input_size = 4
        hidden_size = 64
        output_size = 4
        model_wrapper = TrajectoryModelWrapper(input_size, hidden_size, output_size)
        model_wrapper.train(data, num_epochs=100)
        model_wrapper.evaluate(data)

        # Use the trained model to make predictions
        # new_data_point = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        # predicted_trajectory = model_wrapper.predict(new_data_point)
        # print("Predicted Trajectory:", predicted_trajectory)



