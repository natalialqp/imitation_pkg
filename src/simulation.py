import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd
import robot
import csv
from matplotlib.widgets import Slider
import numpy as np
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
plt.rcParams.update({'font.size': 16})

number_arm_human_joints = 5
number_head_human_joints = 3
robot_head_dimensions = np.array([0, 0.0962])
robot_arm_dimensions = np.array([0.08, 0.0445, 0.07708, 0.184])

human_joints_head  = ['JOINT_LEFT_COLLAR', 'JOINT_NECK', 'JOINT_HEAD']
human_joints_left  = ['JOINT_LEFT_COLLAR', 'JOINT_LEFT_SHOULDER', 'JOINT_LEFT_ELBOW', 'JOINT_LEFT_WRIST', 'JOINT_LEFT_HAND']
human_joints_right = ['JOINT_LEFT_COLLAR', 'JOINT_RIGHT_SHOULDER', 'JOINT_RIGHT_ELBOW', 'JOINT_RIGHT_WRIST', 'JOINT_RIGHT_HAND']

def cartesian_to_spherical(cartesian_points):
    # Convert Cartesian coordinates to spherical coordinates
    r     = np.linalg.norm(cartesian_points, axis=1)                    # Radial distance
    theta = np.arccos(cartesian_points[:, 2] / r)                       # Inclination angle
    phi   = np.arctan2(cartesian_points[:, 1], cartesian_points[:, 0])  # Azimuth angle

    # Set of points in spherical coordinates
    # spherical_points = np.column_stack((r, theta, phi))
    # return spherical_points
    return theta, phi

def spherical_to_cartesian(spherical_points, robot):
    r     = spherical_points[:, 0]      # Radial distance
    theta = spherical_points[:, 1]      # Inclination angle
    phi   = spherical_points[:, 2]      # Azimuth angle
    # l,r,h = robot.forward_kinematics()

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Set of points in Cartesian coordinates
    cartesian_points = np.column_stack((x, y, z))
    return cartesian_points

def human_to_robot(points, isArm):
    #Calculate the n-th discrete difference along the given axis
    vectors = np.diff(points, axis=0)
    spherical_points = cartesian_to_spherical(vectors)
    if isArm:
        spherical_points[:, 0] = robot_arm_dimensions
    else:
        spherical_points[:, 0] = robot_head_dimensions
    coordinates = spherical_to_cartesian(np.vstack((np.array([0, 0, 0]), spherical_points)), robot)
    #Return the cumulative sum of the elements along a given axis
    return np.cumsum(coordinates, axis=0)

def extract_spherical_angles_from_human(points1, points2, points3):
    left_theta, left_phi = cartesian_to_spherical(np.diff(points1, axis=0))
    right_theta, right_phi = cartesian_to_spherical(np.diff(points2, axis=0))
    head_theta, head_phi = cartesian_to_spherical(np.diff(points3, axis=0))
    return left_theta, left_phi, right_theta, right_phi, head_theta, head_phi

# def robot_embodiment(robot, action, user):

#     # left_shoulder_pitch -> 0
#     # left_shoulder_roll --> 1
#     # left_elbow_roll -----> 2
#     # right_shoulder_pitch-> 3
#     # right_shoulder_roll -> 4
#     # right_elbow_roll ----> 5
#     # head_yaw ------------> 6
#     # head_pitch-----------> 7

#     angles = np.array([np.deg2rad(-90), np.deg2rad(-20.3), np.deg2rad(-10.7),
#                        np.deg2rad(-90), np.deg2rad(-40.9), np.deg2rad(-45.4),
#                        np.deg2rad(0), np.deg2rad(-0)])
#     #16 -mixer
#     # angles = np.array([-0.9101365, -0.5816182, -1.3170639, 0.522737, -0.64128196, -1.2126011, -0.13765207,  0.10746284])
#     #16 - pressure cookers
#     # angles = np.array([-0.0620994, -1.0886317, -1.0191336, 0.0456534, -0.9713295 , -0.8338959, 0.00057178,  0.04962267])

#     # save_mapping_csv("./data/robot_angles", action, user, left_theta, left_phi, head_theta, right_theta, right_phi, head_phi, angles)
#     l, r, h = robot.forward_kinematics(angles)
#     return matrix_array_to_list_list(l), matrix_array_to_list_list(r), matrix_array_to_list_list(h)

def robot_embodiment(robot, angles_left, angles_right, angles_head):

    # left_shoulder_pitch -> 0
    # left_shoulder_roll --> 1
    # left_elbow_roll -----> 2
    # right_shoulder_pitch-> 3
    # right_shoulder_roll -> 4
    # right_elbow_roll ----> 5
    # head_yaw ------------> 6
    # head_pitch-----------> 7

    angles = np.concatenate((angles_left, angles_right, angles_head), axis=None)
    l, r, h = robot.forward_kinematics(angles)
    return matrix_array_to_list_list(l), matrix_array_to_list_list(r), matrix_array_to_list_list(h)

def matrix_array_to_list_list(vec):
    new_list = []
    for i in vec:
        new_list.append(list(i))
    return np.array(new_list)

def vectorise_string(vec):
    aux_data = ''.join([i for i in vec if not (i=='[' or i==']' or i==',')])
    data = np.array(aux_data.split())
    return list(data[0:3].astype(float))

def vectorise_spherical_data(vec):
    aux_data = ''.join([i for i in vec if not (i=='[' or i==']' or i==',')])
    data = np.array(aux_data.split())
    return list(data[0:4].astype(float))

def read_csv_combined(df, action, user):
    left_side = []
    right_side = []
    head = []
    aux_arm = [[]] * number_arm_human_joints
    aux_head = [[]] * number_head_human_joints
    for i in range(len(df)):
        if action == df['action'][i] and user == df['participant_id'][i]:
            for count, left in enumerate(human_joints_left):
                aux_arm[count] = vectorise_string(df[left][i])
            left_side.append(aux_arm.copy())
            for count, right in enumerate(human_joints_right):
                aux_arm[count] = vectorise_string(df[right][i])
            right_side.append(aux_arm.copy())
            for count, _head in enumerate(human_joints_head):
                aux_head[count] = vectorise_string(df[_head][i])
            head.append(aux_head.copy())
    return np.array(left_side), np.array(right_side), np.array(head)

def read_training_data(file_name):

    df = read_file(file_name)
    theta_left = np.array([vectorise_spherical_data(i) for i in df['left_arm_human_theta']])
    theta_right = np.array([vectorise_spherical_data(i) for i in df['right_arm_human_theta']])
    theta_head = np.array([vectorise_spherical_data(i) for i in df['head_human_theta']])
    phi_left = np.array([vectorise_spherical_data(i) for i in df['left_arm_human_phi']])
    phi_right = np.array([vectorise_spherical_data(i) for i in df['right_arm_human_phi']])
    phi_head = np.array([vectorise_spherical_data(i) for i in df['head_human_phi']])
    left_arm_robot = np.array([vectorise_spherical_data(i) for i in df['left_arm_robot']])
    right_arm_robot = np.array([vectorise_spherical_data(i) for i in df['right_arm_robot']])
    head_robot = np.array([vectorise_spherical_data(i) for i in df['head_robot']])

    return theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot

def save_mapping_csv(file_name, action, user, lht, lhp, hht, rht, rhp, hhp, ra):

    df = pd.read_csv(file_name + ".csv")
    for i in range(len(df)):
        if action == df['action'][i] and user == df['participant_id'][i]:
            df['left_arm_human_theta'][i] = str(lht)
            df['right_arm_human_theta'][i] = str(rht)
            df['head_human_theta'][i] = str(hht)
            df['left_arm_human_phi'][i] = str(lhp)
            df['right_arm_human_phi'][i] = str(rhp)
            df['head_human_phi'][i] = str(hhp)
            df['left_arm_robot'][i] = str(ra[0:3])
            df['right_arm_robot'][i] = str(ra[3:6])
            df['head_robot'][i] = str(ra[6:8])
        df.to_csv(file_name + ".csv")

def create_3d_plot():
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Set the axis labels and limits for both subplots (adjust as needed)
    ax1.set_xlabel('X', fontsize=18)
    ax1.set_ylabel('Y', fontsize=18)
    ax1.set_zlabel('Z', fontsize=18)
    ax1.set_xlim([1, 3])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    ax2.set_xlabel('X', fontsize=18)
    ax2.set_ylabel('Y', fontsize=18)
    ax2.set_zlabel('Z', fontsize=18)
    ax2.set_xlim([-0.5, 0.5])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_zlim([0, 1])

    return fig, ax1, ax2

def plot_animation_3d(point_clouds_list, base):
    if not isinstance(point_clouds_list, list) or not all(isinstance(pc, tuple) and len(pc) == 6 for pc in point_clouds_list):
        raise ValueError("Invalid input data. Expecting a list of tuples, each containing 5 point clouds")

    fig, ax1, ax2 = create_3d_plot()

    # Line plots for connections in both subplots
    lines1 = [ax1.plot([], [], [], c='b', linewidth=2)[0] for _ in range(len(point_clouds_list[0][0]) - 1)]
    lines2 = [ax1.plot([], [], [], c='g', linewidth=2, linestyle='--')[0] for _ in range(len(point_clouds_list[0][1]) - 1)]
    lines3 = [ax1.plot([], [], [], c='r', linewidth=2)[0] for _ in range(len(point_clouds_list[0][2]) - 1)]
    lines4 = [ax2.plot([], [], [], c='b', linewidth=2)[0] for _ in range(len(point_clouds_list[0][3]) - 1)]
    lines5 = [ax2.plot([], [], [], c='g', linewidth=2, linestyle='--')[0] for _ in range(len(point_clouds_list[0][4]) - 1)]
    lines6 = [ax2.plot([], [], [], c='r', linewidth=2)[0] for _ in range(len(point_clouds_list[0][5]) - 1)]

    def update(frame):
        points1, points2, points3, points4, points5, points6 = point_clouds_list[frame]

        # Set the titles for each subplot
        ax1.set_title('Human', fontsize=20)
        ax2.set_title('Robot', fontsize=20)

        # Update line plots in both subplots
        for i in range(len(points1) - 1):
            lines1[i].set_data([points1[i, 0], points1[i + 1, 0]], [points1[i, 1], points1[i + 1, 1]])
            lines1[i].set_3d_properties([points1[i, 2], points1[i + 1, 2]])

        for i in range(len(points2) - 1):
            lines2[i].set_data([points2[i, 0], points2[i + 1, 0]], [points2[i, 1], points2[i + 1, 1]])
            lines2[i].set_3d_properties([points2[i, 2], points2[i + 1, 2]])

        for i in range(len(points3) - 1):
            lines3[i].set_data([points3[i, 0], points3[i + 1, 0]], [points3[i, 1], points3[i + 1, 1]])
            lines3[i].set_3d_properties([points3[i, 2], points3[i + 1, 2]])

        for i in range(len(points4) - 1):
            lines4[i].set_data([points4[i, 0], points4[i + 1, 0]], [points4[i, 1], points4[i + 1, 1]])
            lines4[i].set_3d_properties([points4[i, 2], points4[i + 1, 2]])

        for i in range(len(points5) - 1):
            lines5[i].set_data([points5[i, 0], points5[i + 1, 0]], [points5[i, 1], points5[i + 1, 1]])
            lines5[i].set_3d_properties([points5[i, 2], points5[i + 1, 2]])

        for i in range(len(points6) - 1):
            lines6[i].set_data([points6[i, 0], points6[i + 1, 0]], [points6[i, 1], points6[i + 1, 1]])
            lines6[i].set_3d_properties([points6[i, 2], points6[i + 1, 2]])

        return *lines1, *lines2, *lines3, *lines4, *lines5, *lines6

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(point_clouds_list), interval=200, blit=True)

    # Add a slider to control the animation
    slider_ax = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_ax, 'Frame', 0, len(point_clouds_list) - 1, valinit=0, valstep=1)
    ax2.scatter(base[0], base[1], base[2], marker="o")

    def update_animation(val):
        frame = int(slider.val)
        update(frame)
        fig.canvas.draw_idle()

    slider.on_changed(update_animation)
    # Show the animation
    plt.show()

def read_file(name):
    return pd.read_csv('./data/' + name + '.csv')

def train_pytorch(theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, num_epochs = 500):

    # Create a custom PyTorch model class
    class MultiOutputModel(nn.Module):
        def __init__(self):
            super(MultiOutputModel, self).__init__()
            self.shared_layer1 = nn.Linear(20, 128)
            self.shared_layer2 = nn.Linear(128, 128)
            self.left_arm_layer = nn.Linear(128, 3)
            self.right_arm_layer = nn.Linear(128, 3)
            self.head_layer = nn.Linear(128, 2)

        def forward(self, x):
            x = torch.relu(self.shared_layer1(x))
            x = torch.relu(self.shared_layer2(x))
            left_arm_output = self.left_arm_layer(x)
            right_arm_output = self.right_arm_layer(x)
            head_output = self.head_layer(x)
            return left_arm_output, right_arm_output, head_output

    # Instantiate the model
    model = MultiOutputModel()

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Prepare training data as PyTorch tensors
    input_data = torch.Tensor(np.concatenate((theta_left, phi_left, theta_right, phi_right, theta_head, phi_head), axis=1))
    output_data_left = torch.Tensor(left_arm_robot)
    output_data_right = torch.Tensor(right_arm_robot)
    output_data_head = torch.Tensor(head_robot)

    # Training loop
    training_losses = []
    num_epochs = 1000  # Adjust as needed
    for epoch in range(num_epochs):
        # Forward pass
        left_arm_pred, right_arm_pred, head_pred = model(input_data)

        # Calculate loss for each output branch
        loss_left = criterion(left_arm_pred, output_data_left)
        loss_right = criterion(right_arm_pred, output_data_right)
        loss_head = criterion(head_pred, output_data_head)

        # Total loss as a combination of individual losses
        total_loss = loss_left + loss_right + loss_head

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        training_losses.append(total_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}')

    # Plot the training loss
    plt.plot(range(1, num_epochs + 1), training_losses, label='Training Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend()
    plt.grid()
    plt.show()
    return model

def predict_pytorch(model, left_input, right_input, head_input):
    theta_left, phi_left, theta_right, phi_right, theta_head, phi_head = extract_spherical_angles_from_human(left_input, right_input, head_input)
    input_data = torch.Tensor(np.concatenate((theta_left, phi_left, theta_right, phi_right, theta_head, phi_head), axis=None))

    with torch.no_grad():
        left_arm_pred, right_arm_pred, head_pred = model(input_data)

    # Convert the predictions to numpy arrays
    left_arm_pred = left_arm_pred.numpy()
    right_arm_pred = right_arm_pred.numpy()
    head_pred = head_pred.numpy()
    return left_arm_pred, right_arm_pred, head_pred

def test_keras(model, test_input_data, test_output_data):
    test_loss = model.evaluate(test_input_data, test_output_data)

if __name__ == "__main__":
    file_path = "./robot_configuration_files/qt.yaml"
    qt = robot.Robot()
    qt.import_robot(file_path)
    base = qt.baseDistance

    # df = read_file("combined_actions")
    # actions = ['teacup', 'teapot', 'spoon', 'ladle', 'shallow_plate',
    #            'dinner_plate', 'knife', 'fork', 'salt_shaker',
    #            'sugar_bowl', 'mixer', 'pressure_cooker']

    # action = "teapot"
    # user = 3
    # users = np.arange(1, 21, 1)
    # left_side, right_side, head = read_csv_combined(df, action, user)


    # for i in range(len(left_side)):
    #     points4, points5, points6 = robot_embodiment(left_side[i], right_side[i], head[i], qt, action, user)
    #     robot_pose.append((left_side[i], right_side[i], head[i], points4, points5, points6))
    # plot_animation_3d(robot_pose, base)

    #NN TRAINING
    file_name = "robot_angles"
    theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = read_training_data(file_name)
    model = train_pytorch(theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, 1000)

    #NN TESTING
    df = read_file("combined_actions")
    action = "teapot"
    user = 5
    robot_pose = []
    left_side, right_side, head = read_csv_combined(df, action, user)

    for i in range(len(left_side)):
        angles_left, angles_right, angles_head = predict_pytorch(model, left_side[i], right_side[i], head[i])
        points4, points5, points6 = robot_embodiment(qt, angles_left, angles_right, angles_head)
        robot_pose.append((left_side[i], right_side[i], head[i], points4, points5, points6))
    plot_animation_3d(robot_pose, base)

    #NN TESTING
    # test(theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, num_epochs = 10)