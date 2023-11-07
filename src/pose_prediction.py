import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import robot
from matplotlib.widgets import Slider
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

plt.rcParams.update({'font.size': 20})

number_arm_human_joints = 5
number_head_human_joints = 3
robot_head_dimensions = np.array([0, 0.0962])
robot_arm_dimensions = np.array([0.08, 0.0445, 0.07708, 0.184])

human_joints_head  = ['JOINT_LEFT_COLLAR', 'JOINT_NECK', 'JOINT_HEAD']
human_joints_left  = ['JOINT_LEFT_COLLAR', 'JOINT_LEFT_SHOULDER', 'JOINT_LEFT_ELBOW', 'JOINT_LEFT_WRIST', 'JOINT_LEFT_HAND']
human_joints_right = ['JOINT_RIGHT_COLLAR', 'JOINT_RIGHT_SHOULDER', 'JOINT_RIGHT_ELBOW', 'JOINT_RIGHT_WRIST', 'JOINT_RIGHT_HAND']

class Prediction(object):

    def __init__(self, file_path, name):

        self.robot = robot.Robot()
        self.robot.import_robot(file_path)
        self.base = self.robot.baseDistance
        self.robotName = name

    def cartesian_to_spherical(self, cartesian_points):
        # Convert Cartesian coordinates to spherical coordinates
        r     = np.linalg.norm(cartesian_points, axis=1)                    # Radial distance
        theta = np.arccos(cartesian_points[:, 2] / r)                       # Inclination angle
        phi   = np.arctan2(cartesian_points[:, 1], cartesian_points[:, 0])  # Azimuth angle

        # Set of points in spherical coordinates
        # spherical_points = np.column_stack((r, theta, phi))
        # return spherical_points
        return theta, phi

    def spherical_to_cartesian(self, spherical_points, robot):
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

    def human_to_robot(self, points, isArm):
        #Calculate the n-th discrete difference along the given axis
        vectors = np.diff(points, axis=0)
        spherical_points = self.cartesian_to_spherical(vectors)
        if isArm:
            spherical_points[:, 0] = robot_arm_dimensions
        else:
            spherical_points[:, 0] = robot_head_dimensions
        coordinates = self.spherical_to_cartesian(np.vstack((np.array([0, 0, 0]), spherical_points)), robot)
        #Return the cumulative sum of the elements along a given axis
        return np.cumsum(coordinates, axis=0)

    def extract_spherical_angles_from_human(self, left, right, head):
        left_theta, left_phi = self.cartesian_to_spherical(np.diff(left, axis=0))
        right_theta, right_phi = self.cartesian_to_spherical(np.diff(right, axis=0))
        head_theta, head_phi = self.cartesian_to_spherical(np.diff(head, axis=0))
        return left_theta, left_phi, right_theta, right_phi, head_theta, head_phi

    def robot_embodiment(self, angles_left, angles_right, angles_head):

        # left_shoulder_pitch -> 0
        # left_shoulder_roll --> 1
        # left_elbow_roll -----> 2
        # right_shoulder_pitch-> 3
        # right_shoulder_roll -> 4
        # right_elbow_roll ----> 5
        # head_yaw ------------> 6
        # head_pitch-----------> 7

        angles = np.concatenate((angles_left, angles_right, angles_head), axis=None)
        if self.robotName == "qt":
            l, r, h = self.robot.forward_kinematics_qt(angles)
        elif self.robotName == "nao":
            l, r, h = self.robot.forward_kinematics_nao(angles)
        elif self.robotName == "gen3":
            l, r, h = self.robot.forward_kinematics_kinova(angles)
        return self.matrix_array_to_list_list(l), self.matrix_array_to_list_list(r), self.matrix_array_to_list_list(h)

    def matrix_array_to_list_list(self, vec):
        new_list = []
        for i in vec:
            new_list.append(list(i))
        return np.array(new_list)

    def vectorise_string(self, vec):
        aux_data = ''.join([i for i in vec if not (i=='[' or i==']' or i==',')])
        data = np.array(aux_data.split())
        return list(data[0:3].astype(float))

    def vectorise_spherical_data(self, vec):
        aux_data = ''.join([i for i in vec if not (i=='[' or i==']' or i==',')])
        data = np.array(aux_data.split())
        return list(data[0:4].astype(float))

    def read_csv_combined(self, df, action, user):
        left_side = []
        right_side = []
        head = []
        aux_arm = [[]] * number_arm_human_joints
        aux_head = [[]] * number_head_human_joints
        for i in range(len(df)):
            if action == df['action'][i] and user == df['participant_id'][i]:
                for count, left in enumerate(human_joints_left):
                    aux_arm[count] = self.vectorise_string(df[left][i])
                left_side.append(aux_arm.copy())
                for count, right in enumerate(human_joints_right):
                    aux_arm[count] = self.vectorise_string(df[right][i])
                right_side.append(aux_arm.copy())
                for count, _head in enumerate(human_joints_head):
                    aux_head[count] = self.vectorise_string(df[_head][i])
                head.append(aux_head.copy())
        return np.array(left_side), np.array(right_side), np.array(head)

    def read_training_data(self, file_name):

        df = self.read_file(file_name)
        theta_left = np.array([self.vectorise_spherical_data(i) for i in df['left_arm_human_theta']])
        theta_right = np.array([self.vectorise_spherical_data(i) for i in df['right_arm_human_theta']])
        theta_head = np.array([self.vectorise_spherical_data(i) for i in df['head_human_theta']])
        phi_left = np.array([self.vectorise_spherical_data(i) for i in df['left_arm_human_phi']])
        phi_right = np.array([self.vectorise_spherical_data(i) for i in df['right_arm_human_phi']])
        phi_head = np.array([self.vectorise_spherical_data(i) for i in df['head_human_phi']])
        left_arm_robot = np.array([self.vectorise_spherical_data(i) for i in df['left_arm_robot']])
        right_arm_robot = np.array([self.vectorise_spherical_data(i) for i in df['right_arm_robot']])
        head_robot = np.array([self.vectorise_spherical_data(i) for i in df['head_robot']])

        return theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot

    def save_mapping_csv(self, file_name, action, user, lht, lhp, hht, rht, rhp, hhp, ra):

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

    def create_3d_plot(self):
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # Set the axis labels and limits for both subplots (adjust as needed)
        ax1.set_xlabel("\n X [m]", linespacing=3.2)
        ax1.set_ylabel("\n Y [m]", linespacing=3.2)
        ax1.set_zlabel("\n Z [m]", linespacing=3.2)
        ax1.set_xlim([1, 3])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])

        ax2.set_xlabel("\n X [m]", linespacing=3.2)
        ax2.set_ylabel("\n Y [m]", linespacing=3.2)
        ax2.set_zlabel("\n Z [m]", linespacing=3.2)
        ax2.set_xlim([-0.5, 0.5])
        ax2.set_ylim([-0.5, 0.5])
        ax2.set_zlim([0, 1])

        return fig, ax1, ax2

    def plot_animation_3d(self, point_clouds_list):
        if not isinstance(point_clouds_list, list) or not all(isinstance(pc, tuple) and len(pc) == 6 for pc in point_clouds_list):
            raise ValueError("Invalid input data. Expecting a list of tuples, each containing 5 point clouds")

        fig, ax1, ax2 = self.create_3d_plot()

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
        ax2.scatter(self.base[0], self.base[1], self.base[2], marker="o")

        def update_animation(val):
            frame = int(slider.val)
            update(frame)
            fig.canvas.draw_idle()

        slider.on_changed(update_animation)
        # Show the animation
        plt.show()

    def read_file(self, name):
        return pd.read_csv('./data/' + name + '.csv')

    def train_pytorch(self, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, num_epochs = 500):

        # Create a custom PyTorch model class
        class MultiOutputModel(nn.Module):
            def __init__(self):
                super(MultiOutputModel, self).__init__()
                self.shared_layer1 = nn.Linear(20, 16)
                # self.shared_layer2 = nn.Linear(128, 128)
                self.left_arm_layer = nn.Linear(16, 3)
                self.right_arm_layer = nn.Linear(16, 3)
                self.head_layer = nn.Linear(16, 2)

            def forward(self, x):
                x = torch.relu(self.shared_layer1(x))
                # x = torch.relu(self.shared_layer2(x))
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
        validation_losses = []
        num_epochs = 1000
        train_input_data, val_input_data, train_output_data_left, val_output_data_left, train_output_data_right, val_output_data_right, train_output_data_head, val_output_data_head = train_test_split(
        input_data, output_data_left, output_data_right, output_data_head, test_size=0.2, random_state=50)

        for epoch in range(num_epochs):
            # Forward pass
            left_arm_pred, right_arm_pred, head_pred = model(train_input_data)

            # Calculate loss for each output branch
            loss_left = criterion(left_arm_pred, train_output_data_left)
            loss_right = criterion(right_arm_pred, train_output_data_right)
            loss_head = criterion(head_pred, train_output_data_head)

            # Total loss as a combination of individual losses
            total_loss = loss_left + loss_right + loss_head

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            training_losses.append(total_loss.item())

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}')

         # For validation, use separate validation data
            with torch.no_grad():  # Disable gradient computation for validation
                val_left_arm_pred, val_right_arm_pred, val_head_pred = model(val_input_data)

                # Calculate validation loss for each output branch
                val_loss_left = criterion(val_left_arm_pred, val_output_data_left)
                val_loss_right = criterion(val_right_arm_pred, val_output_data_right)
                val_loss_head = criterion(val_head_pred, val_output_data_head)

                # Total validation loss as a combination of individual losses
                total_val_loss = val_loss_left + val_loss_right + val_loss_head
                validation_losses.append(total_val_loss.item())

                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {total_val_loss.item()}')

        self.model = model
        # Plot the training loss
        plt.plot(range(1, num_epochs + 1), training_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def dicts_to_lists(self, left, right, head):
        left_side = []
        right_side = []
        head_ = []
        for key in human_joints_head:
            head_.append(head[key])
        for key in human_joints_left:
            left_side.append(left[key])
        for key in human_joints_right:
            right_side.append(right[key])
        return np.array(left_side), np.array(right_side), np.array(head_)

    def predict_pytorch(self, left_input, right_input, head_input):
        theta_left, phi_left, theta_right, phi_right, theta_head, phi_head = self.extract_spherical_angles_from_human(left_input, right_input, head_input)
        input_data = torch.Tensor(np.concatenate((theta_left, phi_left, theta_right, phi_right, theta_head, phi_head), axis=None))

        with torch.no_grad():
            left_arm_pred, right_arm_pred, head_pred = self.model(input_data)

        # Convert the predictions to numpy arrays
        left_arm_pred = left_arm_pred.numpy()
        right_arm_pred = right_arm_pred.numpy()
        head_pred = head_pred.numpy()
        return left_arm_pred, right_arm_pred, head_pred

if __name__ == "__main__":
    robotName = "nao"
    file_path = "./robot_configuration_files/"+ robotName + ".yaml"
    pose_predictor = Prediction(file_path, robotName)

    df = pose_predictor.read_file("combined_actions")
    actions = ['teacup', 'teapot', 'spoon', 'ladle', 'shallow_plate',
               'dinner_plate', 'knife', 'fork', 'salt_shaker',
               'sugar_bowl', 'mixer', 'pressure_cooker']

    robot_pose = []
    action = "teapot"
    user = 3
    users = np.arange(1, 21, 1)
    left_side, right_side, head = pose_predictor.read_csv_combined(df, action, user)


    for i in range(len(left_side)):
        points4, points5, points6 = pose_predictor.robot_embodiment(left_side[i], right_side[i], head[i])
        robot_pose.append((left_side[i], right_side[i], head[i], points4, points5, points6))
    pose_predictor.plot_animation_3d(robot_pose)

    #NN TRAINING
    # file_name = "robot_angles"
    # theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
    # pose_predictor.train_pytorch(theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, 1000)

    #NN TESTING
    # df = pose_predictor.read_file("combined_actions")
    # action = "spoon"
    # user = 5
    # robot_pose = []
    # left_side, right_side, head = pose_predictor.read_csv_combined(df, action, user)

    # for i in range(len(left_side)):
    #     angles_left, angles_right, angles_head = pose_predictor.predict_pytorch(left_side[i], right_side[i], head[i])
    #     points4, points5, points6 = pose_predictor.robot_embodiment(angles_left, angles_right, angles_head)
    #     robot_pose.append((left_side[i], right_side[i], head[i], points4, points5, points6))
    # pose_predictor.plot_animation_3d(robot_pose)