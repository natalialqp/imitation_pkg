import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd
import robot
from matplotlib.widgets import Slider

number_arm_human_joints = 5
number_head_human_joints = 3
human_joints_head = ['JOINT_LEFT_COLLAR', 'JOINT_NECK', 'JOINT_HEAD']
human_joints_left = ['JOINT_LEFT_COLLAR', 'JOINT_LEFT_SHOULDER', 'JOINT_LEFT_ELBOW', 'JOINT_LEFT_WRIST', 'JOINT_LEFT_HAND']
human_joints_right = ['JOINT_LEFT_COLLAR', 'JOINT_RIGHT_SHOULDER', 'JOINT_RIGHT_ELBOW', 'JOINT_RIGHT_WRIST', 'JOINT_RIGHT_HAND']
robot_head_dimensions = np.array([0.04, 0.04])
robot_arm_dimensions = np.array([0.08, 0.12, 0.18, 0.04])

def cartesian_to_spherical(cartesian_points):
    # Convert Cartesian coordinates to spherical coordinates
    r = np.linalg.norm(cartesian_points, axis=1)  # Radial distance
    theta = np.arccos(cartesian_points[:, 2] / r)  # Inclination angle
    phi = np.arctan2(cartesian_points[:, 1], cartesian_points[:, 0])  # Azimuth angle

    # Set of points in spherical coordinates
    spherical_points = np.column_stack((r, theta, phi))
    return spherical_points

def spherical_to_cartesian(spherical_points):
    r = spherical_points[:, 0]  # Radial distance
    theta = spherical_points[:, 1]  # Inclination angle
    phi = spherical_points[:, 2]  # Azimuth angle

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Set of points in Cartesian coordinates
    cartesian_points = np.column_stack((x, y, z))
    return cartesian_points

def human_to_robot(points, isArm):
    vectors = np.diff(points, axis=0)
    spherical_points = cartesian_to_spherical(vectors)
    if isArm:
        spherical_points[:, 0] = robot_arm_dimensions
    else:
        spherical_points[:, 0] = robot_head_dimensions
    coordinates = spherical_to_cartesian(np.vstack((np.array([0, 0, 0]), spherical_points)))
    return np.cumsum(coordinates, axis=0)

def robot_embodiment(points1, points2, points3):
    points4 = human_to_robot(points1, True)
    points5 = human_to_robot(points2, True)
    points6 = human_to_robot(points3, False)
    return points4, points5, points6

def vectorise_string(vec):
    aux_data = ''.join([i for i in vec if not (i=='[' or i==']' or i==',')])
    data = np.array(aux_data.split())
    return list(data[0:3].astype(float))

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

def create_3d_plot():
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Set the axis labels and limits for both subplots (adjust as needed)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([1, 3])
    ax1.set_ylim([-0.8, 0.8])
    ax1.set_zlim([-0.8, 0.8])

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.3, 0.3])
    ax2.set_ylim([-0.3, 0.3])
    ax2.set_zlim([-0.3, 0.3])

    return fig, ax1, ax2

def plot_animation_3d(point_clouds_list):
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
        ax1.set_title('Human')
        ax2.set_title('Robot')

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

    def update_animation(val):
        frame = int(slider.val)
        update(frame)
        fig.canvas.draw_idle()

    slider.on_changed(update_animation)

    # Show the animation
    plt.show()

def read_file(name):
    return pd.read_csv('./data/' + name + '.csv')

if __name__ == "__main__":

    df = read_file("combined_actions")
    actions = ['teacup', 'teapot']
    users = np.arange(1, 21, 1)
    left_side, right_side, head = read_csv_combined(df, "salt_shaker", 2)

    robot_pose = []
    for i in range(len(left_side)):
        points4, points5, points6 = robot_embodiment(left_side[i], right_side[i], head[i])
        robot_pose.append((left_side[i], right_side[i], head[i], points4, points5, points6))
    plot_animation_3d(robot_pose)