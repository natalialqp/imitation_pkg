# -*- encoding: UTF-8 -*-
# CLONE THE KORTEX REPOSITORY TO GET THE KORTEX ENVIRONMENT OUTSIDE OF THE IMITATION WORKSPACE
# PASTE THIS FILE IN THE FOLLOWING DIRECTORY: kortex/api_python/examples/python/102-Movement_high_level/
# PASTE THE DEPENDENCIES FILES: simulate_position.py, robot.py, homogeneous_transformation.py and robot_configuration_files/gen3.yaml
# RUN THE FILE FROM THE KORTEX DIRECTORY CONNECTED TO THE ROBOT, MAKE SURE THE ROBOT IS TURNED ON AND CONNECTED TO THE NETWORK AND THE COMPUTER WITH THE RESPECT API

import sys
import time
import numpy as np
import json
import random
import robot
import os
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

class SelfExploration(object):
    def __init__(self, robot_name):
        """
        Initialize the FreddyController class.

        Parameters:
        - robot_name (str): The name of the robot.

        Attributes:
        - motor_babbling_recording (dict): A dictionary to store motor babbling recordings for the left and right arms.
        - current_ang_pos_Larm (numpy.ndarray): An array to store the current angular positions of the left arm.
        - current_ang_pos_Rarm (numpy.ndarray): An array to store the current angular positions of the right arm.
        - key (str): A variable to store a key value.
        - robot_name (str): The name of the robot.

        Returns:
        - None
        """
        self.motor_babbling_recording = {'left': [], 'right': []}
        self.current_ang_pos_Larm = np.zeros((7))
        self.current_ang_pos_Rarm = np.zeros((7))
        self.key = ''
        self.robot_name = robot_name

    def import_robot(self, file_path):
            """
            Imports a robot from a given file path.

            Args:
                file_path (str): The path to the file containing the robot information.

            Returns:
                None
            """
            self.rob = robot.Robot(self.robot_name)
            self.rob.import_robot(file_path)

    def random_angles(self, delta_angle, limbType):
        """
        Generate an array of random angles within the specified delta angle range for a given limb type.

        Parameters:
        delta_angle (int): The range of angles to generate, specified in degrees.
        limbType (str): The type of limb for which to generate random angles. Can be "left" or "right".

        Returns:
        numpy.ndarray: An array of random angles within the specified range for the given limb type.
        """
        zeros = []
        limits = np.zeros((7))
        if limbType == "left":
            limits = self.rob.physical_limits_left
        elif limbType == "right":
            limits = self.rob.physical_limits_right
        for key in limits:
            zeros.append(random.randrange(limits[key][0], limits[key][1] + delta_angle, delta_angle))
        return np.array(zeros)

    def motor_babbling(self, base, delta_angle, sequence_len):
        """
        Perform motor babbling by randomly moving the arms and recording the configurations.

        Args:
            base (float): The base value for motor movement.
            delta_angle (float): The maximum angle by which the arms can move randomly.
            sequence_len (int): The desired length of the motor babbling sequence.

        Returns:
            bool: True if motor babbling is successfully performed.
        """
        for key in self.motor_babbling_recording:
            self.key = key
            start = time.time()
            while len(self.motor_babbling_recording[key]) < sequence_len:

                print(len(self.motor_babbling_recording[key]))
                if "left" in key:
                    random_move = self.random_angles(delta_angle, "left")
                    new_ang_pos_arm = random_move
                    added = self.add_config(random_move)
                    if added:
                        self.current_ang_pos_Larm = new_ang_pos_arm.copy()
                elif "right" in key:
                    random_move = self.random_angles(delta_angle, "right")
                    new_ang_pos_arm = random_move
                    added = self.add_config(new_ang_pos_arm)
                    if added:
                        self.current_ang_pos_Rarm = new_ang_pos_arm.copy()
                self.motor_publisher(base)
                if len(self.motor_babbling_recording[key]) % 10 == 0 or len(self.motor_babbling_recording[key]) == 7:
                    print(self.motor_babbling_recording[key])
                    end = time.time()
                    elapsed = (end - start) / 60
                    print("Time taken ", self.key, ": ", elapsed, "minutes")
        return True

    def add_config(self, candidate_pos):
        """
        Adds a candidate position to the motor babbling recording if it is within the limits and not already present.

        Args:
            candidate_pos (list): The candidate position to be added.

        Returns:
            bool: True if the candidate position was added successfully, False otherwise.
        """
        added = False
        candidate_pos = list(np.round(candidate_pos, decimals=2))
        inside_limits = self.check_limits(candidate_pos)
        if inside_limits:
            if candidate_pos not in self.motor_babbling_recording[self.key]:
                self.motor_babbling_recording[self.key].append(candidate_pos)
                added = True
        return added

    def check_limits(self, candidate_pos):
        """
        Check if the candidate position is within the physical limits of the robot.

        Args:
            candidate_pos (list): The candidate position to be checked.

        Returns:
            bool: True if the candidate position is within the limits, False otherwise.
        """
        inside_limits = True
        if self.key == "right":
            limits = self.rob.physical_limits_right
        elif self.key == "left":
            limits = self.rob.physical_limits_left
        for count, i in enumerate(limits):
            if candidate_pos[count] < limits[i][0] or candidate_pos[count] > limits[i][1]:
                inside_limits = False
        return inside_limits

    def robot_range(self, vec):
        """
        Converts the given vector of angles to a new vector where negative angles are converted to positive angles.
        Kinova gen3 accepts only possitive angles.

        Args:
            vec (numpy.ndarray): The input vector of angles.

        Returns:
            numpy.ndarray: The new vector with converted angles.
        """
        new_angle_vec = np.zeros_like(vec)
        for i in range(len(vec)):
            if vec[i] > 0:
                new_angle_vec[i] = vec[i]
            else:
                new_angle_vec[i] = 360 + vec[i]
        return new_angle_vec

    def motor_publisher(self, base):
        """
        Publishes motor commands to control the robot's movement.

        Args:
            base: The base object representing the robot's base.

        Returns:
            None
        """
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""
        actuator_count = base.GetActuatorCount()
        n_dof = actuator_count.count
        locked_joints = [0]

        if self.key == "left":
            angles = self.current_ang_pos_Larm
        elif self.key == "right":
            angles = self.current_ang_pos_Rarm

        # Lock joints
        for i in locked_joints:
            angles[i] = 0

        angles = self.robot_range(angles)

        for joint_id, val in enumerate(range(n_dof)):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angles[joint_id]

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        # print("Executing action")
        base.ExecuteAction(action)

        # print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e = e):
            print("EVENT : " + \
                Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def robot_config(self, delta_angle, sequence_len):
        """
        Configures the robot by creating a connection to the device and executing motor babbling.

        Args:
            delta_angle (float): The angle by which the robot's motors will be moved during motor babbling.
            sequence_len (int): The length of the motor babbling sequence.

        Returns:
            int: 0 if the robot configuration is successful, 1 otherwise.
        """
        # Import the utilities helper module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import utilities

        # Parse arguments
        args = utilities.parseConnectionArguments()
        # Create connection to the device and get the router
        with utilities.DeviceConnection.createTcpConnection(args) as router:

            # Create required services
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)

            # Example core
            success = True
            success &= self.motor_babbling(base, delta_angle, sequence_len)

            # You can also refer to the 110-Waypoints examples if you want to execute
            # a trajectory defined by a series of waypoints in joint space or in Cartesian space

            return 0 if success else 1

if __name__ == "__main__":

    robot_name = 'gen3'
    file_path = "robot_configuration_files/" + robot_name + ".yaml"

    delta_angle = 10
    amount_of_points = 30

    self_explorator = SelfExploration(robot_name)
    self_explorator.import_robot(file_path)
    self_explorator.robot_config(delta_angle, amount_of_points)

    file_path = "self_exploration_freddy_" + str(amount_of_points) + "_" + self_explorator.key + ".txt"
    with open(file_path, 'w') as file:
        json.dump(str(self_explorator.motor_babbling_recording), file)