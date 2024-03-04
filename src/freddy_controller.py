# -*- encoding: UTF-8 -*-

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
        # self.motor_babbling_recording = {'left': [], 'right': []}
        self.motor_babbling_recording = {'left': []}
        self.current_ang_pos_Larm = np.zeros((7))
        self.current_ang_pos_Rarm = np.zeros((7))
        self.key = ''
        self.robot_name = robot_name

    def import_robot(self, file_path):
        self.rob = robot.Robot(self.robot_name)
        self.rob.import_robot(file_path)

    def random_angles(self, delta_angle, limbType):
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
        added = False
        candidate_pos = list(np.round(candidate_pos, decimals = 2))
        inside_limits = self.check_limits(candidate_pos)
        if inside_limits:
            if candidate_pos not in self.motor_babbling_recording[self.key]:
                self.motor_babbling_recording[self.key].append(candidate_pos)
                added = True
        return added

    def check_limits(self, candidate_pos):
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
        new_angle_vec = np.zeros_like(vec)
        for i in range(len(vec)):
            if vec[i] > 0:
                new_angle_vec[i] = vec[i]
            else:
                new_angle_vec[i] = 360 - vec[i]

        return new_angle_vec

    def motor_publisher(self, base):

        # print("Starting angular action movement ...")
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

        # if finished:
        #     print("Angular movement completed")
        # else:
        #     print("Timeout on action notification wait")

        # Create closure to set an event after an END or an ABORT
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
