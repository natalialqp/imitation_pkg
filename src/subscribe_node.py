#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import json
import robot
import os
import random

class SelfExploration(object):

    def __init__(self):
        self.motor_babbling_recording = {'left': [], 'right': [], 'head': []}
        self.right_pub = rospy.Publisher('/qt_robot/right_arm_position/command', Float64MultiArray, queue_size = 10)
        self.left_pub = rospy.Publisher('/qt_robot/left_arm_position/command', Float64MultiArray, queue_size = 10)
        self.head_pub = rospy.Publisher('/qt_robot/head_position/command', Float64MultiArray, queue_size = 10)
        self.current_ang_pos_arm = np.zeros((3))
        self.current_ang_pos_head = np.zeros((2))
        self.current_tor_head = np.zeros((2))
        self.current_tor_arm = np.zeros((3))
        self.key = ''

    def import_robot(self, file_path):
        #Read robot configuration from the .yaml file
        self.rob = robot.Robot()
        self.rob.import_robot(file_path)

    def motor_babbling(self, delta_angle, sequence_len):
        waiting = 0
        for key in self.motor_babbling_recording:
            self.key = key
            while len(self.motor_babbling_recording[key]) < sequence_len:
                self.joint_state_sub = rospy.Subscriber('/qt_robot/joints/state', JointState, self.qt_joint_state_cb)
                while self.current_ang_pos_arm[2] == 0:
                    waiting += 1
                random_move = self.random_distribution(delta_angle)
                # rospy.loginfo("RANDOM MOVE: ")
                # rospy.loginfo(random_move)

                if "head" in key:
                    new_ang_pos_head = np.array(self.current_ang_pos_head) + random_move
                    added = self.add_config(new_ang_pos_head)
                    if added:
                        self.current_ang_pos_head = new_ang_pos_head.copy()
                else:
                    new_ang_pos_arm = np.array(self.current_ang_pos_arm) + random_move
                    added = self.add_config(new_ang_pos_arm)
                    if added:
                        self.current_ang_pos_arm = new_ang_pos_arm.copy()
                    rospy.loginfo(new_ang_pos_arm)
                self.joint_angle_publisher()

    def add_config(self, candidate_pos):
        added = False
        candidate_pos = list(np.round(candidate_pos, decimals = 2))
        inside_limits = self.check_limits(candidate_pos)
        rospy.loginfo(self.key)
        rospy.loginfo(self.motor_babbling_recording[self.key])
        if inside_limits:
            if candidate_pos not in self.motor_babbling_recording[self.key]:
                self.motor_babbling_recording[self.key].append(candidate_pos)
                rospy.loginfo(candidate_pos)
                added = True
        return added

    def check_limits(self, candidate_pos):
        inside_limits = True
        if self.key == "head":
            limits = self.rob.physical_limits_head
        else:
            limits = self.rob.physical_limits_left
        for count, i in enumerate(limits):
            if candidate_pos[count] < limits[i][0] or candidate_pos[count] > limits[i][1]:
                inside_limits = False
        return inside_limits

    def qt_joint_state_cb(self, joint_state_msg):
        '''
        This function saves the recorded_positions taken from the robot's sensors
        Input: joint_state_msg
        '''
        if 'head' in self.key:
            self.current_ang_pos_head = [joint_state_msg.position[0], joint_state_msg.position[1]]
            self.current_tor_head = [joint_state_msg.effort[0], joint_state_msg.effort[1]]
            # rospy.loginfo(self.current_ang_pos_head)
        elif 'left' in self.key:
            self.current_ang_pos_arm = [joint_state_msg.position[3], joint_state_msg.position[4], joint_state_msg.position[2]]
            self.current_tor_arm = [joint_state_msg.effort[3], joint_state_msg.effort[4], joint_state_msg.effort[2]]
            # rospy.loginfo(self.current_ang_pos_arm)
        elif 'right' in self.key:
            self.current_ang_pos_arm = [joint_state_msg.position[6], joint_state_msg.position[7], joint_state_msg.position[5]]
            self.current_tor_arm = [joint_state_msg.effort[6], joint_state_msg.effort[7], joint_state_msg.effort[5]]
            # rospy.loginfo(self.current_ang_pos_arm)

    def check_torque(self):
        #torque for HeadPitch (+) torque after force applied
        #torque for HeadYaw (+/-) in different directions -> stable is 0
        collision = False
        collision_joints = []
        if 'head' in self.key:
            if self.current_tor_head[0] > 0:
                collision = True
                collision_joints.append("HeadPitch")

            if self.current_tor_head[1] > 30 or self.current_tor_head[1] < -30:
                collision = True
                collision_joints.append("HeadYaw")
        # LeftShoulderPitch (+) torque after force applied to front, (-) torque after force applied to back
        # LeftShoulderRoll (-) torque after fo  rce applied
        # LeftElbowRoll (-) torque after force applied
        elif 'left' in self.key:
            if self.current_tor_arm[0] < -30:
                collision = True
                collision_joints.append("LeftShoulderPitch+")

            elif self.current_tor_arm[0] > 30:
                collision = True
                collision_joints.append("LeftShoulderPitch-") #crashes with belly, need to reduce angle

            if self.current_tor_arm[1] < -20:
                collision = True
                collision_joints.append("LeftShoulderRoll")

            if self.current_tor_arm[2] < -20:
                collision = True
                collision_joints.append("LeftElbowRoll")
        # RightShoulderPitch (-) torque after force applied to front, (+) torque after force applied to back
        # RightShoulderRoll (-) torque after force applied
        # RightElbowRoll (-) torque after force applied
        elif 'right' in self.key:
            if self.current_tor_arm[0] > 30:
                collision = True
                collision_joints.append("RightShoulderPitch-")

            elif self.current_tor_arm[0] < -30:
                collision = True
                collision_joints.append("RightShoulderPitch+") #crashes with belly, need to increase angle

            if self.current_tor_arm[1] < -20:
                collision = True
                collision_joints.append("RightShoulderRoll")

            if self.current_tor_arm[2] < -20:
                collision = True
                collision_joints.append("RightElbowRoll")
        return collision, collision_joints

    def random_distribution(self, delta_angle):
        collision, joint_list = self.check_torque()
        print("COLLISION: ", collision, joint_list)
        sign = random.choice([-1, 1])

        if "head" in self.key:
            rand = np.random.randint(0, 2)
            zeros = [0] * 2
            zeros[rand] = sign * delta_angle
            if collision and "Pitch" in joint_list:
                zeros = np.abs(zeros)

        elif "left" in self.key or "right" in self.key:
            zeros = [0] * 3
            rand = np.random.randint(0, 3)
            zeros[rand] = sign * delta_angle
            if collision:
                if "Roll" in joint_list:
                    rand = np.random.randint(1, 3)
                    zeros[rand] = sign * delta_angle
                    zeros = np.abs(zeros)
                elif "Pitch+" in joint_list:
                    zeros = np.abs(zeros)
                elif "Pitch-" in joint_list:
                    zeros = -np.abs(zeros)
        return np.array(zeros)

    def joint_angle_publisher(self):
        # Publish angles to robot
        # wait for publisher/subscriber connections
            wtime_begin = rospy.get_time()
            while (self_explorator.right_pub.get_num_connections() == 0):
                rospy.loginfo("waiting for subscriber connections...")
                if rospy.get_time() - wtime_begin > 10.0:
                    rospy.logerr("Timeout while waiting for subscribers connection!")
                    sys.exit()
                rospy.sleep(1)
            rospy.loginfo("publishing motor command...")
            try:
                if 'head' in self.key:
                    ref_head = Float64MultiArray()
                    ref_head.data = [self.current_ang_pos_head[0], self.current_ang_pos_head[1]]
                    self_explorator.head_pub.publish(ref_head)
                elif 'right' in self.key:
                    ref_right = Float64MultiArray()
                    ref_right.data = [self.current_ang_pos_arm[0], self.current_ang_pos_arm[1], self.current_ang_pos_arm[2]]
                    self_explorator.right_pub.publish(ref_right)
                elif 'left' in self.key:
                    ref_left = Float64MultiArray()
                    ref_left.data = [self.current_ang_pos_arm[0], self.current_ang_pos_arm[1], self.current_ang_pos_arm[2]]
                    self_explorator.left_pub.publish(ref_left)
                rospy.sleep(1)
            except rospy.ROSInterruptException:
                rospy.logerr("could not publish motor command!")
            rospy.loginfo("motor command published")

    def execute_online(self, delta_angle, amount_of_points):
        # wait for publisher/subscriber connections
        wtime_begin = rospy.get_time()
        while (self.right_pub.get_num_connections() == 0):
            rospy.loginfo("waiting for subscriber connections...")
            if rospy.get_time() - wtime_begin > 10.0:
                rospy.logerr("Timeout while waiting for subscribers connection!")
                sys.exit()
            rospy.sleep(1)
        rospy.loginfo("publishing motor command...")
        try:
            self.motor_babbling(delta_angle, amount_of_points)
        except rospy.ROSInterruptException:
            rospy.logerr("could not publish motor command!")
        rospy.loginfo("motor command published")

if __name__ == '__main__':
    rospy.init_node('Self-exploration')
    rospy.loginfo("started!")
    directory = os.getcwd()
    robot_name = 'qt'
    file_path = directory  + "/robot_configuration_files/" + robot_name + ".yaml"

    delta_angle = 10
    amount_of_points = 10
    self_explorator = SelfExploration()
    self_explorator.import_robot(file_path)
    self_explorator.execute_online(delta_angle, amount_of_points)
    rospy.loginfo(self_explorator.motor_babbling_recording)

    file_path = directory  + "/data/self_exploration_qt.txt"
    with open(file_path, 'w') as file:
        json.dump(self_explorator.motor_babbling_recording, file)
