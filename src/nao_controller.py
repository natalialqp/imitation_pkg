# -*- encoding: UTF-8 -*-

import sys
import time
from naoqi import ALProxy
import almath
import numpy as np
import json
import random
import robot
import motion

class SelfExploration(object):
    def __init__(self, robot_name):
        self.motor_babbling_recording = {'right': []}
       # self.motor_babbling_recording = {'left': [], 'right': [],'head': []}
        self.current_ang_pos_Larm = np.zeros((5))
        self.current_ang_pos_Rarm = np.zeros((5))
        self.current_ang_pos_head = np.zeros((2))
        self.key = ''
        self.robot_name = robot_name

    def import_robot(self, file_path):
        #Read robot configuration from the .yaml file
        self.rob = robot.Robot(self.robot_name)
        self.rob.import_robot(file_path)

    def random_angles(self, delta_angle):
        zeros = []
        if "head" in self.key:
            limits = self.rob.physical_limits_head
            for key in limits:
		zeros.append(random.randrange(limits[key][0], limits[key][1] + delta_angle, delta_angle))

        elif "left" in self.key or "right" in self.key:
            limits = self.rob.physical_limits_left
            for key in limits:
                zeros.append(random.randrange(limits[key][0], limits[key][1] + delta_angle, delta_angle))
        return np.array(zeros)


    def motor_babbling(self, robotIP, delta_angle, sequence_len):
        PORT = 9559
        try:
            self.motionProxy = ALProxy("ALMotion", robotIP, PORT)
        except Exception as e:
            print("Could not create proxy to ALMotion")
            print ("Error was: ", e)
            sys.exit(1)
        for key in self.motor_babbling_recording:
            self.key = key
            start = time.time()
            while len(self.motor_babbling_recording[key]) < sequence_len:
                print(len(self.motor_babbling_recording[key]))
		random_move = self.random_angles(delta_angle)
                if "head" in key:
                    new_ang_pos_head = random_move
                    added = self.add_config(new_ang_pos_head)
                    if added:
                        self.current_ang_pos_head = new_ang_pos_head.copy()
                elif "left" in key:
                    new_ang_pos_arm = random_move
                    added = self.add_config(new_ang_pos_arm)
                    if added:
                        self.current_ang_pos_Larm = new_ang_pos_arm.copy()
                elif "right" in key:
                    new_ang_pos_arm = random_move
                    added = self.add_config(new_ang_pos_arm)
                    if added:
                        self.current_ang_pos_Rarm = new_ang_pos_arm.copy()
                self.motor_publisher()
            end = time.time()
            elapsed = (end - start) / 60
            print("Time taken ", self.key, ": ", elapsed, "minutes")

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
        if self.key == "head":
            limits = self.rob.physical_limits_head
        else:
            limits = self.rob.physical_limits_left
        for count, i in enumerate(limits):
            if candidate_pos[count] < limits[i][0] or candidate_pos[count] > limits[i][1]:
                inside_limits = False
        return inside_limits

    def motor_publisher(self):
        LeftLeg  = [0, 0, -25, 40, -20, 0]
        RightLeg = [0, 0, -25, 40, -20, 0]
        Head = list(self.current_ang_pos_head)
        LeftArm = list(self.current_ang_pos_Larm)
	LeftArm.append(0)
        RightArm = list(self.current_ang_pos_Rarm)
	RightArm.append(0)
	name = "Body"
        self.motionProxy.setStiffnesses(name, 1.0)
        TargetAngles = Head + LeftArm + LeftLeg + RightLeg + RightArm
        TargetAngles = [ x * motion.TO_RAD for x in TargetAngles]
        MaxSpeedFraction = 0.2
        self.motionProxy.angleInterpolationWithSpeed(name, TargetAngles, MaxSpeedFraction)
#       motionProxy.post.angleInterpolation(names, angleLists, timeLists, isAbsolute)
        time.sleep(0.5)

if __name__ == "__main__":
    robotIp = "192.168.0.101"

    if len(sys.argv) <= 1:
        print ("Usage python almotion_angleinterpolationreactif.py robotIP (optional default: 127.0.0.1)")
    else:
        robotIp = sys.argv[1]

    robot_name = 'nao'
    file_path = robot_name + ".yaml"

    delta_angle = 5
    amount_of_points = 150
    self_explorator = SelfExploration(robot_name)
    self_explorator.import_robot(file_path)
    self_explorator.motor_babbling(robotIp, delta_angle, amount_of_points)
    file_path = "self_exploration_nao_" + str(amount_of_points) + ".txt"
    with open(file_path, 'w') as file:
        json.dump(str(self_explorator.motor_babbling_recording), file)
