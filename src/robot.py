import numpy as np
import yaml
from homogeneous_transformation import HT
import math
import simulate_position

class Robot(object):

    def __init__(self):
        self.robot = None
        self.baseAngles = []
        self.baseDistance = []
        self.leftArmAngles = []
        self.rightArmAngles = []
        self.headAngles = []
        self.leftArmDistance = []
        self.rightArmDistance = []
        self.headDistance = []
        self.physical_limits_left = []
        self.physical_limits_right = []
        self.physical_limits_head = []
        self.robotDict = {}

    def read_yaml_file(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def import_robot(self, file_path):
        data = self.read_yaml_file(file_path)
        self.baseAngles = data["base"]["base-distance"]
        self.baseDistance = data["base"]["base-angle"]
        self.leftArmAngles = data["angles-left-arm"]
        self.leftArmDistance = data["distances-left-arm"]
        self.rightArmAngles = data["angles-right-arm"]
        self.rightArmDistance = data["distances-right-arm"]
        self.headAngles = data["angles-head"]
        self.headDistance = data["distances-head"]
        self.robotDict = data["robot-dict"]
        self.robotAgularDict = data["robot-angular-dict"]
        self.physical_limits_left = data["real-angles-left-arm"]
        self.physical_limits_right = data["real-angles-right-arm"]
        self.physical_limits_head = data["real-angles-head"]

    def forward_kinematics(self, angles):
        '''
        This function receives an angle configuration and calculates the spatial position of the end effector and the elbow
        left_shoulder_pitch -> 0
        left_shoulder_roll --> 1
        left_elbow_roll -----> 2
        right_shoulder_pitch-> 3
        right_shoulder_roll -> 4
        right_elbow_roll ----> 5
        head_yaw ------------> 6
        head_pitch-----------> 7
        Returns the position of all joints of both arms from the closest joint to the base to the end-effector at the end
        '''
        a = HT()
        HT_leftArm = []
        HT_rightArm = []
        HT_head = []
        chain_leftArm = []
        chain_rightArm = []
        chain_head = []
        leftArmAngles = {}
        rightArmAngles = {}
        headAngles = {}
        pos_leftArm = []
        pos_rightArm = []
        pos_head = []

        leftArmAngles["collar"] = self.leftArmAngles["collar"]
        leftArmAngles["elbow"] = self.leftArmAngles["elbow"] - np.array([-angles[2], 0, 0])
        leftArmAngles["shoulder-roll"] = self.leftArmAngles["shoulder-roll"] + np.array([angles[1], 0, 0])
        leftArmAngles["shoulder-pitch"] = self.leftArmAngles["shoulder-pitch"] + np.array([0, angles[0], 0])
        leftArmAngles["hand"] = self.leftArmAngles["hand"]

        rightArmAngles["collar"] = self.rightArmAngles["collar"]
        rightArmAngles["elbow"] = self.rightArmAngles["elbow"] + np.array([angles[5], 0, 0])
        rightArmAngles["shoulder-roll"] = self.rightArmAngles["shoulder-roll"] + np.array([angles[4], 0, 0])
        rightArmAngles["shoulder-pitch"] = self.rightArmAngles["shoulder-pitch"] + np.array([0, -angles[3], 0])
        rightArmAngles["hand"] = self.rightArmAngles["hand"]

        headAngles["head-yaw"] = self.headAngles["head-yaw"] + np.array([0, 0, angles[6]])
        headAngles["head-pitch"] = self.headAngles["head-pitch"] + np.array([0, angles[7], 0])
        headAngles["collar"] = self.headAngles["collar"]
        headAngles["camera"] = self.headAngles["camera"]

        for key in self.leftArmDistance:
            HT_leftArm.append(a.get_homogeneous_transform(leftArmAngles[key], self.leftArmDistance[key]))
            HT_rightArm.append(a.get_homogeneous_transform(rightArmAngles[key], self.rightArmDistance[key]))

        for key in self.headDistance:
            HT_head.append(a.get_homogeneous_transform(headAngles[key], self.headDistance[key]))

        chain_leftArm.append(HT_leftArm[0])
        chain_rightArm.append(HT_rightArm[0])
        chain_head.append(HT_head[0])

        for i in range(len(self.leftArmDistance) - 1):
            chain_leftArm.append(np.dot(chain_leftArm[i], HT_leftArm[i + 1]))
            chain_rightArm.append(np.dot(chain_rightArm[i], HT_rightArm[i + 1]))

        for i in range(len(self.headDistance) - 1):
            chain_head.append(np.dot(chain_head[i], HT_head[i + 1]))

        for i in range(0, len(chain_leftArm)):
            pos_leftArm.append(a.get_translation(chain_leftArm[i]))
            pos_rightArm.append(a.get_translation(chain_rightArm[i]))

        for i in range(0, len(chain_head)):
            pos_head.append(a.get_translation(chain_head[i]))

        return pos_leftArm, pos_rightArm, pos_head

    def pos_vec_to_robot_dict(self, pos_left, pos_right, pos_head):
        for count, i in enumerate(pos_left):
            self.robotDict["jointLeft_" + str(count)] = np.around(i, decimals=2)
        for count, i in enumerate(pos_right):
            self.robotDict["jointRight_" + str(count)] = np.around(i, decimals=2)
        for count, i in enumerate(pos_head):
            self.robotDict["jointHead_" + str(count)] = np.around(i, decimals=2)
        return self.robotDict

    def pos_mat_to_robot_mat_dict(self, pos_left, pos_right, pos_head):
        vec = []
        for i in range(len(pos_left)):
            aux_vec = self.pos_vec_to_robot_dict(pos_left[i], pos_right[i], pos_head[i])
            vec.append(aux_vec.copy())
        return vec

    def angular_vec_to_dict(self, vec):
        angular_dict = {}
        for count, key in enumerate(self.robotAgularDict):
            angular_dict[key] = vec[count]
        return angular_dict

    def angular_mat_to_mat_dict(self, mat):
        vec = []
        for i in range(len(mat)):
            vec.append(self.angular_vec_to_dict(mat[i]))
        return vec

    def calculate_distance_of_joints(self):
        left_distances = [np.linalg.norm(self.leftArmDistance[i]) for i in self.leftArmDistance]
        right_distances = [np.linalg.norm(self.rightArmDistance[i]) for i in self.rightArmDistance]
        head_distances = [np.linalg.norm(self.headDistance[i]) for i in self.headDistance]
        #delete the first element to return the vectors
        return left_distances, right_distances, head_distances

if __name__ == "__main__":
    file_path = "./robot_configuration_files/qt.yaml"
    qt = Robot()
    qt.import_robot(file_path)
    # angles = np.array([ 1.52367249, -1.29154365, -0.2268928, -1.54636169, -1.41022609, -0.14137168, 0.03839724, 0.00523599])
    #default
    angles = np.array([np.deg2rad(90.3), np.deg2rad(-57.3), np.deg2rad(-34.8), np.deg2rad(-90), np.deg2rad(-57.7), np.deg2rad(-34.2), np.deg2rad(0), np.deg2rad(2.2)])
    #left front
    # angles = np.array([np.deg2rad(10.7), np.deg2rad(-74.3), np.deg2rad(-6.5), np.deg2rad(-92), np.deg2rad(-78.5), np.deg2rad(-22.4), np.deg2rad(0), np.deg2rad(2.2)])
    #head
    # angles = np.array([np.deg2rad(-74.3), np.deg2rad(-45.3), np.deg2rad(-70.7), np.deg2rad(74), np.deg2rad(-44.9), np.deg2rad(-69.4), np.deg2rad(0), np.deg2rad(2.2)])
    # #sides
    # angles = np.array([np.deg2rad(-100), np.deg2rad(-12), np.deg2rad(-4.2), np.deg2rad(107.9), np.deg2rad(-12.7), np.deg2rad(-4.8), np.deg2rad(0), np.deg2rad(2.2)])
    # #belly
    # angles = np.array([np.deg2rad(39.1), np.deg2rad(-67.4), np.deg2rad(-74), np.deg2rad(-39.4), np.deg2rad(-60.9), np.deg2rad(-77.9), np.deg2rad(0), np.deg2rad(2.2)])
    angles = np.array([np.deg2rad(-74.3), np.deg2rad(-45.3), np.deg2rad(-70.7), np.deg2rad(74), np.deg2rad(-44.9), np.deg2rad(-69.4), np.deg2rad(0), np.deg2rad(2.2)])

    pos_left, pos_right, pos_head = qt.forward_kinematics(angles)
    print("LEFT: ", pos_left)
    print("RIGHT: ", pos_right)
    print("HEAD: ", pos_head)
    s = simulate_position.RobotSimulation([pos_left], [pos_right], [pos_head])
    s.animate()