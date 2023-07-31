import numpy as np
import yaml
from homogeneous_transformation import HT

class Robot(object):

    def __init__(self):
        self.robot = None
        self.leftArmAngles = []
        self.rightArmAngles = []
        self.headAngles = []
        self.leftArmDistance = []
        self.rightArmDistance = []
        self.headDistance = []

    def read_yaml_file(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def import_robot(self, file_path):
        data = self.read_yaml_file(file_path)
        self.leftArmAngles = data["angles-left-arm"]
        self.leftArmDistance = data["distances-left-arm"]
        self.rightArmAngles = data["angles-right-arm"]
        self.rightArmDistance = data["distances-right-arm"]
        self.headAngles = data["angles-head"]
        self.headDistance = data["distances-head"]

    def forward_kinematics(self, angles):
        '''
        This function receives an angle configuration and calculates the spatial position of the end effector and the elbow
        right_elbow ---------> 0
        left_elbow ----------> 1
        right_shoulder_pitch-> 2
        left_shoulder_pitch -> 3
        right_shoulder_roll -> 4
        left_shoulder_roll --> 5
        neck_pitch ----------> 6
        neck_roll -----------> 7
        '''
        a = HT()
        HT_leftArm = []
        HT_rightArm = []
        chain_leftArm = []
        chain_rightArm = []
        self.pos_leftArm = []
        self.pos_rightArm = []

        self.leftArmAngles["elbow"] -= np.array([angles[1], 0, 0])
        self.rightArmAngles["elbow"] += np.array([angles[0], 0, 0])
        self.leftArmAngles["shoulder-roll"] += np.array([angles[5], angles[3], 0])
        self.rightArmAngles["shoulder-roll"] += np.array([angles[4], angles[2], 0])

        for key in self.leftArmDistance:
            HT_leftArm.append(a.get_homogeneous_transform(self.leftArmAngles[key], self.leftArmDistance[key]))
            HT_rightArm.append(a.get_homogeneous_transform(self.rightArmAngles[key], self.rightArmDistance[key]))

        chain_leftArm.append(HT_leftArm[0])
        chain_rightArm.append(HT_rightArm[0])

        for i in range(len(self.leftArmDistance) - 1):
            chain_leftArm.append(np.dot(chain_leftArm[i], HT_leftArm[i + 1]))
            chain_rightArm.append(np.dot(chain_rightArm[i], HT_rightArm[i + 1]))

        for i in range(1, len(chain_leftArm)):
            self.pos_leftArm.append(a.get_translation(chain_leftArm[i]))
            self.pos_rightArm.append(a.get_translation(chain_rightArm[i]))

        return self.pos_leftArm, self.pos_rightArm

if __name__ == "__main__":
    file_path = "./robot_configuration_files/qt.yaml"
    qt = Robot()
    qt.import_robot(file_path)
    angles = np.array([np.pi, np.pi/2, np.pi/3, np.pi, np.pi/3, np.pi, np.pi, np.pi])
    print("Generic: ", qt.forward_kinematics(angles))