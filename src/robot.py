import numpy as np
import yaml
from homogeneous_transformation import HT
import simulate_position
from scipy.spatial.transform import Rotation as R

class Robot(object):

    def __init__(self, robotName):
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
        self.robotName = robotName

    def check_angle_limits(self, angles, arm_type):
        valid_set = True
        if "left" in arm_type:
            for index, (key, value) in enumerate(self.physical_limits_left.items()):
                if not (value[0] <= angles[index] <= value[0]):
                    valid_set = False
        if "right" in arm_type:
            for index, (key, value) in enumerate(self.physical_limits_right.items()):
                if not (value[0] <= angles[index] <= value[0]):
                    valid_set = False
        if "head" in arm_type:
            for index, (key, value) in enumerate(self.physical_limits_head.items()):
                if not (value[0] <= angles[index] <= value[0]):
                    valid_set = False
        return valid_set

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
        self.dof_arm = data["DOF-right-arm"]

    def forward_kinematics_qt(self, angles):
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
        leftArmAngles["elbow"] = self.leftArmAngles["elbow"] + np.array([angles[2], 0, 0])
        # leftArmAngles["elbow"] = self.leftArmAngles["elbow"] - np.array([-angles[2], 0, 0])
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

    def forward_kinematics_nao(self, angles):
        '''
        This function receives an angle configuration and calculates the spatial position of the end effector and the elbow
        left_shoulder_pitch --> 0
        left_shoulder_roll ---> 1
        left_elbow_yaw -------> 2
        left_elbow_roll ------> 3
        left_wrist-yaw -------> 4
        right_shoulder_pitch -> 5
        right_shoulder_roll --> 6
        right_elbow_yaw ------> 7
        right_elbow_roll -----> 8
        right_wrist-yaw ------> 9
        head_yaw ------------> 10
        head_pitch-----------> 11
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

        leftArmAngles["torso"] = self.leftArmAngles["torso"]
        leftArmAngles["shoulder-pitch"] = self.leftArmAngles["shoulder-pitch"] + np.array([0, angles[0], 0])
        leftArmAngles["shoulder-roll"] = self.leftArmAngles["shoulder-roll"] + np.array([0, 0, angles[1]])
        leftArmAngles["elbow-yaw"] = self.leftArmAngles["elbow-yaw"] + np.array([angles[2], 0, 0])
        leftArmAngles["elbow-roll"] = self.leftArmAngles["elbow-roll"] + np.array([0, 0, angles[3]])
        leftArmAngles["wrist-yaw"] = self.leftArmAngles["wrist-yaw"] + np.array([angles[4], 0, 0])
        leftArmAngles["hand"] = self.leftArmAngles["hand"]

        rightArmAngles["torso"] = self.rightArmAngles["torso"]
        rightArmAngles["shoulder-pitch"] = self.rightArmAngles["shoulder-pitch"] + np.array([0, angles[5], 0])
        rightArmAngles["shoulder-roll"] = self.rightArmAngles["shoulder-roll"] + np.array([0, 0, angles[6]])
        rightArmAngles["elbow-yaw"] = self.rightArmAngles["elbow-yaw"] + np.array([angles[7], 0, 0]) #-
        rightArmAngles["elbow-roll"] = self.rightArmAngles["elbow-roll"] + np.array([0, 0, angles[8]])
        rightArmAngles["wrist-yaw"] = self.rightArmAngles["wrist-yaw"] + np.array([angles[9], 0, 0]) #-
        rightArmAngles["hand"] = self.rightArmAngles["hand"]

        headAngles["torso"] = self.headAngles["torso"]
        headAngles["head-yaw"] = self.headAngles["head-yaw"] + np.array([0, 0, angles[10]])
        headAngles["head-pitch"] = self.headAngles["head-pitch"] + np.array([0, angles[11], 0])
        headAngles["camera-top"] = self.headAngles["camera-top"]

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

    def forward_kinematics_gen3(self, angles):
        '''
        This function receives an angle configuration and calculates the spatial position of the end effector and the elbow
        left_shoulder_pitch --> 0
        left_shoulder_roll ---> 1
        left_elbow_yaw -------> 2
        left_elbow_roll ------> 3
        left_wrist-yaw -------> 4
        right_shoulder_pitch -> 5
        right_shoulder_roll --> 6
        right_elbow_yaw ------> 7
        right_elbow_roll -----> 8
        right_wrist-yaw ------> 9
        head_yaw ------------> 10
        head_pitch-----------> 11
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

        leftArmAngles["torso"] = self.leftArmAngles["torso"]
        leftArmAngles["base"] = self.leftArmAngles["base"]
        leftArmAngles["shoulder_link"] = self.leftArmAngles["shoulder_link"] - np.array([0, 0, angles[0]])
        leftArmAngles["half_arm_1_link"] = self.leftArmAngles["half_arm_1_link"] + np.array([0, angles[1], 0])
        leftArmAngles["half_arm_2_link"] = self.leftArmAngles["half_arm_2_link"] + np.array([0, angles[2], 0])
        leftArmAngles["forearm_link"] = self.leftArmAngles["forearm_link"] + np.array([0, angles[3], 0])
        leftArmAngles["spherical_wrist_1_link"] = self.leftArmAngles["spherical_wrist_1_link"] + np.array([0, angles[4], 0])
        leftArmAngles["spherical_wrist_2_link"] = self.leftArmAngles["spherical_wrist_2_link"] + np.array([0, angles[5], 0])
        leftArmAngles["bracelet_link"] = self.leftArmAngles["bracelet_link"] + np.array([0, angles[6], 0])
        leftArmAngles["end_effector_link"] = self.leftArmAngles["end_effector_link"]

        rightArmAngles["torso"] = self.rightArmAngles["torso"]
        rightArmAngles["base"] = self.rightArmAngles["base"]
        rightArmAngles["shoulder_link"] = self.rightArmAngles["shoulder_link"] - np.array([0, 0, angles[7]])
        rightArmAngles["half_arm_1_link"] = self.rightArmAngles["half_arm_1_link"] + np.array([0, angles[8], 0])
        rightArmAngles["half_arm_2_link"] = self.rightArmAngles["half_arm_2_link"] + np.array([0, angles[9], 0])
        rightArmAngles["forearm_link"] = self.rightArmAngles["forearm_link"] + np.array([0, angles[10], 0])
        rightArmAngles["spherical_wrist_1_link"] = self.rightArmAngles["spherical_wrist_1_link"] + np.array([0,  angles[11], 0])
        rightArmAngles["spherical_wrist_2_link"] = self.rightArmAngles["spherical_wrist_2_link"] + np.array([0, angles[12], 0])
        rightArmAngles["bracelet_link"] = self.rightArmAngles["bracelet_link"] + np.array([0, angles[13], 0])
        rightArmAngles["end_effector_link"] = self.rightArmAngles["end_effector_link"]

        headAngles["torso"] = self.headAngles["torso"]

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
        if self.robotName != "gen3":
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

def linear_angular_mapping_gen3(vec):
    new_vec = []
    for i in vec:
        if i > 0:
            new_vec.append(i)
        else:
            new_vec.append(2*np.pi + i)
    return new_vec

if __name__ == "__main__":
    robotName = "gen3"
    file_path = "./robot_configuration_files/" + robotName + ".yaml"
    qt = Robot(robotName)
    qt.import_robot(file_path)
    # angles = np.array([ 1.52367249, -1.29154365, -0.2268928, -1.54636169, -1.41022609, -0.14137168, 0.03839724, 0.00523599])
    #default
    # angles = np.array([np.deg2rad(90.3), np.deg2rad(-57.3), np.deg2rad(-34.8), np.deg2rad(-90), np.deg2rad(-57.7), np.deg2rad(-34.2), np.deg2rad(0), np.deg2rad(2.2)])
    #left front
    # angles = np.array([np.deg2rad(10.7), np.deg2rad(-74.3), np.deg2rad(-6.5), np.deg2rad(-92), np.deg2rad(-78.5), np.deg2rad(-22.4), np.deg2rad(0), np.deg2rad(2.2)])
    #head
    # angles = np.array([np.deg2rad(-74.3), np.deg2rad(-45.3), np.deg2rad(-70.7), np.deg2rad(74), np.deg2rad(-44.9), np.deg2rad(-69.4), np.deg2rad(0), np.deg2rad(2.2)])
    # #sides
    # angles = np.array([np.deg2rad(-100), np.deg2rad(-12), np.deg2rad(-4.2), np.deg2rad(107.9), np.deg2rad(-12.7), np.deg2rad(-4.8), np.deg2rad(0), np.deg2rad(2.2)])
    # #belly
    # angles = np.array([np.deg2rad(39.1), np.deg2rad(-67.4), np.deg2rad(-74), np.deg2rad(-39.4), np.deg2rad(-60.9), np.deg2rad(-77.9), np.deg2rad(0), np.deg2rad(2.2)])
    # angles = np.array([np.deg2rad(-74.3), np.deg2rad(-45.3), np.deg2rad(-70.7), np.deg2rad(74), np.deg2rad(-44.9), np.deg2rad(-69.4), np.deg2rad(0), np.deg2rad(2.2)])

    # nao angles
    # angles = np.array([np.deg2rad(-90), np.deg2rad(50), np.deg2rad(-0), np.deg2rad(-88), np.deg2rad(-100),
                    #    np.deg2rad(-90), np.deg2rad(-50), np.deg2rad(-0), np.deg2rad(88), np.deg2rad(-100),
                    #    np.deg2rad(0), np.deg2rad(0)])

    le = np.array([
        [0.17101007, -0.03015369, 0.98480775],
        [0.5629971, -0.81728662, -0.1227878],
        [0.80857271, 0.57544186, -0.1227878]])

    ri = np.array([
        [0.15405787, -0.02716456, 0.98768834],
        [-0.71221313, -0.69591499, 0.09194987],
        [0.68484935, -0.71761021, -0.12655814]])

    r = R.from_matrix(ri.T)
    l = R.from_matrix(le.T)
    # print(l.as_euler('xyz', degrees=False))
    # print(r.as_euler('xyz', degrees=False))

    # qt.leftArmAngles["base"] = l.as_euler('zyx', degrees=False)
    # qt.rightArmAngles["base"] = r.as_euler('zyx', degrees=False)

    # qt.leftArmAngles["base"] = [0, np.deg2rad(97.5), np.deg2rad(45)] #= l.as_euler('xyz', degrees=False)
    # qt.rightArmAngles["base"] = [0, np.deg2rad(97.5), np.deg2rad(-45)] #= r.as_euler('xyz', degrees=False)

    left = []
    right = []
    head = []
    for i in range(0, 370, 10):
        angles = np.array([ 1.43989663,  0.78539816,  2.75033203,  1.22173048,  1.25956629,  1.74602669,  -2.86233997, -1.43989663, 0.78539816, -2.72103979,  1.22173048, -1.28781237,  1.73033442,  1.76278254])
        angles = np.array([ 1.43989663,  0.78539816, -1.55458098, 1.89771864 , 0.63187545, 0.16110732, 1.73556517, -1.43989663, 0.78539816, 1.32233537,  1.73033442,  0., 0.05649218, -0.73303829])
        angles = linear_angular_mapping_gen3(angles)

        # NEGATIVE ANGLES RIGHT: SHOULDER_ROLL, ELBOWS and wrist
        # angles = np.array([np.deg2rad(0), np.deg2rad(90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),
        #                    np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),
        #                    np.deg2rad(0), np.deg2rad(0)])

        # angles = np.array([np.deg2rad(-90), np.deg2rad(50), np.deg2rad(-0), np.deg2rad(-88), np.deg2rad(-100),
        #                np.deg2rad(-90), np.deg2rad(-50), np.deg2rad(-0), np.deg2rad(88), np.deg2rad(-100),
        #                np.deg2rad(0), np.deg2rad(0)])

        # angles = np.array([np.deg2rad(80.18695652), np.deg2rad(-71.84782609), np.deg2rad(-10.6), np.deg2rad(-73.8), np.deg2rad(-68.94782609), np.deg2rad(-7.34347826), np.deg2rad(23.76086957), np.deg2rad(23.34347826)])
        # angles = np.array([np.deg2rad(79.99130435), np.deg2rad(-71.56521739), np.deg2rad(-11.6), np.deg2rad(-73.3), np.deg2rad(-69.16521739), np.deg2rad(-7.49565217), np.deg2rad(23.67391304), np.deg2rad(22.49565217)])
        # angles = np.array([np.deg2rad(80.5), np.deg2rad(-72.3), np.deg2rad(-9.0), np.deg2rad(-74.6), np.deg2rad(-68.6), np.deg2rad(-7.1), np.deg2rad(23.9), np.deg2rad(24.7)])
        pos_left, pos_right, pos_head = qt.forward_kinematics_gen3(angles)
        left.append(pos_left)
        right.append(pos_right)
        head.append(pos_head)
    # print(pos_left, pos_right, pos_head)

    # s = simulate_position.RobotSimulation([pos_left], [pos_right], [pos_head])
    s = simulate_position.RobotSimulation(left, right, head)
    s.animate()