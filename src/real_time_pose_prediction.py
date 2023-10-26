#!/usr/bin/env python
import sys
import rospy
from qt_nuitrack_app.msg import Skeletons
from migrave_skeleton_tools.tf_utils import TFUtils
from migrave_skeleton_tools_ros.skeleton_utils import JointUtils

import pose_prediction
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray, Marker

head_pub = rospy.Publisher('/qt_robot/head_position/command', Float64MultiArray, queue_size=10)
right_pub = rospy.Publisher('/qt_robot/right_arm_position/command', Float64MultiArray, queue_size = 10)
left_pub = rospy.Publisher('/qt_robot/left_arm_position/command', Float64MultiArray, queue_size = 10)

rospy.sleep(1.0)

class SkeletonMarkers(object):

    def __init__(self, id, skeleton_frame_id):
        self.id = id
        self.skeleton_frame_id = skeleton_frame_id
        id_pad = self.id * 100
        self.upper_body_marker = self.get_marker(id_pad + 0)
        self.left_hand_marker = self.get_marker(id_pad + 1)
        self.right_hand_marker = self.get_marker(id_pad + 2)
        self.left_leg_marker = self.get_marker(id_pad + 3)
        self.right_leg_marker = self.get_marker(id_pad + 4)

class HumanSkeleton(object):
    def __init__(self, pose_predictor):
        nuitrack_skeleton_topic = rospy.get_param('~nuitrack_skeleton_topic',
                                                  '/qt_nuitrack_app/skeletons')
        self.cam_base_link_translation = rospy.get_param('~cam_base_link_translation', [0., 0., 0.])
        self.skeleton_frame_id = rospy.get_param('~skeleton_frame_id', 'base_link')
        self.cam_base_link_rot = rospy.get_param('cam_base_link_rot', [np.pi/2, 0., np.pi/2])

        self.cam_base_link_tf = TFUtils.get_homogeneous_transform(self.cam_base_link_rot,
                                                                  self.cam_base_link_translation)
        self.skeleton_sub = rospy.Subscriber(nuitrack_skeleton_topic, Skeletons, self.capture_skeletons)
        self.pose_predictor = pose_predictor

    def capture_skeletons(self, skeleton_collection_msg):

        for skeleton_msg in skeleton_collection_msg.skeletons:
            right_arm_points = {}
            left_arm_points = {}
            head_points = {}
            for joint in skeleton_msg.joints:
                # the joint msg contains the position in mm
                position = np.array(joint.real) / 1000.
                position_hom = np.array([[position[0]], [position[1]], [position[2]], [1.]])
                position_base_link = self.cam_base_link_tf.dot(position_hom)
                position_base_link = position_base_link.flatten()[0:3]
                position_base_link[1] = -position_base_link[1]
                joint_name = JointUtils.JOINTS[joint.type]
                if joint_name in JointUtils.JOINTS_TO_IGNORE:
                    continue
                if joint_name in JointUtils.BODY_JOINT_NAMES:
                    head_points[joint_name] = position_base_link
                elif joint_name in JointUtils.LEFT_ARM_JOINT_NAMES:
                    left_arm_points[joint_name] = position_base_link
                elif joint_name in JointUtils.RIGHT_ARM_JOINT_NAMES:
                    right_arm_points[joint_name] = position_base_link

            head_points['JOINT_LEFT_COLLAR'] = left_arm_points['JOINT_LEFT_COLLAR']
            left_side, right_side, head = self.pose_predictor.dicts_to_lists(left_arm_points, right_arm_points, head_points)
            left_arm_pred, right_arm_pred, head_pred = self.pose_predictor.predict_pytorch(left_side, right_side, head)
            joint_angle_publisher(np.rad2deg(left_arm_pred), np.rad2deg(right_arm_pred), np.rad2deg(head_pred))
        return

def joint_angle_publisher(left_arm_ang_pos, right_arm_ang_pos, head_ang_pos):
    # Publish angles to robot
        rospy.loginfo("publishing motor command...")
        try:
            ref_head = Float64MultiArray()
            ref_right = Float64MultiArray()
            ref_left = Float64MultiArray()
            ref_head.data = head_ang_pos
            ref_right.data = right_arm_ang_pos
            ref_left.data = left_arm_ang_pos
            head_pub.publish(ref_head)
            right_pub.publish(ref_right)
            left_pub.publish(ref_left)
            rospy.sleep(0.1)
        except rospy.ROSInterruptException:
            rospy.logerr("could not publish motor command!")
        rospy.loginfo("motor command published")

def initialise():
    file_path = "./robot_configuration_files/qt.yaml"
    pose_predictor = pose_prediction.Prediction(file_path)
    #NN TRAINING
    file_name = "robot_angles"
    theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
    pose_predictor.train_pytorch(theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, 1000)
    return pose_predictor

if __name__ == '__main__':
    rospy.init_node('real_time_pose_estimation_node')
    rospy.loginfo("real_time_pose_estimation_node started!")

    pose_predictor = initialise()
    skeleton = HumanSkeleton(pose_predictor)
    # define ros subscriber
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    rospy.loginfo("finsihed!")

