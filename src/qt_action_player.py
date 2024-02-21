#!/usr/bin/env python

import sys
import rospy
from std_msgs.msg import Float64MultiArray
import pandas as pd
import numpy as np
from math import degrees

left_columns = ['left_1', 'left_2', 'left_3']
right_columns = ['right_1', 'right_2', 'right_3']
head_columns = ['head_1', 'head_2']

left_angle_sequence = []
right_angle_sequence = []
head_angle_sequence = []

def read_action_from_file(file_path):
    df = pd.read_csv("GMM_learned_actions/" + file_path)
    time_column = df['time']
    signal_columns = df.drop(columns=['time'])
    for index, row in df.iterrows():
        left_values = [degrees(row[column]) for column in left_columns]
        right_values = [degrees(row[column]) for column in right_columns]
        head_values = [degrees(row[column]) for column in head_columns]
        left_angle_sequence.append(left_values)
        right_angle_sequence.append(right_values)
        head_angle_sequence.append(head_values)
    return left_angle_sequence, right_angle_sequence, head_angle_sequence

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
            rospy.sleep(0.4)
        except rospy.ROSInterruptException:
            rospy.logerr("could not publish motor command!")
        rospy.loginfo("motor command published")

if __name__ == '__main__':
    rospy.init_node('qt_motor_command')

    # create a publisher
    left_pub = rospy.Publisher('/qt_robot/left_arm_position/command', Float64MultiArray, queue_size=1)
    right_pub = rospy.Publisher('/qt_robot/right_arm_position/command', Float64MultiArray, queue_size=1)
    head_pub = rospy.Publisher('/qt_robot/head_arm_position/command', Float64MultiArray, queue_size=1)
    left_angle_sequence, right_angle_sequence, head_angle_sequence = read_action_from_file("qt_teacup_left_for_execution.csv")

    # wait for publisher/subscriber connections
    for i in range(len(left_angle_sequence)):
        joint_angle_publisher(left_angle_sequence[i], right_angle_sequence[i], head_angle_sequence[i])