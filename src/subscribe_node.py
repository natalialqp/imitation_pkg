#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import json

rospy.init_node('my_tutorial_node')
rospy.loginfo("started!")

instant_position = {'jointLeft_0': [], 'jointLeft_1': [], 'jointLeft_2': [], 'jointRight_0': [],
                     'jointRight_1': [], 'jointRight_2': [], 'jointHead_0': [], 'jointHead_1': []}
instant_torque = {'jointLeft_0': 0, 'jointLeft_1': 0, 'jointLeft_2': 0, 'jointRight_0': 0,
                     'jointRight_1': 0, 'jointRight_2': 0, 'jointHead_0': 0, 'jointHead_1': 0}
recorded_positions = {}
head_pub = rospy.Publisher('/qt_robot/head_position/command', Float64MultiArray, queue_size=10)
rospy.sleep(3.0)

def qt_joint_state_cb(joint_state_msg):
    '''
    This function saves the recorded_positions taken from the robot's sensors
    Input: joint_state_msg
    instant_position: saves the same information for the PID loop
    '''
    current_time = joint_state_msg.header.stamp.secs
    instant_position['jointLeft_0'].append(np.deg2rad(joint_state_msg.position[3]))
    instant_position['jointLeft_1'].append(np.deg2rad(joint_state_msg.position[4]))
    instant_position['jointLeft_2'].append(np.deg2rad(joint_state_msg.position[2]))
    instant_position['jointRight_0'].append(np.deg2rad(joint_state_msg.position[6]))
    instant_position['jointRight_1'].append(np.deg2rad(joint_state_msg.position[7]))
    instant_position['jointRight_2'].append(np.deg2rad(joint_state_msg.position[5]))
    instant_position['jointHead_0'].append(np.deg2rad(joint_state_msg.position[0]))
    instant_position['jointHead_1'].append(np.deg2rad(joint_state_msg.position[1]))

    instant_torque['jointLeft_0'] = np.deg2rad(joint_state_msg.effort[3])
    instant_torque['jointLeft_1'] = np.deg2rad(joint_state_msg.effort[4])
    instant_torque['jointLeft_2'] = np.deg2rad(joint_state_msg.effort[2])
    instant_torque['jointRight_0'] = np.deg2rad(joint_state_msg.effort[6])
    instant_torque['jointRight_1'] = np.deg2rad(joint_state_msg.effort[7])
    instant_torque['jointRight_2'] = np.deg2rad(joint_state_msg.effort[5])
    instant_torque['jointHead_0'] = np.deg2rad(joint_state_msg.effort[0])
    instant_torque['jointHead_1'] = np.deg2rad(joint_state_msg.effort[1])

    print(len(instant_position['jointLeft_0']))

rospy.Subscriber('/qt_robot/joints/state', JointState, qt_joint_state_cb)

if __name__ == '__main__':
    while not rospy.is_shutdown():
        try:
            rospy.sleep(2)
        except KeyboardInterrupt:
            pass
    rospy.loginfo("finsihed!")

    file_path = "dance_2.txt"
    with open(file_path, 'w') as file:
        json.dump(instant_position, file)
