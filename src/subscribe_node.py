#!/usr/bin/env python
import sys
import rospy
from qt_nuitrack_app.msg import Gestures

def gesture_callback(msg):
    rospy.loginfo(msg)

if __name__ == '__main__':
    rospy.init_node('my_tutorial_node')
    rospy.loginfo("my_tutorial_node started!")

    # define ros subscriber
    rospy.Subscriber('/qt_nuitrack_app/gestures', Gestures, gesture_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    rospy.loginfo("finsihed!")

