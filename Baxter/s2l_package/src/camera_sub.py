#!/usr/bin/env python

'''
Written : Leo pauly
Descripion : Subsribes to a camera topic, recieves ROS images which is convered to opencv images (using cv_bridge) and then displayed
Courtesy : http://wiki.ros.org
Enable robot first by running: $ rosrun baxter_tools enable_robot.py -e
'''

import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    cv.imshow('baxter_view',image)
    cv.waitKey(3)
    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/cameras/head_camera/image', Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':

    listener()
