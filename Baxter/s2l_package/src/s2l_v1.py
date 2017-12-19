#!/usr/bin/env python

'''
Written : Leo pauly
Descripion : Program to be implemented inside Baxter for performing observation learning
Version v1 : Created reward_extractor class
Courtesy : http://wiki.ros.org
Enable robot first by running: $ rosrun baxter_tools enable_robot.py -e
'''

import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy
from cv_bridge import CvBridge, CvBridgeError


class reward_extractor():

   def __init__(self):
    rospy.init_node('listener', anonymous=True) # Initialise node

   def callback(self,data):
    bridge = CvBridge()
    self.image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    cv.imshow('baxter_view',self.image)
    cv.waitKey(3)


   def listener(self):
    rospy.Subscriber('/cameras/head_camera/image', Image, self.callback) # Subscribe to topic camera image
    rospy.spin() # spin() simply keeps python from exiting until this node is stopped

if __name__ == '__main__':

    obj=reward_extractor()
    obj.listener()
