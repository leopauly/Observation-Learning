#!/usr/bin/env python

'''
Written : Leo pauly
Descripion : Program to be implemented inside Baxter for performing observation learning.
Version v2:
Wrote following class functions:
obj.listener() # Class funcion for getting fixed length frames from Baxter camera at when needed
obj.load_model() # Class funcion loading model (C3D model trained on Sports1M dataset)
obj.load_data() # Class funcion loading stored data (Demonstrations)
obj.extract_features()  # Class funcion extracting featurs from loaded stored data (TODO in next version: extract features from camera feedback from Baxter)

Courtesy : http://wiki.ros.org
Enable robot first by running: $ rosrun baxter_tools enable_robot.py -e
'''

## Specifing matplotlib backend
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

## Setting theano as backend
import os
os.environ["KERAS_BACKEND"] = "theano"
import keras; import keras.backend

## Imports
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
import random
import numpy as np
from PIL import Image
from os import listdir
from scipy.ndimage import imread
import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

## Custom scripts
import lscript as lsp
import modelling as md
from DataSet import DataSet
import dataset as dset


class reward_extractor():

   def __init__(self):
    rospy.init_node('listener', anonymous=True) # Initialise node

    self.time_step=40
    self.height=112
    self.width=112
    self.height_robo=800
    self.width_robo=800
    self.channel=3
    self.nb_class=3
    self.imagefolderpath= ('/media/leo/PENDRIVELEO/s2l/leeds_rgb/') # Folder pathe where the video sequences are stored
    print(os.path.isdir(self.imagefolderpath))
    self.cluster_length=40
    self.strides=1
    self.total_frames_in_seq=40
    self.num_clusters= int( (self.total_frames_in_seq-self.cluster_length) / self.strides) + 1
    self.feature_size=487
    self.r=rospy.Rate(10)
    self.video_frames=[]


   def load_model(self):
    self.model=md.modelC3D_theano()
    print(self.model.summary())
    print('model printed')

   def load_data(self):
    self.ds=DataSet(self.nb_class,self.time_step, self.height, self.width, self.channel,self.imagefolderpath,'diff')

   def extract_features(self):
    ## Splitting into clusters the video of source demonstration
    x_demo,_ =self.ds.get_batch(1)
    x_demo_original=x_demo
    x_demo =x_demo.reshape(x_demo.shape[0],x_demo.shape[1],self.height,self.width,self.channel)
    clusters_demo=dset.clusters_with_strides(x_demo,self.num_clusters,self.cluster_length,self.height,self.width,self.channel,self.strides)

    ## Extraction of features
    print('shape of clustered demo data:',clusters_demo.shape)
    features_demo=np.zeros([self.num_clusters,self.feature_size])
    for i in range(self.num_clusters):
      t=clusters_demo[i].reshape(1,self.channel,self.cluster_length,self.height,self.width)
      features_demo[i]= self.model.predict(t)
      #print(features)
    print('shape of features from robot actions: ',features_demo.shape)

    ## Splitting into & displaying clusters of target actions
    x_robo,_=self.ds.get_batch(1)
    x_robo_original=x_robo
    x_robo =x_robo.reshape(x_robo.shape[0],x_robo.shape[1],self.height,self.width,self.channel)
    clusters_robo=dset.clusters_with_strides(x_robo,self.num_clusters,self.cluster_length,self.height,self.width,self.channel,self.strides)

    ## Extracting features
    print('shape of clustered robot data:',clusters_robo.shape)
    features_robo=np.zeros([self.num_clusters,self.feature_size])
    for i in range(self.num_clusters):
      t=clusters_robo[i].reshape(1,self.channel,self.cluster_length,self.height,self.width)
      features_robo[i]= self.model.predict(t)
    #print(features)
    print('shape of features from robot actions: ', features_robo.shape)


    distance=np.ones([self.num_clusters,self.feature_size])
    reward=np.ones([self.num_clusters,1])
    for i in range(self.num_clusters):
     distance[i] = features_demo[i]-features_robo[i]
     reward[i]=-(np.linalg.norm(distance[i]))
    print(reward)
    #print(dist)

    y_values=list((range(self.num_clusters)))
    lsp.plot_values_with_legends(y_values,reward,'reward','clusters','value','reward function',color='red',ylim=True)

   def callback(self,data):
       bridge = CvBridge()
       image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
       cv.imshow('baxter_view',image)
       cv.waitKey(3)
       return image

       #rospy.signal_shutdown('reason')
       #return 1
       #self.camera_sub.unregister()


   def listener(self):
    for i in range(self.cluster_length): #
        img_msg = rospy.wait_for_message('/cameras/head_camera/image', Image)

        bridge = CvBridge()
        self.video_frames.append(bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough"))
        #self.frames[i]=self.callback(img_msg)
        #cv.imshow('baxter_camera',self.image)
        #cv.waitKey(3)

        #rospy.init_node('listener', anonymous=True) # Initialise node
        #self.camera_sub=rospy.Subscriber('/cameras/head_camera/image', Image, self.callback) # Subscribe to topic camera image

        #while not rospy.is_shutdown():
             #rospy.sleep(1)
        #rospy.spinOnce()
        #rospy.spin()
        #self.r.sleep()

   def check(self):
        for i in range(self.cluster_length):
            cv.imshow('baxter_check_view',self.video_frames[i])
            cv.waitKey(30)



if __name__ == '__main__':
    ## Checking if theano is backend
    if keras.backend.backend() != 'theano':
      raise BaseException("This script uses other backend")
    else:
      keras.backend.set_image_dim_ordering('th')
      print("Backend ok")


    obj=reward_extractor() # Creating object for class reward_extractor
    obj.listener() # Class funcion for getting frames from Baxter camera
    obj.check()
    #obj.load_model() # Class funcion loading model
    #obj.load_data() # Class funcion loading stored data
    #obj.extract_features()  # Class funcion extracting featurs
