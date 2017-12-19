#!/usr/bin/env python

'''
Written : Leo pauly
Descripion : Program to be implemented inside Baxter for performing observation learning.
Version v3:


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


class s2l():

   def __init__(self):
    rospy.init_node('listener', anonymous=True) # Initialise node
    self.time_step=40
    self.height=112
    self.width=112
    self.height_robo=112
    self.width_robo=112
    self.channel=3
    self.nb_class=3
    self.imagefolderpath= ('/media/leo/PENDRIVELEO/s2l/leeds_rgb/') # Folder pathe where the video sequences are stored
    print(os.path.isdir(self.imagefolderpath))
    self.cluster_length=16
    self.strides=1
    self.total_frames_in_seq=40
    self.num_clusters= int( (self.total_frames_in_seq-self.cluster_length) / self.strides) + 1
    self.feature_size=487
    self.r=rospy.Rate(10)
    self.video_frames=[]
    self.iter=0
    self.distance=np.ones([self.num_clusters,self.feature_size])
    self.reward=np.ones([self.num_clusters,1])
    self.features_robo=np.zeros([self.num_clusters,self.feature_size])
    self.features_demo=np.zeros([self.num_clusters,self.feature_size])


   def load_model(self):
    self.model=md.modelC3D_theano()
    print(self.model.summary())
    print('model printed')

   def load_data(self):
    self.ds=DataSet(self.nb_class,self.time_step, self.height, self.width, self.channel,self.imagefolderpath,'diff')

   def extract_features_demo(self):
    ## Splitting into clusters the video of source demonstration
    x_demo,_ =self.ds.get_batch(1)
    self.x_demo_original=x_demo
    x_demo =x_demo.reshape(x_demo.shape[0],x_demo.shape[1],self.height,self.width,self.channel)
    clusters_demo=dset.clusters_with_strides(x_demo,self.num_clusters,self.cluster_length,self.height,self.width,self.channel,self.strides)
    ## Extraction of features
    print('shape of clustered demo data:',clusters_demo.shape)
    for i in range(self.num_clusters):
      t=clusters_demo[i].reshape(1,self.channel,self.cluster_length,self.height,self.width)
      self.features_demo[i]= self.model.predict(t)
      #print(features)
    print('shape of features from robot actions: ',self.features_demo.shape)

   def extract_features_robo(self):
    x_robo=np.array(self.video_frames)
    self.x_robo_original=x_robo
    print(x_robo.shape)
    t=x_robo.reshape(1,self.channel,self.cluster_length,self.height_robo,self.width_robo)
    self.features_robo[self.iter]= self.model.predict(t)
    print('shape of features from robot actions: ', self.features_robo.shape)

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
        image_=(bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough"))
        image_=cv.resize(image_,(self.height_robo,self.width_robo))
        self.video_frames.append(image_)
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

   def rl(self):
       print('rl run completed')
        # use self.state for obaining state variable
       return 1

   def take_action(self):
       print('action taken')
        # use self.action for obaining state variable
       return 1

   def rl_rewards(self):
        #for i in range(self.num_clusters):
        self.distance[self.iter] = self.features_demo[self.iter]-self.features_robo[self.iter]
        self.reward[self.iter]=-(np.linalg.norm(self.distance[self.iter]))
        print(self.reward)
        #print(dist)
        return 1

   def episode_analysis(self):
        y_values=list((range(self.num_clusters)))
        lsp.plot_values_with_legends(y_values,self.reward,'reward','clusters','value','reward function',color='red',ylim=True)
        return 1

   def iter_update(self):
     self.iter=self.iter+1
     print('iter_updated')

   def clear_video(self):
     print ('video cleared')
     for i in range(self.cluster_length):
         self.video_frames.pop()

   def num_iterations(self):
        return self.num_clusters

   def reset(self):
       return 1

if __name__ == '__main__':
    ## Checking if theano is backend
    if keras.backend.backend() != 'theano':
      raise BaseException("This script uses other backend")
    else:
      keras.backend.set_image_dim_ordering('th')
      print("Backend ok")

    obj=s2l() # Creating object for class s2l
    obj.listener() # Class funcion for getting frames from Baxter camera
    obj.load_model() # Class funcion loading model
    obj.load_data() # Class funcion loading stored data (Demonstrations)
    obj.extract_features_demo()  # Class funcion extracting featurs from demonstration
    obj.extract_features_robo()  # Class funcion extracting featurs from robot acion
    for i in range(obj.num_iterations()):
     obj.rl() # Class function for implementing RL algorithm
     obj.take_action()
     #obj.check()
     obj.clear_video()
     obj.listener() # Class funcion for getting frames from Baxter camera
     obj.extract_features_robo()  # Class funcion extracting featurs
     obj.rl_rewards()  # Class funcion for providing rewads

     obj.iter_update()
    obj.episode_analysis() # Class funcion for analysing the final results after each episode
    obj.reset() # Fo reseting afer every episode
