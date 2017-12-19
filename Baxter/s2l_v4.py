#!/usr/bin/env python

'''
Written : Leo pauly
Descripion : Program to be implemented inside Baxter for performing observation learning.
Version v4:
Implementing reinforcement learning algoithm


Courtesy : http://wiki.ros.org
Enable robot first by running: $ rosrun baxter_tools enable_robot.py -e
'''

## Specifing matplotlib backend
import matplotlib
matplotlib.use('Qt5Agg')
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
from collections import defaultdict
import baxter_interface

## Custom scripts
import lscript as lsp
import modelling as md
from DataSet import DataSet
import dataset as dset
import rl


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
    self.action_num= 10 # Number of possible actions
    self.num_episodes=3 # Number of possible actions
    self.episode_rewards = np.zeros(self.num_episodes)
    self.episode_lengths = np.zeros(self.num_episodes)
    self.limb = baxter_interface.Limb('right')


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

   def take_action(self,action):
       wave_0 = {'right_s0': -0.459, 'right_s1': -0.202, 'right_e0': 1.807, 'right_e1': 1.714, 'right_w0': -0.906, 'right_w1': -1.545, 'right_w2': -0.276}
       if (action==0):
           self.limb.move_to_joint_positions( wave_0)
       print('action taken')

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

   def reset_after_episode(self):
       self.iter=0

   def reset_pose(self):
       angles = limb.joint_angles()
       angles['right_s0']=0.0
       angles['right_s1']=0.0
       angles['right_e0']=0.0
       angles['right_e1']=0.0
       angles['right_w0']=0.0
       angles['right_w1']=0.0
       angles['right_w2']=0.0
       print ('Robotic hands moved to initial pose')
       limb.move_to_joint_positions(angles)



   def q_learning(self, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        self.num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(self.action_num))

    # Keeps track of useful statistics
    # stats = plotting.EpisodeStats(episode_lengths=np.zeros(self.num_episodes),episode_rewards=np.zeros(self.num_episodes))

    # The policy we're following
    policy = rl.make_epsilon_greedy_policy(Q, epsilon, self.action_num)

    for i_episode in range(self.num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, self.num_episodes)," ")

        self.reset_pose()
        obj.listener() # Class funcion for getting frames from Baxter camera
        obj.extract_features_robo()  # Class funcion extracting featurs from robot acion
        self.state= self.features_robo[self.iter]
        self.clear_video()
        # One step in the environment
        # total_reward = 0.0
        for t in range(self.num_iterations()):

            # Take a step
            action_probs = policy(1)  #self.state)  #Problem 1
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs) #Problem 5
            self.take_action(action)
            #obj.check()

            self.listener() # Class funcion for getting frames from Baxter camera
            self.extract_features_robo()  # Class funcion extracting featurs
            self.rl_rewards()  # Class funcion for providing rewads
            next_state=self.features_robo[self.iter]
            reward=self.reward[self.iter]

            # Update statistics
            self.episode_rewards[i_episode] += reward
            self.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[1])   #[next_state])  #Problem 2
            td_target = reward + discount_factor * Q[1][best_next_action] #Q[next_state][best_next_action] #Problem 4
            td_delta = td_target -Q[1][action] #Q[state][action] #Problem 5
             #Q[self.state][action] += alpha * td_delta #Problem 3

            print('iteration : {}finished'.format(t))
            self.state = next_state
            self.iter_update()
            self.clear_video()

        self.episode_analysis() # Class funcion for analysing the final results after each episode
        self.reset_after_episode()

    return Q

if __name__ == '__main__':
    ## Checking if theano is backend
    if keras.backend.backend() != 'theano':
      raise BaseException("This script uses other backend")
    else:
      keras.backend.set_image_dim_ordering('th')
      print("Backend ok")

    obj=s2l() # Creating object for class s2l
    obj.load_model() # Class funcion loading model
    obj.load_data() # Class funcion loading stored data (Demonstrations)
    obj.extract_features_demo()  # Class funcion extracting featurs from demonstration

    Q, stats = obj.q_learning() # Passing argument number of episodes
