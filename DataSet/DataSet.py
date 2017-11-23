import random
import numpy as np
from PIL import Image
from os import listdir
from scipy.ndimage import imread

#dspath = '../../../../../nobackup/leopauly/rgb/'

class DataSet():
        
    def __init__(self,nb_class,time_step, height, width, channel,dspath):   
        
        self.dspath=dspath
        self.height=height
        self.width=width
        self.channel=channel
        self.time_step=time_step
        self.nb_class=nb_class
        self.seq_per_class=5
        
        self.videos = np.empty([nb_class*self.seq_per_class, time_step, height, width, channel], dtype=np.uint8)
        self.activity = []
        vidIndex = -1
        for directory in listdir(dspath): 
            if directory[0] != '.': #Each Activity
                for vid in range(5): #Each video
                    vidpath = dspath+'reach'+'/'+str(vid)+'/'  #directory
                    vidIndex += 1;
                    self.activity.append(directory)
                    for frame in range(40): #Each frame
                        framepath = vidpath+str(frame)+'rgb.png'
                        # print('[',vidIndex, frame, ']', directory)
                        print('Loading ', framepath)
                        self.videos[vidIndex, frame] = imread(framepath)[:,:,:3]
        random.seed(7)
        self.reset()
    
    def get_batch(self,batch_size=5):
        # Initialisation
        videos = np.zeros([batch_size, self.time_step, self.height, self.width, self.channel], dtype=np.uint8)
        labels = []
        # Loading the batch
        for i in range(batch_size):
            if len(self.Indices): 
                ni = self.Indices.pop()
                videos[i,:,:,:,:] = self.videos[ni]
                labels.append(self.activity[ni])
        
        return videos,labels
    
    def reset(self):
        # Reset the data set before starting a new epoch
        self.Indices = list(range(3*5))
        self.size = len(self.Indices)
        random.shuffle(self.Indices)