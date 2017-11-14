import random
import numpy as np
from PIL import Image
from os import listdir
from scipy.ndimage import imread

dspath = './V0/'

class DataSet:
    
    def __init__(self):    
        self.videos = np.empty([3*5, 40, 360, 640, 3], dtype=np.uint8)
        self.activity = []
        vidIndex = -1
        for directory in listdir(dspath): 
            if directory[0] != '.': #Each Activity
                for vid in range(5): #Each video
                    vidpath = dspath+directory+'/'+str(vid)+'/'
                    vidIndex += 1;
                    self.activity.append(directory)
                    for frame in range(40): #Each frame
                        framepath = vidpath+str(frame)+'rgb.png'
                        # print('[',vidIndex, frame, ']', directory)
                        print('Loading ', framepath)
                        self.videos[vidIndex, frame] = imread(framepath)[:,:,:3]
        random.seed(7)
        self.Indices = list(range(3*5))
        random.shuffle(self.Indices)
        self.size = len(self.Indices)
    
    def get_batch(self,batch_size=5):
        # Initialisation
        videos = np.zeros([batch_size, 40, 360, 640, 3], dtype=np.uint8)
        labels = []
        # Loading the batch
        for i in range(batch_size):
            if len(self.Indices): 
                ni = self.Indices.pop()
                videos[i,:,:,:,:] = self.videos[ni]
                labels.append(self.activity[ni])
        
        return videos,labels