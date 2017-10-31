'''
Helper functions for preparing datasets
Author : @leopauly

'''
#Imports
import numpy as np
import cv2

def batch_gen(batch_start,batch_stop,batch_size,time_step,h,w,ch,imagefolderpath,gray,normalisation=True):
    ''' Genreates batches of video sequences suitable for feeding to a ConvLSTM
    '''
    if (ch==1):
        X=np.zeros([batch_size,time_step,h,w])
    else:
        X=np.zeros([batch_size,time_step,h,w,ch])
    for i in range(batch_start,batch_stop):
        j_new=0
        for j in range(0+(50*i),50+(50*i)):
            img=cv2.imread(str(imagefolderpath+str(j)+'.png'))
            if (gray==True):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if(normalisation==True):
                img = img.astype('float32')
                img=img/255
            X[i][j_new]=np.array(img)
            j_new=j_new+1
    return X

