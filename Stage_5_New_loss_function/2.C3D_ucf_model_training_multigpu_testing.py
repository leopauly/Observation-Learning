
# coding: utf-8

# In[3]:

### 2. C3D+ucf_model_training_multigpu_training
# Author : @leopauly | cnlp@leeds.ac.uk <br>
# Description : Training the C3D model using UCF 101 action recognition dataset- Testing Phase

print('Started running the program..!',flush=True)
## Imports
from keras.models import Sequential
import random
import numpy as np
from PIL import Image
from os import listdir
from scipy.ndimage import imread
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K
import datetime
import time
import os 
from datetime import timedelta


# Custom scripts
import lscript as lsp
import modelling as md
#from DataSet.DataSet import DataSet
import dataset as dset
import ucf101_dataset as ucf

print('Loaded libraries...!!',flush=True)


# In[4]:

height=112 
width=112 
channel=3
cluster_length=16
nb_classes=101

lr_rate=.001
next_batch_start=0
batch_size=8
batch_size_test=8
total_train_videos=9991
memory_batch_size_train=50
memory_batch_size_test=3329
iterations= 10 # (int(total_train_videos/memory_batch_size_train)) #10001
custom_global_step=0
LOG_DIR='/nobackup/leopauly/logdir'
saved_path='/nobackup/leopauly/logdirp100'

print('Finished defining variables..!!',flush=True)


# In[ ]:




# In[7]:

## Defining placeholders in tf for images and targets
x_image = tf.placeholder(tf.float32, [None, 16,height,width,channel]) 
y_true = tf.placeholder(tf.float32, [None, nb_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

model_keras = md.C3D_ucf101_training_model_tf(summary=False)
out=model_keras(x_image)
y_pred = tf.nn.softmax(out)
y_pred_cls = tf.argmax(out, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Miscellenious items finished..!!',flush=True)


# In[4]:

#### Training & testing
def testing(iterations,loops):
    #print(test_images.shape)
    #print(test_labels_cls)
    test_score=0
    for j in range(int(memory_batch_size_test/batch_size_test)-1):
        test_score_ = sess.run([accuracy], feed_dict={x_image:test_images[(batch_size_test*j):(batch_size_test*(j+1))],y_true_cls:test_labels_cls[(batch_size_test*j):(batch_size_test*(j+1))],K.learning_phase(): 0 })
        #print('returned value',test_score_)
        test_score=test_score+sum(test_score_)
    print('Test accuracy after iteration:',iterations,',loop:',loops,'is:',test_score/(j+1),flush=True)


# In[ ]:

## Loading Testing data
test_images, test_labels_cls, next_batch_start, _ = ucf.read_vid_and_label('./UCF101_data_preparation/test.list',memory_batch_size_test,-1,16,112,normalisation=False)
test_labels=keras.utils.to_categorical(test_labels_cls, num_classes=nb_classes)
print('testing data loaded',flush=True)


# In[9]:

#### Start the session with logging placement.
init_op = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(init_op)
saver = tf.train.Saver()

## Restore model weights from previously saved model
#saver = tf.train.import_meta_graph(os.path.join(saved_path,'activity_model_1.ckpt-43.meta'))
saver.restore(sess, os.path.join(saved_path,'activity_model_1.ckpt-43'))
print("Model restored from file: %s" % saved_path,flush=True)


# In[ ]:

testing(0,0)


# In[ ]:

sess.close()

