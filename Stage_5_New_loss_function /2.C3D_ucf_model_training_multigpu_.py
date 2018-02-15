
# coding: utf-8

# ### 2. C3D+ucf_model_training_multigpu
# Author : @leopauly | cnlp@leeds.ac.uk <br>
# Description : Training the C3D model using UCF 101 action recognition dataset

# In[ ]:

## Imports
from keras.models import Sequential
import random
import numpy as np
from PIL import Image
from os import listdir
from scipy.ndimage import imread
import keras
import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
from keras import backend as K

# Custom scripts
import lscript as lsp
import modelling as md
#from DataSet.DataSet import DataSet
import dataset as dset
import ucf101_dataset as ucf


# In[ ]:

height=112 
width=112 
channel=3
cluster_length=16
nb_classes=101

lr_rate=.001
next_batch_start=0
batch_size=8
total_train_videos=9999
memory_batch_size_train=9999
memory_batch_size_test=3000
batch_size=8
iterations= (int(total_train_videos/memory_batch_size_train)) #10001
#epoch=int((memory_batch_size_train/batch_size)*10)
custom_global_step=0


# In[ ]:

#!nvidia-smi


# In[ ]:

# Finding how many devices are available
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
num_gpus = len(gpus)
print("GPU nodes found: " + str(num_gpus))
print('Avaialble gpsu:',str(gpus[0]))


# In[ ]:

# Finding how many CPUs are available
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
num_gpus = len(gpus)
print("CPU nodes found: " + str(num_gpus))
print('Avaialble cpus:',str(gpus[0]))


# In[ ]:

## Defining placeholders in tf for images and targets
x_image = tf.placeholder(tf.float32, [None, 16,height,width,channel]) 
y_true = tf.placeholder(tf.float32, [None, nb_classes])
y_true_cls = tf.placeholder(tf.int64, [None])


# #### Getting the model

# In[ ]:

# Define the network in a model function, to make parallelisation across GPUs easier.
def model(x_image_, y_true_):
    ''' Expecting the following parameters, in batches:
        x_image_ - x_image batch
        y_true_ - y_true batch
    '''

    model = md.C3D_ucf101_training_model_tf(summary=False)
    out=model(x_image_)
    
    y_pred = tf.nn.softmax(out)
    y_pred_cls = tf.argmax(out, dimension=1)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=out))
    
    # Outputs to be returned to CPU
    return y_pred, y_pred_cls, loss


# In[ ]:

def make_parallel(fn, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    # An array for every aggregated output
    y_pred_split, y_pred_cls_split, cost_split, fv_split = [], [], [], []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                y_pred_, y_pred_cls_, cost_,  = fn(**{k : v[i] for k, v in in_splits.items()})
                # Adding the output from each device.
                y_pred_split.append(y_pred_)
                y_pred_cls_split.append(y_pred_cls_)
                cost_split.append(cost_)
                #fv_split.append(fv_)

    # Aggregating and returning outputs. tf.concat for multi-dimensional arrays; tf.stack if single values.
    return tf.concat(y_pred_split, axis=0), tf.concat(y_pred_cls_split, axis=0),tf.stack(cost_split, axis=0)
    


# In[ ]:

if num_gpus > 0:
    # There is significant latency for CPU<->GPU copying of shared variables.
    # We want the best balance between speedup and minimal latency.
    y_pred, y_pred_cls, cost = make_parallel(model, x_image_=x_image, y_true_=y_true)
else:
    # CPU-only version
    y_pred, y_pred_cls, cost = model(x_image_=x_image, y_true_=y_true)


# In[ ]:

# Optimisation calculated on CPU on aggregated results.
# NEED the colocate_gradients_with_ops flag TRUE to get the gradient ops to run on same device as original op!
optimizer = tf.train.AdagradOptimizer(learning_rate=2e-4).minimize(cost, colocate_gradients_with_ops=True)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# #### Training 

# In[ ]:

def testing(iterations,epoch):
    test_images, test_labels_cls, next_batch_start, _ = ucf.read_vid_and_label('./UCF101_data_preparation/test.list',memory_batch_size_test,-1,16,112,normalisation=True)
    test_labels=keras.utils.to_categorical(test_labels_cls, num_classes=nb_classes)
    #print(test_images.shape)
    #print(test_labels_cls)
    test_score= sess.run([accuracy], feed_dict={x_image:test_images,y_true_cls:test_labels_cls,K.learning_phase(): 0 })    #print(next_batch_start)
    print('Test accuracy after iteration:',iterations,',epoch:',epoch,'is:',test_score)


# In[ ]:

# Start the session with logging placement.
init_op = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(init_op)
saver = tf.train.Saver() # Creating tf.train.Saver() object

for i in range(iterations*10):
    train_images, train_labels_cls, next_batch_start, _ = ucf. read_vid_and_label('./UCF101_data_preparation/train.list',memory_batch_size_train,-1,cluster_length,112,normalisation=True)
    train_labels=keras.utils.to_categorical(train_labels_cls, num_classes=nb_classes)
    for j in range(memory_batch_size_train/batch_size):
        output_value = sess.run([optimizer], feed_dict={x_image:train_images[0*j:batch_size*j],y_true:train_labels[(batch_size*j):(batch_size*(j+1))],K.learning_phase(): 1 })    #print(next_batch_start)
        if (iterations%1000):
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=custom_global_step)
            custom_global_step=custom_global_step+1
            testing(iterations)


# In[ ]:



