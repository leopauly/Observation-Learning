
def modelC3D(cluster_length, height, width, channel,summary=False, load_weights=True):
    ## Imports
    import numpy as np
    import os
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Activation, Dense, Dropout, Flatten, MaxPooling2D, Input, Concatenate,MaxPooling3D, Reshape, ZeroPadding3D
    from keras.layers.convolutional import Conv3D
    from keras.layers.convolutional_recurrent import ConvLSTM2D
    from keras.layers.normalization import BatchNormalization
    from keras.models import Model
    import numpy as np
    import pylab as plt
    import keras
    import h5py
    import keras.backend as K
    '''
    Model implementation of C3D network used for activity recognition
    Ref : http://arxiv.org/pdf/1412.0767.pdf 
    Courtesy : https://github.com/imatge-upc/activitynet-2016-cvprw/tree/master/data
    '''
    input_cnn = Input(shape=(cluster_length, height, width, channel))
    
    cnn_1=Conv3D(filters=64, kernel_size=(3,3,3),padding='same',activation='relu',name='conv1')(input_cnn)
    pool_1=MaxPooling3D(pool_size=(1, 2, 2),strides=(1, 2, 2),padding='valid',name='pool1')(cnn_1)
    
    cnn_2=Conv3D(filters=128, kernel_size=(3,3,3),padding='same',activation='relu',name='conv2')(pool_1)
    pool_2=MaxPooling3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='valid',name='pool2')(cnn_2)
    
    cnn_3a=Conv3D(filters=256, kernel_size=(3,3,3),padding='same',activation='relu',name='conv3a')(pool_2)
    cnn_3b=Conv3D(filters=256, kernel_size=(3,3,3),padding='same',activation='relu',name='conv3b')(cnn_3a)
    pool_3=MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),padding='valid',name='pool3')(cnn_3b)
    
    cnn_4a=Conv3D(filters=512, kernel_size=(3,3,3),padding='same',activation='relu',name='conv4a')(pool_3)
    cnn_4b=Conv3D(filters=512, kernel_size=(3,3,3),padding='same',activation='relu',name='conv4b')(cnn_4a)
    #zero_4=ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad4')(cnn_4b)
    pool_4=MaxPooling3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='valid',name='pool4')(cnn_4b)
    
    cnn_5a=Conv3D(filters=512, kernel_size=(3,3,0),padding='same',activation='relu',name='conv5a')(pool_4)
    cnn_5b=Conv3D(filters=512, kernel_size=(3,3,0),padding='same',activation='relu',name='conv5b')(cnn_5a)
    zero_5=ZeroPadding3D(padding=((0, 1), (0, 0), (0, 0)), name='zeropad5')(cnn_5b)
    pool_5=MaxPooling3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='valid',name='pool5')(zero_5)
    
    flat= Flatten(name='flatten')(pool_5)
    #flat=Reshape([1,-1])(pool_3)
    
    # FC layers group
    fc_1=Dense(4096, activation='relu',name='fc6')(flat)
    drop_1=Dropout(0.5, name='do1')(fc_1)
    fc_2=Dense(4096, activation='relu',name='fc7')(drop_1)
    drop_2=Dropout(0.5, name='do2')(fc_2)
    fc_2=Dense(487, activation='sigmoid', name='fc8')(drop_2)
    
    model_cnn=Model(input_cnn,fc_2)
    if load_weights:
        model_cnn.load_weights('/nobackup/leopauly/c3d-sports1M_weights.h5')
    if summary:
        print(model_cnn.summary())
    return model_cnn

def modelC3D_theano(load_weights=True,summary=True):
    '''
    '''
    ## Imports
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential
    
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv1',
            subsample=(1, 1, 1),
            input_shape=(3, 16, 112, 112),
            trainable=False))
    model.add(MaxPooling3D(pool_size=(1, 2, 2),strides=(1, 2, 2),border_mode='valid',name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128,3,3,3,activation='relu',border_mode='same',name='conv2',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv3a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(256,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv3b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv4a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv4b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv5a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv5b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool5'))
    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    # Load weights
    if load_weights:
        model.load_weights('/nobackup/leopauly/c3d-sports1M_weights.h5')
    if summary:
        print(model.summary())
        
    return model


def custom_modelC3D_theano(cluster_length,load_weights=True,summary=True):
    '''
    '''
    ## Imports
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential
    
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv1',
            subsample=(1, 1, 1),
            input_shape=(3, cluster_length, 112, 112),
            trainable=False))
    model.add(MaxPooling3D(pool_size=(1, 2, 2),strides=(1, 2, 2),border_mode='valid',name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128,3,3,3,activation='relu',border_mode='same',name='conv2',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv3a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(256,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv3b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv4a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv4b',
            subsample=(1, 1, 1),
            trainable=False))
    #model.add(ZeroPadding3D(padding=(1, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv5a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(Convolution3D(512,3,3,3,
            activation='relu',
            border_mode='same',
            name='conv5b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(ZeroPadding3D(padding=(1, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool5'))
    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    # Load weights
    if load_weights:
        model.load_weights('/nobackup/leopauly/c3d-sports1M_weights.h5')
    if summary:
        print(model.summary())
        
    return model

def get_vgg16_imagenet(summary=True,include_fc=True):
    ''' Returns a vgg16 model with pretrained weights'''
    import keras 
    model=keras.applications.vgg16.VGG16(include_top=include_fc, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    if summary:
        print(model.summary())
    return model
    