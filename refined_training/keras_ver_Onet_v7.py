from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense,Lambda
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU
from keras.optimizers import adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import _pickle as pickle
import random
from keras.losses import mean_squared_error
import tensorflow as tf
import gc
import keras


def mydef_check(*args):
    return True

def irresponsible_slice_array(arrays, start=None, stop=None):
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            # if hasattr(start, 'shape'):
                # start = start.tolist()
            return [None if x is None else x[start[start<x.shape[0]]] for x in arrays]
        else:
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


keras.engine.training._check_array_lengths = mydef_check # this over ride function in module ....
keras.engine.training._slice_arrays = irresponsible_slice_array # this over ride function in module ....




id_train = 1
with open(r'48\cls.imdb','rb') as fid:
    cls = pickle.load(fid)
with open(r'48\pts.imdb', 'rb') as fid:
    pts = pickle.load(fid)
with open(r'48\roi.imdb', 'rb') as fid:
    roi = pickle.load(fid)
ims_cls = []
ims_pts = []
ims_roi = []
cls_score = []
pts_score = []
roi_score = []
for (idx, dataset) in enumerate(cls) :
    ims_cls.append( np.swapaxes(dataset[0],0,2))
    cls_score.append(dataset[1])
for (idx,dataset) in enumerate(roi) :
    ims_roi.append( np.swapaxes(dataset[0],0,2))
    roi_score.append(dataset[2])
for (idx,dataset) in enumerate(pts) :
    ims_pts.append( np.swapaxes(dataset[0],0,2))
    pts_score.append(dataset[3])

ims_cls = np.array(ims_cls)
ims_pts = np.array(ims_pts)
ims_roi = np.array(ims_roi)
cls_score = np.array(cls_score)
pts_score = np.array(pts_score)
roi_score = np.array(roi_score)


one_hot_labels = to_categorical(cls_score, num_classes=2)
rand_threshold = [0.4,0.9]
print_progress = False

def data_input(input):
    x = input[0]
    y = input[1]
    z = input[2]
    random_int = tf.random_uniform([1])
    condition1 = random_int[0] > tf.constant(rand_threshold[1])
    condition0 = random_int[0] > tf.constant(rand_threshold[0])
    val = tf.case({condition1: lambda: x,
                   condition0: lambda: y
                   },
                  default=lambda: z)
    val.set_shape(z.shape)
    return [val,random_int]# tuple (output,random_int ) is NOT allowed

input0 = Input(shape = [48,48,3])
input1 = Input(shape = [48,48,3])
input2 = Input(shape = [48,48,3])
(input,random_int) = Lambda(data_input)([input0,input1,input2])

x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv1')(input)
x = PReLU(shared_axes=[1,2],name='prelu1')(x)
x = MaxPool2D(pool_size=3,strides=2)(x)
x = Conv2D(64,(3,3),strides=1,padding='valid',name='conv2')(x)
x = PReLU(shared_axes=[1,2],name='prelu2')(x)
x = MaxPool2D(pool_size=3,strides=2)(x)
x = Conv2D(64,(3,3),strides=1,padding='valid',name='conv3')(x)
x = PReLU(shared_axes=[1,2],name='prelu3')(x)
x = MaxPool2D(pool_size=2)(x)
x = Conv2D(128,(3,3),strides=1,padding='valid',name='conv4')(x)
x = PReLU(shared_axes=[1,2],name='prelu4')(x)
x = Flatten()(x)
x = Dense(256, activation='relu',name='dense1') (x)
classifier = Dense(2, activation='softmax',name='classifier1')(x)
bbox_regress = Dense(4,name='bbox1')(x)
landmark_regress = Dense(10,name='landmark1')(x)
my_adam = adam(lr = 0.00001)

def myloss (random_int,type): # need to make sure input type
    if print_progress: random_int = tf.Print(random_int, ['random in cls',random_int])
    condition1 = random_int[0] > tf.constant(rand_threshold[1])
    condition0 = random_int[0] > tf.constant(rand_threshold[0])
    condition_default =  condition0 & condition1
    if type =='cls':
        def lossfun(y_true, y_pred):
            if print_progress: condition = tf.Print(condition1, ['rand int[0]:', random_int[0],
                                                    ' tf.constant:', tf.constant(rand_threshold[1]),
                                                    ' condition1:', condition1 ])
            val= tf.case({ condition1: lambda: K.mean(K.square(y_pred - y_true), axis=-1),
                           condition0: lambda: 0 * K.mean(K.square(y_true), axis=-1)
                           },
                         default=lambda:0 * K.mean(K.square(y_true), axis=-1),
                         exclusive=False )

            if print_progress: val = tf.Print(val, ['cls loss out:',val,
                                                    ' rand int received:',random_int,
                                                    'condition',condition1])
            val.set_shape(K.mean(K.square(y_true), axis=-1).shape)
            return val
    elif type =='roi':
        def lossfun(y_true, y_pred):
            if print_progress: condition = tf.Print(condition1, ['rand int[0]:', random_int[0],
                                                                ' tf.constant:', tf.constant(rand_threshold),
                                                                ' condition:', condition1])
            val= tf.case({ condition1: lambda: 0 * K.mean(K.square(y_true), axis=-1),
                           condition0: lambda: K.mean(K.square(y_pred - y_true), axis=-1)
                         },
                         default=lambda: 0 * K.mean(K.square(y_true), axis=-1),exclusive=False)
            if print_progress: val = tf.Print(val, ['roi loss out :', val,
                                                    ' rand int received:', random_int,
                                                    'condition', condition1])
            val.set_shape(K.mean(K.square(y_true), axis=-1).shape)
            return val
    else :
        def lossfun(y_true, y_pred):
            if print_progress: condition = tf.Print(condition1, ['rand int[0]:', random_int[0],
                                                                 ' tf.constant:', tf.constant(rand_threshold),
                                                                 ' condition:', condition1])
            val = tf.case({condition1: lambda: 0 * K.mean(K.square(y_true), axis=-1),
                           condition0: lambda: 0 * K.mean(K.square(y_true), axis=-1)
                           },
                          default=lambda: K.mean(K.square(y_pred - y_true), axis=-1),exclusive=False)
            val.set_shape(K.mean(K.square(y_true), axis=-1).shape)
            if print_progress: val = tf.Print(val, ['pts loss out :', val,
                                                    ' rand int received:', random_int,
                                                    'condition', condition1])
            return val
    return lossfun

def accuracy (y_pred,y_true):
    return K.mean(y_true)

model = Model([input0,input1,input2], [classifier, bbox_regress, landmark_regress])
model.load_weights('onet7.h5',by_name=True)
model.compile(loss=[myloss(random_int,'cls') ,myloss(random_int,'roi') ,myloss(random_int,'pts')],
              optimizer=my_adam, metrics=[accuracy,accuracy,accuracy])

model.fit([ims_cls,ims_roi,ims_pts], [one_hot_labels, roi_score,pts_score], batch_size=128, epochs=100)
gc.collect()
model.save_weights('onet7.h5')
