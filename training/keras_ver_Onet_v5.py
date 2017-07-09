from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense
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
id_train = 1
with open(r'48\cls{0:d}.imdb'.format(id_train),'rb') as fid:
    cls = pickle.load(fid)
# with open(r'48\pts.imdb', 'rb') as fid:
#     pts = pickle.load(fid)
with open(r'48\roi{0:d}.imdb'.format(id_train), 'rb') as fid:
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

input = Input(shape = [48,48,3])
input_randx = Input(shape=[1])
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
my_adam = adam(lr = 0.00003)


model = Model([input], [classifier, bbox_regress, landmark_regress])
model.load_weights('model48.h5', by_name=True)
bbox_dense = model.get_layer('bbox1')
bbox_weight = bbox_dense.get_weights()
classifier_dense = model.get_layer('classifier1')
cls_weight = classifier_dense.get_weights()
landmark_dense = model.get_layer('landmark1')
landmark_weight = landmark_dense.get_weights()


for i_train in range(80):
    randx=random.choice([0,1,1,1,0,0,1])   # still need to run manually on each batch
    batch_size = 64
    print ('currently in training macro cycle: ',i_train)

    if 0 == randx:
        model = Model([input], [classifier])
        model.get_layer('classifier1').set_weights(cls_weight)
        model.compile(loss='binary_crossentropy', optimizer=my_adam, metrics=["accuracy"])
        model.fit(ims_cls, one_hot_labels, batch_size=batch_size, nb_epoch=1)
        classifier_dense = model.get_layer('classifier1')
        cls_weight = classifier_dense.get_weights()
    if 1 == randx:
        model = Model([input], [bbox_regress])
        model.get_layer('bbox1').set_weights(bbox_weight)
        model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
        model.fit(ims_roi, roi_score, batch_size=batch_size, nb_epoch=1)
        bbox_dense = model.get_layer('bbox1')
        bbox_weight = bbox_dense.get_weights()
    if 2 == randx:
        model = Model([input], [landmark_regress])
        model.get_layer('landmark1').set_weights(landmark_weight)
        model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
        model.fit(ims_pts, pts_score, batch_size=batch_size, nb_epoch=1)
        landmark_dense = model.get_layer('landmark1')
        landmark_weight = landmark_dense.get_weights()
gc.collect()


model = Model([input], [classifier, bbox_regress, landmark_regress])
model.get_layer('landmark1').set_weights(landmark_weight)
model.get_layer('bbox1').set_weights(bbox_weight)
model.get_layer('classifier1').set_weights(cls_weight)
model.save_weights('model48.h5')

