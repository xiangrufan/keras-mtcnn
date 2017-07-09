from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense,concatenate
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU
from keras.optimizers import adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import _pickle as pickle
import random
from keras.activations import relu
from keras.losses import mean_squared_error
import tensorflow as tf
import gc

with open(r'24\cls.imdb','rb') as fid:
    cls = pickle.load(fid)
with open(r'24\roi.imdb', 'rb') as fid:
    roi = pickle.load(fid)

ims_cls = []
ims_roi = []
cls_score = []
roi_score = []
for (idx, dataset) in enumerate(cls) :
    ims_cls.append( np.swapaxes(dataset[0],0,2))
    cls_score.append(dataset[1])
for (idx,dataset) in enumerate(roi) :
    ims_roi.append( np.swapaxes(dataset[0],0,2))
    roi_score.append(dataset[2])


ims_cls = np.array(ims_cls)
ims_roi = np.array(ims_roi)
cls_score = np.array(cls_score)
roi_score = np.array(roi_score)


one_hot_labels = to_categorical(cls_score, num_classes=2)

# input = Input(shape = [12,12,3])
input = Input(shape = [24,24,3]) # change this shape to [None,None,3] to enable arbitraty shape input
x = Conv2D(16,(3,3),strides=1,padding='same',name='conv1')(input)
c1out = PReLU(shared_axes=[1,2],name='prelu1')(x)
c1out = concatenate ([c1out,input],axis=3)

c2input = MaxPool2D(pool_size=3)(c1out)

x = Conv2D(32,(3,3),strides=1,padding='same',name='conv2')(c2input)
c2out = PReLU(shared_axes=[1,2],name='prelu2')(x)
c2out = concatenate([c2out,c2input],axis=3)

c3input = MaxPool2D(pool_size=2)(c2out)

x = Conv2D(64,(3,3),strides=1,padding='same',name='conv3')(c3input)
c3out = PReLU(shared_axes=[1,2],name='prelu3')(x)
c3out = concatenate([c3out,c3input],axis=3)

x = Flatten() (c3out)
x = Dense(128,name='dense1')(x)
x = PReLU(shared_axes=[1],name='prelu4')(x)
classifier = Dense(2, activation='softmax',name='classifier1')(x)
bbox_regress = Dense(4,name='bbox1')(x)

my_adam = adam(lr = 0.001)


for i_train in range(80):
    randx=random.choice([0,1,1])  # still need to run manually on each batch
    # randx = 4
    # randx = random.choice([ 4])
    batch_size = 64
    print ('currently in training macro cycle: ',i_train)
    if i_train ==0:
        model = Model([input], [classifier, bbox_regress])
        # model.load_weights('model24.h5',by_name=True)
        bbox = model.get_layer('bbox1')
        bbox_weight = bbox.get_weights()
        classifier_dense = model.get_layer('classifier1')
        cls_weight = classifier_dense.get_weights()

    if 0 == randx:
        model = Model([input], [classifier])
        model.get_layer('classifier1').set_weights(cls_weight)
        model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
        model.fit(ims_cls, one_hot_labels, batch_size=batch_size, epochs=1)
        classifier_softmax = model.get_layer('classifier1')
        cls_weight = classifier_softmax.get_weights()

    if 1 == randx:
        model = Model([input], [bbox_regress])
        model.get_layer('bbox1').set_weights(bbox_weight)
        model.compile(loss='mse', optimizer=my_adam, metrics=["accuracy"])
        model.fit(ims_roi, roi_score, batch_size=batch_size, epochs=1)
        bbox_dense = model.get_layer('bbox1')
        bbox_weight = bbox_dense.get_weights()

gc.collect()

def savemodel():
        model = Model([input], [classifier, bbox_regress])
        model.get_layer('bbox1').set_weights(bbox_weight)
        model.get_layer('classifier1').set_weights(cls_weight)
        model.save_weights('model24.h5')
savemodel()
