# mtcnn-caffe
Keras Implementation of Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks.

This project provide you a method to update multi-task-loss for multi-input source.

Transplanted from MTCNN-caffe from CongweiLin's github repository
github.com/CongWeilin/mtcnn-caffe

# training requires Wider Face Training data set and CelebA data set (same as the caffe version). However,
 the scripted is modified to reduce hard-disk usage. i.e. all intermediate cropped imgs are stored in memory.
 Requires at least 16 Gb memory to precess training data.




