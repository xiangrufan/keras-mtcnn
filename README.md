# mtcnn-caffe
Keras Implementation of Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks.

This project provide you a method to update multi-task-loss for multi-input source.

Transplanted from MTCNN-caffe from CongweiLin's github repository
github.com/CongWeilin/mtcnn-caffe

# training requires Wider Face Training data set and CelebA data set (same as the caffe version). 
However, the scripted is modified to reduce hard-disk usage. i.e. all intermediate cropped imgs are stored in memory.
Requires at least 16 Gb memory to precess training data.

# refined training process
The refined training uses training strategy that closely follows the original caffe code. i.e. randomly select Classification loss, roi regression loss or key point regression losses and minimize it for each batch of data. Slightly improve the performance. But also makes the training code unnecessarrily complex. 
Accuracy measurement is not implemented.


