# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:47:57 2018

@author: 沈鴻儒
"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2

mnist_image = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=False)
pixels, labels = mnist_image.train.next_batch(10)
image = pixels[5, :]
image = np.reshape(image, [28, 28])
pic = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
cv2.imshow('', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()