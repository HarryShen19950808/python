# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 02:28:49 2018

@author: USER
"""

import cv2
import numpy as np

def img_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
path = r"D:\desktop_D\Python\python\OpenCV\License plate recognition"
image_far = cv2.imread(path + "\origin\\far\\far_{}.jpg".format(2))

image_far = image_far[100:, :]
img_show("img", image_far)
cv2.imwrite(path + "\origin\\far\\far_{}.jpg".format(2), image_far)