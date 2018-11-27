# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:05:17 2018

@author: USER
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():exit(-1)

while cv2.waitKey(30)!=ord('q'):
    retval, image = cap.read()
    if not retval:break
    cv2.imshow("video",image)
    cv2.imwrite("video.jpg",image)
#    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()