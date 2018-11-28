# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:31:57 2018

@author: USER
"""

import cv2
import numpy as np

path = r"D:\desktop_D\Python\python\OpenCV\License plate recognition"

def img_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for i in range(1, 11):
    image_AND = cv2.imread(path + "\label\AND\label_AND{}.jpg".format(i))
#    img_show("AND", image_AND)
    
    # 取灰階
    h = image_AND.shape[0]
    w = image_AND.shape[1]
    gray = np.empty((h, w), np.uint8)
    # 分離出R、G、B的分量
    (B,G,R) = cv2.split(image_AND)
#    img_show("origin", image_far)
    # 計算灰階
    gray[:h, :w] = 0.299*R[:h, :w] + 0.587*G[:h, :w] + 0.114*B[:h, :w]
    R = gray[:, :]
    G = gray[:, :]
    B = gray[:, :]

    img,contours,hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    area = cv2.contourArea(cnt)#计算面积
    print(area)
    clone = image_AND.copy()
    cv2.drawContours(clone, contours, -1, (0, 0, 255), 2)
    img_show("area", clone)
    