# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:29:42 2018

@author: 沈鴻儒
"""
import cv2
import numpy as np
#from matplotlib import pyplot as plt

def img_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#def diff_thresh(img):
#    global path
#    img = cv2.resize(img, (300, 300))
#    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#    hmerge = np.hstack((thresh1, thresh2, thresh3, thresh4, thresh5)) #水平拼接
#    cv2.imshow("hmerge", hmerge)
#    
#    #按esc離開，按s儲存
#    k = cv2.waitKey(0)
#    if k == 27:
#        cv2.destroyAllWindows()
#    elif k == ord("s"):
#        cv2.imwrite(path + "\\thresh_R.jpg", hmerge) # 目前用手動調整，非迴圈
#        cv2.destroyAllWindows()
    
path = "D:\desktop_D\ImageProcessing\python_opencv\License plate recognition\origin\\far"
#image_label = cv2.imread(path + "\label\\far\label_far{}.jpg".format(i)) 
img = cv2.imread(path + '\\far_{}.jpg'.format(1))
h = img.shape[0]
w = img.shape[1]
gray = np.empty((h, w), np.uint8)
thresh = np.empty((h, w), np.uint8)
# 分離出R、G、B的分量
(B,G,R) = cv2.split(img)
# 計算灰階
gray[:h, :w] = 0.299*R[:h, :w] + 0.587*G[:h, :w] + 0.114*B[:h, :w]
R = gray[:, :]
G = gray[:, :]
B = gray[:, :]


if gray[:h, :w] < 127:
    thresh[:h, :w] = 0 * int(bool(gray[:h, :w]))
else:
    thresh[:h, :w] = 255 * int(bool(gray[:h, :w]))
    
#diff_thresh(img_R)
img_show("thresh", gray)