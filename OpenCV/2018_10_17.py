# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:29:42 2018

@author: 沈鴻儒
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('L.jpg', 0)
img = cv2.resize(img, (300, 300))
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#for i in range(6):
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#plt.show()
hmerge = np.hstack((thresh1, thresh2, thresh3, thresh4, thresh5)) #水平拼接
#vmerge = np.vstack((thresh1, thresh2)) #垂直拼接

cv2.imshow("hmerge", hmerge)
#cv2.imshow("test2", vmerge)

cv2.waitKey(0)
cv2.destroyAllWindows()