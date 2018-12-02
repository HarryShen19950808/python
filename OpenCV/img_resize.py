# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:35:53 2018

@author: 沈鴻儒
"""

import cv2
#import os
import matplotlib.pyplot as plt



##nameWindow：先創建窗口，再載入圖像，cv2.WINDOW_NORMAL可以調整大小
#cv2.namedWindow("image")
img = cv2.imread("lena.jpg", -1)

height, width = img.shape[:2]

##matplotlib顯示圖像
#plt.imshow(img, cmap = "gray", interpolation = "bicubic")
#plt.xticks([]), plt.yticks([])
#plt.show()


# 缩小图像  
size = (int(width*0.3), int(height*0.3))  
shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
# 放大图像  
fx = 1.2  
fy = 1.2  
enlarge = cv2.resize(shrink, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC) 

cv2.imshow("original", img)
cv2.imshow("shrink", shrink)
cv2.imshow("enlarge", enlarge)

#按esc離開，按s儲存
k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord("s"):
    cv2.imwrite("image_shrink.jpg", shrink)
    cv2.imwrite("image_enlarge.jpg", enlarge)
    cv2.destroyAllWindows()