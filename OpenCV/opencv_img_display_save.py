# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:30:52 2018

@author: 沈鴻儒
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena.jpg")
cv2.imshow("original", img)
##nameWindow：先創建窗口，再載入圖像，cv2.WINDOW_NORMAL可以調整大小
#cv2.namedWindow("image1")

#matplotlib顯示圖像
#plt.imshow(img, cmap = "gray", interpolation = "bicubic")
#plt.xticks([]), plt.yticks([])
#plt.show()


#按esc離開，按s儲存
k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord("s"):
    cv2.imwrite("image_test.jpg", img)
    cv2.destroyAllWindows()