# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:44:19 2018

@author: 沈鴻儒
"""
# =============================================================================
# #車牌辨識
# =============================================================================
#下方的步驟中, 我們使用skimage提供的Adaptive threshold而非OpenCV
from skimage.filters import threshold_adaptive
#Connected-component labeling相關功能就放在skimage的子模組measure
from skimage import measure
import numpy as np
import cv2

def License_plate_recognition(image):
    global hmerge
    # 缩小圖像 
    #height, width = image.shape[:2]
    #size = (int(width*1.0), int(height*1.0))
    #shrink = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    shrink = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    
    #模糊化處理
    plate = cv2.medianBlur(shrink, 5)
    #將圖片由RGB轉為HSV格式，然後取HSV中的Ｖ值，此效果與灰階效果類似。
    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]

    #使用skimage提供的Adaptive threshold
    #skimage.filters.threshold_adaptive(image, block_size, method='gaussian')
    #block_size: 块大小，指当前像素的相邻区域大小，一般是奇数(如3，5，7。。。)
    #bitwise_not—图像非运算
    #函数原型：bitwise_and(src1, src2, dst=None, mask=None)
    #src1：图像矩阵1
    #src1：图像矩阵2
    #dst：默认选项
    #mask：默认选项
    thresh = threshold_adaptive(V, 47, offset= 25).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)

    ##顯示原圖及Threshold處理後的圖片
    #cv2.imshow("License Plate", plate)
    #cv2.imshow("Thresh", thresh)
    
    #針對thresholded圖片進行connected components analysis，
    # neighbors=8表示採用8向方式, background=0表示pixel值為0則認定為背景
    labels = measure.label(thresh, neighbors=8, background=0)

    #建立一個空的圖，存放稍後將篩選出的字母及數字
    mask = np.zeros(thresh.shape, dtype="uint8")
    #顯示一共貼了幾個Lables（即幾個components）
    print("[INFO] Total {} blobs".format(len(np.unique(labels))))
    #依序處理每個labels
    for (i, label) in enumerate(np.unique(labels)):
            #如果label=0，表示它為背景
            if label == 0:
                print("[INFO] label: 0 (background)")
                continue
            #否則為前景，顯示其label編號l
            print("[INFO] label: {} (foreground)".format(i))
            
            #建立該前景的Binary圖
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            
            #有幾個非0的像素?
            numPixels = cv2.countNonZero(labelMask)
            
            #如果像素數目在2500~4000之間認定為車牌字母或數字
            if numPixels >=2000 and numPixels < 10000:
                #放到剛剛建立的空圖中
                mask = cv2.add(mask, labelMask)

    ##顯示該前景物件
    #cv2.imshow("Label", labelMask)
    #cv2.waitKey(0)
                
    ##顯示所抓取到的車牌
    #cv2.imshow("Large Blobs", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #水平拼接
    hmerge = np.hstack((thresh, mask)) 
    #垂直拼接
    #vmerge = np.vstack((thresh1, thresh2))
    cv2.imshow("hmerge", hmerge)
    #cv2.imshow("test2", vmerge)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = r"D:\desktop_D\Python\python\OpenCV"
 
for i in range(1, 11):   
    image = cv2.imread(path + "\{}.jpg".format(i))
    License_plate_recognition(image)
    cv2.imwrite(path + "\label_{}.jpg".format(i), hmerge)