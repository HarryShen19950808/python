# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:54:57 2018

@author: 沈鴻儒
"""

import cv2
import numpy as np
from skimage import measure

# =============================================================================
# #車牌辨識ver2.0
# =============================================================================
# 第一種：灰階 ---> 邊緣偵測 ---> 二值化 ---> 中值濾波 ---> 膨脹侵蝕 ---> 取標籤

path = r"D:\desktop_D\Python\python\OpenCV\License plate recognition"

def img_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def addImage(img1, img2):
    h, w, _ = img1.shape
    # 函数要求两张图必须是同一个size
    img2 = cv2.resize(img2, (w,h), interpolation=cv2.INTER_AREA)
    #print img1.shape, img2.shape
    #alpha，beta，gamma可调
    alpha = 0.7
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    return img_add
    
    
for i in range(1, 11):   
    image_far = cv2.imread(path + "\origin\\far\\far_{}.jpg".format(i))
    h = image_far.shape[0]
    w = image_far.shape[1]
    gray = np.empty((h, w), np.uint8)
    # 分離出R、G、B的分量
    (B,G,R) = cv2.split(image_far)
#    img_show("origin", image_far)
    
    
    # 計算灰階
    gray[:h, :w] = 0.299*R[:h, :w] + 0.587*G[:h, :w] + 0.114*B[:h, :w]
    R = gray[:, :]
    G = gray[:, :]
    B = gray[:, :]
#    img_show("gray", gray)
    
    
    # 用垂直濾鏡取邊緣偵測
#    kernel_edge = np.array([[-1.0, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    edge = cv2.filter2D(gray, -1, kernel_edge)
#    img_show("edge", edge)
    
    # 二值化
    ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
#    img_show("binary", thresh)
#    
    # 中值濾波
    blurred = cv2.medianBlur(thresh, 3)
    
# =============================================================================
#     # 雙邊濾波
#     blurred = cv2.bilateralFilter(thresh,9,41,41)
# =============================================================================
#    img_show("blurred", blurred)
#    
#    # 水平拼接
#    hmerge = np.hstack((edge, thresh, blurred))
#    img_show("merge", hmerge)
    
    # 第一種：先膨脹再侵蝕
    kernel_dilate = np.ones((11, 11), np.uint8)
    dilation = cv2.dilate(blurred, kernel_dilate,iterations = 4)
#    img_show("dilation", dilation)
        
    kernel_erode = np.ones((3,7),np.uint8)
    erosion = cv2.erode(dilation,kernel_erode,iterations = 1)
#    img_show("erosion", erosion)
    
    # 水平拼接
    hmerge = np.hstack((gray, dilation, erosion))
    img_show("merge", hmerge)
    
# =============================================================================
#     # 第二種：先侵蝕再膨脹
#     kernel_erode = np.ones((3,3),np.uint8)
#     erosion = cv2.erode(blurred,kernel_erode,iterations = 1)
# #    img_show("erosion", erosion)
#     
#     kernel_dilate = np.ones((1,11),np.uint8)
#     dilation = cv2.dilate(erosion, kernel_dilate,iterations = 3)
# #    img_show("dilation", dilation)
#     
#     # 水平拼接
#     hmerge = np.hstack((erosion, dilation))
#     img_show("merge", hmerge)
# =============================================================================
    
    labels = measure.label(erosion, neighbors=8, background=0)
    #建立一個空的圖，將篩選出的字母及數字存放至此
    mask = np.zeros(erosion.shape, dtype="uint8")
    #顯示一共貼了幾個Labels（即幾個components）
    print("[INFO] Total {} blobs".format(len(np.unique(labels))))
    #依序處理每個labels
    for (j, label) in enumerate(np.unique(labels)):
            #如果label=0，表示它為背景
            if label == 0:
                print("[INFO] label: 0 (background)")
                continue
            #否則為前景，顯示其label編號l
            print("[INFO] label: {} (foreground)".format(j))
            
            #建立該前景的Binary圖
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            
            #有幾個非0的像素?
            numPixels = cv2.countNonZero(labelMask)
            
            #如果像素數目大於3000認定為車牌字母或數字
            if numPixels >= 9000:
                #放到剛剛建立的空圖中
                mask = cv2.add(mask, labelMask)
#    img_show("mask", mask)
    
    # 將過濾出來的標籤圖儲存
    cv2.imwrite(path + "\label\\far\label_far{}.jpg".format(i), mask)
    
    # 讀取原圖與標籤圖
    image_far = cv2.imread(path + "\origin\\far\\far_{}.jpg".format(i))
    image_label = cv2.imread(path + "\label\\far\label_far{}.jpg".format(i))
    
    # 做疊圖並儲存
    img_add = addImage(image_far, image_label)
    img_show('img_add', img_add)
    cv2.imwrite(path + "\label\\add\label_add{}.jpg".format(i), img_add)
    
    # 取AND並儲存
    bitwiseAnd = cv2.bitwise_and(image_far, image_label)
    img_show("And", bitwiseAnd)
    cv2.imwrite(path + "\label\AND\label_AND{}.jpg".format(i), bitwiseAnd)
    
    # 取輪廓
    img,contours,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 計算面積
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    print(area)
    clone = bitwiseAnd.copy()
    # 畫輪廓
    cv2.drawContours(clone, contours, -1, (0, 0, 255), 2)
    img_show("area", clone)
    
    for c in contours:        
        mask_1 = np.zeros(mask.shape, dtype="uint8")  #依Contours圖形建立mask
        cv2.drawContours(mask_1, [c], -1, 255, -1) #255        →白色, -1→塗滿
        # show the images
        cv2.imshow("Image", mask)
        cv2.imshow("Mask", mask_1)
    #將mask與原圖形作AND運算
    cv2.imshow("Image + Mask", cv2.bitwise_and(mask, mask, mask=mask_1))  
    cv2.waitKey(0)
    cv2.destroyAllWindows()