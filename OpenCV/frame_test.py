# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:54:58 2018

@author: USER
"""

import cv2
import numpy as np
from skimage import measure

cap = cv2.VideoCapture(0) # 参数0表示第一个摄像头

while True:
    grabbed, frame = cap.read()
    resize = cv2.resize(frame, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_AREA)
    H = resize.shape[0]
    W = resize.shape[1]
    gray = np.empty((H, W), np.uint8)
    B = resize[:H, :W, 0]
    G = resize[:H, :W, 1]
    R = resize[:H, :W, 2]
    gray[:, :] = 0.299*R + 0.587*G + 0.114*B
    kernel_edge = np.array([[-1.0, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge = cv2.filter2D(gray, -1, kernel_edge)
    ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
    blurred = cv2.medianBlur(thresh, 3)
    
    kernel_dilate = np.ones((11, 11), np.uint8)
    dilation = cv2.dilate(blurred, kernel_dilate,iterations = 4)   
    kernel_erode = np.ones((3,7),np.uint8)
    erosion = cv2.erode(dilation,kernel_erode,iterations = 1)
    labels = measure.label(erosion, neighbors = 8, background = 0)
    #建立一個空的圖，存放篩選出的輪廓
    mask = np.zeros(erosion.shape, dtype="uint8")
    #顯示總共有幾塊白色輪廓
#    print("Total {} blobs".format(len(np.unique(labels))))
    #依序編號每個白色輪廓
    for (j, label) in enumerate(np.unique(labels)):
            #如果label = 0，表示它為背景
            if label == 0:
#                print("label: 0 (background)")
                continue
            #否則為前景，顯示其label編號l
#            print("label: {} (foreground)".format(j))
            
            #建立該前景的Binary圖
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            
            #有幾個非0的像素?
            numPixels = cv2.countNonZero(labelMask)
            
            #如果像素數目大於9000認定為車牌字母或數字
            if numPixels >= 9000:
                #放到剛剛建立的空圖中
                mask = cv2.add(mask, labelMask)
    img,contours,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 1
    for c in contours:        
        mask_1 = np.zeros(mask.shape, dtype="uint8")  #依前面mask的輪廓圖形建立mask_1
        cv2.drawContours(mask_1, [c], -1, 255, -1)
        
        # 計算輪廓面積並給予數字標籤
        area = cv2.contourArea(c)
#        print("Contour #%d"%cnt)
        clone = resize.copy()
        
        # 畫矩形方框
        (x, y, w, h) = cv2.boundingRect(c)
#        print("— area: %.2f"%area, "\n— rectangle bound width: %.f"%w,
#              "\n— rectangle bound height: %.f"%h) # 顯示輪廓面積與方框的長寬
        cv2.rectangle(clone, (x, y), (x+w, y+h), (223, 199, 22), 2)
        
        # 印出數字標籤
#        clone[y:y+h, x:x+w]
        cv2.putText(clone, "object %d"%cnt, (y, x), cv2.FONT_HERSHEY_SIMPLEX, 1, (223, 199, 22), 2)
                    
        #將mask與灰階圖取AND
        AND_mask = cv2.bitwise_and(mask, mask, mask=mask_1)
        AND_mask = cv2.bitwise_and(AND_mask, gray)
        if (w / h) >= 0.9 and (w / h) <= 3:
            cv2.imshow('frame', clone)
        cnt += 1
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
