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
    # 兩張圖須為同一大小，包括色彩通道
    img2 = cv2.resize(img2, (w,h), interpolation=cv2.INTER_AREA)
    # 調整兩張圖的比例
    alpha = 0.7
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    return img_add

def salt(img, img2, n):
    noff = int((n - 1) / 2) # 設定馬賽克 1 pixel 的大小
    for i in range(noff,img.shape[0]-noff,noff): # 垂直長度，每 1 pixel前進一步
        for j in range(noff,img.shape[1]-noff,noff): # 水平長度，每 1 pixel前進一步
            # 將新 pixel 位置指派給RGB
            (b, g, r) = img[i, j]
            for m in range(-noff,  noff):
                for n in range(-noff, noff):
                    # 指派新色階給RGB
                    img[i+m, j+n, 0] = b
                    img[i + m, j + n, 1] = g
                    img[i + m, j + n, 2] = r

    return img

for i in range(1, 11):   
    image_far = cv2.imread(path + "\origin\\far\\far_{}.jpg".format(i))
    
    # H為垂直長度(長)，W水平長度(寬)
    H = image_far.shape[0]
    W = image_far.shape[1]
    
    # 建立兩個與原圖長寬相等的陣列，用來存放灰階與二值化後的圖
    gray = np.empty((H, W), np.uint8)
    thresh = np.empty((H, W), np.uint8)

    # 分離出R、G、B的分量
    B = image_far[:H, :W, 0]
    G = image_far[:H, :W, 1]
    R = image_far[:H, :W, 2]
    
    # 計算灰階
    gray[:, :] = 0.299*R + 0.587*G + 0.114*B
    cv2.imwrite(path + "\label\\gray\gray_{}.jpg".format(i), gray) 
    
    
    # 邊緣偵測：使用垂直濾鏡
#    kernel_edge = np.array([[-1.0, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    edge = cv2.filter2D(gray, -1, kernel_edge)
    cv2.imwrite(path + "\label\\edge\edge_{}.jpg".format(i), edge) 
    
    # 二值化：閥值為127
    for j in range(0, H):
        for k in range(0, W):
            if edge[j][k] < 127:
                edge[j][k] = 0
                thresh[j][k] = edge[j][k]
            else:
                edge[j][k] = 255
                thresh[j][k] = edge[j][k]
    cv2.imwrite(path + "\label\\thresh\\thresh_{}.jpg".format(i), thresh)
  
    # 第一種：中值濾波
    blurred = cv2.medianBlur(thresh, 3)
    cv2.imwrite(path + "\label\\blurred\\median\\median_{}.jpg".format(i), blurred)
# =============================================================================
#     # 第二種：雙邊濾波
#    blurred = cv2.bilateralFilter(thresh,9,41,41)
#    cv2.imwrite(path + "\label\\blurred\\bilateral\\bilateral_{}.jpg".format(i), blurred)
# =============================================================================
# =============================================================================
#     # 第三種：高斯濾波
#     blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
# =============================================================================
#    img_show("blurred", blurred)
#    
#    # 水平拼接
#    hmerge = np.hstack((edge, thresh))
#    img_show("edge & thresh", hmerge)
#    img_show("blurred", blurred)
    
    # 第一種：先膨脹再侵蝕
    kernel_dilate = np.ones((11, 11), np.uint8)
    dilation = cv2.dilate(blurred, kernel_dilate,iterations = 4)
        
    kernel_erode = np.ones((3,7),np.uint8)
    erosion = cv2.erode(dilation,kernel_erode,iterations = 1)
    
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
#    # 水平拼接
#    hmerge = np.hstack((erosion, dilation))
#    img_show("merge", hmerge)
# =============================================================================
    
    # 利用8區塊連結法將白色(pixel值 = 255)的區塊分別取出，黑色(pixel值 = 0)為背景
    labels = measure.label(erosion, neighbors = 8, background = 0)
    #建立一個空的圖，存放篩選出的輪廓
    mask = np.zeros(erosion.shape, dtype="uint8")
    #顯示總共有幾塊白色輪廓
    print("\nTotal {} blobs".format(len(np.unique(labels))))
    #依序編號每個白色輪廓
    for (j, label) in enumerate(np.unique(labels)):
            #如果label = 0，表示它為背景
            if label == 0:
                print("label: 0 (background)")
                continue
            #否則為前景，顯示其label編號l
            print("label: {} (foreground)".format(j))
            
            #建立該前景的Binary圖
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            
            #有幾個非0的pixel？
            numPixels = cv2.countNonZero(labelMask)
            
            #如果pixel數目大於9000判斷可能為車牌
            if numPixels >= 9000:
                #放到剛剛建立的空圖中
                mask = cv2.add(mask, labelMask)
    
    # 將過濾出來的標籤圖儲存
    cv2.imwrite(path + "\label\\far\label_far{}.jpg".format(i), mask)        
    # 讀取標籤圖
    image_label = cv2.imread(path + "\label\\far\label_far{}.jpg".format(i)) 
    # 做疊圖並儲存
    img_add = addImage(image_far, image_label)
    cv2.imwrite(path + "\label\\add\label_add{}.jpg".format(i), img_add) 
    # 取AND並儲存
    img_AND = cv2.bitwise_and(image_far, image_label)
    cv2.imwrite(path + "\label\AND\label_AND{}.jpg".format(i), img_AND)
    # 水平拼接
    hmerge = np.hstack((img_add, img_AND))
    img_show("add & AND ", hmerge)
    
    # 取輪廓
    img,contours,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 1
    for c in contours:        
        mask_1 = np.zeros(mask.shape, dtype="uint8")  #依前面mask的輪廓圖形建立mask_1
        cv2.drawContours(mask_1, [c], -1, 255, -1)
        
        # 計算輪廓面積並給予數字標籤
        area = cv2.contourArea(c)
        print("Contour #%d"%cnt)
        clone = image_far.copy()
        
        # 畫矩形方框
        (x, y, w, h) = cv2.boundingRect(c)
        print("— area: %.2f"%area, "\n— rectangle bound width: %.f"%w,
              "\n— rectangle bound height: %.f"%h) # 顯示輪廓面積與方框的長寬
        cv2.rectangle(clone, (x, y), (x+w, y+h), (223, 199, 22), 2)
        
        # 印出數字標籤
        cv2.putText(clone, "#%d"%cnt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (223, 199, 22), 3)
                    
        #將mask與灰階圖取AND
        AND_mask = cv2.bitwise_and(mask, mask, mask=mask_1)
        AND_mask = cv2.bitwise_and(AND_mask, gray)
        
        # 水平拼接
        hmerge = np.hstack((mask, mask_1, AND_mask))
        img_show("Image & Mask & Mask_Area", hmerge)
        
        # 車牌的長寬比約為0.9 ~ 3之間
        if (w / h) >= 0.9 and (w / h) <= 3:
            img_show("License plate", clone)
            cv2.imwrite(path + "\label\\bound\plate_%d_#%d.jpg"%(i, cnt), clone)
           
            # 車牌區間
            plate = clone[y:y+h, x:x+w]
            image_plate = plate.copy()
            mosaic = salt(plate, image_plate, 40)
            clone[y:y+h, x:x+w] = mosaic
            img_show("plate mosaic", clone)
            cv2.imwrite(path + "\label\\mosaic\\mosaic_%d_#%d.jpg"%(i, cnt), clone)
        cnt += 1