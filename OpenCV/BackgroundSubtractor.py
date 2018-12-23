# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 01:43:13 2018

@author: USER
"""

import cv2
#import numpy as np

camera = cv2.VideoCapture(0) # 参数0表示第一个摄像头
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
es_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

while True:
    grabbed, frame_lwpCV = camera.read()
    frame_lwpCV = cv2.resize(frame_lwpCV, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_AREA)
    fgmask = bs.apply(frame_lwpCV) # 背景分割器，该函数计算了前景掩码

    th = cv2.threshold(fgmask, 240, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.medianBlur(th, 3)
    
    dilated = cv2.dilate(blurred, es_1, iterations=4) 
    erode = cv2.erode(dilated, es, iterations=1)
    
    image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) > 30000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame_lwpCV, "object", (y, x), cv2.FONT_HERSHEY_SIMPLEX, 1, (223, 199, 22), 2)

    cv2.imshow('blurred', blurred)
    cv2.imshow('dilated', dilated)
    cv2.imshow('detection', frame_lwpCV)
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
