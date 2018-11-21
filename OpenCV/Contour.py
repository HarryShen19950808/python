# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:37:15 2018

@author: 沈鴻儒
"""

import numpy as np
import cv2

path = r"D:\desktop_D\ImageProcessing\python_opencv\License plate recognition"

for i in range(1, 4):   
    image = cv2.imread(path + "\\far_{}.jpg".format(i))
    shrink = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(shrink, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    binaryIMG = cv2.Canny(blurred, 20, 160)
    (_, cnts, _) = cv2.findContours(binaryIMG.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clone = shrink.copy()
    
    
#    cnt = cnts[0]
#    M = cv2.moments(cnt)
#    print(M)
    
    
#    for c in cnts:
#        # CV2.moments會傳回一系列的moments值，我們只要知道中點X, Y的取得方式是如下進行即可。
#        M = cv2.moments(c)
#        cX = int(M["m10"] / M["m00"])
#        cY = int(M["m01"] / M["m00"]) 
#        # 在中心點畫上黃色實心圓
#        cv2.circle(clone, (cX, cY), 10, (1, 227, 254), -1)

    
    cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
    
#    #計算面積
#    area = cv2.contourArea(cnts)
#    #計算周長
#    perimeter = cv2.arcLength(cnts, True)
#    #印出結果    
#    print("Contour #%d — area: %.2f, perimeter: %.2f" % (i + 1, area, perimeter))
    #各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
#    cv2.putText(clone, '#%d'%(i + 1), (50,150), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 12)
                
    cv2.imshow("binary image", binaryIMG)
    cv2.imshow("All Contour", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#    cv2.imwrite(path + "\Contour_{}.jpg".format(i), contour)