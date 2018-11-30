# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 02:49:14 2018

@author: USER
"""

import cv2
import numpy as np


def salt(img, img2, n):
    noff = int((n - 1) / 2)
    for i in range(noff,img.shape[0]-noff,noff):
        for j in range(noff,img.shape[1]-noff,noff):
            # img.shape[0] -- 取得img 的列（图片的高）
            # img.shape[1] -- 取得img 的行（图片的宽）
            # i = int(np.random.random() * img.shape[1]);
            # j = int(np.random.random() * img.shape[0]);

            (b, g, r) = img[i, j]
            #b = img[j, i, 0]
            #g = img[j, i, 1]
            #r = img[j, i, 2]
            for m in range(-noff,  noff):
                for n in range(-noff, noff):
                    img[i+m, j+n, 0] = b
                    img[i + m, j + n, 1] = g
                    img[i + m, j + n, 2] = r
                    #img[i+m, j+n] = 255
            #img[j, i, 1] = 255
            #img[j, i, 2] = 255
    return img

path = r"D:\desktop_D\Python\python\OpenCV\License plate recognition"
image_far = cv2.imread(path + "\origin\\far\\far_{}.jpg".format(2))
#img = cv2.imread("100900.jpg")
img2 = image_far.copy()
saltImage = salt(image_far, img2, 17)

cv2.imshow("Salt", saltImage)
cv2.waitKey(0)
cv2.destroyAllWindows()