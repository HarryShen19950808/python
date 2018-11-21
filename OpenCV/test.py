import cv2
import numpy as np
#import matplotlib.pyplot as plt
import pylab as pl

# 
def addImage(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img = cv2.imread(img2_path)
    h, w, _ = img1.shape
    # 函数要求两张图必须是同一个size
    img2 = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
    #print img1.shape, img2.shape
    #alpha，beta，gamma可调
    alpha = 0.7
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    cv2.namedWindow('img_add')
    cv2.imshow('img_add',img_add)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("add_{:02d}.jpg".format(3), img_add)

image_1 = r"D:\desktop_D\ImageProcessing\python_opencv\preprocess_hole\school_3.jpg"
image_2 = r"D:\desktop_D\ImageProcessing\python_opencv\preprocess_hole\sch_3.jpg"
addImage(image_1, image_2)

## 兩圖取AND
#img_1 = cv2.imread(image_1, 1)
#img_2 = cv2.imread(image_2, 1)
#
#bitwiseAnd = cv2.bitwise_and(img_1, img_2)
#cv2.imshow("AND",bitwiseAnd)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#cv2.imwrite("road_{:02d}.jpg".format(3), bitwiseAnd)

#img = cv2.imread(image, 0)

## 二維旋積
#src = cv2.imread("lena.jpg", 1)
#
#kernels = [ 
#    (u"低通濾波器",np.array([[1,  1, 1],[1, 2, 1],[1, 1, 1]])*0.1),
#    (u"高通濾波器",np.array([[0.0, -1, 0],[-1, 5, -1],[0, -1, 0]])),
#    (u"邊緣檢驗",np.array([[-1.0, -1, -1],[-1, 8, -1],[-1, -1, -1]]))
#]
#
#index = 0
#fig, axes = pl.subplots(1, 3, figsize=(12, 4.3))
#for ax, (name, kernel) in list(zip(axes, kernels)):
#    dst = cv2.filter2D(src, -1, kernel)
#    dst = cv2.sepFilter2D(src, -1, kernel)
#    # 由於matplotlib的彩色順序和OpenCV的順序相反
#    ax.imshow(dst[:, :, ::-1])
#    ax.set_title(name)
#    ax.axis("off")
#fig.subplots_adjust(0.02, 0, 0.98, 1, 0.02, 0)

## 圖像尺吋變換
#res = cv2.resize(img, (600, 800), interpolation=cv2.INTER_LINEAR)
#cv2.imshow('resize',res)
##cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

## 影像輸出
#for quality in [90, 60, 30]:
#    cv2.imwrite("pic_q{:02d}.jpg".format(quality), img,
#                [cv2.IMWRITE_JPEG_QUALITY, quality])

## 以灰階讀入圖像
#img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # 或是輸入0

## 轉灰階
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(type(img), img.shape, img.dtype)
#print(type(img_gray), img_gray.shape, img_gray.dtype)
#cv2.namedWindow("demo1")
#cv2.imshow("demo1", img_gray)

#cv2.namedWindow("demo1")
#cv2.imshow("demo1", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
