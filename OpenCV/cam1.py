import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
#img = cv2.imread("C://Users//USER//deep-learning-from-scratch-master//dataset//lena.png")			
#cv2.imshow("Image", img)
#cv2.waitKey(0)
#cv2.destroyWindow("Image")

#gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image", gray_img)
#cv2.waitKey(0)
#cv2.destroyWindow("Image")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
	raise IOError("CANNOT OPEN CAM")
while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_AREA)

	cv2.imshow("Input", frame)
	c = cv2.waitKey(1)
	if c == 27:
		break
cap.release()
cv2.destroyAllWindows()