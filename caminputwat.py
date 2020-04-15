import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cam = cv.VideoCapture(0)
#fgbg = cv.createBackgroundSubtractorMOG2()

cv.namedWindow("Camera")
cv.namedWindow("Shadow")

while True:
	ret, frame = cam.read()
	frame = cv.flip(frame, 1)
	gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
	val, fgmask = cv.threshold(gray,25,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
	#fgmask = fgbg.apply(frame)
	#noise
	kernel = np.ones((3,3),np.uint8)
	opening = cv.morphologyEx(fgmask,cv.MORPH_OPEN,kernel, iterations =2)
	sure_bg = cv.dilate(opening,kernel,iterations=3)
	dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
	ret, sure_fg = cv.threshold(dist_transform,0.3*dist_transform.max(),255,0)
	sure_fg = np.uint8(sure_fg)
	unknown = cv.subtract(sure_bg,sure_fg)
	# Marker labelling
	ret, markers = cv.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	markers = cv.watershed(frame,markers)
	frame[markers == -1] = [255,0,0]	

	cv.imshow("Camera", frame)
	cv.imshow("Shadow", fgmask)
	if not ret:
		break
	key = cv.waitKey(1)

cam.release()

cv.destroyAllWindows()