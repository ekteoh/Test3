import numpy as np
import cv2

cap=cv2.VideoCapture(0)

fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
	ret,frame=cap.read()
	fgmask=fgbg.apply(frame)
	bgremove=cv2.bitwise_and(frame,frame,mask=fgmask)
	cv2.imshow('frame',bgremove)
	if cv2.waitKey(1) & 0xff:
		continue
cap.release()
cv2.destroyAllWindows()