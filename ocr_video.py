import cv2
import numpy as np
import pickle

with open('trained.dump','rb') as f:
	ann=pickle.load(f)

# img = cv2.imread("num1.jpg")
cam = cv2.VideoCapture(0)
while True:
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)

	ret, img_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

	ctrs,hier = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]

	for rect in rects:
		cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2) 
		y = rect[1]
		x = rect[0]
		th_x=rect[3]//8
		th_y=rect[3]//10
		roi = img_th[y-th_y:y+rect[3]+th_y, x-th_x:x+rect[2]+th_x]
		try:
			roi = cv2.resize(roi, (20, 20))
			roi = cv2.dilate(roi, (6, 6))
			roi = cv2.copyMakeBorder(roi,4,4,4,4,cv2.BORDER_CONSTANT,(255,255,255))
			roi = (roi.reshape(784,))/255
			out=ann.feed_forward(roi)
			nbr=out.argmax()

			cv2.putText(img, str(nbr), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
		except:
			pass

	cv2.imshow("Results", img)
	if cv2.waitKey(1) & 0xff == 27:
		break

cam.release()
cv2.destroyAllWindows()