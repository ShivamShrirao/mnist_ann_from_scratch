import cv2
import numpy as np
import pickle

with open('trained.dump','rb') as f:
    ann=pickle.load(f)

img = cv2.imread("num1.jpg")
# img = cv2.imread("num.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

ret, img_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

img2,ctrs,hier = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    y = rect[1]
    x = rect[0]
    roi = img_th[y:y+rect[3], x:x+rect[2]]
    roi = cv2.resize(roi, (20, 20))
    roi = cv2.copyMakeBorder(roi,4,4,4,4,cv2.BORDER_CONSTANT,(255,255,255))
    roi = cv2.dilate(roi, (3, 3))
    roi = (roi.reshape(784,))/255
    out=ann.feed_forward(roi)
    nbr=out.argmax()
#     plt.imshow(roi.reshape(28,28), cmap='Greys')
#     plt.text(19, 1,'Prediction: {}'.format(nbr))
#     plt.show()

    cv2.putText(img, str(nbr), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

cv2.imshow("Results", img)
cv2.waitKey(0)
cv2.destroyAllWindows()