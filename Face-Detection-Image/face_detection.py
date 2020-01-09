import numpy as np
import cv2
from matplotlib import pyplot as plt

faceCascade = cv2.CascadeClassifier('C:\Program Files (x86)\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')#절대경로
image = cv2.imread('face.jfif')

faces = faceCascade.detectMultiScale(image, 1.01, 10)

print(faces.shape)
print("Number of faces detected: " + str(faces.shape[0]))
print(faces)

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.rectangle(image, ((0,image.shape[0] -35)),
(250, image.shape[0]), (255,255,255), -1);
cv2.putText(image, "Face Detection", (0,image.shape[0] -10),
cv2.QT_FONT_NORMAL, 1.0, (0,0,0), 1);

plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()

cv2.waitKey(0)