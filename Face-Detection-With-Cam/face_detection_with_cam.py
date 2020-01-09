import numpy as np
import cv2

detector = cv2.CascadeClassifier('C:\Program Files (x86)\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
image = cv2.imread('face.jpg')

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    if faces is None:
        return
    # for (x, y, w, h) in faces:
    #     cropped_face = img[y:y+h, x:x+w]
    return faces
while True:
    ret, img = cap.read()
    faces = face_detector(img)
    cropped_face = face_detector(image)

    if cropped_face is None or (cropped_face is not None and len(cropped_face) != 1):
        cropped_face = image
    # if face_detector(image) is not None and len(face_detector(image) != 1):
    #     cropped_face = image
    else:
        for (x, y, w, h) in cropped_face:
            cropped_face = image[y:y + h, x:x + w]

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 2)
        face_img = cv2.resize(cropped_face, (w, h), interpolation=cv2.INTER_AREA)
        img[y:y+h, x:x+w] = face_img
    cv2.imshow('cropped', cropped_face)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()