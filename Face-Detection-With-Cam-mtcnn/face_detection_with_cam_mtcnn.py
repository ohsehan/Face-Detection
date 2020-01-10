import face_recognition as face_recognition
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
image = face_recognition.load_image_file('image.jpg')
# MTCNN 모델로 얼굴 인식을 수행했다.
results = detector.detect_faces(image)  # highlight-line
for result in results:
    print(result)
    bounding_box = result['box']
    keypoints = result['keypoints']


    # 1. 얼굴 위치 표시
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (255, 255, 255), 2)

    # 2. 주요 부분 landmark 표시
    cv2.circle(image, (keypoints['left_eye']), 2, (255, 255, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (255, 255, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (255, 255, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (255, 255, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (255, 255, 255), 2)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()