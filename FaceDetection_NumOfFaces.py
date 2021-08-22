import cv2     # opencv-python==4.2.0.32
import numpy as np
import dlib  # machine learning lib  ... Cmake ... wheel

video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

while video.isOpened():
    _, frame = video.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    num = 0
    for face in faces:
        x, y = face.left(), face.top()
        h, w = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (h, w), (0, 0, 255), 2)
        num = num + 1

        cv2.putText(frame, 'face' + str(num), (x - 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('faces', frame)
    print("there are "+ str(num) + " face/s")
    if cv2.waitKey(10) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
