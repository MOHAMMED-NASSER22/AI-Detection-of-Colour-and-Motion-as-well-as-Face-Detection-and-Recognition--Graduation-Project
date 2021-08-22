import time

import cv2
import numpy as np
import winsound

threshold = 5000
def stackImages(scale, imgArray):
    rows = len(imgArray)  # setting the rows
    cols = len(imgArray[0])  # setting the colums
    rowsAvailable = isinstance(imgArray[0], list)  # determinig available rows
    width = imgArray[0][0].shape[1]  # determining width
    height = imgArray[0][0].shape[0]  # determining height
    if rowsAvailable:  # checking for available rows
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

cam = cv2.VideoCapture(0)
_, frame1 = cam.read()
_, frame2 = cam.read()
time.sleep(1)
_, frame1 = cam.read()
_, frame2 = cam.read()

while cam.isOpened():
    _, frame1 = cam.read()
    _, frame2 = cam.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    # cv2.drawContours(frame1 , contours , -1, (0,255,0), 2 )
    num = 0
    for c in contours:
        num = num + 1
        if cv2.contourArea(c) < threshold:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2) #(G, B, R)
        # winsound.Beep(500, 200)
        print("there is a motion detected")
        # sent notif
        break
    imgStack = stackImages(.5, ([frame1, diff,gray], [blur, thresh, dilated]))
    # print(num)
    if cv2.waitKey(10) == ord('q'):
        break
    # cv2.imshow('frame', diff)
    cv2.imshow('frame', imgStack)


cam.release()
cv2.destroyAllWindows()
