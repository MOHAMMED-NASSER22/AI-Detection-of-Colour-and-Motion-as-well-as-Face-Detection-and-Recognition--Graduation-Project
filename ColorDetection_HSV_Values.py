import cv2
import numpy as np
import time


def empty(a):
    pass


# function to combine shown windows

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
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


cap = cv2.VideoCapture(0)

_, img = cap.read()
time.sleep(1)  # warm up the camera (reduce noise in the first second)
_, img = cap.read()  # take the photo from the cam
# path = "D:\Screenshot_1.png" # uncomment if u have the photo saved

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 320)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 129, 179, empty)
cv2.createTrackbar("sat Min", "TrackBars", 75, 255, empty)
cv2.createTrackbar("sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("val Min", "TrackBars", 150, 255, empty)
cv2.createTrackbar("val Max", "TrackBars", 255, 255, empty)

while True:
    # img = cv2.imread(path) # uncomment if u have the photo saved

    # track bar for all variables
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("val Max", "TrackBars")

    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])  # get the min values from the track bars
    upper = np.array([h_max, s_max, v_max])  # get the max values from the track bars
    mask = cv2.inRange(imgHSV, lower, upper)  # adjust the the frame according to the new variables
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find the contours
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)  # sort them form largest to smallest
    imgResult = cv2.bitwise_and(img, img, mask=mask)  # add color to the mask

    # cv2.imshow("origin", img)
    # cv2.imshow("HSV", imgHSV)
    # cv2.imshow("mask", mask)
    # cv2.imshow("imgResult", imgResult)

    imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult])) # stacked all processing steps in the frames to demonstrate
    cv2.imshow("imgStack", imgStack)

    if cv2.waitKey(10) == ord('q'): # Exit when press q
        break
