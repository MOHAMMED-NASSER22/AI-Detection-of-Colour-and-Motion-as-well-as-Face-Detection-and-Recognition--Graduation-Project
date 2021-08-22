
import time
import cv2
import numpy as np


#adjusting motor parameters 

    

# Set frequency to 60hz, good for servos.

#function to compine shown windows
def stackImages(scale,imgArray):
   rows = len(imgArray) #setting the rows
   cols = len(imgArray[0]) #setting the colums 
   rowsAvailable = isinstance(imgArray[0], list) #determinig available rows 
   width = imgArray[0][0].shape[1] #determining width
   height = imgArray[0][0].shape[0] #determining height
   if rowsAvailable: # checking for available rows 
       for x in range ( 0, rows):
           for y in range(0, cols):
               if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                   imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
               else:
                   imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
               if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
       imageBlank = np.zeros((height, width, 3), np.uint8)
       hor = [imageBlank]*rows
       hor_con = [imageBlank]*rows
       for x in range(0, rows):
           hor[x] = np.hstack(imgArray[x])
       ver = np.vstack(hor)
   else:
       for x in range(0, rows):
           if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
               imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
           else:
               imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
           if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
       hor= np.hstack(imgArray)
       ver = hor
   return ver


cap = cv2.VideoCapture(0) #initiate camera capture 
x_medium = 320 #detected line position
y_medium = 320 #detected line position
position = 320 # variable to set initial position in middle of screen range

while True:
    _ , frame = cap.read() #making copy of each frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert frame from RGB to HSV

    # selecting red color 
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    #selecting orange color 
    low_orange = np.array([0, 125, 152])
    high_orange = np.array([179, 255, 255])
    #creating masks to isolate the prefered color 
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange) #selected color is orange
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    orange = cv2.bitwise_and(frame, frame, mask=orange_mask)


    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # to drow rectangel
        x_medium = int((x + x + w) / 2)  # determine x center of rectangular
        y_medium = int((y + y + w) / 2)  # determine x center of rectangular

        break # to exit the loop when the biggest value is detected
# 
    cv2.line(frame, (x_medium, 0), (x_medium, 480), (0, 255, 0), 2) #create line at the center of rectangular
    cv2.line(frame, (320, 0), (320, 480), (0, 255, 0), 2)#create line at the center of frame

    cv2.line(frame, (0, y_medium), (640, y_medium), (0, 255, 0), 2)  # create line at the center of rectangular
    cv2.line(frame, (0, 240), (640, 240), (0, 255, 0), 2)  # create line at the center of frame
#     

    #cv2.imshow("red", orange)
    cv2.imshow("hopa", frame) # show the frame
    # imgStack = stackImages(0.7,([orange,frame])) #calling the imgstack function to compine the mask and the frame
    #cv2.imshow("imgStack", imgStack) # show both the frame and the mask

    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release() #stop racording frame
cv2.destroyAllWindows() #close all windows 


