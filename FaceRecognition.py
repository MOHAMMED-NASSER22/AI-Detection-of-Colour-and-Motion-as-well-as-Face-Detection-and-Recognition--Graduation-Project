import cv2
import numpy as np
import face_recognition
import os
import threading
import time

# for initialization
def config():
    print("reading known...")
    path = 'known'
    images = []
    Names = []
    X_Center = 320
    Y_Center = 240

    cap = cv2.VideoCapture(0)
    _, img = cap.read()
    _, imgS = cap.read()

    return img, imgS, images, Names, Y_Center, X_Center, path, cap
# load all the known images
def LoadImages():
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        Names.append(os.path.splitext(cl)[0])
    print(Names)
# find all faces
def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
# save the results of encoded faced to get them faster next time
def EncodingImage():
    import pickle

    # to read
    with open('EncodingImage.txt', 'rb') as file:
        Outlist = pickle.load(file)

    if len(Names) != len(Outlist):
        print("It seems that you added or removed an image  ")
        print("Begin start new encoding... ,  it will take awhile ")
        print("Don't worry  it will load faster next time")
        Outlist = findEncoding(images)
        # to write
        with open('EncodingImage.txt', 'wb') as file:
            pickle.dump(Outlist, file)
    return Outlist
    print(len(Names))
    print(len(Outlist))

# live feed without any face reconciliation (i did it for the limitation of Raspberry pi 4 computation power )
def LiveFeed():
    while True:
        try:  # there is many problem occur here from threading  exception  so i except the exception XD And it work !!!
            while True:
                _, LiveFeed = cap.read()
                # _, LiveFeed = cap.read()
                cv2.imshow("LiveFeed", LiveFeed)
                if cv2.waitKey(10) & 0xFF == ord('s'):
                    break
        except:
            print("there is something wrong happend in thread 1 , trying again ....")

# live feed but now with all the faces been recognized
def faceRecognition():
    while True:
        try:  # there is many problem occur here from threading exception  so i except the exception XD And it work !!!
            while True:
                time.sleep(1)
                _, imgS = cap.read()
                # _, img = cap.read()

                #
                # _, imgS = cap.read()
                # _, imgS = cap.read()
                # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                # turn  to gray for reducing the cpu needs (as no need color for detection)
                imgSGRAY = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
                imgSRGB = cv2.cvtColor(imgSGRAY, cv2.COLOR_BGR2RGB)  # transfer the frame from BGR to RGB

                facesCurFrame = face_recognition.face_locations(imgSRGB)  # get all faces
                encodesCurFrame = face_recognition.face_encodings(imgSRGB, facesCurFrame)  # encode all faces
                # compare the known faces with the faces in the frame
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(Outlist, encodeFace)
                    # print(matches)
                    faceDis = face_recognition.face_distance(Outlist, encodeFace)
                    # print(faceDis)
                    matchIndex = np.argmin(faceDis)
                    # print(matchIndex)
                    # print(matches[matchIndex])
                    nameFlag = False  # there is no known face
                    if matches[matchIndex]:
                        nameFlag = True  # there is known face
                        name = Names[matchIndex].upper()
                        print(name)
                        y1, x2, y2, x1 = faceLoc
                        # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(imgS, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.rectangle(imgS, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(imgS, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                        # cv2.putText(imgS, f'{name} {round(1 - (faceDis[matchIndex]), 2)}', (x1 + 6, y2 - 6),
                        #             cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255),
                        #             2)  # if i want to print the matches percent %
                        # markAttendce(name) # save the when the face detect in the app
                        # faceloc1 = faceLoc
                        # print(faceLoc)
                        X_Center = (x1 + x2) / 2
                        Y_Center = (y1 + y2) / 2
                        print(" X_Center :  " + str(X_Center) + "  Y_Center :  " + str(Y_Center))
                    else:
                        name = 'UNKNOWN' # if there is unknown face print unknown under his face
                        print(name)
                        y1, x2, y2, x1 = faceLoc
                        # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(imgS, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.rectangle(imgS, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(imgS, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)

                cv2.imshow('faceRecognition', imgS)
                if cv2.waitKey(1) & 0xFF == ord('f'):
                    break

        except:
            print("there is something wrong happend in thread 2 , trying again ....")

def finaly():
    start = time.perf_counter()
    LoadImages()
    print('start encoding... ')
    Outlist = EncodingImage()
    finish = time.perf_counter()
    print('encoding complete....')
    print(f'Finished in {round(finish - start, 2)} second(s)')

    return Outlist


img, imgS, images, Names, Y_Center, X_Center, path, cap = config()
Outlist = finaly()


# threading
t1 = threading.Thread(target=LiveFeed)  # Thread 1
t2 = threading.Thread(target=faceRecognition)  # Thread 2
t1.start()
t2.start()
t1.join()
t2.join()
