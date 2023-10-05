import cv2
import time
import mediapipe as mp
import Hand_Tracking_Module as htm
import os
import numpy as np

########
brushThickness = 10
eraserThickness = 100
########

xp,yp = 0,0
imgCanvas = np.zeros((720,1280,3), np.uint8)

pTime = 0

wCam, hCam = 1280, 750

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = "Color_Img"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]

drawColor = (0,165,255)


detector = htm.handDetector(detectionCon=0.75)

lmList = []

while True:
    # Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Finding hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList)

        # Tip of index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        # Checking which finngers are up
        fingers = detector.fingersUp()
        #print(fingers)

        # Selection mode- If two fingers are up
        if fingers[1] and fingers[2]:
            xp,yp = 0,0

            print("Selection mode")

            # Checking for click
            if y1 < 125:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor = (0,165,255)
                elif 450<x1<750:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 750<x1<950:
                    header = overlayList[2]
                    drawColor = (0,128,0)
                elif 950<x1<1280:
                    header = overlayList[3]
                    drawColor = (0,0,0)
        
            cv2.rectangle(img, (x1,y1-30), (x2,y2+30), drawColor, cv2.FILLED)

        # Drawing mode- If index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 10, drawColor, cv2.FILLED)
            print("Drawing mode")

            if xp == 0 and yp == 0:
                xp,yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)
                
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)

            xp,yp = x1,y1


    
    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_RGB2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2RGB)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting header image
    h, w, c = header.shape
    img[0:h,0:w] = header


    # Calculating fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}',(10,155), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Paint",img)
    #cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)