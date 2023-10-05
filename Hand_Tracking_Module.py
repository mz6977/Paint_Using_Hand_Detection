import cv2
import time
import mediapipe as mp
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.complexity,self.detectionCon,self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12 ,16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        #ADD if need position of X and Y axis
        #print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
            
        self.lmList = []


        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)    
                     
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 25, (255,0,255), cv2.FILLED)
        return self.lmList
    
    def fingersUp(self):
        fingers = []

        # Thumb Right Hand
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
                fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def findDistance(self, index1, index2, img, draw=True, r=3, t=2):
        x1, y1 = self.lmList[index1][1],self.lmList[index1][2]
        x2, y2 = self.lmList[index2][1],self.lmList[index2][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        indexes = [x1,y1,x2,y2,cx,cy]

        if draw == True:
            cv2.circle(img, (x1,y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx,cy), r, (255,0,255), cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), t)

        length = math.hypot(x2-x1, y2-y1)

        return length, img, indexes

    

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        
        if len(lmlist) != 0:
            print(lmlist[2])
    
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
       
        cv2.imshow("Camera", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()