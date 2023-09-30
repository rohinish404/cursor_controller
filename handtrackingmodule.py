import cv2
import mediapipe as mp
import pyautogui

upper_points = [4, 8, 12, 16, 20]

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self,image, handNo=0, draw=True):
        pt_x,pt_y = 0,0
        pt_x2,pt_y2 = 0,0
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                if id==upper_points[0]:
                    pt_x,pt_y = cx,cy

                if id==upper_points[1]:
                    pt_x2,pt_y2 = cx,cy    

                if id==upper_points[1]:
                    pt_x3,pt_y3 = cx,cy  


                if id == upper_points[1]:
                    if draw:
                        cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)   
                    pyautogui.moveTo(cx,cy)

            if ((pt_x2 - pt_x)**2 + (pt_y2 - pt_y)**2) ** 0.5 < 20:
                pyautogui.click()   

            elif ((pt_x3 - pt_x2)**2 + (pt_y3 - pt_y2)**2) ** 0.5 < 20 and pt_y2 > pt_y:
                pyautogui.scroll(10)  

            elif ((pt_x3 - pt_x2)**2 + (pt_y3 - pt_y2)**2) ** 0.5 < 20 and pt_y2 < pt_y:
                pyautogui.scroll(-10)           

                 
        return lmlist


def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success,image = cap.read()
        image = tracker.handFinder(image)
        lmList = tracker.positionFinder(image)
        cv2.imshow("Video",image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()    
