#importing libraries
import cv2
import mediapipe as mp

#creating an object for video capture, 0 arguement for in built device camera
cap = cv2.VideoCapture(0)
#mediapipe imports
#initialising the hand tracking model
mpHands = mp.solutions.hands
hands = mpHands.Hands()
#drawing_utils lets you visually track the hands
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read() #returns two values, first is the boolean value for whether the frame is returned or not, and the other one is numpy array of image 

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#converting the color from BGR to RGB bcoz mediapipe works with RGB

    #processing to detect the hands
    results = hands.process(imageRGB)

    # checking whether a hand is detected
    if results.multi_hand_landmarks:
      
        for handLms in results.multi_hand_landmarks: # working with each hand
            print(handLms)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h) 
                #selecting the ids of the tip of fingers
                if id == 20 or id==8 or id==12 or id==16 :
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Output", image)
    cv2.waitKey(1)           