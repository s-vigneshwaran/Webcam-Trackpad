import cv2
import numpy as np
import handtrackingmodule as htm
import mouse

cap = cv2.VideoCapture(0)
w_cam, h_cam = 640, 480
frame = 100
smoothness = 7
w_screen, h_screen = 1920, 1080
cap.set(3, w_cam)
cap.set(4, h_cam)

detector = htm.HandDetector(detection_con=0.85, max_hands=1)
xp, yp, xc, yc = 0, 0, 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Landmarks
    img = detector.find_hands(img)
    lmList, bounding_box = detector.find_position(img, draw=True, bounding=True)

    if len(lmList) != 0:
        # Tip of two fingers
        x1, y1 = lmList[8][1:]  # Index Finger
        x2, y2 = lmList[12][1:]  # Middle Finger

        # Active Fingers
        fingers = detector.active_fingers()
        cv2.rectangle(img, (frame, frame), (w_cam - frame, h_cam - frame), (255, 0, 255), 2)

        # Moving Mode - Index Finger
        if fingers[1] == 1 and fingers[2] ==0:
            # Convert Coordinates
            x3 = np.interp(x1, (frame, w_cam - frame), (0, w_screen))
            y3 = np.interp(y1, (frame, h_cam - frame), (0, w_screen))

            # Smooth Values
            xc = xp + (x3 - xp) / smoothness
            yc = yp + (y3 - yp) / smoothness

            # Move Cursor
            mouse.move(xc, yc)
            xp, yp = xc, yp
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        # Clicking Mode - Index and Middle
        if fingers[1] == 1 and fingers[2] == 1:
            # Distance
            length, img, line = detector.compute_distance(8, 12, img)

            # Click if distance is short
            if length < 10:
                cv2.circle(img, (line[4], line[5]), 7, (0, 255, 0), cv2.FILLED)
                mouse.click('left')

    cv2.imshow('Gesture Trackpad', img)
    key = cv2.waitKey(1)
    if key == 27:
        break
