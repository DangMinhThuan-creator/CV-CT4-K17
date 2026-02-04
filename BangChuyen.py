import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)
# cam = cv.VideoCapture("bang_chuyen.mp4")
base_frame = None

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21,21), 0)

    if base_frame is None:
        base_frame = gray
        continue

    chenh_lech = cv.absdiff(base_frame, gray)
    nguong = cv.threshold(chenh_lech, 50, 255, cv.THRESH_BINARY)[1]
    nguong = cv.dilate(nguong, None, iterations=2)

    contours, info = cv.findContours(nguong.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv.contourArea(c) <= 800:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("Camera", frame)
    cv.imshow("Threshold", nguong)
    if cv.waitKey(100)== ord('q'):
        break

cam.release()
cv.destroyAllWindows()
