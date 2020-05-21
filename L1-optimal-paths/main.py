import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frames
    cv.imshow('Color', frame)
    cv.imshow('Grayscale', gray)
    cv.moveWindow('Color', 0, 0)
    cv.moveWindow('Grayscale', 700, 0)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
