import cv2
import numpy as np


# File locations, etc. for input data
def main():
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('0.avi')

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video  file")

    # Read until video is completed
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
