import argparse
import cv2
import imutils
from imutils.video import VideoStream
import pyvirtualcam
import numpy as np

def argumentParse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--alarm", type=str, default="", help="path alarm .WAV file")
    ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
    return vars(ap.parse_args())

with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
    print(f'Using virtual camera: {cam.device}')
    # frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB

    args = argumentParse()
    vs = VideoStream(src=args["webcam"]).start()
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    while True:
        frame = vs.read()
        # frame[:] = cam.frames_sent % 255 * 255  # grayscale animation
        cv2.putText(frame, "aaXXXX", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cv2.COLOR_BGR2RGB, 2)
        cam.send(frame)
        # cam.sleep_until_next_frame()
        cv2.imshow("Drowsiness Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()