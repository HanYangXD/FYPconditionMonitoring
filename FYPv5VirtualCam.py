from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime;
import time;
import pyvirtualcam;

def alertUser(alarmName):
    playsound.playsound(alarmName)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    earValue = (A+B)/(2.0*C)
    return earValue

def mouth_aspect_ratio(mouth):
    D = dist.euclidean(mouth[13], mouth[19])
    E = dist.euclidean(mouth[14], mouth[18])
    F = dist.euclidean(mouth[15], mouth[17])
    G = dist.euclidean(mouth[13], mouth[16])
    marValue = (D+E+F)/(2.0*G)
    return marValue

def getCurrentTimee():
    return datetime.datetime.now()

def set_EAR_threshold(leftEye, rightEye):
    leftEyeAspectRatio = eye_aspect_ratio(leftEye)
    rightEyeAspectRatio = eye_aspect_ratio(rightEye)
    return (((leftEyeAspectRatio + rightEyeAspectRatio) / 2.0) * (80 / 100))

def sett(eye):
    singletresh = eye_aspect_ratio(eye)
    return singletresh

def argumentParse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--alarm", type=str, default="", help="path alarm .WAV file")
    ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
    return vars(ap.parse_args())

def set_Mouth_Treshold():
    return 1.0

def initDetector():
    return dlib.get_frontal_face_detector()
    
def initPredictor():
    return dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def setLandmarks(part):
    return face_utils.FACIAL_LANDMARKS_IDXS[part]

def launchVideoStream():
    vs = VideoStream(src=args["webcam"]).start()
    return vs

def getUserName():
    return input("Enter your name:")

def initGsheet(sheetName):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
    client = gspread.authorize(creds)
    return client.open(sheetName).sheet1

def readResizeVS():
    frame = vs.read()
    return imutils.resize(frame, width=450)

def grayScale(color):
    
    return detector(color, 0)

def initShape():
    shape = predictor(gray, rect)
    return face_utils.shape_to_np(shape)

def assignShape(start, end):
    return shape[start:end]

def calibrateEAR(leftEye, rightEye):
    leftEyeAspectRatio = eye_aspect_ratio(leftEye)
    rightEyeAspectRatio = eye_aspect_ratio(rightEye)
    EARthreshold = ((leftEyeAspectRatio + rightEyeAspectRatio) / 2.0) * (80 / 100)
    return EARthreshold , leftEyeAspectRatio, rightEyeAspectRatio

def calculateCurrentEAR(leftEAR, rightEAR):
    return (leftEAR+rightEAR)/2.0

def drawHull(part):
    cv2.convexHull(part)
    cv2.drawContours(frame, [part], -1, (0, 255, 0), 1)

def displayText(frame, texts, variables, position1, position, font):
    cv2.putText(frame, texts.format(variables), (position1, position), font, 0.7, (0, 0, 255), 2)

def insertData():
    row = [timestamp, ear, mar, userName] 
    index = 1
    sheet.insert_row(row, index)
    return now
def displayStats():
    displayText(frame, "EAR: {:.2f}", ear, 10, 30, cv2.FONT_HERSHEY_SIMPLEX)
    displayText(frame, "MAR: {:.2f}", mar, 10, 60, cv2.FONT_HERSHEY_SIMPLEX)
    displayText(frame, "detected left eye: {:.2f}", leftEyeAspectRatio, 10, 180, cv2.FONT_HERSHEY_SIMPLEX)
    displayText(frame, "detected left eye: {:.2f}", rightEyeAspectRatio, 10, 210, cv2.FONT_HERSHEY_SIMPLEX)        
    displayText(frame, "EAR threshold: {:.2f}", EARthreshold, 10, 120, cv2.FONT_HERSHEY_SIMPLEX)     
    
EYE_AR_CONSEC_FRAMES = 48
MOUTH_AR_CONSEC_FRAMES = 48
MAR_THRESHOLD = set_Mouth_Treshold()
COUNTER = 0

args = argumentParse()

detector = initDetector()
predictor = initPredictor()

(lStart, lEnd) = setLandmarks("left_eye")
(rStart, rEnd) = setLandmarks("right_eye")
(mStart, mEnd) = setLandmarks("mouth")

vs = launchVideoStream()

lastUpdateTime = getCurrentTimee()
lastAlertTime = getCurrentTimee()
sheet = initGsheet("FYPconditionMonitoring")    
EARcalibrated = False

# userName = getUserName()
userName = "haha"
with pyvirtualcam.Camera(width=640,height=480,fps=30) as cam:
    print(f'Using virtual camera: {cam.device}')
    frames = np.zeros((cam.height, cam.width, 3), np.uint8)
    while True:
        
        # cam.sleep_until_next_frame()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            EARcalibrated = False

        frame = readResizeVS()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = grayScale(gray)

        now = getCurrentTimee()
        
        for rect in rects:
            shape = initShape()
            leftEye = assignShape(lStart, lEnd)
            rightEye = assignShape(rStart, rEnd)

            if not EARcalibrated:
                EARthreshold, leftEyeAspectRatio, rightEyeAspectRatio = calibrateEAR(leftEye, rightEye)
                EARcalibrated = True

            mouth = assignShape(mStart, mEnd)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mar = mouth_aspect_ratio(mouth)
            ear = calculateCurrentEAR(leftEAR, rightEAR)
            

            drawHull(leftEye)
            drawHull(rightEye)
            drawHull(mouth)

            displayText(frame, "Consec frame: {:.2f}", COUNTER, 10, 90, cv2.FONT_HERSHEY_SIMPLEX)
            if ear < EARthreshold or mar > MAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES or COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    if now >= lastAlertTime + datetime.timedelta(seconds=5):
                        alertUser("alert.mp3")
                        lastAlertTime = now
                    
                    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
                    cv2.putText(frame, "DROWSINESS ALERT!", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                    if now >= lastUpdateTime + datetime.timedelta(seconds=20):
                        
                        lastUpdateTime = insertData()

            else:
                COUNTER = 0

            displayStats()   

        cv2.putText(frame, "Current Time:" + now.strftime("%H:%M:%S"), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detector", frame)
        frames[:] = cam.frames_sent  # grayscale animation
        cam.send(vs.frame)
    cv2.destroyAllWindows()
    vs.stop()

