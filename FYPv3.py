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

def getCurrentTime():
    now = datetime.datetime.now()
    currentTime = now.strftime("%H:%M:%S")

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



EYE_AR_CONSEC_FRAMES = 48
MOUTH_AR_CONSEC_FRAMES = 48



EYE_AR_THRESH = 0.3 #insert default value (will be calibrated once the system launch) 
MOUTH_AR_THRESH = set_Mouth_Treshold()

args = argumentParse()



leftEyeAspectRatio = 0.0
rightEyeAspectRatio = 0.0

COUNTER = 0
GSHEETCOUNTER = 0
insertCounter = 0
ALARM_ON = False



detector = initDetector()
predictor = initPredictor()




(lStart, lEnd) = setLandmarks("left_eye")
(rStart, rEnd) = setLandmarks("right_eye")
(mStart, mEnd) = setLandmarks("mouth")



vs = launchVideoStream()
time.sleep(1.0)
getCurrentTime()

lastUpdateTime = getCurrentTimee()
userName = getUserName()




    
sheet = initGsheet("FYPconditionMonitoring")    


EARcalibrated = False


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    NOW = getCurrentTimee()
    
    


    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        if not EARcalibrated:
            EYE_AR_THRESHOLD = set_EAR_threshold(leftEye,rightEye)
            leftEyeAspectRatio = sett(leftEye)
            rightEyeAspectRatio = sett(rightEye)
            EYE_AR_THRESHOLD = ((leftEyeAspectRatio + rightEyeAspectRatio) / 2.0) * (80 / 100)
            # print("jehe")
            EARcalibrated = True

        

        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mar = mouth_aspect_ratio(mouth)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
        cv2.putText(frame, "mar frame: {:.2f}".format(COUNTER), (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES or COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    alertUser("alert.mp3")
                timestamp = datetime.datetime.now()
                dt_string = timestamp.strftime("%d/%m/%Y %H:%M:%S")
                
                cv2.putText(frame, "DROWSINESS ALERT!", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # if GSHEETCOUNTER > 300 or insertCounter == 0:
                #     row = [dt_string,"EAR:",ear,"MAR:",mar,"Student Name:", studentName]
                #     index = 1
                #     sheet.insert_row(row, index)
                #     GSHEETCOUNTER = 0
                #     insertCounter += 1

                if NOW >= lastUpdateTime + datetime.timedelta(seconds=20):
                    row = [dt_string, ear, mar, userName] 
                    index = 1
                    sheet.insert_row(row, index)
                    lastUpdateTime = NOW

        else:
            COUNTER = 0
            ALARM_ON = False
        cv2.putText(frame, "v3: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "detected left eye: {:.2f}".format(leftEyeAspectRatio), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "detected right eye: {:.2f}".format(rightEyeAspectRatio), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR threshold: {:.2f}".format(EYE_AR_THRESHOLD), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # cv2.putText(frame, "gsheet frame: {:.2f}".format(GSHEETCOUNTER), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Current Time:" + NOW.strftime("%H:%M:%S"), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Drowsiness Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        EARcalibrated = False

cv2.destroyAllWindows()
vs.stop()