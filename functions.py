from pyvirtualcam.camera import PixelFormat
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
import datetime
import time
import pyvirtualcam
import itertools

def alertUser(alarmName):
    playsound.playsound(alarmName)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    earValue = (A+B)/(2.0*C)
    return earValue

def calculateCurrentMAR(mouth):
    D = dist.euclidean(mouth[13], mouth[19])
    E = dist.euclidean(mouth[14], mouth[18])
    F = dist.euclidean(mouth[15], mouth[17])
    G = dist.euclidean(mouth[12], mouth[16])
    marValue = (D+E+F)/(2.0*G)
    return marValue

def getCurrentTime():
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
    name = input("\nEnter your name: ")
    
    while validateName(name):
        name = input("\nInvalid Name! Please enter your name: ")
        validateName(name)
    print("Welcome "+ name + ", launching Condition Monitoring System...")
    return name

def validateName(name):
    contain_digit = any(map(str.isdigit, name))
    input_length = len(name)
    is_name = any(map(str.isalpha, name))
    if contain_digit or not is_name or input_length==0:
        return True
    else:
        return False


def initGsheet(sheetName):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
    client = gspread.authorize(creds)
    return client.open(sheetName).sheet1

def readResizeVS():
    frame = vs.read()
    return imutils.resize(frame, width=640)

def grayScale(color):
    
    return detector(color, 0)

def initShape(gray,rect):
    shape = predictor(gray, rect)
    return face_utils.shape_to_np(shape)

def assignShape(shape, start, end):
    return shape[start:end]

def calibrateEAR(leftEye, rightEye):
    leftEyeAspectRatio = eye_aspect_ratio(leftEye)
    rightEyeAspectRatio = eye_aspect_ratio(rightEye)
    EARthreshold = ((leftEyeAspectRatio + rightEyeAspectRatio) / 2.0) * (80 / 100)
    return EARthreshold, leftEyeAspectRatio, rightEyeAspectRatio

def calculateCurrentEAR(leftEAR, rightEAR):
    return (leftEAR+rightEAR)/2.0

def drawHull(frame, part):
    cv2.convexHull(part)
    cv2.drawContours(frame, [part], -1, (255, 255, 255), 1)

def displayText(frame, texts, variables, position1, position, font):
    cv2.putText(frame, texts.format(variables), (position1, position), font, 1, (0, 0, 255), 2)

def insertData(timestamp, ear, mar):
    row = [timestamp, ear, mar, userName] 
    index = 1
    sheet.insert_row(row, index)
    return getCurrentTime()

def displayStats(frame, ear, mar, leftEyeAspectRatio, rightEyeAspectRatio, EARthreshold):
    displayText(frame, "EAR: {:.2f}", ear, 30, 160, cv2.FONT_HERSHEY_SIMPLEX) #usethis
    displayText(frame, "MAR: {:.2f}", mar, 30, 190, cv2.FONT_HERSHEY_SIMPLEX)
    # displayText(frame, "detected left eye: {:.2f}", leftEyeAspectRatio, 10, 180, cv2.FONT_HERSHEY_SIMPLEX)
    # displayText(frame, "detected right eye: {:.2f}", rightEyeAspectRatio, 10, 210, cv2.FONT_HERSHEY_SIMPLEX)        
    # displayText(frame, "EAR threshold: {:.2f}", EARthreshold, 10, 120, cv2.FONT_HERSHEY_SIMPLEX)     


####~~~~####  ####~~~~####  ####~~~~####
####~~~~#### Initialisation ####~~~~####
####~~~~####  ####~~~~####  ####~~~~####
print("Welcome to Condition Monitoring System")
userName = getUserName()

EYE_AR_CONSEC_FRAMES = 48
MOUTH_AR_CONSEC_FRAMES = 48
MAR_THRESHOLD = set_Mouth_Treshold()
COUNTER = 0
tiredCounter = 0

args = argumentParse()

detector = initDetector()
predictor = initPredictor()

(lStart, lEnd) = setLandmarks("left_eye")
(rStart, rEnd) = setLandmarks("right_eye")
(mStart, mEnd) = setLandmarks("mouth")

vs = launchVideoStream()

lastUpdateTime = getCurrentTime()
lastAlertTime = getCurrentTime()
sheet = initGsheet("FYPconditionMonitoring")    
EARcalibrated = False

toggle = itertools.cycle([True, False]).__next__
startTimer = False
ensureOnce = True
showHull = False
displayTextOnScreen = True
drowsyTimer = getCurrentTime()


