from functions import *

#using virtual camera
with pyvirtualcam.Camera(width=640,height=480,fps=30,fmt=PixelFormat.RGB) as cam:
    frames = np.zeros((cam.height, cam.width, 3), np.uint8)
    while True:

        # shortcut keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): # "q" to Quit the program
            break
        if key == ord("r"): # "r" to recalibrate EAR (Eye Aspect Ratio)
            EARcalibrated = False

        #accessing camera and grayscale it for a more accurate face detector
        frame = readResizeVS()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = grayScale(gray)

        now = getCurrentTimee()
        
        for rect in rects:
            #getting the landmark of Eye and Mouth
            shape = initShape(gray, rect)
            leftEye = assignShape(shape, lStart, lEnd)
            rightEye = assignShape(shape, rStart, rEnd)
            mouth = assignShape(shape, mStart, mEnd)        

            #calibrate EAR if it is not calibrated
            if not EARcalibrated:
                EARthreshold, leftEyeAspectRatio, rightEyeAspectRatio = calibrateEAR(leftEye, rightEye)
                EARcalibrated = True

            #getting EAR value for each eye
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            #calculating EAR and MAR value
            ear = calculateCurrentEAR(leftEAR, rightEAR)
            mar = calculateCurrentMAR(mouth)
            # drawHull(frame, leftEye)
            # drawHull(frame, rightEye)
            # drawHull(frame, mouth)

            #if EAR is higher than threshold, or MAR is higher than threshold
            if ear < EARthreshold or mar > MAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES or COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    if now >= lastAlertTime + datetime.timedelta(seconds=5):
                        alertUser("alert.mp3")
                        lastAlertTime = now
                        drowsyCounter += 1
                    
                    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
                    cv2.putText(frame, "TIREDNESS ALERT!", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 4)
                
                    if now >= lastUpdateTime + datetime.timedelta(seconds=20):
                        
                        lastUpdateTime = insertData(timestamp, ear, mar)
                        
            else:
                COUNTER = 0

            displayStats(frame, ear, mar, leftEyeAspectRatio, rightEyeAspectRatio, EARthreshold)   

        # cv2.putText(frame, "Current Time:" + now.strftime("%H:%M:%S"), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "(q - quit) | (r - recalibrate EAR)", (60, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.putText(frame, "Tiredness counter: {:.2f}".format(drowsyCounter), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        
        cv2.imshow("Tiredness Detector", frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cam.send(frame)
    cv2.destroyAllWindows()
    vs.stop()

