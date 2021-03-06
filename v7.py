from functions import *
import sys

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
        if key == ord("s"):
            showHull = toggle()
        
        faceDetected = False
        #accessing camera and grayscale it for a more accurate face detector
        frame = readResizeVS()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = grayScale(gray)
        now = getCurrentTime()
        for rect in rects:
            faceDetected = True
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
            
            if showHull:
                drawHull(frame, leftEye)
                drawHull(frame, rightEye)
                drawHull(frame, mouth)

            #if EAR is higher than threshold, or MAR is higher than threshold
            if ear < EARthreshold or mar > MAR_THRESHOLD:
                startTimer = True
                if startTimer and ensureOnce:
                    drowsyTimer = now
                    ensureOnce = False

                if now >= drowsyTimer + datetime.timedelta(seconds=3):
                    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
                    cv2.putText(frame, "TIREDNESS ALERT!", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 4)
                    if now >= lastAlertTime + datetime.timedelta(seconds=5):
                        alertUser("alert.mp3")
                        lastAlertTime = now
                        tiredCounter += 1
                
                    if now >= lastUpdateTime + datetime.timedelta(seconds=20):
                        lastUpdateTime = insertData(timestamp, ear, mar)
                        
            else:
                startTimer = False
                ensureOnce = True

            displayStats(frame, ear, mar, leftEyeAspectRatio, rightEyeAspectRatio, EARthreshold)   

        cv2.putText(frame, "(q - quit) | (r - recalibrate EAR)", (60, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.putText(frame, "Tiredness counter: {:.2f}".format(tiredCounter), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        del rects

        if not faceDetected:		
            cv2.putText(frame, "No face detected!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("Condition Monitoring System", frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cam.send(frame)
    cv2.destroyAllWindows()
    vs.stop()

