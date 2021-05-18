from functions import *

with pyvirtualcam.Camera(width=640,height=480,fps=30,fmt=PixelFormat.RGB) as cam:
    print(f'Using virtual camera: {cam.device}')
    frames = np.zeros((cam.height, cam.width, 3), np.uint8)
    while True:
        
        # shortcut keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): # "q" to Quit the program
            break
        if key == ord("r"): # "r" to recalibrate EAR (Eye Aspect Ratio)
            EARcalibrated = False

        frame = readResizeVS()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = grayScale(gray)


        now = getCurrentTimee()
        
        for rect in rects:
            shape = initShape(gray, rect)
            leftEye = assignShape(shape, lStart, lEnd)
            rightEye = assignShape(shape, rStart, rEnd)

            if not EARcalibrated:
                EARthreshold, leftEyeAspectRatio, rightEyeAspectRatio = calibrateEAR(leftEye, rightEye)
                EARcalibrated = True

            mouth = assignShape(shape, mStart, mEnd)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mar = calculateCurrentMAR(mouth)
            ear = calculateCurrentEAR(leftEAR, rightEAR)
            
            drawHull(frame, leftEye)
            drawHull(frame, rightEye)
            drawHull(frame, mouth)

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
                        drowsyCounter += 1
                        lastUpdateTime = insertData(timestamp, ear, mar)
                        cv2.putText(frame, "Total Drowsiness: " + drowsyCounter, (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                COUNTER = 0

            displayStats(frame, ear, mar, leftEyeAspectRatio, rightEyeAspectRatio, EARthreshold)   

        cv2.putText(frame, "Current Time:" + now.strftime("%H:%M:%S"), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "q - quit | r - recalibrate EAR", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detector", frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cam.send(frame)
    cv2.destroyAllWindows()
    vs.stop()

