import cv2
from deepface import DeepFace

#load training data from classifier file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#video capturing star
cap = cv2.VideoCapture(0)

while True:
    #to capture frame-by0frame
    ret, frame = cap.read()

    #frame converting to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #convert that to rgb format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    #for detecting faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        #extract face ROI
        face_roi = rgb_frame[y:y + h, x:x + w]

        #perform emotion anaylsis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        #determine the facial emotion based on traingn data
        emlotion = result[0]['dominant_emotion']

        #draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emlotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #display the resulting frame 
    cv2.imshow('Real-time Emotion Detection', frame)

    #press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the capture and close all windows 
cap.release()
cv2.destroyAllWindows()
