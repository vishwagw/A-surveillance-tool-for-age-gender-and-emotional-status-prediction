# detector model:
import cv2
import numpy as np
import time
from datetime import datetime

def main():
    # Load pre-trained models
    # Face detection model
    face_net = cv2.dnn.readNetFromCaffe(
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel'
    )
    
    # Age detection model
    age_net = cv2.dnn.readNetFromCaffe(
        'age_deploy.prototxt',
        'age_net.caffemodel'
    )
    
    # Gender detection model
    gender_net = cv2.dnn.readNetFromCaffe(
        'gender_deploy.prototxt',
        'gender_net.caffemodel'
    )
    
    # Emotion detection model
    emotion_net = cv2.dnn.readNetFromCaffe(
        'emotion_deploy.prototxt',
        'emotion_net.caffemodel'
    )
    
    # Define classes
    age_classes = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
    gender_classes = ['Male', 'Female']
    emotion_classes = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry', 'Disgust', 'Fear']
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Facial recognition system started. Press 'q' to quit.")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create a 4D blob from frame
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Detect faces
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # Process each detection
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > 0.5:
                # Get face box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the face box is within frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Extract face ROI
                face = frame[startY:endY, startX:endX]
                
                # Skip if face ROI is empty
                if face.size == 0:
                    continue
                
                # Prepare face for age, gender, and emotion prediction
                face_blob = cv2.dnn.blobFromImage(
                    cv2.resize(face, (227, 227)),
                    1.0,
                    (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                
                # Predict age
                age_net.setInput(face_blob)
                age_preds = age_net.forward()
                age = age_classes[age_preds[0].argmax()]
                
                # Predict gender
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                gender = gender_classes[gender_preds[0].argmax()]
                
                # Predict emotion
                emotion_net.setInput(face_blob)
                emotion_preds = emotion_net.forward()
                emotion = emotion_classes[emotion_preds[0].argmax()]
                
                # Calculate emotional intelligence score (simplified approach)
                emotional_intelligence = calculate_ei_score(emotion_preds, time.time())
                
                # Draw face box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
                # Display predictions
                label = f"Age: {age}, Gender: {gender}"
                emotion_label = f"Emotion: {emotion}, EI: {emotional_intelligence:.1f}"
                
                # Position text above face rectangle
                y_pos = startY - 10 if startY > 20 else startY + 10
                cv2.putText(frame, label, (startX, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (startX, y_pos + 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Facial Recognition', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Global variables for EI calculation
emotion_history = []
last_emotion_change = time.time()

def calculate_ei_score(emotion_preds, current_time):
    """
    Calculate a simplified emotional intelligence score based on:
    1. Emotion recognition confidence
    2. Emotional stability (consistency over time)
    3. Emotional diversity (range of emotions detected)
    """
    global emotion_history, last_emotion_change
    
    # Get the confidence of the top emotion
    top_confidence = np.max(emotion_preds)
    
    # Get current dominant emotion
    current_emotion = np.argmax(emotion_preds)
    
    # Add current emotion to history (keep last 50 entries)
    emotion_history.append((current_time, current_emotion, top_confidence))
    if len(emotion_history) > 50:
        emotion_history.pop(0)
    
    # Check if emotion changed
    if len(emotion_history) > 1 and emotion_history[-1][1] != emotion_history[-2][1]:
        last_emotion_change = current_time
    
    # Calculate emotional stability (time since last change)
    stability = min(10, (current_time - last_emotion_change))
    
    # Calculate emotional diversity (unique emotions in history)
    if len(emotion_history) > 10:
        unique_emotions = len(set([e[1] for e in emotion_history[-10:]]))
        diversity = unique_emotions / 3  # Normalize
    else:
        diversity = 1.0  # Default
    
    # Calculate final score (scale 1-10)
    ei_score = (top_confidence * 5) + (stability) + (diversity * 2)
    ei_score = min(10, max(1, ei_score))
    
    return ei_score

if __name__ == "__main__":
    main()