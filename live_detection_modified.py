# using alternative emotion detection method instaed of standard  model:
# ater debugging the first model:
# for live detection only:
import cv2
import numpy as np
import time
from datetime import datetime

def main():
    # Load pre-trained models for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load pre-trained models for age and gender detection
    # You'll need to download these files separately
    age_net = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
    gender_net = cv2.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')
    
    # For emotion detection we'll use a pre-trained face landmarks detector from dlib
    # and estimate emotions based on facial geometry
    try:
        import dlib
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        has_dlib = True
        print("Dlib loaded successfully")
    except ImportError:
        has_dlib = False
        print("Dlib not available - emotion detection will be simulated")
    
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
    
    frame_count = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Process each face
        for (x, y, w_face, h_face) in faces:
            # Extract face ROI
            face = frame[y:y+h_face, x:x+w_face]
            
            # Skip if face ROI is empty
            if face.size == 0:
                continue
            
            # Prepare face for age and gender prediction
            blob = cv2.dnn.blobFromImage(
                cv2.resize(face, (227, 227)),
                1.0,
                (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            
            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_classes[age_preds[0].argmax()]
            
            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_classes[gender_preds[0].argmax()]
            
            # Emotion detection (using facial landmarks or simulation)
            if has_dlib:
                # Use dlib for facial landmarks to detect emotion
                rect = dlib.rectangle(x, y, x + w_face, y + h_face)
                landmarks = predictor(gray, rect)
                emotion, confidence = analyze_emotion_from_landmarks(landmarks)
                emotion_index = emotion_classes.index(emotion)
                emotion_confidence = confidence
            else:
                # Simulate emotion detection (rotate through emotions for demo purposes)
                frame_count += 1
                emotion_index = (frame_count // 30) % len(emotion_classes)
                emotion = emotion_classes[emotion_index]
                emotion_confidence = 0.7 + (np.random.random() * 0.3)  # Random confidence between 0.7 and 1.0
            
            # Create one-hot encoded emotion vector for EI calculation
            emotion_preds = np.zeros(len(emotion_classes))
            emotion_preds[emotion_index] = emotion_confidence
            
            # Calculate emotional intelligence score
            emotional_intelligence = calculate_ei_score(emotion_preds, time.time())
            
            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 255, 0), 2)
            
            # Display predictions
            label = f"Age: {age}, Gender: {gender}"
            emotion_label = f"Emotion: {emotion}, EI: {emotional_intelligence:.1f}"
            
            # Position text above face rectangle
            y_pos = y - 10 if y > 20 else y + 10
            cv2.putText(frame, label, (x, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y_pos + 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Facial Recognition', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def analyze_emotion_from_landmarks(landmarks):
    """
    Analyze facial landmarks to determine emotion
    This is a simplified approach - real emotion detection would use ML models
    """
    # Get key facial measurements
    # These are approximations - a real system would be more sophisticated
    
    # Extract mouth width and height ratio (smile detection)
    mouth_width = abs(landmarks.part(54).x - landmarks.part(48).x)
    mouth_height = abs((landmarks.part(51).y + landmarks.part(57).y)/2 - 
                       (landmarks.part(62).y + landmarks.part(66).y)/2)
    
    smile_ratio = mouth_width / max(mouth_height, 1)
    
    # Get eyebrow position (surprise, anger)
    left_eyebrow_height = landmarks.part(24).y - landmarks.part(27).y
    right_eyebrow_height = landmarks.part(19).y - landmarks.part(27).y
    eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
    
    # Mouth openness (surprise, fear)
    mouth_open = abs(landmarks.part(62).y - landmarks.part(66).y)
    
    # Simple decision tree for emotions
    if smile_ratio > 4:
        return "Happy", 0.85
    elif eyebrow_height > 15:
        return "Surprised", 0.75
    elif eyebrow_height < -5:
        return "Angry", 0.7
    elif mouth_open > 15:
        return "Fear", 0.6
    elif smile_ratio < 2:
        return "Sad", 0.65
    else:
        return "Neutral", 0.9

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
