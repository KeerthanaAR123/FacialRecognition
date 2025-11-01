"""
Module 4: Real-Time Face Detection and Recognition
Implements live webcam-based face detection and recognition
Uses trained CNN-based face encodings for identification
Saves screenshots in JPG format
UPDATED: Stricter threshold to prevent false matches
"""

import cv2
import face_recognition
import pickle
import numpy as np
import os
from datetime import datetime


def detect_faces_realtime():
    """
    Perform real-time face recognition using webcam
    Displays detected faces with names and confidence scores
    """
    
    # Load trained model
    model_path = 'models/face_encodings.pkl'
    
    if not os.path.exists(model_path):
        print("Error: Trained model not found! Please run training first.")
        print("Run: python 3_train_model.py")
        return
    
    print("\n" + "="*60)
    print("REAL-TIME FACE DETECTION & RECOGNITION")
    print("="*60)
    
    # Load model data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    known_face_encodings = model_data['encodings']
    known_face_names = model_data['names']
    detection_method = model_data.get('detection_method', 'hog')
    
    print(f" Model loaded successfully")
    print(f"Total face encodings: {len(known_face_encodings)}")
    print(f"Unique persons: {len(set(known_face_names))}")
    print(f"Detection method: {detection_method.upper()}")
    
    # Display trained persons
    print(f"\nTrained persons:")
    for name in set(known_face_names):
        count = known_face_names.count(name)
        print(f"  - {name}: {count} encodings")
    
    print("="*60 + "\n")
    
    print("Starting webcam...")
    print("Press 'Q' to quit")
    print("Press 'S' to save screenshot (JPG format)")
    print("\n  Recognition Threshold: STRICT (0.45)")
    print("   Lower threshold = More accurate, fewer false positives\n")
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # For performance optimization
    process_this_frame = True
    frame_count = 0
    
    # Initialize variables for face tracking
    face_locations = []
    face_names = []
    face_distances = []
    
    while True:
        # Capture frame
        ret, frame = video_capture.read()
        
        if not ret:
            print(" Error: Cannot access webcam")
            break
        
        frame_count += 1
        
        # Process every other frame for better performance
        if process_this_frame:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces in frame
            face_locations = face_recognition.face_locations(rgb_small_frame, model=detection_method)
            
            # Generate face encodings for detected faces
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            face_distances = []
            
            # Compare detected faces with known faces
            for face_encoding in face_encodings:
                # Calculate distances between face and all known faces
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                # Find the best match
                best_match_index = np.argmin(distances)
                min_distance = distances[best_match_index]
                
                # ⭐ STRICTER THRESHOLD FOR ACCURACY ⭐
                # Changed from 0.6 to 0.45 for better accuracy
                # Lower value = more strict = fewer false positives
                RECOGNITION_THRESHOLD = 0.45  # ← MAIN CHANGE HERE
                
                if min_distance < RECOGNITION_THRESHOLD:
                    name = known_face_names[best_match_index]
                    confidence = (1 - min_distance) * 100  # Convert to percentage
                else:
                    name = "Unknown"
                    confidence = 0
                
                face_names.append(name)
                face_distances.append(confidence)
        
        process_this_frame = not process_this_frame
        
        # Display results
        for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_distances):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Choose color based on recognition
            if name != "Unknown":
                box_color = (0, 255, 0)  # Green for known faces
                text_color = (255, 255, 255)
            else:
                box_color = (0, 0, 255)  # Red for unknown faces
                text_color = (255, 255, 255)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            
            # Display name and confidence
            if name != "Unknown":
                label = f"{name} ({confidence:.1f}%)"
            else:
                label = "Unknown"
            
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)
        
        # Display FPS and frame count
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display threshold info
        cv2.putText(frame, "Threshold: STRICT (0.45)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'Q' to quit | 'S' to save screenshot", (10, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Real-Time Face Detection & Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Quit on 'Q' key
        if key == ord('q'):
            break
        
        # Save screenshot on 'S' key (JPG format)
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"✓ Screenshot saved: {screenshot_path}")
    
    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Face detection stopped")
    print(f"Total frames processed: {frame_count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    detect_faces_realtime()
