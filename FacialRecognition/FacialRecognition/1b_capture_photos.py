"""
Module 1B: Capture Photos via Webcam
Capture multiple images of a person using webcam for training
Real-time face detection with proper preprocessing
Saves all images in JPG format
"""

import cv2
import os
import time


def capture_training_photos(person_name, num_images=50):
    """
    Capture training images for a person using webcam
    
    Args:
        person_name (str): Name of the person
        num_images (int): Number of images to capture (default: 50)
    """
    
    # Create directory for person if it doesn't exist
    person_dir = os.path.join('dataset', person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load Haar Cascade for face detection (preprocessing)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print(f"\n{'='*60}")
    print(f"CAPTURE PHOTOS VIA WEBCAM - Training Images for: {person_name}")
    print(f"{'='*60}")
    print(f"Total images to capture: {num_images}")
    print(f"Storage location: {person_dir}")
    print(f"Image format: JPG")
    print(f"\nInstructions:")
    print("  - Look at the camera")
    print("  - Move your face slightly between captures")
    print("  - Press 'SPACE' to capture image")
    print("  - Press 'Q' to quit early")
    print(f"{'='*60}\n")
    
    count = 0
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot access webcam")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display progress
        progress_text = f"Images Captured: {count}/{num_images}"
        cv2.putText(frame, progress_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press SPACE to capture | Q to quit", (10, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(f'Capturing Photos - {person_name}', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capture image on SPACE key
        if key == ord(' ') and len(faces) > 0:
            # Save the captured frame as JPG
            image_path = os.path.join(person_dir, f'{person_name}_{count+1}.jpg')
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
            print(f"✓ Image {count}/{num_images} saved: {image_path}")
            time.sleep(0.2)  # Small delay to avoid duplicate captures
        
        # Quit on 'Q' key
        elif key == ord('q'):
            print("\nCapture stopped by user")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Photo capture completed!")
    print(f"Total images saved: {count}")
    print(f"Location: {person_dir}")
    print(f"Format: JPG")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FACIAL RECOGNITION SYSTEM - CAPTURE PHOTOS")
    print("="*60)
    
    # Input person name
    person_name = input("\nEnter the person's name: ").strip()
    
    if not person_name:
        print("Error: Name cannot be empty")
        exit()
    
    # Input number of images (optional)
    try:
        num_images = input("Enter number of images to capture (default 50): ").strip()
        num_images = int(num_images) if num_images else 50
    except ValueError:
        num_images = 50
        print("Invalid input. Using default: 50 images")
    
    # Start capture
    capture_training_photos(person_name, num_images)
