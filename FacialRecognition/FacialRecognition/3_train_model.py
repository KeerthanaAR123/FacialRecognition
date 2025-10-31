"""
Complete CNN Training with Image Format Fix
Automatically converts problematic images to 8-bit RGB
"""

import os
import cv2
import numpy as np
import face_recognition
import pickle
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelEncoder


def fix_and_load_image(path):
    """
    Load image and ensure it's proper 8-bit RGB for face_recognition
    """
    try:
        # Open with PIL
        with Image.open(path) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            
            # Convert to numpy uint8 array
            arr = np.array(im, dtype=np.uint8)
            
            # Ensure contiguous memory
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            
            # Verify shape
            if len(arr.shape) != 3 or arr.shape[2] != 3:
                raise ValueError(f"Invalid shape: {arr.shape}")
            
            return arr
    
    except Exception as e:
        print(f"    Error loading: {e}")
        return None


def train_model_cnn():
    """Train CNN-based face recognition model"""
    
    dataset = "dataset"
    
    if not os.path.exists(dataset):
        print("❌ Error: 'dataset' folder not found!")
        return False
    
    print("\n" + "="*60)
    print("CNN FACE RECOGNITION - TRAINING")
    print("="*60)
    print(f"Dataset: {dataset}/")
    print("Method: CNN (Deep Learning)")
    print("Image handling: Auto-fix to 8-bit RGB")
    print("="*60 + "\n")
    
    encodings = []
    names = []
    total = 0
    success = 0
    failed = 0
    
    # Process each person folder
    for person in os.listdir(dataset):
        person_dir = os.path.join(dataset, person)
        
        if not os.path.isdir(person_dir):
            continue
        
        print(f"Training: {person}")
        
        # Get all image files
        files = [f for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if not files:
            print(f"  ⚠ No images found\n")
            continue
        
        # Process each image
        for img_file in files:
            total += 1
            img_path = os.path.join(person_dir, img_file)
            
            try:
                # Load and fix image
                img_rgb = fix_and_load_image(img_path)
                
                if img_rgb is None:
                    print(f"  ✗ Cannot load: {img_file}")
                    failed += 1
                    continue
                
                # Detect faces with CNN
                locations = face_recognition.face_locations(img_rgb, model='cnn')
                
                if len(locations) == 0:
                    print(f"  ✗ No face: {img_file}")
                    failed += 1
                    continue
                
                # Generate face encodings
                encs = face_recognition.face_encodings(img_rgb, known_face_locations=locations)
                
                if len(encs) == 0:
                    print(f"  ✗ No encoding: {img_file}")
                    failed += 1
                    continue
                
                # Take first encoding
                encodings.append(encs[0])
                names.append(person)
                success += 1
                print(f"  ✓ {img_file}")
                
            except Exception as e:
                print(f"  ✗ Error in {img_file}: {e}")
                failed += 1
        
        print(f"  Total: {len(files)}\n")
    
    # Check if we got any encodings
    if len(encodings) == 0:
        print("❌ No encodings generated!")
        return False
    
    # Convert to numpy array
    encodings = np.array(encodings)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(names)
    
    # Create model dictionary
    model = {
        'encodings': encodings,
        'names': names,
        'label_encoder': label_encoder,
        'detection_method': 'cnn'
    }
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/face_encodings.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Total images: {total}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Unique persons: {len(set(names))}")
    print(f"Model saved: {model_path}")
    print(f"Total encodings: {len(encodings)}")
    print("="*60 + "\n")
    
    # Show statistics per person
    print("PERSON STATISTICS:")
    print("-" * 60)
    for name in set(names):
        count = names.count(name)
        print(f"  {name}: {count} face encodings")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FACIAL RECOGNITION - CNN TRAINING")
    print("="*60)
    print("\nStarting training from dataset/ folder...")
    
    success = train_model_cnn()
    
    if success:
        print("\n✓ Training complete! You can now run detection.")
        print("Next step: Run 4_detect_faces.py to test recognition")
    else:
        print("\n✗ Training failed. Please check errors above.")
