"""
Complete CNN Training - Converts all images to RGB only
Works with JPG/JPEG and other formats from dataset/ folder
Standardized training for facial recognition
"""

import os
import cv2
import numpy as np
import face_recognition
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def load_as_rgb_uint8(path):
    """
    Load any image and force it to 8-bit RGB (3 channels)
    Handles all color modes and bit depths
    """
    # Method 1: Try PIL first (most reliable for format conversion)
    try:
        with Image.open(path) as im:
            # Convert to RGB regardless of original mode
            # This handles: L, LA, P, PA, RGBA, CMYK, LAB, HSV, I, F, etc.
            rgb_im = im.convert("RGB")
            
            # Convert to numpy array as uint8
            arr = np.array(rgb_im, dtype=np.uint8)
            
            # Verify it's 3-channel RGB
            if len(arr.shape) != 3 or arr.shape[2] != 3:
                raise ValueError(f"Unexpected shape: {arr.shape}")
            
            # Ensure contiguous memory for dlib
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            
            return arr
    
    except Exception as e:
        print(f"    PIL failed: {e}")
    
    # Method 2: Try OpenCV with enhanced conversion
    try:
        # Read image with all channels
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return None
        
        # Check number of dimensions
        if len(img.shape) == 2:
            # Grayscale - convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        elif len(img.shape) == 3:
            channels = img.shape[2]
            
            if channels == 1:
                # Single channel - convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            elif channels == 3:
                # BGR - convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            elif channels == 4:
                # BGRA - convert to RGB (remove alpha)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            else:
                raise ValueError(f"Unsupported channel count: {channels}")
        
        else:
            raise ValueError(f"Unexpected image dimensions: {img.shape}")
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            # Normalize to 0-255 range
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
        
        # Ensure contiguous memory
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        
        # Final verification
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Final image has wrong shape: {img.shape}")
        
        return img
    
    except Exception as e:
        print(f"    OpenCV failed: {e}")
        return None


def verify_image_format(path):
    """
    Verify and print image information for debugging
    """
    try:
        with Image.open(path) as im:
            print(f"    Format: {im.format}, Mode: {im.mode}, Size: {im.size}")
            return True
    except Exception as e:
        print(f"    Cannot open with PIL: {e}")
        return False


def train_model_cnn():
    """Train CNN-based face recognition model"""
    
    dataset = "dataset"
    
    if not os.path.exists(dataset):
        print("❌ Error: 'dataset' folder not found!")
        print("Please create dataset/PersonName/ folders with photos")
        return False
    
    print("\n" + "="*60)
    print("CNN FACE RECOGNITION - TRAINING")
    print("="*60)
    print(f"Dataset: {dataset}/")
    print("Method: CNN (Deep Learning)")
    print("Conversion: All images → RGB (3 channels)")
    print("Supported formats: JPG, JPEG, PNG, BMP, TIFF")
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
        
        # Get all image files (both .jpg and .jpeg supported)
        files = [f for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if not files:
            print(f"  ⚠ No images found\n")
            continue
        
        # Process each image
        for img_file in files:
            total += 1
            img_path = os.path.join(person_dir, img_file)
            
            print(f"  Processing: {img_file}")
            
            # Verify image first
            verify_image_format(img_path)
            
            try:
                # Load and convert to RGB
                img_rgb = load_as_rgb_uint8(img_path)
                
                if img_rgb is None:
                    print(f"  ✗ Cannot read: {img_file}")
                    failed += 1
                    continue
                
                print(f"    Loaded: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
                
                # Detect faces with CNN
                print(f"    Detecting faces...")
                locations = face_recognition.face_locations(img_rgb, model='cnn')
                
                if len(locations) == 0:
                    print(f"  ✗ No face detected: {img_file}")
                    failed += 1
                    continue
                
                print(f"    Found {len(locations)} face(s)")
                
                # Generate face encodings
                print(f"    Generating encodings...")
                encs = face_recognition.face_encodings(img_rgb, locations)
                
                if len(encs) == 0:
                    print(f"  ✗ No encoding generated: {img_file}")
                    failed += 1
                    continue
                
                # Take first encoding
                encodings.append(encs[0])
                names.append(person)
                success += 1
                print(f"  ✓ Success: {img_file}\n")
                
            except Exception as e:
                print(f"  ✗ Error in {img_file}: {e}\n")
                failed += 1
        
        print(f"  Completed: {len(files)} images\n")
    
    # Check if we got any encodings
    if len(encodings) == 0:
        print("❌ No encodings generated!")
        print("Please check your images are valid JPG/PNG files")
        print("\nTroubleshooting:")
        print("1. Make sure images contain visible faces")
        print("2. Try re-capturing images with better lighting")
        print("3. Ensure images are not corrupted")
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
