"""
Module 3: Train Face Recognition Model (INCREMENTAL + FAST)
Only trains NEW persons, keeps existing encodings
Uses HOG detection for 10x faster training
"""

import os
import cv2
import numpy as np
import face_recognition
import pickle
from pathlib import Path
from PIL import Image, ImageOps
from datetime import datetime


def load_as_rgb_uint8(img_path):
    """
    Load image as proper 8-bit RGB array
    """
    try:
        with Image.open(img_path) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            return arr
    except Exception as e:
        return None


def train_model_incremental():
    """
    Train only NEW persons/images, keep existing encodings
    OPTIMIZED: Uses HOG for speed
    """
    
    dataset_path = 'dataset'
    model_path = 'models/face_encodings.pkl'
    
    if not os.path.exists(dataset_path):
        print("❌ Error: 'dataset' folder not found!")
        return
    
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*60)
    print("INCREMENTAL FACE RECOGNITION TRAINING")
    print("="*60)
    print(f"Dataset: {dataset_path}/")
    print(f"Method: HOG (Fast Detection)")
    print("Mode: INCREMENTAL (only new data)")
    print("="*60 + "\n")
    
    # Load existing model if it exists
    existing_encodings = []
    existing_names = []
    trained_persons = set()
    
    if os.path.exists(model_path):
        print("📦 Loading existing model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            existing_encodings = model_data['encodings']
            existing_names = model_data['names']
            trained_persons = set(existing_names)
        print(f"✓ Found {len(existing_encodings)} existing encodings")
        print(f"✓ Already trained persons: {', '.join(sorted(trained_persons))}\n")
    else:
        print("📦 No existing model found. Training from scratch.\n")
    
    person_folders = [f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))]
    
    if not person_folders:
        print("❌ No person folders found in dataset!")
        return
    
    new_persons = [p for p in person_folders if p not in trained_persons]
    
    if not new_persons:
        print("✓ All persons already trained. No new data to process.")
        print(f"\nCurrent persons: {', '.join(sorted(trained_persons))}")
        return
    
    print(f"🆕 New persons to train: {', '.join(new_persons)}\n")
    
    all_encodings = list(existing_encodings)
    all_names = list(existing_names)
    
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    for person_name in new_persons:
        person_folder = os.path.join(dataset_path, person_name)
        
        print(f"Training: {person_name}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in os.listdir(person_folder) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        if not image_files:
            print(f"  ⚠️  No images found\n")
            continue
        
        person_success = 0
        
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(person_folder, img_file)
            total_processed += 1
            
            # Progress indicator
            print(f"  Processing {i}/{len(image_files)}...", end='\r')
            
            img_rgb = load_as_rgb_uint8(img_path)
            if img_rgb is None:
                total_failed += 1
                continue
            
            try:
                # ⚡ USE HOG FOR SPEED (10x faster than CNN)
                locations = face_recognition.face_locations(img_rgb, model='hog')
                
                if not locations:
                    total_failed += 1
                    continue
                
                encodings = face_recognition.face_encodings(img_rgb, known_face_locations=locations)
                
                if not encodings:
                    total_failed += 1
                    continue
                
                all_encodings.append(encodings[0])
                all_names.append(person_name)
                person_success += 1
                total_success += 1
                
            except Exception as e:
                total_failed += 1
        
        print(f"  ✓ Success: {person_success}/{len(image_files)}")
        print(f"  Total: {len(image_files)}\n")
    
    if total_success > 0:
        model_data = {
            'encodings': all_encodings,
            'names': all_names,
            'detection_method': 'hog',  # ⚡ Changed to HOG
            'trained_date': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("="*60)
        print("✓ TRAINING COMPLETED")
        print("="*60)
        print(f"Total images processed: {total_processed}")
        print(f"Successfully encoded: {total_success}")
        print(f"Failed: {total_failed}")
        print(f"Model saved: {model_path}")
        print("="*60)
        
        print("\nPERSON STATISTICS:")
        unique_persons = sorted(set(all_names))
        for person in unique_persons:
            count = all_names.count(person)
            status = "🆕 NEW" if person in new_persons else "✓ existing"
            print(f"  {person}: {count} encodings ({status})")
        
        print("\n✓ Ready for recognition!")
        print("Next: Run 4_detect_faces.py")
        print("="*60 + "\n")
        
    else:
        print("❌ No encodings generated!")
        print("✗ Training failed. Please check errors above.")


if __name__ == "__main__":
    train_model_incremental()
