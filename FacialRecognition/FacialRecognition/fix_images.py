"""
Utility: Fix Image Format
Converts all dataset images to proper 8-bit RGB JPEG format
Run this BEFORE training if you have image format issues
"""

from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import os


def fix_image_to_rgb_jpeg(img_path, quality=95):
    """
    Convert any image to proper 8-bit RGB JPEG
    
    Args:
        img_path: Path to the image file
        quality: JPEG quality (0-100, default 95)
    """
    try:
        with Image.open(img_path) as im:
            # Load image fully
            im.load()
            
            # Fix orientation from EXIF
            im = ImageOps.exif_transpose(im)
            
            # Force convert to RGB (removes alpha, handles all modes)
            im = im.convert("RGB")
            
            # Ensure uint8 via numpy
            arr = np.array(im, dtype=np.uint8)
            im = Image.fromarray(arr, mode="RGB")
            
            # Save as JPEG
            im.save(img_path, format="JPEG", quality=quality, optimize=True)
            
            return True
    except Exception as e:
        print(f"  ✗ Failed to fix {img_path.name}: {e}")
        return False


def fix_all_dataset_images():
    """
    Fix all images in the dataset folder
    """
    dataset = Path("dataset")
    
    if not dataset.exists():
        print("❌ Error: 'dataset' folder not found!")
        return
    
    print("\n" + "="*60)
    print("IMAGE FORMAT FIXER - Converting to 8-bit RGB JPEG")
    print("="*60)
    print(f"Dataset folder: {dataset}")
    print("="*60 + "\n")
    
    # Supported formats
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    total = 0
    fixed = 0
    failed = 0
    
    # Process each person folder
    for person_folder in dataset.iterdir():
        if not person_folder.is_dir():
            continue
        
        print(f"Processing: {person_folder.name}")
        
        # Get all image files
        images = [f for f in person_folder.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_exts]
        
        if not images:
            print(f"  No images found\n")
            continue
        
        for img_path in images:
            total += 1
            if fix_image_to_rgb_jpeg(img_path):
                fixed += 1
                print(f"  ✓ Fixed: {img_path.name}")
            else:
                failed += 1
        
        print(f"  Total in folder: {len(images)}\n")
    
    print("="*60)
    print("IMAGE FIXING COMPLETED")
    print("="*60)
    print(f"Total images processed: {total}")
    print(f"Successfully fixed: {fixed}")
    print(f"Failed: {failed}")
    print("="*60 + "\n")
    
    if fixed > 0:
        print("✓ Images are now ready for training!")
        print("Next step: Run 3_train_model.py")
    else:
        print("⚠ No images were fixed. Please check your dataset folder.")


if __name__ == "__main__":
    fix_all_dataset_images()
