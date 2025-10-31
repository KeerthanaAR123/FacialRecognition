"""
Fix Image Format - Convert to proper 8-bit RGB JPEG
Run this to fix problematic images before training
"""

from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import os


def fix_image_to_rgb_jpeg(img_path, quality=95):
    """
    Convert any image to proper 8-bit RGB JPEG format
    
    Args:
        img_path: Path to the image file
        quality: JPEG quality (0-100, default 95)
    """
    p = Path(img_path)
    
    # Check if file exists
    if not p.exists():
        print(f"❌ Error: File not found: {p}")
        print(f"\nPlease check:")
        print(f"  1. File exists at this location")
        print(f"  2. File name is spelled correctly")
        print(f"  3. Path has correct folder names")
        return False
    
    try:
        print(f"\n{'='*60}")
        print(f"FIXING IMAGE FORMAT")
        print(f"{'='*60}")
        print(f"File: {p.name}")
        print(f"Location: {p.parent}")
        print(f"{'='*60}\n")
        
        # Open and process image
        with Image.open(p) as im:
            print(f"Original format: {im.format}")
            print(f"Original mode: {im.mode}")
            print(f"Original size: {im.size}")
            
            # Load image fully into memory
            im.load()
            
            # Fix orientation from EXIF data
            im = ImageOps.exif_transpose(im)
            
            # Force convert to RGB (removes alpha, handles all modes)
            im = im.convert("RGB")
            print(f"Converted to: RGB")
            
            # Ensure uint8 data type via numpy
            arr = np.array(im, dtype=np.uint8)
            im = Image.fromarray(arr, mode="RGB")
            
            # Save as high-quality JPEG
            im.save(p, format="JPEG", quality=quality, optimize=True, progressive=True)
        
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS: Image fixed and saved!")
        print(f"{'='*60}")
        print(f"File: {p}")
        print(f"Format: JPEG (8-bit RGB)")
        print(f"Quality: {quality}%")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error fixing image: {e}")
        return False


def fix_all_images_in_folder(folder_path):
    """
    Fix all images in a specific folder
    
    Args:
        folder_path: Path to folder containing images
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Error: Folder not found: {folder}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"❌ No images found in: {folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"BATCH FIX - Found {len(images)} images")
    print(f"{'='*60}\n")
    
    fixed = 0
    failed = 0
    
    for img in images:
        if fix_image_to_rgb_jpeg(img):
            fixed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"BATCH FIX COMPLETED")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    print(f"Successfully fixed: {fixed}")
    print(f"Failed: {failed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("IMAGE FORMAT FIXER")
    print("="*60)
    print("\nChoose option:")
    print("  1. Fix single image (enter full path)")
    print("  2. Fix all images in a folder")
    print("  3. Fix all images in dataset folder")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        # Single image fix
        img_path = input("\nPaste full image path: ").strip()
        # Remove quotes if user copied with quotes
        img_path = img_path.strip('"').strip("'")
        fix_image_to_rgb_jpeg(img_path)
        
    elif choice == '2':
        # Fix folder
        folder_path = input("\nPaste folder path: ").strip()
        folder_path = folder_path.strip('"').strip("'")
        fix_all_images_in_folder(folder_path)
        
    elif choice == '3':
        # Fix entire dataset
        dataset = Path("dataset")
        if not dataset.exists():
            print("❌ Error: 'dataset' folder not found in current directory!")
        else:
            print(f"\n{'='*60}")
            print(f"FIXING ALL DATASET IMAGES")
            print(f"{'='*60}\n")
            
            for person_folder in dataset.iterdir():
                if person_folder.is_dir():
                    print(f"\nProcessing: {person_folder.name}")
                    fix_all_images_in_folder(person_folder)
    else:
        print("Invalid choice")
    
    print("\n✓ Done! You can now run training.")
3