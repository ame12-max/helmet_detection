import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / 'dataset' / 'helmet'
TEMP_DIR = Path('temp_helmet')

def organize_dataset():
    """Sort images into with_helmet/without_helmet folders"""
    
    images_dir = TEMP_DIR / 'images'
    annotations_dir = TEMP_DIR / 'annotations'
    
    if not images_dir.exists():
        print("❌ temp_helmet/images not found!")
        print("   Make sure you extracted the dataset first.")
        return
    
    # Create target folders
    with_helmet_dir = DATASET_DIR / 'with_helmet'
    without_helmet_dir = DATASET_DIR / 'without_helmet'
    
    with_helmet_dir.mkdir(parents=True, exist_ok=True)
    without_helmet_dir.mkdir(parents=True, exist_ok=True)
    
    with_count = 0
    without_count = 0
    
    # Process each XML annotation file
    for xml_file in annotations_dir.glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Check if the image has a helmet
            has_helmet = False
            for obj in root.findall('object'):
                name = obj.find('name').text
                # The dataset uses "with helmet" and "without helmet" as class names
                if 'with helmet' in name.lower() or 'helmet' in name.lower():
                    has_helmet = True
                    break
            
            # Find the corresponding image
            img_name = xml_file.stem + '.jpg'
            img_path = images_dir / img_name
            
            if img_path.exists():
                if has_helmet:
                    shutil.copy2(img_path, with_helmet_dir / img_name)
                    with_count += 1
                else:
                    shutil.copy2(img_path, without_helmet_dir / img_name)
                    without_count += 1
                    
        except Exception as e:
            print(f"Error: {xml_file} - {e}")
    
    print(f"\n✅ Dataset organized!")
    print(f"   With helmet: {with_count} images")
    print(f"   Without helmet: {without_count} images")
    print(f"\n📁 Location: {DATASET_DIR}")

def cleanup():
    """Remove temporary files"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("🧹 Cleaned up temporary files")
    
    zip_file = Path('helmet-detection.zip')
    if zip_file.exists():
        zip_file.unlink()
        print("🧹 Removed zip file")

if __name__ == '__main__':
    print("="*50)
    print("  Organizing Kaggle Helmet Dataset")
    print("="*50)
    
    organize_dataset()
    cleanup()
    
    print("\n✅ Done! Now run: python train_cnn.py")