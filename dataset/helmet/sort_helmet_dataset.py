import os
import shutil
import xml.etree.ElementTree as ET

# --- UPDATE THESE PATHS ---
# Point to where you extracted the Kaggle dataset
kaggle_images = r"C:\Users\MR AME\Downloads\archive\images"
kaggle_annotations = r"C:\Users\MR AME\Downloads\archive\annotations"
# --------------------------

target_with = "./with_helmet"
target_without = "./without_helmet"

os.makedirs(target_with, exist_ok=True)
os.makedirs(target_without, exist_ok=True)

# Counters for verification
with_count = 0
without_count = 0

for xml_file in os.listdir(kaggle_annotations):
    if not xml_file.endswith(".xml"):
        continue
    
    xml_path = os.path.join(kaggle_annotations, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Check the class name - it might be "with helmet" or "helmet"
    has_helmet = False
    for obj in root.findall("object"):
        name = obj.find("name").text
        # The dataset might use different naming
        if name.lower() in ["with helmet", "helmet", "with_helmet"]:
            has_helmet = True
            break
    
    # Corresponding image file (check both .jpg and .png)
    img_name = xml_file.replace(".xml", ".jpg")
    img_path = os.path.join(kaggle_images, img_name)
    
    # Try .png if .jpg not found
    if not os.path.exists(img_path):
        img_name = xml_file.replace(".xml", ".png")
        img_path = os.path.join(kaggle_images, img_name)
    
    if os.path.exists(img_path):
        if has_helmet:
            shutil.copy(img_path, target_with)
            with_count += 1
        else:
            shutil.copy(img_path, target_without)
            without_count += 1
    else:
        print(f"⚠️ Image not found for: {xml_file}")

print(f"✅ Dataset sorted successfully!")
print(f"   📁 with_helmet: {with_count} images")
print(f"   📁 without_helmet: {without_count} images")
print(f"   📂 Location: {os.path.abspath('.')}")