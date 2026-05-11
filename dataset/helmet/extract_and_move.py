import os
import shutil
import xml.etree.ElementTree as ET

# --- CONFIGURE THESE PATHS ---
images_source = "path_to_extracted/JPEGImages"  # Folder with all .jpg files
annotations_source = "path_to_extracted/Annotations" # Folder with all .xml files
target_base = "." # Your current 'helmet' folder
# -----------------------------

os.makedirs(f"{target_base}/with_helmet", exist_ok=True)
os.makedirs(f"{target_base}/without_helmet", exist_ok=True)

for xml_file in os.listdir(annotations_source):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(annotations_source, xml_file))
    root = tree.getroot()

    has_helmet = False
    for obj in root.findall("object"):
        if obj.find("name").text == "hat": # The class for helmets in SHWD[reference:3]
            has_helmet = True
            break

    # Find the corresponding .jpg file
    img_name = xml_file.replace(".xml", ".jpg")
    img_path = os.path.join(images_source, img_name)

    if os.path.exists(img_path):
        if has_helmet:
            shutil.copy(img_path, f"{target_base}/with_helmet/")
        else:
            shutil.copy(img_path, f"{target_base}/without_helmet/")

print("Dataset sorted successfully!")