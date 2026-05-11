import os
import xml.etree.ElementTree as ET

kaggle_annotations = r"C:\Users\MR AME\Downloads\archive\annotations"

# Check first 5 XML files to see class names
for i, xml_file in enumerate(os.listdir(kaggle_annotations)):
    if i >= 5:
        break
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(kaggle_annotations, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        print(f"\n📄 {xml_file}:")
        for obj in root.findall("object"):
            name = obj.find("name").text
            print(f"   Class: '{name}'")