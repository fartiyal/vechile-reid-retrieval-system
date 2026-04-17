# diagnose_xml.py - Inspect VeRi XML structure
import os
from pathlib import Path

xml_path = "VeRi/train_label.xml"

print("="*60)
print("XML DIAGNOSTIC")
print("="*60)

if not os.path.exists(xml_path):
    print(f"ERROR: {xml_path} not found!")
    exit(1)

# Read the first 2000 characters
with open(xml_path, 'rb') as f:
    raw = f.read(2000)

print(f"File size: {os.path.getsize(xml_path) / 1024 / 1024:.2f} MB")

# Try different encodings
for encoding in ['utf-8', 'utf-16', 'utf-8-sig', 'iso-8859-1', 'cp1252']:
    try:
        text = raw.decode(encoding)
        print(f"\n✅ Success with encoding: {encoding}")
        print(f"\nFirst 1000 characters:")
        print("-"*40)
        print(text[:1000])
        print("-"*40)
        
        # Look for patterns
        if '<Item' in text:
            print("Found '<Item' tags")
        if '<vehicle' in text:
            print("Found '<vehicle' tags")
        if 'imageName' in text:
            print("Found 'imageName' attribute")
        if 'vehicleID' in text:
            print("Found 'vehicleID' attribute")
            
        break
    except:
        continue

# Try parsing with ElementTree
import xml.etree.ElementTree as ET

print("\n" + "="*60)
print("ATTEMPTING TO PARSE")
print("="*60)

with open(xml_path, 'rb') as f:
    content = f.read()

for encoding in ['utf-8', 'utf-16', 'utf-8-sig', 'iso-8859-1']:
    try:
        text = content.decode(encoding)
        if text.startswith('\ufeff'):
            text = text[1:]
        root = ET.fromstring(text)
        print(f"\n✅ Parsed successfully with {encoding}")
        print(f"Root tag: {root.tag}")
        print(f"Root attributes: {root.attrib}")
        
        # List all child tags
        child_tags = set()
        for child in root:
            child_tags.add(child.tag)
        print(f"Child tags: {child_tags}")
        
        # Show first 3 items
        count = 0
        for child in root:
            if count < 3:
                print(f"\nItem {count + 1}:")
                print(f"  Tag: {child.tag}")
                print(f"  Attributes: {child.attrib}")
            count += 1
        print(f"\nTotal children: {count}")
        
        break
    except Exception as e:
        print(f"Failed with {encoding}: {e}")