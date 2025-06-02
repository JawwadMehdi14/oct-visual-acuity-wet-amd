import os
import json
import shutil
from tqdm import tqdm

# === CONFIGURATION ===
source_root = r"E:\Dataset 01"
json_path = r"AMD_Label_New.json"
output_dir = r"E:\Labeled_PNGs"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the annotations
with open(json_path, "r", encoding="utf-8") as f:
    annotations = json.load(f)

copied_files = 0
skipped_files = 0

# Loop through all labeled patient-eyes
for patient_key, records in tqdm(annotations.items(), desc="Copying labeled PNGs"):
    surname, name, _ = patient_key.split(", ")
    patient_folder = f"{surname}, {name}"

    for record in records:
        for image in record.get("images", []):
            # Build source path
            patient_path = os.path.join(source_root, patient_folder)
            destination_path = os.path.join(output_dir, image)

            if os.path.exists(destination_path):
                skipped_files += 1
                continue

            found = False
            for subfolder in os.listdir(patient_path):
                candidate_path = os.path.join(patient_path, subfolder, image)
                if os.path.isfile(candidate_path):
                    shutil.copy(candidate_path, destination_path)
                    copied_files += 1
                    found = True
                    break

            if not found:
                print(f"⚠️ File not found: {image} for {patient_key}")

print(f"\n✅ Copied {copied_files} new PNGs to {output_dir}")
print(f"⏭️ Skipped {skipped_files} already existing PNGs.")
