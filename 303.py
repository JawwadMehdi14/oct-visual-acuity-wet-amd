import json
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
image_root = Path(r"E:\Labeled_PNGs")  # folder containing all .png images
label_path = Path("trajectory_labels.json")
data_path = Path("AMD_Label_New.json")
output_path = Path("flattened_all_years.json")

# === LOAD LABELS ===
with open(label_path, "r", encoding="utf-8") as f:
    trajectory_labels = json.load(f)

with open(data_path, "r", encoding="utf-8") as f:
    all_records = json.load(f)

# === GENERATE FLAT PATIENT-EYE DATASET ===
output = []

for patient_eye, records in tqdm(all_records.items(), desc="Building patient-eye dataset"):
    label = trajectory_labels.get(patient_eye)
    if label is None:
        continue

    image_paths = []
    for year_data in records:
        image_list = year_data.get("images", [])
        image_paths.extend([str(image_root / img) for img in image_list])

    if not image_paths:
        continue  # skip entries with no valid images

    output.append({
        "patient_eye": patient_eye,
        "label": label,
        "images": image_paths
    })

# === SAVE DATASET ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Dataset saved to: {output_path} with {len(output)} patient-eyes")
