import json
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
image_root = Path("E:\Labeled_PNGs")
data_path = Path("AMD_Label_New.json")
output_path = Path("year1_2_dataset.json")

# === LOAD RECORDS ===
with open(data_path, "r", encoding="utf-8") as f:
    all_records = json.load(f)

# === GENERATE YEAR 1-2 DATASET WITH LOCAL BCVA-BASED LABELING ===
output = []

for patient_eye, records in tqdm(all_records.items(), desc="Building year 1-2 dataset"):
    # Get records for year 1 and 2 only
    filtered_records = sorted([r for r in records if r.get("year") in [1, 2]], key=lambda r: r.get("year"))
    if len(filtered_records) < 2:
        continue

    # Extract BCVA values
    bcva_sequence = [r.get("BCVA") for r in filtered_records if isinstance(r.get("BCVA"), (float, int))]
    if len(bcva_sequence) < 2:
        continue

    avg_bcva = sum(bcva_sequence) / len(bcva_sequence)
    final_bcva = bcva_sequence[-1]
    label = "improved" if final_bcva >= 0.5 and final_bcva > avg_bcva else "not_improved"

    # Collect image paths
    image_paths = []
    for r in filtered_records:
        image_list = r.get("images", [])
        image_paths.extend([str(image_root / img) for img in image_list if (image_root / img).exists()])

    if not image_paths:
        continue

    output.append({
        "patient_eye": patient_eye,
        "label": label,
        "images": image_paths,
        "years": 2
    })

# === SAVE DATASET ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Year 1-2 dataset saved to: {output_path} with {len(output)} patient-eyes")
