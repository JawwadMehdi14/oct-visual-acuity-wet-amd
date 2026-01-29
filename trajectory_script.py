import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Load your existing labeled JSON
data_path = Path("AMD_Label_New.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define binary trajectory labeling logic: improved vs not_improved
def generate_trajectory_label(bcva_sequence, inj_sequence):
    bcva_sequence = [v for v in bcva_sequence if v is not None]
    if len(bcva_sequence) < 3:
        return "not_improved"

    avg_bcva = sum(bcva_sequence) / len(bcva_sequence)
    final_bcva = bcva_sequence[-1]

    if final_bcva >= 0.5 and final_bcva > avg_bcva:
        return "improved"
    else:
        return "not_improved"

# Build the output dictionary
trajectory_labels = {}

for patient_eye, records in tqdm(data.items(), desc="Labeling patient-eyes"):
    bcva_seq = []
    inj_seq = []
    for year_record in sorted(records, key=lambda r: r["year"]):
        bcva = year_record.get("BCVA")
        n_inj = year_record.get("n_inj")
        bcva_seq.append(bcva if isinstance(bcva, (float, int)) else None)
        inj_seq.append(n_inj if isinstance(n_inj, (int, float)) else 0)

    trajectory_labels[patient_eye] = generate_trajectory_label(bcva_seq, inj_seq)

# Save to new JSON
output_path = Path("trajectory_labels.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(trajectory_labels, f, indent=2)

# Print summary statistics
label_counts = Counter(trajectory_labels.values())
print("\n✅ Refined trajectory labels saved to:", output_path)
print("\n🔍 Label Distribution Summary:")
for label, count in label_counts.items():
    print(f"{label}: {count} samples")
