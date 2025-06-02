import os
import json
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import unicodedata

# === CONFIGURATION ===
excel_path = r"D:\MS Computer Engineering\Thesis\Resources_1\Tabella AMD real-life.xlsx"
bscan_root = r"E:\Dataset 01"
output_json = "AMD_Label_New.json"

# Load Excel
df = pd.read_excel(excel_path)
eye_map = {1: "RIGHT", 2: "LEFT"}

# Output holders
final_annotations = {}
skipped_patients = []

# Full BCVA and injection column names
bcva_cols = {
    2: "BCVA 1 anno", 3: "BCVA 2 anni", 4: "BCVA 3 anni", 5: "BCVA 4 anni",
    6: "BCVA 5 anni", 7: "BCVA 6 anni", 8: "BCVA 7 anni", 9: "BCVA 8 anni",
    10: "BCVA 9 anni", 11: "BCVA 10 anni", 12: "BCVA 11 anni", 13: "BCVA 12 anni"
}
inj_cols = {i: f"N. inj {i-1} anno" for i in range(2, 14)}

def clean(text):
    return unicodedata.normalize("NFKD", str(text)).strip().upper().replace('\xa0', ' ')

# Loop over patients
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing patients"):
    surname = clean(row["AACognome"])
    name = clean(row["Nome"])
    occhio = row["OCCHIO"]
    eye = eye_map.get(occhio)
    patient_key = f"{surname}, {name}, {eye}"

    patient_folder = f"{surname}, {name}"
    full_folder_path = os.path.join(bscan_root, patient_folder)

    if not os.path.isdir(full_folder_path):
        skipped_patients.append(patient_key)
        continue

    records = []
    all_images = []
    baseline_date = None
    fallback_images = []

    subfolders = sorted(os.listdir(full_folder_path))

    for i, subfolder in enumerate(subfolders):
        sub_path = os.path.join(full_folder_path, subfolder)
        metadata_path = os.path.join(sub_path, "FilteredAnnotations.json")
        if not os.path.isfile(metadata_path):
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        png_files = [f for f in os.listdir(sub_path) if f.lower().endswith(".png")]

        valid_entries = []
        for png in png_files:
            meta = next((entry for entry in metadata if entry.get("filename") == png), None)
            if not meta:
                continue

            laterality = meta.get("Laterality", "").strip().upper()
            if laterality != eye[0]:
                continue

            scan_date_str = meta.get("scan_date")
            if not scan_date_str:
                continue

            scan_date = None
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                try:
                    scan_date = datetime.strptime(scan_date_str, fmt)
                    break
                except ValueError:
                    continue

            if scan_date is None:
                continue

            valid_entries.append({
                "image": png,
                "scan_date": scan_date,
                "folder": subfolder
            })

        if valid_entries:
            all_images.extend(valid_entries)
            if not baseline_date:
                baseline_date = valid_entries[0]["scan_date"]
            continue

        if i == 0 and not png_files and not baseline_date:
            try:
                first_meta = metadata[0]
                scan_date_str = first_meta.get("scan_date")
                baseline_date = None
                for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                    try:
                        baseline_date = datetime.strptime(scan_date_str, fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                continue

            for j in range(1, len(subfolders)):
                future_path = os.path.join(full_folder_path, subfolders[j])
                meta_path = os.path.join(future_path, "FilteredAnnotations.json")
                if not os.path.isfile(meta_path):
                    continue

                with open(meta_path, "r", encoding="utf-8") as f:
                    future_meta = json.load(f)

                future_pngs = [f for f in os.listdir(future_path) if f.lower().endswith(".png")]

                for png in future_pngs:
                    meta = next((entry for entry in future_meta if entry.get("filename") == png), None)
                    if not meta:
                        continue
                    laterality = meta.get("Laterality", "").strip().upper()
                    if laterality != eye[0]:
                        continue

                    scan_date_str = meta.get("scan_date")
                    scan_date = None
                    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                        try:
                            scan_date = datetime.strptime(scan_date_str, fmt)
                            break
                        except ValueError:
                            continue
                    if scan_date is None:
                        continue

                    if baseline_date and abs((scan_date - baseline_date).days) <= 92:
                        fallback_images.append({
                            "image": png,
                            "scan_date": baseline_date,
                            "folder": subfolders[j]
                        })

            all_images.extend(fallback_images)

    if not all_images:
        print(f"⚠️ No valid scans for: {patient_key}")
        continue

    all_images.sort(key=lambda x: x["scan_date"])
    year_buckets = defaultdict(list)

    for img in all_images:
        scan_date = img["scan_date"]
        delta = (scan_date - baseline_date).days
        if delta <= 30:
            year = 1
        else:
            year = ((delta - 31) // 365) + 2
        year_buckets[year].append((scan_date, img["image"]))

    for year in range(1, 14):
        if year not in year_buckets:
            continue

        bcva_col = "BCVA baseline" if year == 1 else bcva_cols.get(year)
        if bcva_col not in row or pd.isna(row[bcva_col]):
            continue
        bcva_val = float(row[bcva_col])

        inj_col = inj_cols.get(year)
        try:
            n_inj_val = int(row.get(inj_col, 0)) if inj_col else 0
        except (ValueError, TypeError):
            n_inj_val = 0

        label = "good" if bcva_val >= 0.4 else "bad"

        sorted_scans = sorted(year_buckets[year], key=lambda x: x[0])
        images = [img for _, img in sorted_scans]
        latest_scan_date = sorted_scans[-1][0].strftime("%d/%m/%Y")

        records.append({
            "year": year,
            "images": images,
            "n_inj": n_inj_val,
            "BCVA": bcva_val,
            "label": label,
            "scan_date": latest_scan_date
        })

    if records:
        final_annotations[patient_key] = records

# Save output JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(final_annotations, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Annotated {len(final_annotations)} patient-eyes.")
if skipped_patients:
    print(f"❌ Skipped {len(skipped_patients)} patients (missing folders):")
    for p in skipped_patients:
        print(f"  - {p}")
