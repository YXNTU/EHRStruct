import pandas as pd
from pathlib import Path
from tqdm import tqdm

def generate_comparison_labels_eicu():
    current_dir = Path(__file__).resolve().parent
    input_dir = current_dir / "comparison"
    output_dir = current_dir / "Label"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison.csv"

    label_rows = []

    for file in tqdm(sorted(input_dir.glob("sample_*.csv")), desc="Generating comparison labels"):
        df = pd.read_csv(file)

        # Filter for MAP (mmHg) rows
        map_df = df[
            df["nursingchartcelltypevallabel"].str.strip().str.upper() == "MAP (MMHG)"
        ]

        if map_df["patientunitstayid"].nunique() != 2 or len(map_df) != 2:
            continue  # Must have exactly 2 patients with MAP

        grouped = map_df.groupby("patientunitstayid")["nursingchartvalue"].first()
        patient_ids = list(grouped.index)
        values = list(pd.to_numeric(grouped, errors="coerce"))

        if any(pd.isna(values)):
            continue  # Skip if non-numeric

        p1, p2 = patient_ids[0], patient_ids[1]
        v1, v2 = values[0], values[1]

        label_rows.append({
            "File": file.name,
            "PATIENT1": p1,
            "PATIENT2": p2,
            "Greater": int(v1 > v2),
            "Less": int(v1 < v2),
            "Equal": int(v1 == v2),
            "Unequal": int(v1 != v2),
        })

    label_df = pd.DataFrame(label_rows, columns=[
        "File", "PATIENT1", "PATIENT2", "Greater", "Less", "Equal", "Unequal"
    ])
    label_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved label file to: {output_path}")

if __name__ == "__main__":
    generate_comparison_labels_eicu()
