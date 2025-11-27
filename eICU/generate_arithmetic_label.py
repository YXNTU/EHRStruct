import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def generate_arithmetic_labels_eicu():
    # Set paths
    current_dir = Path(__file__).resolve().parent
    input_dir = current_dir / "arithmetic"
    output_dir = current_dir / "Label"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "arithmetic.csv"

    label_rows = []

    for file in tqdm(sorted(input_dir.glob("sample_*.csv")), desc="Generating arithmetic labels"):
        df = pd.read_csv(file)

        # Ensure required columns exist
        df = df[[
            "patientunitstayid",
            "admissionweight",
            "dischargeweight",
            "cost",
            "tax"
        ]].dropna()

        # Convert to numeric
        for col in ["admissionweight", "dischargeweight", "cost", "tax"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()

        if len(df) == 0:
            continue

        # Randomly select one patient
        selected = df.sample(n=1, random_state=hash(file.name) % 10000).iloc[0]
        pid = selected["patientunitstayid"]

        # Compute one subtraction and one addition
        diff = round(selected["dischargeweight"] - selected["admissionweight"], 1)
        add = round(selected["cost"] + selected["tax"], 1)

        label_rows.append({
            "Filename": file.name,
            "PATIENT": pid,
            "Diff": diff,
            "Add": add
        })

    label_df = pd.DataFrame(label_rows, columns=["Filename", "PATIENT", "Diff", "Add"])
    label_df.to_csv(output_file, index=False)
    print(f"[INFO] Saved label file to: {output_file}")

if __name__ == "__main__":
    generate_arithmetic_labels_eicu()
