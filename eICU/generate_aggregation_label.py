import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def fahrenheit_to_celsius(f):
    return (f - 32) * 5.0 / 9.0

def generate_temperature_labels():
    # Define input/output paths
    current_dir = Path(__file__).resolve().parent
    agg_dir = current_dir / "aggregation"
    label_dir = current_dir / "Label"
    label_dir.mkdir(parents=True, exist_ok=True)
    output_path = label_dir / "aggregation.csv"

    label_rows = []

    # Clean all files in aggregation: convert Temperature (F) → (C), round to 1 decimal
    for file in tqdm(sorted(agg_dir.glob("sample_*.csv")), desc="Cleaning temperature units"):
        df = pd.read_csv(file)

        is_fahrenheit = df["DESCRIPTION"].str.contains("F", na=False) | df["UNITS"].str.contains("F", na=False)
        df.loc[is_fahrenheit, "VALUE"] = pd.to_numeric(
            df.loc[is_fahrenheit, "VALUE"], errors="coerce"
        ).apply(fahrenheit_to_celsius).round(1)

        df.loc[is_fahrenheit, "DESCRIPTION"] = df.loc[is_fahrenheit, "DESCRIPTION"].str.replace("F", "C", regex=False)
        df.loc[is_fahrenheit, "UNITS"] = df.loc[is_fahrenheit, "UNITS"].str.replace("F", "C", regex=False)

        df.to_csv(file, index=False)

        # Clean: convert Fahrenheit to Celsius
        is_fahrenheit = df["DESCRIPTION"].str.contains("F", na=False) | df["UNITS"].str.contains("F", na=False)
        df.loc[is_fahrenheit, "VALUE"] = pd.to_numeric(df.loc[is_fahrenheit, "VALUE"], errors="coerce").apply(fahrenheit_to_celsius)
        df.loc[is_fahrenheit, "DESCRIPTION"] = df.loc[is_fahrenheit, "DESCRIPTION"].str.replace("F", "C", regex=False)
        df.loc[is_fahrenheit, "UNITS"] = df.loc[is_fahrenheit, "UNITS"].str.replace("F", "C", regex=False)

        patient_ids = df["PATIENT"].unique()
        if len(patient_ids) != 2:
            continue

        selected_patient = random.choice(patient_ids)
        patient_data = df[df["PATIENT"] == selected_patient]

        # Filter Temperature records (now all are in Celsius)
        temp_data = patient_data[
            patient_data["DESCRIPTION"].str.contains("temperature", case=False, na=False) |
            patient_data["UNITS"].str.contains("temperature", case=False, na=False)
        ]

        values = pd.to_numeric(temp_data["VALUE"], errors="coerce").dropna()

        count = len(values)
        avg = round(values.mean(), 1) if count > 0 else 0.0
        ssum = round(values.sum(), 1) if count > 0 else 0.0
        maxx = round(values.max(), 1) if count > 0 else 0.0
        minn = round(values.min(), 1) if count > 0 else 0.0

        label_rows.append({
            "File": file.name,
            "Name": selected_patient,
            "Count": count,
            "Avg": avg,
            "Sum": ssum,
            "Max": maxx,
            "Min": minn
        })

    label_df = pd.DataFrame(label_rows, columns=["File", "Name", "Count", "Avg", "Sum", "Max", "Min"])
    label_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved temperature labels to: {output_path}")

if __name__ == "__main__":
    generate_temperature_labels()
