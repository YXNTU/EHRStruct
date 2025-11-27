import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from MysqlConnection import get_mysql_connection

def generate_comparison_samples_eicu():
    # Set paths
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect and fetch from nurseCharting
    conn = get_mysql_connection("eicu")
    try:
        with conn.cursor() as cursor:
            query = """
                SELECT patientunitstayid, nursingchartcelltypevallabel,
                    nursingchartcelltypevalname, nursingchartvalue
                        FROM nurseCharting
                                WHERE nursingchartvalue IS NOT NULL
                        LIMIT 500000 OFFSET 0

            """
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()

    print(f"[INFO] Loaded {len(df)} nurseCharting rows")

    # Standardize column names
    df = df.dropna(subset=["patientunitstayid", "nursingchartvalue"])
    df["MAP_flag"] = df["nursingchartcelltypevallabel"].str.strip().str.upper() == "MAP (MMHG)"
    df["nursingchartvalue"] = pd.to_numeric(df["nursingchartvalue"], errors="coerce")
    df = df.dropna(subset=["nursingchartvalue"])

    grouped = df.groupby("patientunitstayid")
    valid_patients = []

    print("[INFO] Scanning for first 200 patients with MAP (mmHg)...")
    for pid, group in grouped:
        if group["MAP_flag"].any():
            valid_patients.append(pid)
        if len(valid_patients) >= 200:
            break

    if len(valid_patients) < 200:
        raise ValueError("Not enough patients with MAP (mmHg) available.")

    def sample_patient(pid, seed):
        data = df[df["patientunitstayid"] == pid].copy()
        map_part = data[data["MAP_flag"]].head(1)
        other_part = data[~data["MAP_flag"]].head(40 - len(map_part))
        sample = pd.concat([map_part, other_part]).sample(frac=1, random_state=seed)
        return sample

    for i in tqdm(range(100), desc="Generating comparison samples"):
        p1, p2 = valid_patients[2 * i], valid_patients[2 * i + 1]
        sample1 = sample_patient(p1, seed=i)
        sample2 = sample_patient(p2, seed=i + 100)
        df_sample = pd.concat([sample1, sample2]).sample(frac=1, random_state=i + 200)

        df_sample = df_sample[[
            "patientunitstayid",
            "nursingchartcelltypevallabel",
            "nursingchartcelltypevalname",
            "nursingchartvalue"
        ]]

        out_path = output_dir / f"sample_{i+1:03d}.csv"
        df_sample.to_csv(out_path, index=False)

    print(f"[INFO] Saved 100 comparison samples to: {output_dir}")

if __name__ == "__main__":
    generate_comparison_samples_eicu()
