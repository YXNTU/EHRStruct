import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from MysqlConnection import get_mysql_connection

def generate_filter_samples_eicu():
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "filter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load patient info from eICU
    conn = get_mysql_connection("eicu")
    try:
        with conn.cursor() as cursor:
            query = """
                SELECT patientunitstayid, gender, age, ethnicity, hospitaldischargestatus
                FROM patient
                WHERE gender IS NOT NULL AND age IS NOT NULL
                      AND ethnicity IS NOT NULL AND hospitaldischargestatus IS NOT NULL
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()

    print(f"[INFO] Loaded {len(df)} patient rows with required fields.")

    # Standardize text fields
    df["hospitaldischargestatus"] = df["hospitaldischargestatus"].astype(str).str.strip().str.lower()
    df["ethnicity"] = df["ethnicity"].astype(str).str.strip().str.lower()

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    count = 0
    idx = 0
    max_samples = 100
    chunk_size = 20

    while count < max_samples and idx + chunk_size <= len(df):
        chunk = df.iloc[idx:idx + chunk_size]

        # Count conditions
        status_counts = chunk["hospitaldischargestatus"].value_counts()
        alive_count = status_counts.get("alive", 0)
        expired_count = status_counts.get("expired", 0)

        ethnicity = chunk["ethnicity"]
        american_count = ethnicity.str.contains("american").sum()
        caucasian_count = (ethnicity == "caucasian").sum()


        # Check relaxed balance condition
        if alive_count >= 4 and expired_count >= 4 and american_count >= 4 and caucasian_count >= 4:
            out_file = output_dir / f"sample_{count+1:03d}.csv"
            chunk.to_csv(out_file, index=False)
            count += 1

        idx += chunk_size

    print(f"[INFO] Generated {count} relaxed-balanced 20-row filter samples in: {output_dir}")

if __name__ == "__main__":
    generate_filter_samples_eicu()
