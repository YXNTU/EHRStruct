import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from MysqlConnection import get_mysql_connection

def generate_arithmetic_samples_eicu():
    # Set paths
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "arithmetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to eICU and fetch required fields
    conn = get_mysql_connection("eicu")
    try:
        with conn.cursor() as cursor:
            query = """
                SELECT patientunitstayid, admissionweight, dischargeweight,
                       gender, age, unitvisitnumber
                FROM patient
                WHERE admissionweight IS NOT NULL AND dischargeweight IS NOT NULL
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()

    print(f"[INFO] Loaded {len(df)} valid rows from eICU.")

    # Deduplicate and shuffle
    df = df.drop_duplicates(subset=["patientunitstayid"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total_required = 100 * 10
    if len(df) < total_required:
        raise ValueError(f"Not enough unique patients: need {total_required}, found {len(df)}.")

    for i in tqdm(range(100), desc="Generating arithmetic samples"):
        start = i * 10
        end = start + 10
        df_chunk = df.iloc[start:end].copy()

        # Generate cost and tax (tax is a percentage of cost)
        np.random.seed(i)
        df_chunk["cost"] = np.round(np.random.uniform(1000, 20000, size=10), 1)
        tax_rate = np.random.uniform(0.03, 0.15, size=10)
        df_chunk["tax"] = np.round(df_chunk["cost"] * tax_rate, 1)

        # Shuffle column order
        df_chunk = df_chunk.sample(frac=1, axis=1, random_state=i)

        output_file = output_dir / f"sample_{i+1:03d}.csv"
        df_chunk.to_csv(output_file, index=False)

    print(f"[INFO] Saved 100 arithmetic samples with randomized column order to: {output_dir}")

if __name__ == "__main__":
    generate_arithmetic_samples_eicu()
