import pandas as pd
from pathlib import Path
from tqdm import tqdm
from MysqlConnection import get_mysql_connection

def generate_time_samples_eicu():
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "time"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load patient info
    conn = get_mysql_connection("eicu")
    try:
        with conn.cursor() as cursor:
            query = """
                SELECT patientunitstayid, gender, age, hospitaldischargeyear
                FROM patient
                WHERE gender IS NOT NULL AND age IS NOT NULL AND hospitaldischargeyear IS NOT NULL
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()

    print(f"[INFO] Loaded {len(df)} valid rows from eICU.")

    # Normalize fields
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df["hospitaldischargeyear"] = pd.to_numeric(df["hospitaldischargeyear"], errors="coerce")
    df = df.dropna(subset=["gender", "age", "hospitaldischargeyear"])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    count = 0
    idx = 0
    max_samples = 100
    chunk_size = 10

    while count < max_samples and idx + chunk_size <= len(df):
        chunk = df.iloc[idx:idx + chunk_size]
        if chunk["gender"].nunique() >= 2:
            out_file = output_dir / f"sample_{count+1:03d}.csv"
            chunk.to_csv(out_file, index=False)
            count += 1
        idx += chunk_size

    print(f"[INFO] Generated {count} time samples in: {output_dir}")

if __name__ == "__main__":
    generate_time_samples_eicu()
