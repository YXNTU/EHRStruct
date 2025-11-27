import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from MysqlConnection import get_mysql_connection

def generate_constrained_temperature_samples():
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    conn = get_mysql_connection("eicu")

    seen_patients = {}
    collected = set()
    offset = 0
    batch_size = 5000

    print("[INFO] Iteratively scanning rows to collect 200 patients with temperature data...")

    while len(collected) < 200:
        with conn.cursor() as cursor:
            query = f"""
                SELECT patientunitstayid, nursingchartcelltypevalname, 
                       nursingchartcelltypevallabel, nursingchartvalue
                FROM nurseCharting
                WHERE nursingchartvalue IS NOT NULL
                LIMIT {batch_size} OFFSET {offset}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                break  # End of table
            cols = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=cols)

        df = df.rename(columns={
            "patientunitstayid": "PATIENT",
            "nursingchartcelltypevalname": "DESCRIPTION",
            "nursingchartcelltypevallabel": "UNITS",
            "nursingchartvalue": "VALUE"
        }).dropna(subset=["PATIENT", "VALUE"])

        for pid, group in df.groupby("PATIENT"):
            if pid in collected:
                continue
            temp_mask = group["DESCRIPTION"].str.contains("temperature", case=False, na=False) | \
                        group["UNITS"].str.contains("temperature", case=False, na=False)
            if temp_mask.sum() >= 2:
                seen_patients[pid] = group.copy()
                collected.add(pid)
                if len(collected) >= 200:
                    break

        offset += batch_size
        print(f"[INFO] Checked {offset} rows, collected {len(collected)} patients.")

    conn.close()
    print(f"[INFO] Done collecting {len(collected)} patients with temperature data.")

    patient_ids = list(seen_patients.keys())
    np.random.shuffle(patient_ids)

    for i in tqdm(range(100), desc="Generating samples"):
        p1, p2 = patient_ids[2 * i], patient_ids[2 * i + 1]
        df1 = seen_patients[p1]
        df2 = seen_patients[p2]

        def sample_df(df):
            temp_mask = df["DESCRIPTION"].str.contains("temperature", case=False, na=False) | \
                        df["UNITS"].str.contains("temperature", case=False, na=False)
            temp_part = df[temp_mask].head(20)
            non_temp_part = df[~temp_mask]
            return pd.concat([temp_part, non_temp_part]).head(40)

        sample1 = sample_df(df1)
        sample2 = sample_df(df2)
        combined = pd.concat([sample1, sample2]).sample(frac=1, random_state=i + 42)

        out_path = output_dir / f"sample_{i+1:03d}.csv"
        combined.to_csv(out_path, index=False)
        print(f"[INFO] Saved: {out_path}")

if __name__ == "__main__":
    generate_constrained_temperature_samples()
