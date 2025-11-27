import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from MysqlConnection import get_mysql_connection

def is_abnormal_temp(value):
    try:
        return float(value) >= 37.2
    except:
        return False

def generate_single_temp_per_sample():
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "abnormal"
    output_dir.mkdir(parents=True, exist_ok=True)
    label_file = Path(current_dir / "Label" / "abnormal.csv")
    label_file.parent.mkdir(parents=True, exist_ok=True)

    conn = get_mysql_connection("eicu")
    offset = 0
    batch_size = 5000
    normal, abnormal = [], []
    all_labels = []

    print("[INFO] Searching for temperature records...")

    while len(normal) < 50 or len(abnormal) < 50:
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
                break
            cols = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=cols)

        df = df.rename(columns={
            "patientunitstayid": "ID",
            "nursingchartcelltypevalname": "DESCRIPTION",
            "nursingchartcelltypevallabel": "UNITS",
            "nursingchartvalue": "VALUE"
        }).dropna()

        for pid, group in df.groupby("ID"):
            if len(normal) >= 50 and len(abnormal) >= 50:
                break

            temp_df = group[
                group["DESCRIPTION"].str.strip().str.lower() == "temperature (c)"
                ].copy()
            temp_df["VALUE"] = pd.to_numeric(temp_df["VALUE"], errors="coerce")
            temp_df = temp_df.dropna(subset=["VALUE"])

            if temp_df.empty:
                continue

            temp_row = temp_df.sample(n=1, random_state=42).iloc[0]
            is_abn = is_abnormal_temp(temp_row["VALUE"])

            if is_abn and len(abnormal) < 50:
                target_list = abnormal
                label = 1
            elif not is_abn and len(normal) < 50:
                target_list = normal
                label = 0
            else:
                continue

            non_temp = group[~group["DESCRIPTION"].str.contains("temperature", case=False, na=False)].copy()
            non_temp = non_temp.head(9)  # 1 temp + 9 others = 10 total
            final = pd.concat([pd.DataFrame([temp_row]), non_temp]).sample(frac=1, random_state=len(all_labels))
            final = final.drop(columns=["ID"])
            target_list.append(final)

            all_labels.append({
                "Filename": f"sample_{len(all_labels)+1:03d}.csv",
                "abnormal": label
            })

        offset += batch_size
        print(f"[INFO] Offset {offset} | Normal: {len(normal)} | Abnormal: {len(abnormal)}")

    conn.close()

    # Save files
    for i, df in enumerate(normal + abnormal):
        df.to_csv(output_dir / f"sample_{i+1:03d}.csv", index=False)

    pd.DataFrame(all_labels).to_csv(label_file, index=False)
    print(f"[INFO] Saved {len(all_labels)} samples and labels to: {output_dir} / {label_file}")

if __name__ == "__main__":
    generate_single_temp_per_sample()
