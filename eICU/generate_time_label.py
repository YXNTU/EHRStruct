import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

def generate_time_labels_discharge_year():
    current_dir = Path(__file__).resolve().parent
    input_dir = current_dir / "time"
    output_dir = current_dir / "Label"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "time.csv"

    label_rows = []

    for file in tqdm(sorted(input_dir.glob("sample_*.csv")), desc="Generating time labels (discharge year)"):
        df = pd.read_csv(file)

        df = df[["patientunitstayid", "hospitaldischargeyear"]].rename(columns={"patientunitstayid": "Id"})
        df["hospitaldischargeyear"] = pd.to_numeric(df["hospitaldischargeyear"], errors="coerce")
        df = df.dropna()

        if len(df) < 2:
            print(f"[SKIP] {file.name}: less than 2 valid rows.")
            continue

        # 默认标记是否找到了不同 discharge year 的两个病人
        found = False
        for _ in range(50):
            selected = df.sample(n=2, random_state=random.randint(0, 100000))
            y1 = selected.iloc[0]["hospitaldischargeyear"]
            y2 = selected.iloc[1]["hospitaldischargeyear"]
            if y1 != y2:
                found = True
                break

        if not found:
            # fallback: 取前两个
            selected = df.iloc[:2]
            y1 = selected.iloc[0]["hospitaldischargeyear"]
            y2 = selected.iloc[1]["hospitaldischargeyear"]

        p1 = selected.iloc[0]["Id"]
        p2 = selected.iloc[1]["Id"]
        diff = int(abs(y2 - y1))
        greater = int(y1 < y2)

        label_rows.append({
            "File": file.name,
            "PATIENT1": p1,
            "PATIENT2": p2,
            "greater": greater,
            "diff": diff
        })

    pd.DataFrame(label_rows).to_csv(output_file, index=False)
    print(f"[INFO] Saved label file with {len(label_rows)} rows to: {output_file}")

if __name__ == "__main__":
    generate_time_labels_discharge_year()
