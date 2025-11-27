import pandas as pd
from pathlib import Path
from tqdm import tqdm


def parse_age(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except ValueError:
        val = str(value).strip().replace("<", "").replace(">", "").replace("+", "")
        try:
            return float(val)
        except ValueError:
            return None


def generate_filter_labels_eicu():
    current_dir = Path(__file__).resolve().parent
    input_dir = current_dir / "Filter"
    output_dir = current_dir / "Label"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "filter.csv"

    label_rows = []

    for file in tqdm(sorted(input_dir.glob("sample_*.csv")), desc="Generating filter labels"):
        df = pd.read_csv(file)

        # Standardize fields
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()
        df["ethnicity"] = df["ethnicity"].astype(str).str.strip().str.lower()
        df["hospitaldischargestatus"] = df["hospitaldischargestatus"].astype(str).str.strip().str.lower()
        df["parsed_age"] = df["age"].apply(parse_age)

        # 1. Female patients
        eq_ids = df[df["gender"] == "female"]["patientunitstayid"].astype(str).tolist()

        # 2. Female American patients older than 86 years AND expired
        greater_ids = df[
            (df["gender"] == "female") &
            (df["parsed_age"] > 60) &
            (df["ethnicity"].str.contains("american", na=False)) &
            (df["hospitaldischargestatus"] == "expired")
        ]["patientunitstayid"].astype(str).tolist()

        # 3. Caucasian male patients younger than 60 years AND still alive
        less_ids = df[
            (df["gender"] == "male") &
            (df["parsed_age"] < 90) &
            (df["ethnicity"] == "caucasian") &
            (df["hospitaldischargestatus"] == "alive")
        ]["patientunitstayid"].astype(str).tolist()

        label_rows.append({
            "Filename": file.name,
            "filter_eq": ",".join(eq_ids) if eq_ids else "NULL",
            "filter_greater": ",".join(greater_ids) if greater_ids else "NULL",
            "filter_less": ",".join(less_ids) if less_ids else "NULL"
        })

    pd.DataFrame(label_rows).to_csv(
        output_file,
        index=False,
        quoting=1  # csv.QUOTE_NONNUMERIC: ensure all non-numeric values are quoted
    )

    print(f"[INFO] Saved filter label file to: {output_file}")


if __name__ == "__main__":
    generate_filter_labels_eicu()
