import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
import re
import glob
import contextlib
import time
from typing import Dict, Any, List
from LLMCaller import LLMCaller
from sklearn.metrics import roc_auc_score

LLM_MAP = {"Qwen7B": "Qwen/Qwen2.5-7B-Instruct","Qwen14B": "Qwen/Qwen2.5-14B-Instruct","Qwen32B": "Qwen/Qwen2.5-32B-Instruct","Qwen72B": "Qwen/Qwen2.5-72B-Instruct","deepseekV2.5": "deepseek-ai/DeepSeek-V2.5","deepseekV3": "deepseek-ai/DeepSeek-V3",}
TASK_DIR_MAP = {"filter": "filter","aggregation": "aggregation","arithmetic": "arithmetic","snomed": "snomed","death": "death","disorder": "disorder","medications": "medications",}
SINGLE_TASK_MAPPING = {"snomed": {"label_suffix": "K-U1", "report_label": "K-U1"},"death": {"label_suffix": "K-R1", "report_label": "K-R1"},"disorder": {"label_suffix": "K-R2", "report_label": "K-R2"},"medications": {"label_suffix": "K-R3", "report_label": "K-R3"},}
POST_INSTRUCTION_MAP = {
    "filter": (
        f"\nReturn only the list of IDs for each filter, separated by commas. Use NULL if no result.\n"
        f"filter1: ID1,ID2,...\n"
        f"filter2: ID3,ID4,...\n"
        f"Do NOT explain. Only return the lines above."
    ),
    "aggregation": (
        f"\nReturn exactly three numbers, separated by commas: Agg1, Agg2, Agg3\n"
        f"For example: -10, 50.5, -200\n"
        f"Do NOT explain. Do NOT add extra text. Only return the numbers."
    ),
    "arithmetic": (
        f"\nReturn exactly two numbers, separated by a comma: Diff, Add\n"
        f"For example: -13725.74, 2967017.62\n"
        f"Do NOT explain your reasoning. Do NOT add extra text after the numbers."
    ),
    "snomed": (
        f"\nReturn in the format:\nDiabetes: <0 or 1>\nDo NOT explain. Only return the prediction line."
    ),
    "death": (
        f"\nReturn in the format:\nDeath: <0 or 1>\nDo NOT explain. Only return the prediction line."
    ),
    "disorder": (
        f"\nReturn in the format:\nDisorder: <0 or 1>\nDo NOT explain. Only return the prediction line."
    ),
    "medications": (
        f"\nReturn in the format:\nRecommend: <0 or 1>\nDo NOT explain. Only return the prediction line."
    ),
}


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def format_input(df, sample_id, input_type):
    if input_type == "txt" or input_type == "sgen":
        return df.to_csv(sep="\t", index=False)
    elif input_type == "latex":
        return df.to_latex(index=False)
    elif input_type == "hyper":
        return format_hypergraph(df, sample_id)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")


def format_hypergraph(df, sample_id):
    lines = []
    for _, row in df.iterrows():
        edge = " | ".join([f"{col}:{row[col]}" for col in df.columns])
        lines.append(f"[{sample_id}] {edge}")
    return "\n".join(lines)


def compute_accuracy(pred_str: str, gt_str: str) -> int:
    def normalize_list(s: str) -> str:
        if pd.isna(s) or s.strip().upper() == "NULL" or s.strip() == "":
            return "NULL"
        ids = [i.strip() for i in s.upper().replace(" ", "").split(",") if i.strip()]
        return ",".join(sorted(list(set(ids))))
    pred_norm = normalize_list(pred_str)
    gt_norm = normalize_list(gt_str)
    return int(pred_norm == gt_norm)


def build_instruction(df: pd.DataFrame, task_queries: List[str], task: str, input_type: str, sample_id: str) -> str:
    table_str = format_input(df, sample_id=sample_id, input_type=input_type)
    if input_type == "sgen":
        pre_instruction = (
            f"You are given a patient table. "
            f"First, briefly understand the table in natural language: how many rows it has, what columns are included,"
            f"and what kind of values are recorded. "
            f"Then, based on the table, answer the following questions: "
            f"Table:\n{table_str}\n\n"
        )
    else:
        pre_instruction = (
            f"You are given a patient table. based on the table, answer the following questions: "
            f"Table:\n{table_str}\n\n"
        )
    query_parts = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(task_queries)])
    query_section = f"Questions:\n{query_parts}\n"
    post_instruction = POST_INSTRUCTION_MAP.get(task, f"Do NOT explain. Return your answer in the requested format.")
    return pre_instruction + query_section + post_instruction


parser = argparse.ArgumentParser(description='LLM Evaluation Framework for New Logic Tasks')
parser.add_argument('--llm', type=str, required=True, help='LLM alias name.')
parser.add_argument('--task', type=str, required=True,
                    choices=list(TASK_DIR_MAP.keys()),
                    help='Name of the task to evaluate.')
parser.add_argument('--type', type=str, default="txt",
                    choices=["txt", "latex", "hyper", "sgen"], help="Input formatting type for prompt content.")
parser.add_argument('--k', type=int, default=0, help='Number of few-shot examples (0 for zero-shot)')
args = parser.parse_args()

if args.task == "filter":
    llm_client = LLMCaller(llm=args.llm, max_tokens=100)
else:
    llm_client = LLMCaller(llm=args.llm)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_root = os.path.join(project_root, "data")
task_subdir = TASK_DIR_MAP[args.task]
task_dir = os.path.join(data_root, task_subdir)
output_dir = os.path.join(os.path.dirname(__file__), "results", args.task)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{args.llm}_{args.task}_{args.type}_{args.k}_shot_output.txt")

with open(output_file, "w", encoding="utf-8") as f, contextlib.redirect_stdout(Tee(sys.stdout, f)):
    print(
        f"⚡️ Command: python run_task_framework.py --llm {args.llm} --task {args.task} --type {args.type} --k {args.k}")
    print(f"## ⚙️ Configuration")
    print(f"* 🤖 LLM: **{args.llm}**")
    print(f"* 📝 Task: **{args.task}**")
    print(f"* 📄 Input Type: **{args.type}**")
    print(f"* 🔢 Few-shot K: **{args.k}**")
    print(f"* 📁 Data Directory: {task_dir}")
    print("---")

    is_filter_task = args.task == "filter"
    is_aggregation_task = args.task == "aggregation"
    is_arithmetic_task = args.task == "arithmetic"
    is_single_binary_task = args.task in ["snomed", "death", "disorder", "medications"]

    if is_filter_task:
        try:
            label_path1 = os.path.join(data_root, "filter", f"query_answer_D-U1.csv")
            label_df1 = pd.read_csv(label_path1, usecols=["file name", "answer", "query"]).rename(columns={"answer": "gt_filter1", "query": "query_filter1"})
            label_path2 = os.path.join(data_root, "filter", f"query_answer_D-U2.csv")
            label_df2 = pd.read_csv(label_path2, usecols=["file name", "answer", "query"]).rename(columns={"answer": "gt_filter2", "query": "query_filter2"})
            label_df = pd.merge(label_df1, label_df2, on="file name", how="inner")
            total_labels = len(label_df)
            if total_labels == 0:
                print(f"❌ ERROR: Loaded labels are empty after merging D-U1 and D-U2 data.")
                sys.exit(1)
            print(f"✅ Loaded and merged labels for D-U1 and D-U2, total {total_labels} entries.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load and merge label files for filter task: {e}")
            sys.exit(1)

    elif is_aggregation_task:
        try:
            label_path1 = os.path.join(data_root, "aggregation", f"query_answer_D-R1.csv")
            label_df1 = pd.read_csv(label_path1, usecols=["file name", "answer", "query"]).rename(columns={"answer": "gt_agg1", "query": "query_agg1"})
            label_path2 = os.path.join(data_root, "aggregation", f"query_answer_D-R2.csv")
            label_df2 = pd.read_csv(label_path2, usecols=["file name", "answer", "query"]).rename(columns={"answer": "gt_agg2", "query": "query_agg2"})
            label_path3 = os.path.join(data_root, "aggregation", f"query_answer_D-R3.csv")
            label_df3 = pd.read_csv(label_path3, usecols=["file name", "answer", "query"]).rename(columns={"answer": "gt_agg3", "query": "query_agg3"})
            label_df = pd.merge(pd.merge(label_df1, label_df2, on="file name", how="inner"), label_df3, on="file name", how="inner")
            total_labels = len(label_df)
            if total_labels == 0:
                print(f"❌ ERROR: Loaded labels are empty after merging D-R1, D-R2, and D-R3 data.")
                sys.exit(1)
            print(f"✅ Loaded and merged labels for D-R1, D-R2, and D-R3, total {total_labels} entries.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load and merge label files for aggregation task: {e}")
            sys.exit(1)

    elif is_arithmetic_task:
        try:
            label_path1 = os.path.join(data_root, "arithmetic", f"query_answer_D-R4.csv")
            label_df1 = pd.read_csv(label_path1, usecols=["file name", "answer", "query"]).rename(columns={"answer": "gt_diff", "query": "query_diff"})
            label_path2 = os.path.join(data_root, "arithmetic", f"query_answer_D-R5.csv")
            label_df2 = pd.read_csv(label_path2, usecols=["file name", "answer", "query"]).rename(columns={"answer": "gt_add", "query": "query_add"})
            label_df = pd.merge(label_df1, label_df2, on="file name", how="inner")
            total_labels = len(label_df)
            if total_labels == 0:
                print(f"❌ ERROR: Loaded labels are empty after merging D-R4 and D-R5 data.")
                sys.exit(1)
            print(f"✅ Loaded and merged labels for D-R4 and D-R5, total {total_labels} entries.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load and merge label files for arithmetic task: {e}")
            sys.exit(1)

    elif is_single_binary_task:
        try:
            task_suffix = SINGLE_TASK_MAPPING[args.task]["label_suffix"]
            label_filename = f"query_answer_{task_suffix}.csv"
            label_path = os.path.join(task_dir, label_filename)
            label_df = pd.read_csv(label_path)
            if not all(col in label_df.columns for col in ["file name", "query", "answer"]):
                print(f"❌ ERROR: Label file {label_filename} missing required columns (file name, query, answer).")
                sys.exit(1)
            label_df = label_df.rename(columns={"answer": "gt_answer", "query": "task_query"})
            total_labels = len(label_df)
            print(f"✅ Loaded labels from {label_filename}, total {total_labels} entries.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load label file {label_path}: {e}")
            sys.exit(1)
    else:
        print(f"--- ⚠️ Task {args.task} not recognized in loading phase. ---")
        sys.exit(1)

    print("\n## 🔍 Starting Evaluation Loop")

    if args.task == "filter":
        correct_filter1 = 0
        correct_filter2 = 0
        total_samples = len(label_df)
        sub_task_labels = ["D-U1_Filter", "D-U2_Filter"]

        for i, row in tqdm(label_df.iterrows(), total=total_samples, desc=f"🔍 {args.task} Evaluation (D-U1/D-U2)"):
            filename = row["file name"]
            file_path = os.path.join(task_dir, filename)

            if not os.path.exists(file_path):
                continue

            try:
                df_sample = pd.read_csv(file_path)
                gt_filter1 = str(row["gt_filter1"]).strip().replace(" ", "") or "NULL"
                gt_filter2 = str(row["gt_filter2"]).strip().replace(" ", "") or "NULL"
                task_queries = [row["query_filter1"], row["query_filter2"]]

                full_instruction = build_instruction(
                    df=df_sample, task_queries=task_queries, task=args.task, input_type=args.type,
                    sample_id=filename.replace(".csv", "")
                )

                while True:
                    try:
                        response = llm_client.query(full_instruction)
                        print(f"[{filename}] 🤖 LLM Response:\n{response.strip()}")
                        pred_filter1 = "NULL"
                        pred_filter2 = "NULL"
                        for line in response.strip().splitlines():
                            if "filter1:" in line.lower():
                                pred_filter1 = line.split(":", 1)[-1].strip().replace(" ", "") or "NULL"
                            elif "filter2:" in line.lower():
                                pred_filter2 = line.split(":", 1)[-1].strip().replace(" ", "") or "NULL"
                        is_correct1 = compute_accuracy(pred_filter1, gt_filter1)
                        is_correct2 = compute_accuracy(pred_filter2, gt_filter2)
                        correct_filter1 += is_correct1
                        correct_filter2 += is_correct2
                        break
                    except Exception as e:
                        print(f"[❌ ERROR] {filename}: {e}")
                        if "429" in str(e) or "503" in str(e):
                            print("[⏸️ INFO] Rate limit hit. Sleeping 60s and retrying...")
                            time.sleep(60)
                            continue
                        else:
                            break
            except Exception as e:
                print(f"[❌ ERROR] Failed to process {filename}: {e}")
                continue

        print(f"\n== Task {args.task} Evaluation (Accuracy) ==")
        print(f"📊 {sub_task_labels[0]} Correct: {correct_filter1} / {total_samples}")
        print(f"📊 {sub_task_labels[1]} Correct: {correct_filter2} / {total_samples}")
        if total_samples > 0:
            print(f"🏆 {sub_task_labels[0]} Accuracy: {correct_filter1 / total_samples:.2f}")
            print(f"🏆 {sub_task_labels[1]} Accuracy: {correct_filter2 / total_samples:.2f}")

    elif args.task == "aggregation":
        correct_agg1 = 0
        correct_agg2 = 0
        correct_agg3 = 0
        total_samples = len(label_df)
        sub_task_labels = ["D-R1_Agg", "D-R2_Agg", "D-R3_Agg"]

        for i, row in tqdm(label_df.iterrows(), total=total_samples, desc=f"📈 {args.task} Evaluation (D-R1/R2/R3)"):
            filename = row["file name"]
            file_path = os.path.join(task_dir, filename)

            if not os.path.exists(file_path):
                continue

            try:
                df_sample = pd.read_csv(file_path)
                gt_agg1 = float(str(row["gt_agg1"]).strip())
                gt_agg2 = float(str(row["gt_agg2"]).strip())
                gt_agg3 = float(str(row["gt_agg3"]).strip())
                task_queries = [row["query_agg1"], row["query_agg2"], row["query_agg3"]]

                full_instruction = build_instruction(
                    df=df_sample, task_queries=task_queries, task=args.task, input_type=args.type,
                    sample_id=filename.replace(".csv", "")
                )

                while True:
                    try:
                        response = llm_client.query(full_instruction)
                        print(f"[{filename}] 🤖 LLM Response: {response.strip()}")
                        numbers = [s.strip() for s in response.strip().split(",")]

                        if len(numbers) < 3:
                            print(f"[⚠️ WARN] Bad response format: Expected 3 numbers, found {len(numbers)} in {response}")
                            break

                        pred_agg1 = float(numbers[0])
                        pred_agg2 = float(numbers[1])
                        pred_agg3 = float(numbers[2])

                        is_correct1 = int(abs(pred_agg1 - gt_agg1) < 1e-4)
                        is_correct2 = int(abs(pred_agg2 - gt_agg2) < 1e-4)
                        is_correct3 = int(abs(pred_agg3 - gt_agg3) < 1e-4)

                        correct_agg1 += is_correct1
                        correct_agg2 += is_correct2
                        correct_agg3 += is_correct3

                        break

                    except Exception as e:
                        print(f"[❌ ERROR] {filename}: {e}")
                        if "429" in str(e) or "503" in str(e):
                            print("[⏸️ INFO] Rate limit hit. Sleeping 60s and retrying...")
                            time.sleep(60)
                            continue
                        else:
                            break

            except Exception as e:
                print(f"[❌ ERROR] Failed to process {filename}: {e}")
                continue

        print(f"\n== Task {args.task} Evaluation (Accuracy) ==")
        print(f"📊 {sub_task_labels[0]} Correct: {correct_agg1} / {total_samples}")
        print(f"📊 {sub_task_labels[1]} Correct: {correct_agg2} / {total_samples}")
        print(f"📊 {sub_task_labels[2]} Correct: {correct_agg3} / {total_samples}")
        if total_samples > 0:
            print(f"🏆 {sub_task_labels[0]} Accuracy: {correct_agg1 / total_samples:.2f}")
            print(f"🏆 {sub_task_labels[1]} Accuracy: {correct_agg2 / total_samples:.2f}")
            print(f"🏆 {sub_task_labels[2]} Accuracy: {correct_agg3 / total_samples:.2f}")

    elif args.task == "arithmetic":
        correct_diff = 0
        correct_add = 0
        total_samples = len(label_df)

        sub_task_labels = ["D-R4_Diff", "D-R5_Add"]

        for i, row in tqdm(label_df.iterrows(), total=total_samples, desc=f"🧮 {args.task} Evaluation (D-R4/D-R5)"):
            filename = row["file name"]
            file_path = os.path.join(task_dir, filename)

            if not os.path.exists(file_path):
                continue

            try:
                df_sample = pd.read_csv(file_path)
                gt_diff = float(str(row["gt_diff"]).strip())
                gt_add = float(str(row["gt_add"]).strip())
                task_queries = [row["query_diff"], row["query_add"]]
                sample_id = filename.replace(".csv", "")

                full_instruction = build_instruction(
                    df=df_sample, task_queries=task_queries, task=args.task, input_type=args.type,
                    sample_id=sample_id
                )

                while True:
                    try:
                        response = llm_client.query(full_instruction)
                        print(f"[{filename}] 🤖 LLM Response: {response.strip()}")
                        numbers = [s.strip() for s in response.strip().split(",")]

                        if len(numbers) != 2:
                            print(
                                f"[⚠️ WARN] Bad response format: Expected 2 numbers, found {len(numbers)} in {response}")
                            break

                        pred_diff = round(float(numbers[0]), 2)
                        pred_add = round(float(numbers[1]), 2)

                        gt_diff_rounded = round(gt_diff, 2)
                        gt_add_rounded = round(gt_add, 2)

                        correct_diff += int(abs(pred_diff - gt_diff_rounded) < 1e-2)
                        correct_add += int(abs(pred_add - gt_add_rounded) < 1e-2)

                        break

                    except Exception as e:
                        print(f"[❌ ERROR] {filename}: {e}")
                        if "429" in str(e) or "503" in str(e):
                            print("[⏸️ INFO] Rate limit hit. Sleeping 60s and retrying...")
                            time.sleep(60)
                            continue
                        else:
                            break

            except Exception as e:
                print(f"[❌ ERROR] Failed to process {filename}: {e}")
                continue

        print(f"\n== Task {args.task} Evaluation (Accuracy) ==")
        print(f"📊 {sub_task_labels[0]} Correct: {correct_diff} / {total_samples}")
        print(f"📊 {sub_task_labels[1]} Correct: {correct_add} / {total_samples}")
        if total_samples > 0:
            print(f"🏆 {sub_task_labels[0]} Accuracy: {correct_diff / total_samples:.2f}")
            print(f"🏆 {sub_task_labels[1]} Accuracy: {correct_add / total_samples:.2f}")

    elif is_single_binary_task:
        y_true = []
        y_pred = []
        total_samples = len(label_df)

        LABEL_MAPPING = {
            "snomed": "K-U1",
            "death": "K-R1",
            "disorder": "K-R2",
            "medications": "K-R3",
        }

        task_label_prefix = LABEL_MAPPING.get(args.task, f"{args.task.upper()}")

        if args.task == "snomed":
            match_key = "diabetes"
        elif args.task == "death":
            match_key = "death"
        elif args.task == "disorder":
            match_key = "disorder"
        elif args.task == "medications":
            match_key = "recommend"
        else:
            match_key = ""

        for i, row in tqdm(label_df.iterrows(), total=total_samples, desc=f"🧠 {args.task} Evaluation (AUC)"):
            filename = row["file name"]
            file_path = os.path.join(task_dir, filename)

            if not os.path.exists(file_path):
                continue

            try:
                df_sample = pd.read_csv(file_path)
                gt_answer = int(str(row["gt_answer"]).strip())
                task_queries = [row["task_query"]]
                sample_id = filename.replace(".csv", "")

                full_instruction = build_instruction(
                    df=df_sample, task_queries=task_queries, task=args.task, input_type=args.type,
                    sample_id=sample_id
                )

                while True:
                    try:
                        response = llm_client.query(full_instruction)
                        print(f"[{filename}] 🤖 Response:\n{response.strip()}")

                        pred = None
                        for line in response.strip().splitlines():
                            if match_key in line.lower():
                                match = re.findall(r"\b[01]\b", line)
                                if match:
                                    pred = int(match[0])
                                    break

                        if pred is None:
                            print(f"[⚠️ WARN] No valid prediction found for {filename}")
                            break

                        y_true.append(gt_answer)
                        y_pred.append(pred)

                        break

                    except Exception as e:
                        print(f"[❌ ERROR] LLM failed on {filename}: {e}")
                        if "429" in str(e) or "503" in str(e):
                            print("[⏸️ INFO] Rate limit hit. Sleeping 60 seconds before retrying...")
                            time.sleep(60)
                            continue
                        else:
                            break

            except Exception as e:
                print(f"[❌ ERROR] Failed to process {filename}: {e}")
                continue

        print(f"\n== Task {args.task} Evaluation (AUC) ==")

        if len(y_true) == 0:
            print("⚠️ AUC skipped: No valid samples processed.")
        elif len(set(y_true)) < 2:
            print(f"⚠️ AUC skipped: only one class present in ground truth ({task_label_prefix}).")
        else:
            auc = roc_auc_score(y_true, y_pred)
            print(f"🏆 {task_label_prefix} AUC = {auc:.3f}")

    else:
        print(f"--- ⚠️ Task {args.task} not recognized. ---")