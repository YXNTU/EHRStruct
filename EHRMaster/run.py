import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
import re
import contextlib
import time
from typing import Dict, Any, List
from LLMCaller import LLMCaller

LLM_MAP = {"Qwen7B": "Qwen/Qwen2.5-7B-Instruct", "Qwen14B": "Qwen/Qwen2.5-14B-Instruct",
           "Qwen32B": "Qwen/Qwen2.5-32B-Instruct", "Qwen72B": "Qwen/Qwen2.5-72B-Instruct",
           "deepseekV2.5": "deepseek-ai/DeepSeek-V2.5", "deepseekV3": "deepseek-ai/DeepSeek-V3", }
TASK_DIR_MAP = {"D-U1": "filter", "D-U2": "filter", "D-R1": "aggregation", "D-R2": "aggregation", "D-R3": "aggregation",
                "D-R4": "arithmetic", "D-R5": "arithmetic", }


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


def jaccard_score(pred_list: list, gt_str: str) -> float:
    if pd.isna(gt_str) or gt_str.strip().lower() == "nan" or gt_str.strip() == "":
        gt_str = "NULL"

    if isinstance(pred_list, list) and len(pred_list) == 0:
        pred_list = "NULL"

    if pred_list == "NULL" and gt_str == "NULL":
        return 1.0
    if pred_list == "NULL" or gt_str == "NULL":
        return 0.0

    if isinstance(pred_list, str):
        pred_list = [s.strip() for s in pred_list.split(",") if s.strip()]

    gt_list = [s.strip() for s in gt_str.split(",") if s.strip()]

    if not isinstance(pred_list, list):
        pred_list = []

    pred_set = set(pred_list)
    gt_set = set(gt_list)

    intersection = pred_set & gt_set
    union = pred_set | gt_set
    return len(intersection) / len(union) if union else 1.0


def extract_task_logic(instruction: str, task_type: str, llm_client) -> str:
    prompt = f"""
    You are a table question analyzer.

    Given a question, describe its logic as a short natural language summary that includes:
    - The goal: what field or result the user wants
    - The conditions: what constraints the data must satisfy (e.g., gender is female, race is white)

    Use full sentences in English. Keep it short and clear.

    Example:
    Question: What is the maximum pain score for patient P001?
    Output: Find the maximum value of pain score for records where patient is P001.

    Now analyze this question:
    {instruction}
    Output:
    """
    response = llm_client.query(prompt)
    return response.strip()


def map_logic_with_table_str(logic_str: str, table_str: str, llm_client) -> str:
    prompt = f"""
    You are a table-aware logic mapper.

    You are given:
    - A task logic string in natural language (logic_str)
    - A TSV-format table (table_str) with real column names and sample values

    Your job:
    - Carefully read the table's column names and values
    - Replace in the logic_str any approximate or synonymous column names or values
      with the **exact column names or values** that appear in the table
    - ⭐️ You MUST replace phrases like "patient ID" or "identifier" with the actual column name from the table (e.g., 'Id')
    - ⭐️ Ensure that every field name and value you write exists in the table
    - ⭐️ If no match is found for a value or field, leave it unchanged and do NOT guess
    - Do NOT change structural keywords like "return", "filter", etc.

    Table (TSV):
    {table_str}

    Original logic:
    {logic_str}

    Return the rewritten logic using **only** table column names and values:
    """
    response = llm_client.query(prompt)
    return response.strip()


def generate_and_run_code(map_logic_str: str, df_sample, llm_client, instruction: str, task_type: str) -> Any:
    if task_type.startswith("D-U"):
        return_focus_hint = (
            "- For Filter tasks (D-U*), the final 'result' MUST be a list or Series containing "
            "ONLY the **patient identifier values** from the 'Id' column (e.g., ['P001', 'P002', ...]). "
            "Do NOT return the entire filtered DataFrame."
        )
    else:
        return_focus_hint = (
            "- For Aggregation/Arithmetic tasks (D-R*), the final 'result' MUST be a single numerical value (float or int)."
        )

    return_hint = (
        f"- Carefully analyze the instruction and determine which value(s) should be returned.\n"
        f"{return_focus_hint}\n"
        "- You will be provided the table header for reference.\n"
    )
    table_columns = ", ".join(df_sample.columns)

    prompt = f'''
    You are a Python coding assistant.

    You are given:
    - A natural language instruction describing the goal
    - A structured task logic in natural language, already matched to column names
    - A pandas DataFrame named `df`, already loaded in memory
    - The DataFrame has the following columns: {table_columns}

    Your job:
    - Write Python code that completes the task using the DataFrame
    - Do NOT include import statements
    - Do NOT redefine or reload the DataFrame
    - Use only the DataFrame variable named `df`
    - Assign the final result to a variable named `result`
    {return_hint}
    - Only use column names that exist in the DataFrame
    - Do NOT include any explanation or markdown (e.g., no ```python)

    Original instruction:
    {instruction}

    Mapped logic:
    {map_logic_str}

    Now write the code:
    '''

    response = llm_client.query(prompt)
    code = response.strip()

    if code.startswith("```"):
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()
    if code.lower().startswith("python"):
        code = code[len("python"):].strip()

    if not code or "result" not in code:
        print("[WARN] LLM returned invalid or incomplete code. Skipping execution.")
        return f"# Code skipped:\n{code}"

    print("Generated code:\n", code)

    local_vars = {}
    global_vars = {"df": df_sample, "pd": pd}
    try:
        exec(code, global_vars, local_vars)
        result = local_vars.get("result", [])

        if isinstance(result, pd.Series):
            result = result.tolist()

        if isinstance(result, (int, float)):
            return str(result)

        if isinstance(result, list) and len(result) == 0:
            return "NULL"

        return result
    except Exception as e:
        print(f"[ERROR] Failed to run generated code.\nReason: {e}\nCode:\n{code}")
        return f"# ERROR: {str(e)}\n{code}"


GT_COLUMN_MAP = {
    "D-U1": "answer",
    "D-U2": "answer",
    "D-R1": "answer",
    "D-R2": "answer",
    "D-R3": "answer",
    "D-R4": "answer",
    "D-R5": "answer",
}

parser = argparse.ArgumentParser(description='LLM Evaluation Framework for Sub-Tasks with SEMaster Logic')
parser.add_argument('--llm', type=str, required=True, help='LLM alias name.')
parser.add_argument('--task', type=str, required=True,
                    choices=list(TASK_DIR_MAP.keys()),
                    help='Name of the sub-task to evaluate (D-U1, D-U2, D-R1-R5).')
parser.add_argument('--type', type=str, default="txt",
                    choices=["txt"], help='Input formatting type for prompt content (fixed to txt for SEMaster).')
parser.add_argument('--k', type=int, default=0, help='Number of few-shot examples (0 for zero-shot)')
args = parser.parse_args()

if args.task.startswith("D-U"):
    llm_client = LLMCaller(llm=args.llm, max_tokens=200)
else:
    llm_client = LLMCaller(llm=args.llm, max_tokens=200)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_root = os.path.join(project_root, "data")
task_subdir = TASK_DIR_MAP[args.task]
task_dir = os.path.join(data_root, task_subdir)

label_filename = f"query_answer_{args.task}.csv"
label_path = os.path.join(data_root, task_subdir, label_filename)

try:
    label_df = pd.read_csv(label_path)
    if not all(col in label_df.columns for col in ["file name", "query", "answer"]):
        print(f"❌ ERROR: Label file {label_filename} missing required columns (file name, query, answer).")
        sys.exit(1)

    label_df = label_df.rename(columns={"file name": "file_name", "query": "task_query", "answer": "gt_answer"})
    total_labels = len(label_df)
    print(f"✅ Loaded labels for {args.task} from {label_path}, total {total_labels} entries.")
except Exception as e:
    print(f"❌ ERROR: Failed to load label file {label_path}: {e}")
    sys.exit(1)

output_dir = os.path.join(os.path.dirname(__file__), "results", args.task)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{args.llm}_{args.task}_semaster_output.txt")

with open(output_file, "w", encoding="utf-8") as f, contextlib.redirect_stdout(Tee(sys.stdout, f)):
    print(
        f"⚡️ Command: python run_subtask_semaster.py --llm {args.llm} --task {args.task} --type {args.type} --k {args.k}")
    print(f"## ⚙️ Configuration")
    print(f"* 🤖 LLM: **{args.llm}**")
    print(f"* 📝 Sub-Task: **{args.task}** (Main Task: {task_subdir})")
    print(f"* 🧠 Logic: **SEMaster (3-Step)**")
    print(f"* 📄 Input Type: **{args.type}**")
    print("---")

    correct_count = 0

    is_filter_task = args.task.startswith("D-U")
    is_aggregation_task = args.task in ["D-R1", "D-R2", "D-R3"]
    is_arithmetic_task = args.task in ["D-R4", "D-R5"]

    for i, row in tqdm(label_df.iterrows(), total=total_labels, desc=f"🔍 {args.task} Evaluation"):
        filename = row["file_name"]
        file_path = os.path.join(task_dir, filename)

        if not os.path.exists(file_path):
            print(f"[SKIP] Missing file: {file_path}")
            continue

        try:
            df_sample = pd.read_csv(file_path)
            table_str = df_sample.to_csv(sep="\t", index=False)

            instruction = str(row["task_query"]).strip()

            logic_str = extract_task_logic(instruction, args.task, llm_client)
            print(f"[{filename}] Step 1 - Raw Logic:\n{logic_str}\n")

            map_logic_str = map_logic_with_table_str(logic_str, table_str, llm_client)
            print(f"[{filename}] Step 2 - Mapped Logic:\n{map_logic_str}\n")

            result = generate_and_run_code(map_logic_str, df_sample, llm_client, instruction, args.task)
            print(f"[{filename}] Step 3 - Final Results:\n{result}\n")

            gt_answer = str(row["gt_answer"]).strip()
            match = 0

            if isinstance(result, str) and result.startswith(("# ERROR:", "# Code skipped:")):
                print(f"[❌ EVAL] Code execution failed/skipped. Match: 0")

            elif is_filter_task:
                gt_normalized = gt_answer.replace(" ", "") or "NULL"
                pred_normalized = (",".join(map(str, result)) if isinstance(result, list) else str(result)).replace(" ",
                                                                                                                    "") or "NULL"
                match = int(jaccard_score(pred_normalized, gt_normalized) == 1.0)
                print(f"[{filename}] Pred: {pred_normalized} | GT: {gt_normalized} | Match: {match}")

            elif is_aggregation_task or is_arithmetic_task:
                try:
                    pred_float = float(result)
                    gt_float = float(gt_answer)

                    if is_aggregation_task:
                        match = int(abs(pred_float - gt_float) < 1e-4)
                    elif is_arithmetic_task:
                        pred_rounded = round(pred_float, 2)
                        gt_rounded = round(gt_float, 2)
                        match = int(abs(pred_rounded - gt_rounded) < 1e-2)

                    print(f"[{filename}] Pred: {result} | GT: {gt_answer} | Match: {match}")

                except ValueError:
                    print(f"[⚠️ WARN] Non-numerical output for numerical task. Pred: {result} | GT: {gt_answer}")
                    match = 0

            correct_count += match

        except Exception as e:
            print(f"[FATAL ERROR] {filename}: {e}")

    n = total_labels
    metric_name = "Accuracy"

    print(f"\n== Task {args.task} Evaluation (Metric: {metric_name}) ==")
    print(f"📊 Correct: {correct_count} / {n}")
    if n > 0:
        print(f"🏆 {metric_name}: {correct_count / n:.4f}")