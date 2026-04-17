import pandas as pd
from tqdm import tqdm
from pathlib import Path

from config import LLM_CONFIG
from memory_store import MemoryStore
from tools import FUNCTION_MAP
from qa_agent import QAAgent


COP_TO_LABEL = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    "0": "A",
    "1": "B",
    "2": "C",
    "3": "D",
}


def extract_option_label(text: str) -> str:
    text = str(text).strip().upper()

    if text in {"A", "B", "C", "D"}:
        return text

    for ch in ["A", "B", "C", "D"]:
        if text.startswith(ch):
            return ch
        
        if f" {ch}" in text:
            return ch

    return ""


def build_mcq_question(row) -> str:
    question_text = str(row.get("question", "")).strip()

    options = [
        f"A. {str(row.get('opa', ''))}",
        f"B. {str(row.get('opb', ''))}",
        f"C. {str(row.get('opc', ''))}",
        f"D. {str(row.get('opd', ''))}",
    ]

    return question_text + "\n" + "\n".join(options)


def run_from_csv(agent, input_csv):
    input_path = Path(input_csv)

    category_from_path = input_path.parts[-3]
    dataset_name = input_path.parts[-2]
    file_name = input_path.stem

    config_category = LLM_CONFIG.get("kb_category", "general")

    output_dir = Path("results") / config_category
    output_dir.mkdir(parents = True, exist_ok = True)

    output_csv = output_dir / f"{dataset_name}_{file_name}.csv"

    df = pd.read_csv(input_csv)

    required_cols = {"question", "opa", "opb", "opc", "opd", "cop", "exp"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

    results = []

    for _, row in tqdm(df.iterrows(), total = len(df), desc = f"Running QA Agent on {dataset_name}"):
        full_question = build_mcq_question(row)

        result = agent.answer_question(full_question)

        raw_predicted_answer = result["answer"]
        predicted_answer = extract_option_label(raw_predicted_answer)
        predicted_reasoning = result["thought"]

        correct_answer_index = row.get("cop", "")
        correct_answer = COP_TO_LABEL.get(correct_answer_index, "")
        correct_reasoning = row.get("exp", "")

        is_correct = ""

        if correct_answer != "":
            is_correct = predicted_answer == correct_answer

        row_dict = row.to_dict()

        row_dict.update({
            "question_with_options": full_question,

            "correct_answer_index": correct_answer_index,
            "correct_answer": correct_answer,
            "correct_reasoning": correct_reasoning,
            "subject_name": row.get("subject_name", ""),
            "topic_name": row.get("topic_name", ""),

            "predicted_answer_raw": raw_predicted_answer,
            "predicted_answer": predicted_answer,
            "predicted_reasoning": predicted_reasoning,

            "plan": str(result["plan"]),
            "knowledge": result["knowledge"],
            "examples": result["examples"],
            "tool_result": result["tool_result"],

            "is_correct": is_correct
        })

        results.append(row_dict)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index = False)

    print(f"\nSaved results to: {output_csv}")
    return result_df


def main():
    config_category = LLM_CONFIG.get("kb_category", "general")
    input_csv_name = LLM_CONFIG.get("input_csv_name", "train.csv")

    memory_file = f"memory_{config_category}.json"

    memory_store = MemoryStore(
        path = memory_file,
        max_size = LLM_CONFIG.get("memory_max_size", 100)
    )

    agent = QAAgent(
        config = LLM_CONFIG,
        memory_store = memory_store,
        function_map = FUNCTION_MAP,
        num_shots = LLM_CONFIG.get("example_top_k", 3),
    )

    dataset_root = Path("datasets") / config_category

    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    dataset_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]

    if not dataset_dirs:
        raise ValueError(f"No datasets found in {dataset_root}")

    for dataset_dir in dataset_dirs:
        input_csv = dataset_dir / input_csv_name

        if not input_csv.exists():
            print(f"Skipping {dataset_dir.name}: missing {input_csv_name}")
            continue

        run_from_csv(agent, input_csv)

    print(f"Processing complete for category: {config_category}")
    print(f"Memory file used: {memory_file}")


if __name__ == "__main__":
    main()