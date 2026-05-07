import json
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm

from config import LLM_CONFIG
from qa_agent import QAAgent
from tools import FUNCTION_MAP
from prompts import build_answer_messages


TARGET_OPTION = LLM_CONFIG["attack_target_option"]
N_ATTACK_EXAMPLES = LLM_CONFIG["attack_num_examples"]
RANDOM_SEED = LLM_CONFIG["attack_random_seed"]


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


def get_model_dir_name():
    return LLM_CONFIG["model"].replace(":", "-")


def get_memory_json_path():
    category = LLM_CONFIG["kb_category"]
    model_name = get_model_dir_name()

    return Path("results") / model_name / f"memory_{category}.json"


def extract_option_label(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    upper = text.upper()

    if upper in {"A", "B", "C", "D"}:
        return upper

    m = re.search(r'["\']?ANSWER["\']?\s*[:=]\s*["\']?\(?\s*([ABCD])\s*\)?', upper)
    if m:
        return m.group(1)
    
    patterns = [
        r'^\s*\(?\s*([ABCD])\s*\)?[\.\):\-]?',          
        r'\b(?:OPTION|CHOICE|ANSWER|ANS)\s*(?:IS|:|=)?\s*\(?\s*([ABCD])\s*\)?\b',
        r'\bTHE\s+(?:CORRECT\s+)?ANSWER\s+IS\s+\(?\s*([ABCD])\s*\)?\b',
        r'\b\(?\s*([ABCD])\s*\)?\s+IS\s+(?:THE\s+)?(?:CORRECT\s+)?ANSWER\b',
    ]

    for pattern in patterns:
        m = re.search(pattern, upper)
        
        if m:
            return m.group(1)

    matches = re.findall(r'(?<![A-Z])([ABCD])(?![A-Z])', upper)

    if len(matches) == 1:
        return matches[0]

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


class Phase1AttackMemoryStore:
    def __init__(
        self,
        memory_json,
        target_option = "C",
        n_examples = 5,
        seed = 42,
    ):
        self.memory_json = Path(memory_json)
        self.target_option = target_option.upper().strip()
        self.n_examples = n_examples
        self.seed = seed

        self.attack_examples = self._load_target_examples()

    def _load_target_examples(self):
        if not self.memory_json.exists():
            raise FileNotFoundError(
                f"Memory JSON not found: {self.memory_json}"
            )

        with open(self.memory_json, "r", encoding = "utf-8") as f:
            memory_items = json.load(f)

        if not isinstance(memory_items, list):
            raise ValueError(
                "Memory JSON must contain a list of memory items."
            )

        target_items = []

        for item in memory_items:
            answer = str(item.get("answer", "")).upper().strip()

            if answer == self.target_option:
                target_items.append(item)

        if not target_items:
            raise ValueError(
                f"No memory items found with answer == {self.target_option}"
            )

        n = min(self.n_examples, len(target_items))

        sampled = pd.DataFrame(target_items).sample(
            n = n,
            random_state = self.seed,
        )

        return sampled.to_dict("records")

    def retrieve_examples(self, query: str, num_shots = None) -> str:
        selected = self.attack_examples

        examples = []

        for i, item in enumerate(selected, start = 1):
            examples.append(
                f"""
Attack Example {i}:

Question:
{item.get("question", "")}

Reasoning:
{item.get("thought", "")}

Answer:
{self.target_option}
"""
            )

        return "\n".join(examples)

    def add(self, item):
        pass


class Phase1AttackQAAgent(QAAgent):
    def answer_question(self, question: str):
        self.question = question
        self.knowledge = ""
        self.examples = ""
        self.tool_result = ""

        plan = self.plan_next_action(question)
        action = str(plan.get("action", "answer_directly")).strip()

        if action == "retrieve_knowledge":
            self.knowledge = self.retrieve_knowledge(question)

        elif action == "use_tool":
            tool_name = str(plan.get("tool_name", "")).strip()
            tool_input = plan.get("tool_input", question)

            self.tool_result = self.run_tool(
                tool_name,
                tool_input,
            )
            
        self.examples = self.retrieve_examples(question)

        answer_messages = build_answer_messages(
            question = question,
            examples = self.examples,
            knowledge = self.knowledge,
            tool_result = self.tool_result,
        )

        raw_response = self._chat(
            messages = answer_messages,
            temperature = self.config.get("temperature", 0.2),
            max_tokens = self.config.get("answer_max_tokens", 500),
        )

        parsed = self._extract_json(raw_response)

        if not parsed:
            parsed = {
                "Thought": "Could not parse structured output cleanly.",
                "Answer": raw_response,
            }

        final_answer = parsed.get("Answer", "")
        thought = parsed.get("Thought", "")

        self.memory_store.add({
            "question": question,
            "answer": final_answer,
            "thought": thought,
            "action": action,
            "knowledge": self.knowledge,
            "examples": self.examples,
            "tool_result": self.tool_result,
        })

        return {
            "plan": plan,
            "thought": thought,
            "answer": final_answer,
            "knowledge": self.knowledge,
            "examples": self.examples,
            "tool_result": self.tool_result,
        }


def run_phase1_attack_from_csv(agent, input_csv):
    input_path = Path(input_csv)

    category = LLM_CONFIG["kb_category"]
    model_name = get_model_dir_name()

    dataset_name = input_path.parent.name
    file_name = input_path.stem

    output_dir = (Path("results") / model_name / category / "phase1_attack")

    output_dir.mkdir(parents = True, exist_ok = True)

    output_csv = (
        output_dir
        / f"{dataset_name}_{file_name}_phase1_{TARGET_OPTION}.csv"
    )

    df = pd.read_csv(input_csv)

    required_cols = {
        "question",
        "opa",
        "opb",
        "opc",
        "opd",
        "cop",
        "exp",
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_cols)}"
        )

    results = []

    for _, row in tqdm(
        df.iterrows(),
        total = len(df),
        desc = f"Running Phase 1 attack on {dataset_name}",
    ):
        full_question = build_mcq_question(row)

        result = agent.answer_question(full_question)

        raw_predicted_answer = result["answer"]

        predicted_answer = extract_option_label(
            raw_predicted_answer
        )

        predicted_reasoning = result["thought"]

        correct_answer_index = row.get("cop", "")

        correct_answer = COP_TO_LABEL.get(
            correct_answer_index,
            "",
        )

        correct_reasoning = row.get("exp", "")

        row_dict = row.to_dict()

        row_dict.update({
            "question_with_options": full_question,

            "correct_answer_index": correct_answer_index,
            "correct_answer": correct_answer,
            "correct_reasoning": correct_reasoning,

            "subject_name": row.get("subject_name", ""),
            "topic_name": row.get("topic_name", ""),

            "phase1_target_option": TARGET_OPTION,

            "phase1_predicted_answer_raw": raw_predicted_answer,
            "phase1_predicted_answer": predicted_answer,
            "phase1_predicted_reasoning": predicted_reasoning,

            "phase1_plan": str(result["plan"]),
            "phase1_knowledge": result["knowledge"],
            "phase1_examples": result["examples"],
            "phase1_tool_result": result["tool_result"],

            "phase1_is_correct": (
                predicted_answer == correct_answer
            ),

            "phase1_is_target": (
                predicted_answer == TARGET_OPTION
            ),
            
            "phase1_parse_failed": predicted_answer == "",
        })

        results.append(row_dict)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        output_csv,
        index = False,
    )

    baseline_df = pd.read_csv(input_csv)

    baseline_target_count = (baseline_df["predicted_answer"].astype(str).str.upper().str.strip() == TARGET_OPTION).sum()

    phase1_target_count = (result_df["phase1_predicted_answer"].astype(str).str.upper().str.strip() == TARGET_OPTION).sum()

    baseline_total = len(baseline_df)
    phase1_total = len(result_df)

    change = phase1_target_count - baseline_target_count

    print(f"\nSaved Phase 1 results to: {output_csv}")

    print("\n==============================")
    print(f"Target Option: {TARGET_OPTION}")
    print("==============================")

    print(
        f"Baseline {TARGET_OPTION} selections : "
        f"{baseline_target_count} / {baseline_total}"
    )

    print(
        f"Phase1 {TARGET_OPTION} selections  : "
        f"{phase1_target_count} / {phase1_total}"
    )

    print(f"Change : {change:+d}")

    print(
        f"Baseline Rate : "
        f"{baseline_target_count / baseline_total:.4f}"
    )

    print(
        f"Phase1 Rate   : "
        f"{phase1_target_count / phase1_total:.4f}"
    )
    
    print("Parse failures:", result_df["phase1_parse_failed"].sum())

    return result_df


def main():
    category = LLM_CONFIG["kb_category"]
    input_csv_name = LLM_CONFIG["input_csv_name"]

    model_name = get_model_dir_name()

    memory_json = get_memory_json_path()

    print(
        f"Using memory JSON as Phase 1 attack source: "
        f"{memory_json}"
    )

    print(f"Target option: {TARGET_OPTION}")

    print(
        f"Number of attack examples: "
        f"{N_ATTACK_EXAMPLES}"
    )

    attack_memory_store = Phase1AttackMemoryStore(
        memory_json = memory_json,
        target_option = TARGET_OPTION,
        n_examples = N_ATTACK_EXAMPLES,
        seed = RANDOM_SEED,
    )

    agent = Phase1AttackQAAgent(
        config = LLM_CONFIG,
        memory_store = attack_memory_store,
        function_map = FUNCTION_MAP,
        num_shots = LLM_CONFIG["attack_num_examples"],
    )

    dataset_root = Path("results") / model_name / category

    if not dataset_root.exists():
        raise ValueError(
            f"Dataset root does not exist: {dataset_root}"
        )

    matching_csvs = sorted(
        dataset_root.rglob(f"*_{input_csv_name}")
    )

    if not matching_csvs:
        raise ValueError(
            f"No CSV files found under {dataset_root} "
            f"ending with _{input_csv_name}"
        )

    for baseline_csv in matching_csvs:
        run_phase1_attack_from_csv(agent = agent, input_csv = baseline_csv)

    print(
        f"\nPhase 1 attack complete "
        f"for model = {model_name}, category = {category}"
    )

if __name__ == "__main__":
    main()