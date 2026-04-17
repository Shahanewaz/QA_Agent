import json
import re
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher

from config import LLM_CONFIG


BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
EXAMPLES_PATH = BASE_DIR / "examples.json"
ENTITIES_PATH = BASE_DIR / "entities.json"


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    text = _normalize(text)
    return re.findall(r"[a-zA-Z0-9_]+", text)


def _load_json_file(path: Path, default: Any):
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding = "utf-8") as f:
            return json.load(f)
        
    except Exception:
        return default


def _get_active_kb_dir() -> Path:
    category = LLM_CONFIG.get("kb_category", "").strip()

    if not category:
        return KB_DIR

    return KB_DIR / category


def _load_text_files(folder: Path) -> List[Dict[str, str]]:
    docs = []

    if not folder.exists():
        return docs

    for path in folder.rglob("*.txt"):
        try:
            content = path.read_text(encoding = "utf-8")
            relative_path = path.relative_to(folder)

            docs.append({
                "id": path.stem,
                "source": str(relative_path),
                "domain": LLM_CONFIG.get("kb_category", "general") or "general",
                "text": content
            })

        except Exception:
            continue

    return docs


def _score_text(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))

    if not q_tokens:
        return 0.0

    overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)

    query_norm = _normalize(query)
    text_norm = _normalize(text)

    substring_bonus = 0.0

    if query_norm in text_norm:
        substring_bonus += 1.0

    for token in q_tokens:
        if token in text_norm:
            substring_bonus += 0.1

    similarity = SequenceMatcher(None, query_norm, text_norm[:2000]).ratio()

    return overlap * 2.0 + substring_bonus + similarity


def _split_into_passages(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    text = text.strip()

    if not text:
        return []

    passages = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            passages.append(chunk)

        if end == n:
            break

        start = max(end - overlap, start + 1)

    return passages


def retrieve_context(query: str) -> str:
    active_dir = _get_active_kb_dir()
    docs = _load_text_files(active_dir)

    if not docs:
        return f"No knowledge base documents found in {active_dir}"

    scored = []

    for doc in docs:
        score = _score_text(query, doc["text"])
        scored.append((score, doc))

    scored.sort(key = lambda x: x[0], reverse = True)
    best_score, best_doc = scored[0]

    if best_score <= 0:
        return "No relevant context found."

    text = best_doc["text"].strip()
    short_text = text[:LLM_CONFIG.get("context_char_limit", 2000)]

    return (
        f"Domain: {best_doc['domain']}\n"
        f"Source: {best_doc['source']}\n"
        f"Context:\n{short_text}"
    )


def retrieve_passages(query: str, top_k: int = None) -> str:
    active_dir = _get_active_kb_dir()
    docs = _load_text_files(active_dir)

    if not docs:
        return f"No knowledge base documents found in {active_dir}"

    if top_k is None:
        top_k = LLM_CONFIG.get("passage_top_k", 3)

    candidates = []

    for doc in docs:
        passages = _split_into_passages(doc["text"])

        for idx, passage in enumerate(passages):
            score = _score_text(query, passage)

            if score > 0:
                candidates.append({
                    "score": score,
                    "source": doc["source"],
                    "domain": doc["domain"],
                    "passage_id": idx,
                    "text": passage
                })

    if not candidates:
        return "No relevant passages found."

    candidates.sort(key = lambda x: x["score"], reverse = True)
    top = candidates[:max(1, top_k)]

    results = []

    for item in top:
        results.append(
            f"[Domain: {item['domain']} | Source: {item['source']} | Passage: {item['passage_id']} | Score: {item['score']:.2f}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(results)


def retrieve_examples(query: str, top_k: int = None) -> str:
    examples = _load_json_file(EXAMPLES_PATH, default=[])

    if not examples:
        return "No examples found in examples.json"

    if top_k is None:
        top_k = LLM_CONFIG.get("example_top_k", 3)

    scored = []

    for item in examples:
        q = item.get("question", "")
        a = item.get("answer", "")
        score = _score_text(query, q)
        scored.append((score, q, a))

    scored.sort(key = lambda x: x[0], reverse = True)
    top = scored[:max(1, top_k)]

    if not top or top[0][0] <= 0:
        return "No similar examples found."

    lines = []

    for idx, (score, q, a) in enumerate(top, start = 1):
        lines.append(
            f"Example {idx} | Score: {score:.2f}\n"
            f"Question: {q}\n"
            f"Answer: {a}"
        )

    return "\n\n".join(lines)


def verify_grounding(question: str, answer: str, evidence: str) -> str:
    q_tokens = set(_tokenize(question))
    a_tokens = set(_tokenize(answer))
    e_tokens = set(_tokenize(evidence))

    important = (q_tokens | a_tokens) - {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
        "for", "and", "or", "by", "with", "what", "which", "who", "when",
        "where", "why", "how"
    }

    if not important:
        result = {
            "supported": "partial",
            "overlap_score": 0.0,
            "matched_terms": [],
            "explanation": "Not enough informative tokens to verify grounding."
        }

        return json.dumps(result, indent = 2)

    matched = sorted(list(important & e_tokens))
    overlap_score = len(matched) / max(len(important), 1)

    if overlap_score >= 0.6:
        supported = "yes"
        explanation = "The answer appears to be well supported by the evidence."

    elif overlap_score >= 0.3:
        supported = "partial"
        explanation = "The answer is partially supported, but some claims may be missing from the evidence."
    
    else:
        supported = "no"
        explanation = "The evidence does not appear to sufficiently support the answer."

    result = {
        "supported": supported,
        "overlap_score": round(overlap_score, 3),
        "matched_terms": matched[:20],
        "explanation": explanation
    }

    return json.dumps(result, indent = 2)


def lookup_entity(name: str) -> str:
    entities = _load_json_file(ENTITIES_PATH, default = {})

    if not entities:
        return "No entities found in entities.json"

    name_norm = _normalize(name)

    if name_norm in entities:
        return json.dumps(entities[name_norm], indent = 2)

    best_key = None
    best_score = 0.0

    for key in entities.keys():
        score = SequenceMatcher(None, name_norm, _normalize(key)).ratio()

        if score > best_score:
            best_score = score
            best_key = key

    if best_key and best_score >= 0.7:
        return json.dumps({
            "matched_entity": best_key,
            "match_score": round(best_score, 3),
            "data": entities[best_key]
        }, indent = 2)

    return f"No entity found for: {name}"


FUNCTION_MAP = {
    "retrieve_context": retrieve_context,
    "retrieve_passages": retrieve_passages,
    "retrieve_examples": retrieve_examples,
    "verify_grounding": verify_grounding,
    "lookup_entity": lookup_entity,
}