import json
import os
import Levenshtein


class MemoryStore:
    def __init__(self, path = "memory.json", max_size = 100):
        self.path = path
        self.max_size = max_size
        self.memory = self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding = "utf-8") as f:
                return json.load(f)
            
        return []

    def save(self):
        with open(self.path, "w", encoding = "utf-8") as f:
            json.dump(self.memory, f, indent = 4, ensure_ascii = False)

    def add(self, item):
        self.memory.append(item)

        if len(self.memory) > self.max_size:
            self.memory = self.memory[-self.max_size:]

        self.save()

    def retrieve_examples(self, query: str, num_shots: int = 3) -> str:
        if not self.memory:
            return ""

        distances = []

        for i, item in enumerate(self.memory):
            old_q = item.get("question", "")
            d = Levenshtein.distance(query, old_q)
            distances.append((i, d))

        distances = sorted(distances, key = lambda x: x[1])
        selected = [idx for idx, _ in distances[:min(num_shots, len(distances))]]

        examples = []

        for idx in selected:
            item = self.memory[idx]
            
            examples.append(
                "Question: {}\nAnswer: {}\n".format(
                    item.get("question", ""),
                    item.get("answer", "")
                )
            )

        return "\n".join(examples)