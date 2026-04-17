import json
import re
import time
from typing import Any, Dict, List, Optional

from prompts import (
    build_planner_messages,
    build_retrieval_messages,
    build_answer_messages,
)

from openai import OpenAI, AzureOpenAI
import anthropic
from google import genai
from google.genai import types as genai_types


class QAAgent:
    def __init__(self, config: dict, memory_store, function_map = None, num_shots: int = 3):
        self.config = config
        self.memory_store = memory_store
        self.function_map = function_map or {}
        self.num_shots = num_shots

        self.question = ""
        self.knowledge = ""
        self.examples = ""
        self.tool_result = ""

        self.provider = self.config["provider"].lower()
        self.client = self._build_client()

    def _build_client(self):
        if self.provider == "openai":
            return OpenAI(api_key=self.config["openai_api_key"])

        if self.provider == "azure":
            return AzureOpenAI(
                api_key=self.config["azure_api_key"],
                azure_endpoint=self.config["azure_base_url"],
                api_version=self.config["azure_api_version"],
            )

        if self.provider == "anthropic":
            return anthropic.Anthropic(
                api_key=self.config["anthropic_api_key"]
            )

        if self.provider == "gemini":
            return genai.Client(api_key=self.config["gemini_api_key"])

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _normalize_messages_for_plain_text(self, messages: List[Dict[str, str]]) -> str:
        chunks = []

        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            chunks.append(f"{role}:\n{content}")

        return "\n\n".join(chunks)

    def _extract_anthropic_text(self, response) -> str:
        parts = []

        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                parts.append(block.text)

        return "\n".join(parts).strip()

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)

            if match:
                candidate = match.group(1).strip()

                try:
                    return json.loads(candidate)
                
                except json.JSONDecodeError:
                    pass

        try:
            return json.loads(text)
        
        except json.JSONDecodeError:
            return None

    def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3,
    ) -> str:
        temperature = self.config.get("temperature", 0.2) if temperature is None else temperature
        max_tokens = self.config.get("max_tokens", 800) if max_tokens is None else max_tokens

        last_error = None

        for _ in range(retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model = self.config["model"],
                        messages = messages,
                        temperature = temperature,
                        max_tokens = max_tokens,
                        top_p = 1,
                    )
                    return response.choices[0].message.content.strip()

                if self.provider == "azure":
                    response = self.client.chat.completions.create(
                        model = self.config["model"],
                        messages = messages,
                        temperature = temperature,
                        max_tokens = max_tokens,
                        top_p = 1,
                    )
                    return response.choices[0].message.content.strip()

                if self.provider == "anthropic":
                    system_chunks = []
                    anthro_messages = []

                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")

                        if role == "system":
                            system_chunks.append(content)

                        elif role == "assistant":
                            anthro_messages.append({"role": "assistant", "content": content})

                        else:
                            anthro_messages.append({"role": "user", "content": content})

                    response = self.client.messages.create(
                        model = self.config["model"],
                        system = "\n\n".join(system_chunks) if system_chunks else None,
                        messages = anthro_messages,
                        temperature = temperature,
                        max_tokens = max_tokens,
                    )
                    return self._extract_anthropic_text(response)

                if self.provider == "gemini":
                    prompt = self._normalize_messages_for_plain_text(messages)

                    response = self.client.models.generate_content(
                        model = self.config["model"],
                        contents = prompt,
                        config = genai_types.GenerateContentConfig(
                            temperature = temperature,
                            max_output_tokens = max_tokens,
                        ),
                    )
                    return (response.text or "").strip()

                raise ValueError(f"Unsupported provider: {self.provider}")

            except Exception as e:
                last_error = e
                print(f"_chat error: {e}")
                time.sleep(2)

        raise RuntimeError(f"LLM call failed after retries. Last error: {last_error}")

    def retrieve_knowledge(self, question: str) -> str:
        messages = build_retrieval_messages(question)
        return self._chat(
            messages = messages,
            temperature = 0,
            max_tokens = self.config.get("retrieval_max_tokens", 300),
        )

    def retrieve_examples(self, question: str) -> str:
        return self.memory_store.retrieve_examples(question, self.num_shots)

    def plan_next_action(self, question: str) -> Dict[str, Any]:
        tool_list = ", ".join(self.function_map.keys()) if self.function_map else "None"
        messages = build_planner_messages(question, tool_list)

        response = self._chat(
            messages = messages,
            temperature = 0,
            max_tokens = self.config.get("planner_max_tokens", 250),
        )

        parsed = self._extract_json(response)

        if not parsed:
            return {
                "action": "answer_directly",
                "reason": "Planner output was invalid, so fallback to direct answer.",
                "tool_name": "",
                "tool_input": "",
            }

        return parsed

    def run_tool(self, tool_name: str, tool_input):
        func = self.function_map.get(tool_name)

        if not func:
            return f"Tool '{tool_name}' not found."

        try:
            if isinstance(tool_input, dict):
                if tool_name == "retrieve_examples" and "top_k" not in tool_input:
                    tool_input["top_k"] = self.config.get("example_top_k", 3)

                if tool_name == "retrieve_passages" and "top_k" not in tool_input:
                    tool_input["top_k"] = self.config.get("passage_top_k", 3)

                return func(**tool_input)

            return func(tool_input)

        except Exception as e:
            return f"Tool execution error: {e}"

    def answer_question(self, question: str) -> Dict[str, Any]:
        self.question = question
        self.knowledge = ""
        self.examples = ""
        self.tool_result = ""

        plan = self.plan_next_action(question)
        action = str(plan.get("action", "answer_directly")).strip()

        if action == "retrieve_knowledge":
            self.knowledge = self.retrieve_knowledge(question)
            self.examples = self.retrieve_examples(question)

        elif action == "retrieve_examples":
            self.examples = self.retrieve_examples(question)

        elif action == "use_tool":
            tool_name = str(plan.get("tool_name", "")).strip()
            tool_input = plan.get("tool_input", question)
            self.tool_result = self.run_tool(tool_name, tool_input)
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