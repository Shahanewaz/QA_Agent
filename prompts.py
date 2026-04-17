def build_planner_messages(question: str, tool_list: str):
    return [
        {
            "role": "system",
            "content": "You are a planning agent that decides the next action."
        },
        {
            "role": "user",
            "content": f"""
Choose one action:

1. retrieve_knowledge
2. retrieve_examples
3. use_tool
4. answer_directly

Available tools:
{tool_list}

If using a tool:
- tool_input may be a string
- or a JSON object if the tool needs multiple arguments

Use these tools when appropriate:
- retrieve_context(query): get broad relevant context
- retrieve_passages(query, top_k = 3): get the most relevant evidence passages
- retrieve_examples(query): get similar solved QA examples
- verify_grounding(question, answer, evidence): check whether an answer is supported
- lookup_entity(name): find information about a domain-specific entity

Return ONLY JSON in a fenced code block:

```json
{{
  "action": "retrieve_knowledge | retrieve_examples | use_tool | answer_directly",
  "reason": "short explanation",
  "tool_name": "",
  "tool_input": ""
}}
```

Question:
{question}
"""
        }
    ]


def build_retrieval_messages(question: str):
    return [
        {
            "role": "system",
            "content": "You are a retrieval assistant. Return concise background knowledge useful for answering the user's question."
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Return short relevant knowledge only.
"""
        }
    ]


def build_answer_messages(question: str, examples: str, knowledge: str, tool_result: str):
    return [
        {
            "role": "system",
            "content": (
                "You are a precise multiple-choice QA assistant. "
                "You will be given a question with options A, B, C, and D. "
                "Your final answer must be exactly one of: A, B, C, or D."
            )
        },
        {
            "role": "user",
            "content": f"""
Answer the multiple-choice question using the available context.

Examples:
{examples if examples else "None"}

Knowledge:
{knowledge if knowledge else "None"}

Tool result:
{tool_result if tool_result else "None"}

Question:
{question}

Return ONLY JSON in a fenced code block:

```json
{{
  "Thought": "brief reasoning",
  "Answer": "A or B or C or D"
}}
```
"""
        }
    ]