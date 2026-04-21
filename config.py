LLM_CONFIG = {
    # Supported providers: "openai", "azure", "anthropic", "gemini"
    "provider": "openai",

    # Model name for the selected provider
    "model": "gpt-4o",

    # Shared generation defaults
    "temperature": 0,
    "max_tokens": 800,
    "planner_max_tokens": 250,
    "retrieval_max_tokens": 300,
    "answer_max_tokens": 500,

    # Retrieval settings
    "example_top_k": 3,
    "passage_top_k": 3,
    "context_char_limit": 2000,

    # Memory settings
    "memory_max_size": 100,

    # Dataset / KB settings
    # Allowed examples: "healthcare", "cybersecurity", "machine_learning", "networking"
    "kb_category": "healthcare",
    "input_csv_name": "train.csv",

    # OpenAI
    "openai_api_key": "YOUR_OPENAI_API_KEY",

    # Azure OpenAI
    "azure_api_key": "YOUR_AZURE_API_KEY",
    "azure_base_url": "https://YOUR-RESOURCE.openai.azure.com/",
    "azure_api_version": "2024-02-01",

    # Anthropic
    "anthropic_api_key": "YOUR_ANTHROPIC_API_KEY",

    # Gemini
    "gemini_api_key": "YOUR_GEMINI_API_KEY",
}
