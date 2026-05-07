LLM_CONFIG = {
    # Supported providers: "openai", "azure", "anthropic", "gemini", "ollama"
    "provider": "ollama",

    # Model name for the selected provider
    # "model": "gpt-5.4-mini",
    "model": "phi3:3.8b",

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
    
    # Phase 1 attack settings
    "attack_target_option": "C",
    "attack_num_examples": 5,
    "attack_random_seed": 42,
    
    # Phase 2 attack settings
    "phase2_max_retries": 1,
    "phase2_save_full_interaction": True,
    
    # Phase 3 attack settings
    "phase3_target_bias": 2.0,
    "phase3_score_max": 10.0,
    "phase3_save_full_reasoning": True,
    
    # Overall structure
    "phase_order": ["phase3", "phase2"],

    # Dataset / KB settings
    # Allowed examples: "healthcare", "cybersecurity", "machine_learning", "networking"
    "kb_category": "networking",
    "input_csv_name": "dataset.csv",

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

    # Ollama
    "ollama_host": "http://127.0.0.1:11434"
}
