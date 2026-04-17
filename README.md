Setup:
1. Put your API settings in config.py
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python main.py

Notes:
- prompts.py contains all prompt/message-building logic
- qa_agent.py contains the controller logic
- tools.py contains required tools
- memory_store.py handles persistent memory 
- config.py supports openai, azure, anthropic, and gemini
