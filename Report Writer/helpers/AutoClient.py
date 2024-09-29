import os
import openai

# azure_client = openai.OpenAI(base_url="https://api.xiaoai.plus/v1", api_key=os.getenv("XIAOAI_KEY"))
azure_client = openai.OpenAI(base_url="https://api.holdai.top/v1", api_key=os.getenv("HOLDAI_AZURE_KEY"))
anthropic_client = openai.OpenAI(base_url="https://api.holdai.top/v1", api_key=os.getenv("HOLDAI_CLAUDE_KEY"))
o1_client = openai.OpenAI(base_url="https://api.holdai.top/v1", api_key=os.getenv("HOLDAI_O1_KEY"))
global_client = openai.OpenAI(base_url="https://api.holdai.top/v1", api_key=os.getenv("HOLDAI_GLOBAL_KEY"))
ollama_client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def auto_client(model_name: str):
    if "gpt" in model_name:
        return azure_client
    elif "claude" in model_name:
        return anthropic_client
    elif "o1" in model_name:
        return o1_client
    elif "ollama" in model_name:
        return ollama_client
    elif "gemini" in model_name:
        return global_client
    else:
        return global_client