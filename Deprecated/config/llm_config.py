import os

def set_temperature(config_dict, temperature):
    config_dict["temperature"] = temperature
    return config_dict

groqllama8b_config = {
    "model": "llama3-8b-8192",
    "api_type" : "groq",
    "api_key": os.environ["GROQ_API_KEY"]
}

groqllama70b_config = {
    "model": "llama3-70b-8192",
    "api_type" : "groq",
    "api_key": os.environ["GROQ_API_KEY"]
}

pplxllama8b_config = {
    "model": "llama-3-8b-instruct",
    "base_url" : "https://api.perplexity.ai/",
    "api_key": os.environ["PPLX_API_KEY"]
}

pplxllama70b_config = {
    "model": "llama-3-70b-instruct",
    "base_url" : "https://api.perplexity.ai/",
    "api_key": os.environ["PPLX_API_KEY"]
}

pplxwebsonarsmall_config = {
    "model": "llama-3-sonar-small-32k-online",
    "base_url" : "https://api.perplexity.ai/",
    "api_key": os.environ["PPLX_API_KEY"]
}

pplxwebsonarlarge_config = {
    "model": "llama-3-sonar-large-32k-online",
    "base_url" : "https://api.perplexity.ai/",
    "api_key": os.environ["PPLX_API_KEY"]
}

xiaoai_gpt4o_config = {
    "model": "gpt-4o",
    "base_url" : "https://api.xiaoai.plus/v1",
    "api_key": os.environ["XIAOAI_API_KEY"]
}

lmstudio_gemma2_2b_config = {
    "model" : "lmstudio-community/gemma-2-2b-it-GGUF",
    "base_url" : "http://localhost:1234/v1",
    "api_key" : "lm-studio"
}