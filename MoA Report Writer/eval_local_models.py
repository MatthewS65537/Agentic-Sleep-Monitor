from tqdm import tqdm
import pickle
import os
import openai
from main import generate_markdown_reports, DataPool, template
from evaluator import EvalAgent

def evaluate_reports(ollama_client, ollama_model_name, openai_client, openai_model_name, num_agents=2):
    # Instantiate the evaluation agent
    evaluator = EvalAgent(openai_client, openai_model_name)
    
    # Generate the reports
    reports = generate_markdown_reports(ollama_client, ollama_model_name, num_agents)
    
    # Extract the final response to be evaluated
    final_response = reports.get('checked_final_response')
    
    if not final_response:
        return {"error": "No final response to evaluate."}
    
    # Perform evaluation
    ratings = evaluator.evaluate_response(final_response, DataPool, template)
    
    # Prepare the results dictionary
    results = {
        "ratings": ratings,
        "final_response": final_response,
        "total_tokens": reports['agent_tokens'] + reports['other_tokens']
    }
    
    return results

if __name__ == "__main__":
    num_agents = 2
    # Initialize Ollama client
    ollama_client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    ollama_model_names = [
        # "gemma2:2b",
        # "phi3.5:latest",
        "qwen2:1.5b"
    ]

    openai_client = openai.OpenAI(base_url=os.getenv("HOLDAI_URL"), api_key=os.getenv("HOLDAI_API_KEY"))
    openai_model_name = "gpt-4o-2024-08-06"

    for model in ollama_model_names:
        results = {
            model: []
        }
        print(f"Evaluating {model}")
        for i in tqdm(range(10)):
            res_dict = evaluate_reports(ollama_client, model, openai_client, openai_model_name, num_agents)
            results[model].append(res_dict)
        with open(f'./results/open_evaluation_results_MoA_{num_agents}_{model}.pkl', 'wb') as f:
            pickle.dump(results, f)
    
