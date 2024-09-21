from tqdm import tqdm
import pickle
import os
import openai
from evaluator import EvalAgent

# client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# model_name = "gemma2:2b"

DataPool = """## Sleep Data
Date: August 11, 2024
### Sleep Duration
- Total Sleep Time: 7 hours 23 minutes
- Time in Bed: 8 hours 5 minutes
- Sleep Efficiency: 91%
### Sleep Stages
- Light Sleep: 3 hours 42 minutes (50.2%)
- Deep Sleep: 1 hour 51 minutes (25.1%)
- REM Sleep: 1 hour 50 minutes (24.7%)
### Snoring
- Total Snoring Time: 42 minutes
- Snoring Episodes: 14
- Average Snoring Intensity: 48 dB
### Sleep Apnea
- Apnea-Hypopnea Index (AHI): 3.2 events/hour
- Total Apnea Events: 24
- Longest Apnea Duration: 18 seconds
### Heart Rate
- Average Heart Rate: 62 bpm
- Lowest Heart Rate: 54 bpm
- Highest Heart Rate: 78 bpm
### Other Metrics
- Room Temperature: 20.5°C
- Room Humidity: 45%
- Noise Level: Average 32 dB, Peak 58 dB
"""

template = """## Recommendations
### Recommendation 1
- Bullet A
- Bullet B
### Recommendation 2
- Bullet A
- Bullet B
### Recommendation 3
- Bullet A
- Bullet B
## Points of Concern
- Mention anomalies in data here
## Potential Issues
- Mention potential health concerns here
## Further Action
- Recommend the user to an expert if required
"""

def generate_markdown_report(client, model_name, silent=True):
    prompt = (
        "You are a professional sleep doctor. Analyze the given data to write a comprehensive sleep report. "
        "You MUST strictly adhere to the given markdown template, or else you will be fired. "
        "You will get sacked if you do not use MARKDOWN formatting consistent with the template. "
        "You will be demoted for dereliction of duty if you do not replace ALL {bracketed} information "
        "with your own content. Be SPECIFIC and aim to PERSONALIZE your report, rather than give general advice. More is better. "
        "Do not mention common knowledge or anything that is not tied to specific things in the data. "
        "Do not give template-style responses like [Mention possible diseases] or [Mention possible treatments].\n"
        f"Here are the sleep stats for a patient. Help write a concise report of the patient's sleep health based on the data provided:\n{DataPool}\n"
        f"Remember your job. Please strictly adhere to this markdown template (do not add any other information or titles, etc.):\n{template}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    total_tokens = response.usage.total_tokens  # Compute total tokens used

    if not silent:
        print(f"# Final Response\n{response.choices[0].message.content}\n\n")
        print(f"# Total Tokens Used: {total_tokens}\n")  # Print total tokens used

    return {
        "final_response": response.choices[0].message.content,
        "total_tokens": total_tokens
    }

if __name__ == "__main__":
    client = openai.OpenAI(base_url=os.getenv("HOLDAI_URL"), api_key=os.getenv("HOLDAI_CLAUDE_KEY"))
    # models = ["gpt-4o-2024-08-06", "gpt-4o-mini", "claude-3-5-sonnet", "o1-preview", "o1-mini"]
    models = ["claude-3-haiku", "claude-3-opus"]
    
    evaluator = EvalAgent(openai.OpenAI(base_url=os.getenv("HOLDAI_URL"), api_key=os.getenv("HOLDAI_AZURE_KEY")), "gpt-4o-2024-08-06")

    for model in models:
        results = {
            model: []
        }
        for i in tqdm(range(10)):
            final_response = generate_markdown_report(client, model)
            ratings = evaluator.evaluate_response(final_response, DataPool, template)
            result = {
                "ratings": ratings,
                "final_response": final_response,
                "total_tokens": final_response['total_tokens']
            }
            results[model].append(result)
            
        with open(f'./results/closed_evaluation_results_{model}.pkl', 'wb') as f:
            pickle.dump(results, f)