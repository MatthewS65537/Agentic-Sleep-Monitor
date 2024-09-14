import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def read_pkl_files():
    results = {}
    for filename in os.listdir():
        if filename.startswith('local_evaluation_results_') and filename.endswith('.pkl'):
            model_name = filename.replace('local_evaluation_results_', '').replace('.pkl', '')
            with open(filename, 'rb') as f:
                results[model_name] = pickle.load(f)
    
    for filename in os.listdir():
        if filename.startswith('closed_evaluation_results_') and filename.endswith('.pkl'):
            model_name = filename.replace('closed_evaluation_results_', '').replace('.pkl', '')
            with open(filename, 'rb') as f:
                results[model_name] = pickle.load(f)
    return results

evaluation_results = read_pkl_files()
models = list(evaluation_results.keys())

# Define colors for each model
colors = {
    "phi3.5:latest": "blue",
    "gemma2:2b": "orange",
    "qwen2:1.5b": "green",
    "gpt-4o-2024-08-06": "red",
    "gpt-4o-mini": "yellow"
}

class Result:
    def __init__(self, ratings, final_response, total_tokens):
        self.ratings = ratings
        self.final_response = final_response
        self.total_tokens = total_tokens

results_data = {}

for model in models:
    accuracy = 0.0
    clarity = 0.0
    relevancy = 0.0
    style = 0.0
    avg = 0.0
    for eval in evaluation_results[model][model]:
        accuracy += eval["ratings"]["Accuracy"]
        clarity += eval["ratings"]["Clarity"]
        relevancy += eval["ratings"]["Relevancy"]
        style += eval["ratings"]["Style"]
    avg = (accuracy + clarity + relevancy + style) / 4

    num_evals = len(evaluation_results[model][model])
    accuracy /= num_evals
    clarity /= num_evals
    relevancy /= num_evals
    style /= num_evals
    avg /= num_evals

    results_data[model] = {
        "Accuracy": accuracy,
        "Clarity": clarity,
        "Relevancy": relevancy,
        "Style": style,
        "Average": avg
    }

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

attributes = ["Accuracy", "Clarity", "Relevancy", "Style", "Average"]

x = np.arange(len(attributes))
width = 0.15
multiplier = 0

for model in models:
    suffix = " @ 1-Pass" if model in ["gpt-4o-2024-08-06", "gpt-4o-mini"] else " @ MoA"
    model_scores = [results_data[model][attribute] for attribute in attributes]
    offset = width * multiplier
    rects = ax.bar(x + offset, model_scores, width, label=model+suffix, color=colors[model])
    ax.bar_label(rects, padding=3, rotation=90, fmt='%.2f')
    multiplier += 1

ax.set_ylabel('Scores')
ax.set_title('Model Evaluation Results')
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(attributes)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_ylim(0, 5.9)

plt.tight_layout()
plt.savefig('model_evaluation_results.png')
plt.close()

print("Visualization saved as 'model_evaluation_results.png'")

# Efficiency Analysis

speeds = {
    "phi3.5:latest": 22,
    "gemma2:2b": 25,
    "qwen2:1.5b": 40,
    "gpt-4o-2024-08-06": 91,
    "gpt-4o-mini": 141
}

times = {
    "phi3.5:latest": 340,
    "gemma2:2b": 107,
    "qwen2:1.5b": 73,
    "gpt-4o-2024-08-06": 7,
    "gpt-4o-mini": 4
}

price_1k = {
    "phi3.5:latest": 0.00004,
    "gemma2:2b": 0.00003,
    "qwen2:1.5b": 0.00002,
    "gpt-4o-2024-08-06": 0.01,
    "gpt-4o-mini": 0.000375
}

latency_data = {}
for model in models:
    total_tokens = 0
    for eval in evaluation_results[model][model]:
        total_tokens += eval["total_tokens"]

    latency_data[model] = {
        "total_tokens": total_tokens/len(evaluation_results[model][model]),
        "time": times[model],
        "speed": speeds[model],
        "cost": total_tokens/len(evaluation_results[model][model]) * price_1k[model]/1000 * 100
    }

# Visualization of latency data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

# Plot 1: Total Tokens
tokens = [latency_data[model]["total_tokens"] for model in models]
ax1.bar(models, tokens, color=[colors[model] for model in models])
ax1.set_title('Total Tokens')
ax1.set_ylabel('Tokens')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Time
times = [latency_data[model]["time"] for model in models]
ax2.bar(models, times, color=[colors[model] for model in models])
ax2.set_title('Time per Response')
ax2.set_ylabel('Seconds')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Speed
speeds = [latency_data[model]["speed"] for model in models]
ax3.bar(models, speeds, color=[colors[model] for model in models])
ax3.set_title('Speed')
ax3.set_ylabel('Tokens/second')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Cost
costs = [latency_data[model]["cost"] for model in models]
ax4.bar(models, costs, color=[colors[model] for model in models])
ax4.set_title('Cost per Response')
ax4.set_ylabel('Cents (USD)')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('latency_analysis.png')
plt.close()

print("Latency analysis visualization saved as 'latency_analysis.png'")

# Combined Visualization of Evaluation Results and Latency Data
fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(2, 4)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])
ax5 = fig.add_subplot(gs[1, 3])

x = np.arange(len(attributes))
width = 0.15
multiplier = 0

for model in models:
    suffix = " @ 1-Pass" if model in ["gpt-4o-2024-08-06", "gpt-4o-mini"] else " @ MoA"
    model_scores = [results_data[model][attribute] for attribute in attributes]
    offset = width * multiplier
    rects = ax1.bar(x + offset, model_scores, width, label=model+suffix, color=colors[model])
    ax1.bar_label(rects, padding=3, rotation=90, fmt='%.2f')
    multiplier += 1

ax1.set_ylabel('Scores')
ax1.set_title('Model Evaluation Results')
ax1.set_xticks(x + width * (len(models) - 1) / 2)
ax1.set_xticklabels(attributes)
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.set_ylim(0, 5.5)

# Plot 2: Total Tokens
tokens = [latency_data[model]["total_tokens"] for model in models]
ax2.bar(models, tokens, color=[colors[model] for model in models])
ax2.set_title('Total Tokens')
ax2.set_ylabel('Tokens')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Time
times = [latency_data[model]["time"] for model in models]
ax3.bar(models, times, color=[colors[model] for model in models])
ax3.set_title('Time per Response')
ax3.set_ylabel('Seconds')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Speed
speeds = [latency_data[model]["speed"] for model in models]
ax4.bar(models, speeds, color=[colors[model] for model in models])
ax4.set_title('Speed')
ax4.set_ylabel('Tokens/second')
ax4.tick_params(axis='x', rotation=45)

# Plot 5: Cost
costs = [latency_data[model]["cost"] for model in models]
ax5.bar(models, costs, color=[colors[model] for model in models])
ax5.set_title('Cost per Response')
ax5.set_ylabel('Cents (USD)')
ax5.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('combined_visualization.png')