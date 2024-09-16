import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def read_pkl_files():
    results = {}
    for filename in os.listdir("./results"):
        if filename.startswith('open_evaluation_results_zeroshot_') and filename.endswith('.pkl'):
            model_name = filename.replace('open_evaluation_results_zeroshot_', '').replace('.pkl', '') + " @ 1 Pass"
            with open(f"./results/{filename}", 'rb') as f:
                results[model_name] = pickle.load(f)
        elif filename.startswith('local_evaluation_results_MoA_') and filename.endswith('.pkl'):
            moa_number = filename.split('_')[4]  # Extract the MOA number from the filename
            model_name = filename.replace(f'local_evaluation_results_MoA_{moa_number}_', '').replace('.pkl', '') + f" @ MoA {moa_number}"
            with open(f"./results/{filename}", 'rb') as f:
                results[model_name] = pickle.load(f)
            
    
    for filename in os.listdir("./results"):
        if filename.startswith('closed_evaluation_results_') and filename.endswith('.pkl'):
            model_name = filename.replace('closed_evaluation_results_', '').replace('.pkl', '') + " @ 1 Pass"
            with open(f"./results/{filename}", 'rb') as f:
                results[model_name] = pickle.load(f)
    return results

evaluation_results = read_pkl_files()
models = [
    # "phi3.5:latest @ 1 Pass",
    "gemma2:2b @ 1 Pass",
    # "qwen2:1.5b @ 1 Pass",
    # "phi3.5:latest @ MoA 2",
    "gemma2:2b @ MoA 2",
    # "qwen2:1.5b @ MoA 2",
    "gemma2:2b @ MoA 3",
    # "qwen2:1.5b @ MoA 3",
    "gemma2:2b @ MoA 4",
    # "qwen2:1.5b @ MoA 4",
    "gemma2:2b @ MoA 5",
    # "qwen2:1.5b @ MoA 5",
    "gpt-4o-mini @ 1 Pass",
    "gpt-4o-2024-08-06 @ 1 Pass",
    "o1-mini @ 1 Pass",
    "o1-preview @ 1 Pass",
    "claude-3-haiku @ 1 Pass",
    "claude-3-opus @ 1 Pass",
    "claude-3-5-sonnet @ 1 Pass",
]

print(models)

# Define colors for each model
colors = {
    "phi3.5:latest @ 1 Pass": "#001aff",
    "gemma2:2b @ 1 Pass": "#006aff",
    "qwen2:1.5b @ 1 Pass": "#00d9ff",
    "phi3.5:latest @ MoA 2": "#8000ff",
    "gemma2:2b @ MoA 2": "#bb00ff",
    "qwen2:1.5b @ MoA 2": "#c75bcf",
    "phi3.5:latest @ MoA 3": "#FFB347",
    "gemma2:2b @ MoA 3": "#FFA500",
    "qwen2:1.5b @ MoA 3": "#FF8C00",
    "gemma2:2b @ MoA 4": "#FF4500",
    "qwen2:1.5b @ MoA 4": "#FF0000",
    "gemma2:2b @ MoA 5": "#0000FF",
    "qwen2:1.5b @ MoA 5": "#0066FF",
    "gpt-4o-mini @ 1 Pass": "#98FF98",
    "gpt-4o-2024-08-06 @ 1 Pass": "#32CD32",
    "o1-mini @ 1 Pass": "#228B22",
    "o1-preview @ 1 Pass": "#006400",
    "claude-3-haiku @ 1 Pass": "#E2725B",
    "claude-3-opus @ 1 Pass": "#B7410E",
    "claude-3-5-sonnet @ 1 Pass": "#800020",
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
    for eval in evaluation_results[model][model.split(" @")[0]]:
        accuracy += eval["ratings"]["Accuracy"]
        clarity += eval["ratings"]["Clarity"]
        relevancy += eval["ratings"]["Relevancy"]
        style += eval["ratings"]["Style"]
    avg = (accuracy + clarity + relevancy + style) / 4

    num_evals = len(evaluation_results[model][model.split(" @")[0]])
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

attributes = ["Accuracy", "Clarity", "Relevancy", "Style", "Average"]

speeds = {
    "phi3.5:latest": 22,
    "gemma2:2b": 25,
    "qwen2:1.5b": 40,
    "gpt-4o-2024-08-06": 91,
    "gpt-4o-mini": 141,
    "o1-preview": 10,
    "o1-mini": 10,
    "claude-3-haiku": 10,
    "claude-3-opus": 10,
    "claude-3-5-sonnet": 10,
}

times = {
    "phi3.5:latest @ 1 Pass": 54,
    "gemma2:2b @ 1 Pass": 15,
    "qwen2:1.5b @ 1 Pass": 15,
    "phi3.5:latest @ MoA": 340,
    "gemma2:2b @ MoA 2": 60,
    "qwen2:1.5b @ MoA 2": 50,
    "gemma2:2b @ MoA 3": 107,
    "qwen2:1.5b @ MoA 3": 73,
    "gemma2:2b @ MoA 4": 107,
    "qwen2:1.5b @ MoA 4": 73,
    "gemma2:2b @ MoA 5": 76,
    "qwen2:1.5b @ MoA 5": 64,
    "gpt-4o-2024-08-06 @ 1 Pass": 5,
    "gpt-4o-mini @ 1 Pass": 4,
    "o1-preview @ 1 Pass": 35,
    "o1-mini @ 1 Pass": 15,
    "claude-3-haiku @ 1 Pass": 14,
    "claude-3-opus @ 1 Pass": 26,
    "claude-3-5-sonnet @ 1 Pass": 16,
}

price_1k = {
    "phi3.5:latest": 0.00004,
    "gemma2:2b": 0.00003,
    "qwen2:1.5b": 0.00002,
    "gpt-4o-2024-08-06": 0.01,
    "gpt-4o-mini": 0.000375,
    "o1-preview": 0.075,
    "o1-mini": 0.015,
    "claude-3-haiku": 0.006,
    "claude-3-opus": 0.135,
    "claude-3-5-sonnet": 0.027,
}

latency_data = {}
for model in models:
    total_tokens = 0
    for eval in evaluation_results[model][model.split(" @")[0]]:
        total_tokens += eval["total_tokens"]

    latency_data[model] = {
        "total_tokens": total_tokens/len(evaluation_results[model][model.split(" @")[0]]),
        "time": times[model],
        "speed": speeds[model.split(" @")[0]],
        "cost": total_tokens/len(evaluation_results[model][model.split(" @")[0]]) * price_1k[model.split(" @")[0]]/1000 * 100
    }

# # Visualization of latency data
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# ax1, ax2, ax3, ax4 = axes.flatten()

# # Plot 1: Total Tokens
# tokens = [latency_data[model]["total_tokens"] for model in models]
# ax1.bar(models, tokens, color=[colors[model] for model in models])
# ax1.set_title('Total Tokens')
# ax1.set_ylabel('Tokens')
# ax1.tick_params(axis='x', rotation=45)

# # Plot 2: Time
# times = [latency_data[model]["time"] for model in models]
# ax2.bar(models, times, color=[colors[model] for model in models])
# ax2.set_title('Time per Response')
# ax2.set_ylabel('Seconds')
# ax2.tick_params(axis='x', rotation=45)

# # Plot 3: Speed
# speeds = [latency_data[model]["speed"] for model in models]
# ax3.bar(models, speeds, color=[colors[model] for model in models])
# ax3.set_title('Speed')
# ax3.set_ylabel('Tokens/second')
# ax3.tick_params(axis='x', rotation=45)

# # Plot 4: Cost
# costs = [latency_data[model]["cost"] for model in models]
# ax4.bar(models, costs, color=[colors[model] for model in models])
# ax4.set_title('Cost per Response')
# ax4.set_ylabel('Cents (USD)')
# ax4.tick_params(axis='x', rotation=45)

# plt.tight_layout()
# plt.savefig('latency_analysis.png')
# plt.close()

# print("Latency analysis visualization saved as 'latency_analysis.png'")

# Combined Visualization of Evaluation Results and Latency Data
fig = plt.figure(figsize=(25, 12))
gs = fig.add_gridspec(2, 4)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])
ax5 = fig.add_subplot(gs[1, 3])

x = np.arange(len(attributes))
width = 0.06
multiplier = 0

for model in models:
    model_scores = [results_data[model][attribute] for attribute in attributes]
    offset = width * multiplier
    rects = ax1.bar(x + offset, model_scores, width, label=model, color=colors[model])
    ax1.bar_label(rects, padding=3, rotation=90, fmt='%.2f')
    multiplier += 1

ax1.set_ylabel('Scores')
ax1.set_title('Model Evaluation Results')
ax1.set_xticks(x + width * (len(models) - 1) / 2)
ax1.set_xticklabels(attributes)
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.set_ylim(0, 11)

# Plot 2: Total Tokens
tokens = [latency_data[model]["total_tokens"] for model in models]
ax2.bar(models, tokens, color=[colors[model] for model in models])
ax2.set_title('Total Tokens')
ax2.set_ylabel('Tokens')
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Plot 3: Time
times = [latency_data[model]["time"] for model in models]
ax3.bar(models, times, color=[colors[model] for model in models])
ax3.set_title('Time per Response')
ax3.set_ylabel('Seconds')
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Plot 4: Speed
speeds = [latency_data[model]["speed"] for model in models]
ax4.bar(models, speeds, color=[colors[model] for model in models])
ax4.set_title('Speed')
ax4.set_ylabel('Tokens/second')
ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Plot 5: Cost
costs = [latency_data[model]["cost"] for model in models]
ax5.bar(models, costs, color=[colors[model] for model in models])
ax5.set_title('Cost per Response')
ax5.set_ylabel('Cents (USD)')
ax5.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.tight_layout()
plt.savefig('./graphs/combined_visualization.png')
plt.close()

print("Combined visualization saved as 'combined_visualization.png'")

# Print all results at the end
print("\nResults Data:")
for model, data in results_data.items():
    print(f"{model}: {data}")

print("\nLatency Data:")
for model, data in latency_data.items():
    print(f"{model}: {data}")