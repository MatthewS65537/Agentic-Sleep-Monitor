import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from Ablation_Defines import names, colors

models = [
    "gemma2:2b @ 1Pass",
    "gemma2:2b @ MoA 1",
    "gemma2:2b @ MoA 2",
    "gemma2:2b @ MoA 4",
    "gemma2:2b @ MoA 6",
    "gemma2:2b @ MoA 8",
    "qwen2:1.5b @ 1Pass",
    "qwen2:1.5b @ MoA 1",
    "qwen2:1.5b @ MoA 2",
    "qwen2:1.5b @ MoA 4",
    # "qwen2:1.5b @ MoA 6",
    # "qwen2:1.5b @ MoA 8",
    "phi3.5:latest @ 1Pass",
    # "phi3.5:latest @ MoA 1",
    # "phi3.5:latest @ MoA 2",
    # "phi3.5:latest @ MoA 4",
    # "phi3.5:latest @ MoA 6",
    # "phi3.5:latest @ MoA 8",
    "gpt-4o-mini @ 1Pass",
    "gpt-4o-mini @ MoA 1",
    "gpt-4o-mini @ MoA 2",
    "gpt-4o-mini @ MoA 4",
    "gpt-4o-mini @ MoA 6",
    "gpt-4o-mini @ MoA 8",
    "gpt-4o-2024-08-06 @ 1Pass",
    "o1-mini @ 1Pass",
    "o1-preview @ 1Pass",
    "claude-3-haiku @ 1Pass",
    "claude-3-opus @ 1Pass",
    "claude-3-5-sonnet @ 1Pass",
]

open_models = []
closed_models = []

for model in models:
    if ("gemma" in model) or ("phi" in model) or ("qwen" in model):
        open_models.append(model)
    else:
        closed_models.append(model)

# Assume each report is 1k tokens (generally true)
cost_per_1M_tokens_output = {
    "phi3.5:latest @ 1Pass" : 0.0625,
    "phi3.5:latest @ MoA 1" : 0.0625 * (1 + 2),
    "phi3.5:latest @ MoA 2" : 0.0625 * (2 + 2),
    "phi3.5:latest @ MoA 4" : 0.0625 * (4 + 2),
    "phi3.5:latest @ MoA 6" : 0.0625 * (6 + 2),
    "phi3.5:latest @ MoA 8" : 0.0625 * (8 + 2),
    "gemma2:2b @ 1Pass" : 0.0532,
    "gemma2:2b @ MoA 1" : 0.0532 * (1 + 2),
    "gemma2:2b @ MoA 2" : 0.0532 * (2 + 2),
    "gemma2:2b @ MoA 4" : 0.0532 * (4 + 2),
    "gemma2:2b @ MoA 6" : 0.0532 * (6 + 2),
    "gemma2:2b @ MoA 8" : 0.0532 * (8 + 2),
    "qwen2:1.5b @ 1Pass" : 0.0347,
    "qwen2:1.5b @ MoA 1" : 0.0347 * (1 + 2),
    "qwen2:1.5b @ MoA 2" : 0.0347 * (2 + 2),
    "qwen2:1.5b @ MoA 4" : 0.0347 * (4 + 2),
    "qwen2:1.5b @ MoA 6" : 0.0347 * (6 + 2),
    "qwen2:1.5b @ MoA 8" : 0.0347 * (8 + 2),
    "gpt-4o-mini @ 1Pass" : 0.94,
    "gpt-4o-mini @ MoA 1" : 0.94 * (1 + 2),
    "gpt-4o-mini @ MoA 2" : 0.94 * (2 + 2),
    "gpt-4o-mini @ MoA 4" : 0.94 * (4 + 2),
    "gpt-4o-mini @ MoA 6" : 0.94 * (6 + 2),
    "gpt-4o-mini @ MoA 8" : 0.94 * (8 + 2),
    "gpt-4o-2024-08-06 @ 1Pass" : 25.0,
    "o1-mini @ 1Pass" : 15.0,
    "o1-preview @ 1Pass" : 75.0,
    "claude-3-haiku @ 1Pass" : 6.0,
    "claude-3-opus @ 1Pass" : 135.0,
    "claude-3-5-sonnet @ 1Pass" : 27.0,
}

accuracies = []
costs = []

for model in models:
    with open(f"./results/EVAL_{model}.pkl", "rb") as f:
        eval_data = pickle.load(f)
    accuracies.append(round(eval_data["avg_ratings"]["accuracy"], 2))
    cost = cost_per_1M_tokens_output.get(model, 0) / 1000  # Convert to cost per 1k tokens
    costs.append(np.log10(cost) if cost > 0 else np.nan)  # Use log scale, handle zero costs

fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f0f0f0')

# Separate data for open and closed models
open_costs = [cost for model, cost in zip(models, costs) if model in open_models]
open_accuracies = [acc for model, acc in zip(models, accuracies) if model in open_models]
closed_costs = [cost for model, cost in zip(models, costs) if model in closed_models]
closed_accuracies = [acc for model, acc in zip(models, accuracies) if model in closed_models]

# Run regressions
open_slope, open_intercept, _, _, _ = stats.linregress(open_costs, open_accuracies)
closed_slope, closed_intercept, _, _, _ = stats.linregress(closed_costs, closed_accuracies)
all_slope, all_intercept, _, _, _ = stats.linregress(costs, accuracies)

# Plot data points
for model, accuracy, cost in zip(models, accuracies, costs):
    if not np.isnan(cost):
        ax.scatter(cost, accuracy, color=colors[names[model]], s=100, label=names[model], edgecolors='black', linewidth=1)

# Plot regression lines
x_range = np.linspace(min(costs) - 0.5, max(costs) + 0.5, 100)
ax.plot(x_range, open_slope * x_range + open_intercept, color='blue', linestyle='--', label='Open Models Regression')
ax.plot(x_range, closed_slope * x_range + closed_intercept, color='green', linestyle='--', label='Closed Models Regression')
ax.plot(x_range, all_slope * x_range + all_intercept, color='black', linestyle='--', label='Overall Regression')

ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('Log Cost ($ per 1k tokens)', fontsize=14, fontweight='bold')
ax.set_title('Log Response Cost vs Accuracy', fontsize=18, fontweight='bold', pad=20)

ax.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

legend = ax.legend(title='Models', loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))
legend.get_title().set_fontsize('12')
legend.get_title().set_fontweight('bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.set_ylim(5, 10)
ax.set_xlim(min(costs) - 0.5, max(costs) + 0.5)  # Adjust x-axis limits for log scale

# Set custom x-axis ticks for better readability
log_ticks = [-5, -4, -3, -2, -1, 0]
ax.set_xticks(log_ticks)
ax.set_xticklabels([f'$10^{{{x}}}$' for x in log_ticks])

plt.tight_layout()
plt.savefig("./graphs/Log_Cost_vs_Accuracy.png", dpi=300, bbox_inches='tight')