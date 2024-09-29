import pickle
import matplotlib.pyplot as plt
import numpy as np

from ablations.Ablation_Defines import names, colors

models = [
    "gemma2:2b @ 1Pass",
    "gemma2:2b @ MoA 2",
    "gemma2:2b @ MoA 4",
    "gemma2:2b @ MoA 6",
    "gemma2:2b @ MoA 8",
    "gpt-4o-mini @ 1Pass",
    "gpt-4o-mini @ MoA 1",
    "gpt-4o-mini @ MoA 2",
    "gpt-4o-mini @ MoA 3",
    "gpt-4o-mini @ MoA 4",
    "gpt-4o-2024-08-06 @ 1Pass",
    "o1-mini @ 1Pass",
    "o1-preview @ 1Pass",
    "claude-3-haiku @ 1Pass",
    "claude-3-opus @ 1Pass",
    "claude-3-5-sonnet @ 1Pass",
]

avg_ratings = []
latencies = []

for model in models:
    with open(f"./results/EVAL_{model}.pkl", "rb") as f:
        eval_data = pickle.load(f)
    avg_ratings.append(round(eval_data["avg_ratings"]["avg"], 2))
    latencies.append(eval_data["time"])

fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f0f0f0')

for model, avg_rating, latency in zip(models, avg_ratings, latencies):
    ax.scatter(latency, avg_rating, color=colors[names[model]], s=100, label=names[model], edgecolors='black', linewidth=1)

ax.set_ylabel('Average Rating', fontsize=14, fontweight='bold')
ax.set_xlabel('Latency (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Model Latency vs Quality', fontsize=18, fontweight='bold', pad=20)

ax.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# for model, avg_rating, latency in zip(models, avg_ratings, latencies):
    # ax.annotate(names[model], (latency, avg_rating), xytext=(5, 5), textcoords='offset points', fontsize=8, rotation=45)

legend = ax.legend(title='Models', loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))
legend.get_title().set_fontsize('12')
legend.get_title().set_fontweight('bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.set_ylim(5, 10)
ax.set_xlim(0, 200)
plt.tight_layout()
plt.savefig("./graphs/Latency_vs_Quality.png", dpi=300, bbox_inches='tight')