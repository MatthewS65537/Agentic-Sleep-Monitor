import pickle
import matplotlib.pyplot as plt
import numpy as np
from Ablation_Defines import names, colors

models = [
    "gemma2:2b @ 1Pass",
    "gemma2:2b @ MoA 1",
    "gemma2:2b @ MoA 2",
    "gemma2:2b @ MoA 4",
    # "gemma2:2b @ MoA 8",
    "qwen2:1.5b @ 1Pass",
    "qwen2:1.5b @ MoA 1",
    "qwen2:1.5b @ MoA 2",
    "qwen2:1.5b @ MoA 4",
    # "phi3.5:latest @ 1Pass",
    "gpt-4o-mini @ 1Pass",
    "gpt-4o-2024-08-06 @ 1Pass",
    "o1-mini @ 1Pass",
    "o1-preview @ 1Pass",
    "claude-3-haiku @ 1Pass",
    "claude-3-sonnet @ 1Pass",
    "claude-3-opus @ 1Pass",
    "claude-3-5-sonnet @ 1Pass",
    "gemini-1.5-flash @ 1Pass",
    "gemini-1.5-pro @ 1Pass",
]

acc_ratings = []
clarity_ratings = []
relevance_ratings = []
style_ratings = []
quality_ratings = []
avg_ratings = []
for model in models:
    with open(f"./results/EVAL_{model}.pkl", "rb") as f:
        eval = pickle.load(f)
    avg_ratings.append(round(eval["avg_ratings"]["avg"], 1))
    acc_ratings.append(round(eval["avg_ratings"]["accuracy"], 1))
    clarity_ratings.append(round(eval["avg_ratings"]["clarity"], 1))
    relevance_ratings.append(round(eval["avg_ratings"]["relevancy"], 1))
    style_ratings.append(round(eval["avg_ratings"]["style"], 1))
    quality_ratings.append(round((eval["avg_ratings"]["accuracy"] + eval["avg_ratings"]["clarity"] + eval["avg_ratings"]["relevancy"]) / 3, 1))

# Plotting the ratings
metrics = ['Accuracy', 'Clarity', 'Relevance', 'Style', 'Quality', 'Average']
x = np.arange(len(metrics))  # the label locations
width = 0.05  # the width of the bars

fig, ax = plt.subplots(figsize=(30, 12), facecolor='#f0f0f0')

# Number of models
num_models = len(models)

# Adjust the positions of the bars
for i, (model, acc, clarity, relevance, style, quality, avg) in enumerate(zip(models, acc_ratings, clarity_ratings, relevance_ratings, style_ratings, quality_ratings, avg_ratings)):
    ax.bar(x + i * width, [acc, clarity, relevance, style, quality, avg], width, label=names[model], color=colors[names[model]], edgecolor='black', linewidth=0.5)

# Setting labels and title
ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='gray')
ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')
ax.set_ylabel('Ratings', fontsize=16, fontweight='bold')
ax.set_title('Model Ratings by Metric', fontsize=20, fontweight='bold', pad=20)
ax.set_ylim(0, 10)
ax.set_xticks(x + width * (num_models / 2))
ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')

# Customize the legend
legend = ax.legend(title='Models', loc='upper left', fontsize=10, ncol=4, bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand")
legend.get_title().set_fontsize('14')
legend.get_title().set_fontweight('bold')

# Add value labels on top of each bar
for i, container in enumerate(ax.containers):
    ax.bar_label(container, label_type='edge', fontsize=8, fontweight='bold', padding=2, fmt='%.1f')

# Customize the appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig("./graphs/Ablation_All_Metrics.png", dpi=300, bbox_inches='tight')