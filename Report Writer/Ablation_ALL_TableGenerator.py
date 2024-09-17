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

# Generate LaTeX table
latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|c|c|c|c|c|c|}\n\\hline\n"
latex_table += "Model & Accuracy & Clarity & Relevance & Style & Quality & Average \\\\ \\hline\n"

for model, acc, clarity, relevance, style, quality, avg in zip(models, acc_ratings, clarity_ratings, relevance_ratings, style_ratings, quality_ratings, avg_ratings):
    latex_table += f"{names[model]} & {acc} & {clarity} & {relevance} & {style} & {quality} & {avg} \\\\ \\hline\n"

latex_table += "\\end{tabular}\n\\caption{Model Performance Comparison}\n\\label{tab:model_comparison}\n\\end{table}"

# Print the LaTeX table
print(latex_table)

# Optionally, save the LaTeX table to a file
# with open("model_comparison_table.tex", "w") as f:
#     f.write(latex_table)
