import pickle
import matplotlib.pyplot as plt
import numpy as np
import pyperclip  # Importing pyperclip to handle clipboard operations
from ablations.Ablation_Defines import names, colors

models = [
    "qwen2.5:3b-instruct @ 1Pass",
    "qwen2.5:3b-instruct @ MoA 1",
    "qwen2.5:3b-instruct @ MoA 2",
    "qwen2.5:3b-instruct @ MoA 3",
    "qwen2.5:3b-instruct @ MoA 4",
    "qwen2.5:3b-instruct @ MoA 5",
    "qwen2.5:3b-instruct @ MoA 6",
    "qwen2.5:3b-instruct @ MoA 7",
    "qwen2.5:3b-instruct @ MoA 8",
    "gpt-4o-mini @ 1Pass",
    # "gpt-4o-mini @ MoA 1",
    # "gpt-4o-mini @ MoA 2",
    # "gpt-4o-mini @ MoA 3",
    # "gpt-4o-mini @ MoA 4",
    # "gpt-4o-mini @ MoA 5",
    # "gpt-4o-mini @ MoA 6",
    # "gpt-4o-mini @ MoA 7",
    # "gpt-4o-mini @ MoA 8",
    "gpt-4o-2024-08-06 @ 1Pass",
    "o1-mini @ 1Pass",
    "o1-preview @ 1Pass",
    "claude-3-haiku @ 1Pass",
    "claude-3-opus @ 1Pass",
    "claude-3-5-sonnet @ 1Pass",
    # "claude-3-5-sonnet @ MoA 1",
    # "claude-3-5-sonnet @ MoA 2",
    # "claude-3-5-sonnet @ MoA 3",
    # "claude-3-5-sonnet @ MoA 4",
    # "claude-3-5-sonnet @ MoA 5",
    # "claude-3-5-sonnet @ MoA 6",
    # "claude-3-5-sonnet @ MoA 7",
    # "claude-3-5-sonnet @ MoA 8",
    "gemini-1.5-pro @ 1Pass",
]

acc_ratings = []
clarity_ratings = []
relevance_ratings = []
style_ratings = []
quality_ratings = []
avg_ratings = []

max_accuracy = 0
max_clarity = 0
max_relevance = 0
max_style = 0
max_quality = 0
max_avg = 0

base_acc = {}
base_clarity = {}
base_relevance = {}
base_style = {}
base_quality = {}
base_avg = {}

for model in models:
    with open(f"./results/EVAL_{model}.pkl", "rb") as f:
        eval = pickle.load(f)
    
    accuracy_ = []
    clarity_ = []
    relevancy_ = []
    style_ = []
    quality_ = []
    avg_ = []
    for entry in eval["detailed"]:
        accuracy_.append(entry["ratings"]["accuracy"])
        clarity_.append(entry["ratings"]["clarity"])
        relevancy_.append(entry["ratings"]["relevancy"])
        style_.append(entry["ratings"]["style"])
        quality_.append(np.mean([entry["ratings"]["accuracy"], entry["ratings"]["clarity"], 
                                 entry["ratings"]["relevancy"]]))
        avg_.append(entry["ratings"]["avg"])
    
    # Drop 2 lowest and 2 highest
    accuracy_.sort()
    clarity_.sort()
    relevancy_.sort()
    style_.sort()
    quality_.sort()
    avg_.sort()
    
    acc_ratings.append(round(np.mean(accuracy_[1:-1]), 2))
    clarity_ratings.append(round(np.mean(clarity_[1:-1]), 2))
    relevance_ratings.append(round(np.mean(relevancy_[1:-1]), 2))
    style_ratings.append(round(np.mean(style_[1:-1]), 2))
    quality_ratings.append(round(np.mean(quality_[1:-1]), 2))
    avg_ratings.append(round(np.mean(avg_[1:-1]), 2))
    
    max_accuracy = max(max_accuracy, round(np.mean(accuracy_[1:-1]), 2))
    max_clarity = max(max_clarity, round(np.mean(clarity_[1:-1]), 2))
    max_relevance = max(max_relevance, round(np.mean(relevancy_[1:-1]), 2))
    max_style = max(max_style, round(np.mean(style_[1:-1]), 2))
    max_quality = max(max_quality, round(np.mean(quality_[1:-1]), 2))
    max_avg = max(max_avg, round(np.mean(avg_[1:-1]), 2))

    if "1Pass" in model:
        base_acc[model.split(" @ ")[0]] = round(np.mean(accuracy_[1:-1]), 2)
        base_clarity[model.split(" @ ")[0]] = round(np.mean(clarity_[1:-1]), 2)
        base_relevance[model.split(" @ ")[0]] = round(np.mean(relevancy_[1:-1]), 2)
        base_style[model.split(" @ ")[0]] = round(np.mean(style_[1:-1]), 2)
        base_quality[model.split(" @ ")[0]] = round(np.mean(quality_[1:-1]), 2)
        base_avg[model.split(" @ ")[0]] = round(np.mean(avg_[1:-1]), 2)

# Plotting the ratings
metrics = ['Accuracy', 'Clarity', 'Relevance', 'Style', 'Quality', 'Average']

# Generate LaTeX table
latex_table = """\\begin{table*}[h]
    \\centering
    \\renewcommand{\\arraystretch}{1.5}
    \\begin{tabular}{|l|*{6}{>{\\centering\\arraybackslash$}m{1.5cm}<{$}|}}
    \\hline
    \\textbf{Model} & \\textbf{Accuracy} & \\textbf{Clarity} & \\textbf{Relevance} & \\textbf{Style} & \\textbf{Quality} & \\textbf{Overall} \\\\ \\hline\n"""

format_arrow = lambda base, now: "\color{green!50!black}{\\; \\blacktriangle " if now > base else ("\color{red!50!black}{\\; \\blacktriangledown " if now < base else "\color{gray}{\\; \\approx}")
calculate_delta = lambda base, now: abs(round(now - base, 2))
arrow_entry = lambda base, now: format_arrow(base, now) + str(calculate_delta(base, now)).format("{:0.2f}") + "} \\color{" + "black" + "}"


for model, acc, clarity, relevance, style, quality, avg in zip(models, acc_ratings, clarity_ratings, relevance_ratings, style_ratings, quality_ratings, avg_ratings):
    if not "1Pass" in model:
        latex_table += f"{names[model]} & \\large {acc:0.2f} {arrow_entry(base_acc[model.split(' @ ')[0]], acc)} \\atop \\normalsize ({acc/max_accuracy * 100:0.2f} \\%) & \\large {clarity:0.2f} {arrow_entry(base_clarity[model.split(' @ ')[0]], clarity)} \\atop \\normalsize ({clarity/max_clarity * 100:0.2f} \\%) & \\large {relevance:0.2f} {arrow_entry(base_relevance[model.split(' @ ')[0]], relevance)} \\atop \\normalsize ({relevance/max_relevance * 100:0.2f} \\%) & \\large {style:0.2f} {arrow_entry(base_style[model.split(' @ ')[0]], style)} \\atop \\normalsize ({style/max_style * 100:0.2f} \\%) & \\large {quality:0.2f} {arrow_entry(base_quality[model.split(' @ ')[0]], quality)} \\atop \\normalsize ({quality/max_quality * 100:0.2f} \\%) & \\large {avg:0.2f} {arrow_entry(base_avg[model.split(' @ ')[0]], avg)} \\atop \\normalsize ({avg/max_avg * 100:0.2f} \\%) \\\\ \\hline\n"
    else:
        latex_table += f"{names[model]} & \\large {acc:0.2f} \\atop \\normalsize ({acc/max_accuracy * 100:0.2f} \\%) & \\large {clarity:0.2f} \\atop \\normalsize ({clarity/max_clarity * 100:0.2f} \\%) & \\large {relevance:0.2f} \\atop \\normalsize ({relevance/max_relevance * 100:0.2f} \\%) & \\large {style:0.2f} \\atop \\normalsize ({style/max_style * 100:0.2f} \\%) & \\large {quality:0.2f} \\atop \\normalsize ({quality/max_quality * 100:0.2f} \\%) & \\large {avg:0.2f} \\atop \\normalsize ({avg/max_avg * 100:0.2f} \\%) \\\\ \\hline\n"

latex_table += "\\end{tabular}\n\\caption{Model Performance Comparison}\n\\label{tab:model_comparison}\n\\end{table*}"

# Copy the LaTeX table to the clipboard
pyperclip.copy(latex_table)

# Print the LaTeX table
print(latex_table)

# Optionally, save the LaTeX table to a file
# with open("model_comparison_table.tex", "w") as f:
#     f.write(latex_table)
