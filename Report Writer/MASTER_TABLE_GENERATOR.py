import pickle
import numpy as np
import pyperclip
from tqdm import tqdm
from local_ablations.Ablation_Defines import names, colors

models = [
    ("qwen2.5_0.5b-instruct", "Qwen 2.5 0.5B Instruct", "./gdrive_results", True),
    ("qwen2.5_1.5b-instruct", "Qwen 2.5 1.5B Instruct", "./gdrive_results", True),
    ("qwen2.5_3b-instruct", "Qwen 2.5 3B Instruct", "./gdrive_results", True),
    ("gemma2_2b", "Gemma 2 2B Instruct", "./gdrive_results", True),
    ("llama3.2_1b-instruct-q4_0", "Llama 3.2 1B Instruct", "./gdrive_results", True),
    ("llama3.2_3b-instruct-q4_0", "Llama 3.2 3B Instruct", "./gdrive_results", True),
    ("gpt-4o-mini", "GPT-4o Mini", "./results", True),
    ("gpt-4o-2024-08-06", "GPT-4o", "./results", True),
    ("claude-3-5-sonnet", "Claude 3.5 Sonnet", "./results", True), # END OF MOA Models
    ("o1-mini", "O1 Mini", "./results", False),
    ("o1-preview", "O1 Preview", "./results", False),
    ("claude-3-haiku", "Claude 3 Haiku", "./results", False),
    ("claude-3-sonnet", "Claude 3 Sonnet", "./results", False),
    ("claude-3-opus", "Claude 3 Opus", "./results", False),
    ("gemini-1.5-flash", "Gemini 1.5 Pro", "./results", False),
    ("gemini-1.5-pro", "Gemini 1.5 Flash", "./results", False),
]

exp_settings = [
    "1Pass",
    "MoA 1",
    "MoA 2",
    "MoA 3",
    "MoA 4",
    "MoA 5",
    "MoA 6",
    "MoA 7",
    "MoA 8",
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

for model_prefix, name, path, use_MoA in tqdm(models):
    for exp_setting in exp_settings:
        model = f"{model_prefix} @ {exp_setting}"
        
        try:
            with open(f"{path}/EVAL_{model}.pkl", "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            if not use_MoA and exp_setting != "1Pass":
                break
            continue

        acc_ = []
        cla_ = []
        rel_ = []
        sty_ = []
        qua_ = []
        avg_ = []

        for d in data["detailed"]:
            acc_.append(d["ratings"]["accuracy"])
            cla_.append(d["ratings"]["clarity"])
            rel_.append(d["ratings"]["relevancy"])
            sty_.append(d["ratings"]["style"])
            qua_.append((d["ratings"]["accuracy"] + d["ratings"]["clarity"] + d["ratings"]["relevancy"])/3)
            avg_.append((d["ratings"]["accuracy"] + d["ratings"]["clarity"] + d["ratings"]["relevancy"] + d["ratings"]["style"])/4)

        trim_num = int(len(acc_) * 0.1)
        acc = round(np.mean(acc_[trim_num:-trim_num]), 2)
        clarity = round(np.mean(cla_[trim_num:-trim_num]), 2)
        relevance = round(np.mean(rel_[trim_num:-trim_num]), 2)
        style = round(np.mean(sty_[trim_num:-trim_num]), 2)
        quality = round(np.mean(qua_[trim_num:-trim_num]), 2)
        avg = round(np.mean(avg_[trim_num:-trim_num]), 2)

        acc_ratings.append(acc)
        clarity_ratings.append(clarity)
        relevance_ratings.append(relevance)
        style_ratings.append(style)
        quality_ratings.append(quality)
        avg_ratings.append(avg)

        max_accuracy = max(max_accuracy, acc)
        max_clarity = max(max_clarity, clarity)
        max_relevance = max(max_relevance, relevance)
        max_style = max(max_style, style)
        max_quality = max(max_quality, quality)
        max_avg = max(max_avg, avg)

        if exp_setting == "1Pass":
            base_acc[model_prefix] = acc
            base_clarity[model_prefix] = clarity
            base_relevance[model_prefix] = relevance
            base_style[model_prefix] = style
            base_quality[model_prefix] = quality
            base_avg[model_prefix] = avg

        if not use_MoA and exp_setting == "1Pass":
            break

# Generate LaTeX table
latex_table = """\\begin{table*}[h]
\\centering
\\renewcommand{\\arraystretch}{1.5}
\\begin{tabular}{|l|*{6}{>{\\centering\\arraybackslash$}m{1.5cm}<{$}|}}
\\hline
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Clarity} & \\textbf{Relevance} & \\textbf{Style} & \\textbf{Quality} & \\textbf{Overall} \\\\ \\hline\n"""

format_arrow = lambda base, now: "\color{green!50!black}{\\; \\blacktriangle " if now > base else ("\color{red!50!black}{\\; \\blacktriangledown " if now < base else "\color{gray}{\\; \\approx ")
calculate_delta = lambda base, now: abs(round(now - base, 2))
arrow_entry = lambda base, now: format_arrow(base, now) + str(calculate_delta(base, now)).format("{:0.2f}") + "} \\color{" + "black" + "}"

for model_prefix, name, path, use_MoA in models:
    for exp_setting in exp_settings:
        model = f"{model_prefix} @ {exp_setting}"

        acc = acc_ratings.pop(0)
        clarity = clarity_ratings.pop(0)
        relevance = relevance_ratings.pop(0)
        style = style_ratings.pop(0)
        quality = quality_ratings.pop(0)
        avg = avg_ratings.pop(0)

        if exp_setting != "1Pass":
            latex_table += f"{name} @ {exp_setting} & \\large {acc:0.2f} {arrow_entry(base_acc[model_prefix], acc)} \\atop \\normalsize ({acc/max_accuracy * 100:0.2f} \\%) & \\large {clarity:0.2f} {arrow_entry(base_clarity[model_prefix], clarity)} \\atop \\normalsize ({clarity/max_clarity * 100:0.2f} \\%) & \\large {relevance:0.2f} {arrow_entry(base_relevance[model_prefix], relevance)} \\atop \\normalsize ({relevance/max_relevance * 100:0.2f} \\%) & \\large {style:0.2f} {arrow_entry(base_style[model_prefix], style)} \\atop \\normalsize ({style/max_style * 100:0.2f} \\%) & \\large {quality:0.2f} {arrow_entry(base_quality[model_prefix], quality)} \\atop \\normalsize ({quality/max_quality * 100:0.2f} \\%) & \\large {avg:0.2f} {arrow_entry(base_avg[model_prefix], avg)} \\atop \\normalsize ({avg/max_avg * 100:0.2f} \\%) \\\\ \\hline\n"
        else:
            latex_table += f"{name} @ {exp_setting} & \\large {acc:0.2f} \\atop \\normalsize ({acc/max_accuracy * 100:0.2f} \\%) & \\large {clarity:0.2f} \\atop \\normalsize ({clarity/max_clarity * 100:0.2f} \\%) & \\large {relevance:0.2f} \\atop \\normalsize ({relevance/max_relevance * 100:0.2f} \\%) & \\large {style:0.2f} \\atop \\normalsize ({style/max_style * 100:0.2f} \\%) & \\large {quality:0.2f} \\atop \\normalsize ({quality/max_quality * 100:0.2f} \\%) & \\large {avg:0.2f} \\atop \\normalsize ({avg/max_avg * 100:0.2f} \\%) \\\\ \\hline\n"

        if not use_MoA and exp_setting == "1Pass":
            break

latex_table += "\\end{tabular}\n\\caption{Model Performance Comparison}\n\\label{tab:model_comparison}\n\\end{table*}"

# Copy the LaTeX table to the clipboard
pyperclip.copy(latex_table)

# Print the LaTeX table
print(latex_table)

# Optionally, save the LaTeX table to a file
# with open("model_comparison_table.tex", "w") as f:
#     f.write(latex_table)
