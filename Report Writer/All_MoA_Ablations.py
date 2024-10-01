from tqdm import tqdm
import csv
import pickle
import numpy as np
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

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
    ("gemini-1.5-pro", "Gemini 1.5 Pro", "./results", False),
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

import matplotlib.pyplot as plt

for model_prefix, name, path, use_MoA in tqdm(models):
    acc = []
    cla = []
    rel = []
    sty = []
    qua = []
    avg = []
    acc_ci = []
    cla_ci = []
    rel_ci = []
    sty_ci = []
    qua_ci = []
    avg_ci = []
    acc_std = []
    cla_std = []
    rel_std = []
    sty_std = []
    qua_std = []
    avg_std = []
    acc_se = []
    cla_se = []
    rel_se = []
    sty_se = []
    qua_se = []
    avg_se = []
    all_avg_data = []  # List to store all average data for violin plot

    for exp_setting in exp_settings:
        with open(f"{path}/EVAL_{model_prefix} @ {exp_setting}.pkl", "rb") as f:
            data = pickle.load(f)
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

        all_avg_data.append(avg_)  # Store all average data for this experiment setting

        # Calculate the CIs for each metric using t-distribution
        def calculate_ci(data):
            n = len(data)
            mean = np.mean(data)
            se = stats.sem(data)
            ci = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=se)
            return (ci[1] - ci[0]) / 2  # Return half-width of CI

        acc_ci.append(calculate_ci(acc_))
        cla_ci.append(calculate_ci(cla_))
        rel_ci.append(calculate_ci(rel_))
        sty_ci.append(calculate_ci(sty_))
        qua_ci.append(calculate_ci(qua_))
        avg_ci.append(calculate_ci(avg_))

        # Calculate Standard Deviation
        acc_std.append(np.std(acc_))
        cla_std.append(np.std(cla_))
        rel_std.append(np.std(rel_))
        sty_std.append(np.std(sty_))
        qua_std.append(np.std(qua_))
        avg_std.append(np.std(avg_))

        # Calculate Standard Error
        acc_se.append(stats.sem(acc_))
        cla_se.append(stats.sem(cla_))
        rel_se.append(stats.sem(rel_))
        sty_se.append(stats.sem(sty_))
        qua_se.append(stats.sem(qua_))
        avg_se.append(stats.sem(avg_))

        trim_num = int(len(acc_) * 0.1)
        # Append for Graphing Later
        acc.append(np.mean(acc_[trim_num:-trim_num]))
        cla.append(np.mean(cla_[trim_num:-trim_num]))
        rel.append(np.mean(rel_[trim_num:-trim_num]))
        sty.append(np.mean(sty_[trim_num:-trim_num]))
        qua.append(np.mean(qua_[trim_num:-trim_num]))
        avg.append(np.mean(avg_[trim_num:-trim_num]))

        if not use_MoA:
            break

    if use_MoA:
        # Create violin plot
        fig, ax = plt.subplots(figsize=(9, 3))
        parts = ax.violinplot(all_avg_data, showmeans=True, showextrema=True, showmedians=True)

        # Customize the violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('blue')
        parts['cbars'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')

        # Set labels and title
        ax.set_title(f'Distribution of \"Average\" Ratings for {name} Across MoA Iterations', fontsize=16)
        ax.set_xlabel('Experiment Setting', fontsize=12)
        ax.set_ylabel('Average Rating', fontsize=12)
        ax.set_xticks(range(1, len(exp_settings) + 1))
        ax.set_xticklabels(exp_settings, rotation=45, ha='right')

        # Set y-axis limits
        ax.set_ylim(7.5, 9)

        # Add grid
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

        # Tight layout and save
        plt.tight_layout()
        plt.savefig(f'./graphs/violin_plots/{model_prefix}_violin_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    with open(f'./results_csv/{model_prefix}_Ablations.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Experiment', 'Accuracy', 'Accuracy CI', 'Accuracy StdDev', 'Accuracy StdErr',
                        'Clarity', 'Clarity CI', 'Clarity StdDev', 'Clarity StdErr',
                        'Relevancy', 'Relevancy CI', 'Relevancy StdDev', 'Relevancy StdErr',
                        'Style', 'Style CI', 'Style StdDev', 'Style StdErr',
                        'Average', 'Average CI', 'Average StdDev', 'Average StdErr'])
        
        for i in range(len(acc)):
            writer.writerow([
                model_prefix,
                exp_settings[i],
                f"{acc[i]:.2f}",
                f"{acc_ci[i]:.2f}",
                f"{acc_std[i]:.2f}",
                f"{acc_se[i]:.2f}",
                f"{cla[i]:.2f}",
                f"{cla_ci[i]:.2f}",
                f"{cla_std[i]:.2f}",
                f"{cla_se[i]:.2f}",
                f"{rel[i]:.2f}",
                f"{rel_ci[i]:.2f}",
                f"{rel_std[i]:.2f}",
                f"{rel_se[i]:.2f}",
                f"{sty[i]:.2f}",
                f"{sty_ci[i]:.2f}",
                f"{sty_std[i]:.2f}",
                f"{sty_se[i]:.2f}",
                f"{avg[i]:.2f}",
                f"{avg_ci[i]:.2f}",
                f"{avg_std[i]:.2f}",
                f"{avg_se[i]:.2f}"
            ])