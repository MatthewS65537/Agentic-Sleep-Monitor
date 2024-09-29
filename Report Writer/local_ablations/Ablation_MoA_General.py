import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats

def plot_moa_performance(models, names, output_file, title, results_dir="./results", lower_y=5, upper_y=10):
    accuracy = []
    clarity = []
    relevancy = []
    style = []
    quality = []
    avg = []
    print(models[0].split("@")[0].strip())
    for model in models:
        with open(f"{results_dir}/EVAL_{model}.pkl", "rb") as f:
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
        # Drop lowest and highest
        accuracy_.sort()
        clarity_.sort()
        relevancy_.sort()
        style_.sort()
        quality_.sort()
        avg_.sort()
        if "1Pass" not in model:
            accuracy.append(np.mean(accuracy_[1:-1]))
            clarity.append(np.mean(clarity_[1:-1]))
            relevancy.append(np.mean(relevancy_[1:-1]))
            style.append(np.mean(style_[1:-1]))
            quality.append(np.mean(quality_[1:-1]))
        avg.append(np.mean(avg_[1:-1]))

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f0f0f0')

    # Define colors and make them darker
    colors = plt.cm.tab10(np.linspace(0, 1, 6))  # Increased to 6 for the new metric
    darker_colors = [mcolors.rgb_to_hsv(c[:3]) for c in colors]
    for c in darker_colors:
        c[2] *= 0.7  # Reduce brightness to make color darker
    darker_colors = [mcolors.hsv_to_rgb(c) for c in darker_colors]

    ax.plot(names[1:], accuracy, marker='o', label='Accuracy', color=darker_colors[0], alpha=0.3, linewidth=1)
    ax.plot(names[1:], clarity, marker='s', label='Clarity', color=darker_colors[1], alpha=0.3, linewidth=1)
    ax.plot(names[1:], relevancy, marker='^', label='Relevancy', color=darker_colors[2], alpha=0.3, linewidth=1)
    ax.plot(names[1:], style, marker='D', label='Style', color=darker_colors[3], alpha=0.3, linewidth=1)
    ax.plot(names[1:], quality, marker='x', label='Quality', color=darker_colors[4], linewidth=2)
    ax.plot(names[1:], avg[1:], marker='*', label='Average', color="blue", linewidth=2, markersize=12)

    # Plot 1-pass average as a dotted blue line
    ax.axhline(y=avg[0], color='blue', linestyle=':', linewidth=2, label='1-Pass Average')

    # Add linear regression for average
    x = np.arange(len(avg[1:]))
    slope_avg, intercept_avg, r_value_avg, p_value_avg, std_err_avg = stats.linregress(x, avg[1:])

    def linear_func_avg(x):
        return slope_avg * x + intercept_avg

    x_smooth = np.linspace(0, len(avg[1:]) - 1, 100)
    y_smooth_avg = linear_func_avg(x_smooth)

    ax.plot(x_smooth, y_smooth_avg, 'r--', label='Average Regression')

    # Add linear regression for quality
    slope_quality, intercept_quality, r_value_quality, p_value_quality, std_err_quality = stats.linregress(x, quality)

    def linear_func_quality(x):
        return slope_quality * x + intercept_quality

    y_smooth_quality = linear_func_quality(x_smooth)

    ax.plot(x_smooth, y_smooth_quality, 'g--', label='Quality Regression')

    # Print regression statistics
    print("Average Regression:")
    print(f"Slope: {slope_avg:.4f}")
    print(f"Intercept: {intercept_avg:.4f}")
    print(f"R-squared: {r_value_avg**2:.4f}")
    print(f"P-value: {p_value_avg:.4f}")

    print("\nQuality Regression:")
    print(f"Slope: {slope_quality:.4f}")
    print(f"Intercept: {intercept_quality:.4f}")
    print(f"R-squared: {r_value_quality**2:.4f}")
    print(f"P-value: {p_value_quality:.4f}")

    ax.set_ylabel('Rating', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    legend = ax.legend(title='Metrics', loc='best', fontsize=10)
    legend.get_title().set_fontsize('12')
    legend.get_title().set_fontweight('bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_ylim(lower_y, upper_y)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Merge Ablation_MoA files for different models

    # GPT-4o
    models_4o = [
        "gpt-4o-2024-08-06 @ 1Pass",
        "gpt-4o-2024-08-06 @ MoA 1",
        "gpt-4o-2024-08-06 @ MoA 2",
        "gpt-4o-2024-08-06 @ MoA 3",
        "gpt-4o-2024-08-06 @ MoA 4",
        "gpt-4o-2024-08-06 @ MoA 5",
        "gpt-4o-2024-08-06 @ MoA 6",
        "gpt-4o-2024-08-06 @ MoA 7",
        "gpt-4o-2024-08-06 @ MoA 8",
    ]

    # GPT-4o-Mini
    models_4o_mini = [
        "gpt-4o-mini @ 1Pass",
        "gpt-4o-mini @ MoA 1",
        "gpt-4o-mini @ MoA 2",
        "gpt-4o-mini @ MoA 3",
        "gpt-4o-mini @ MoA 4",
        "gpt-4o-mini @ MoA 5",
        "gpt-4o-mini @ MoA 6",
        "gpt-4o-mini @ MoA 7",
        "gpt-4o-mini @ MoA 8",
    ]

    # Gemma2
    models_gemma2 = [
        "gemma2:2b @ 1Pass",
        "gemma2:2b @ MoA 1",
        "gemma2:2b @ MoA 2",
        "gemma2:2b @ MoA 3",
        "gemma2:2b @ MoA 4",
        "gemma2:2b @ MoA 5",
        "gemma2:2b @ MoA 6",
        "gemma2:2b @ MoA 7",
        "gemma2:2b @ MoA 8",
    ]

    # Claude 3.5 Sonnet
    models_claude35 = [
        "claude-3-5-sonnet @ 1Pass",
        "claude-3-5-sonnet @ MoA 1",
        "claude-3-5-sonnet @ MoA 2",
        "claude-3-5-sonnet @ MoA 3",
        "claude-3-5-sonnet @ MoA 4",
        "claude-3-5-sonnet @ MoA 5",
        "claude-3-5-sonnet @ MoA 6",
        "claude-3-5-sonnet @ MoA 7",
        "claude-3-5-sonnet @ MoA 8",
    ]

    models_qwen25 = [
        "qwen2.5_3b-instruct @ 1Pass",
        "qwen2.5_3b-instruct @ MoA 1",
        "qwen2.5_3b-instruct @ MoA 2",
        "qwen2.5_3b-instruct @ MoA 3",
        "qwen2.5_3b-instruct @ MoA 4",
        "qwen2.5_3b-instruct @ MoA 5",
        "qwen2.5_3b-instruct @ MoA 6",
        "qwen2.5_3b-instruct @ MoA 7",
        "qwen2.5_3b-instruct @ MoA 8",
    ]

    names = [
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

    # Plot performance for each model
    plot_moa_performance(models_4o, names, "./graphs/GPT4o_MoA_Performance.png", "GPT4o Performance Across MoA Iterations", lower_y=6.75, upper_y=8.5)
    plot_moa_performance(models_4o_mini, names, "./graphs/GPT4o_mini_MoA_Performance.png", "GPT-4o-Mini Performance Across MoA Iterations", lower_y=6.5, upper_y=9.5)
    plot_moa_performance(models_gemma2, names, "./graphs/Gemma2_MoA_Performance.png", "Gemma2-2b Performance Across MoA Iterations", lower_y=5, upper_y=8.5)
    plot_moa_performance(models_claude35, names, "./graphs/Claude35Sonnet_MoA_Performance.png", "Claude 3.5 Sonnet Performance Across MoA Iterations", lower_y=6.5, upper_y=9.5)
    # plot_moa_performance(models_qwen25, names, "./graphs/Qwen25_MoA_Performance.png", "Qwen2.5-3b Performance Across MoA Iterations", lower_y=6.5, upper_y=9.5, results_dir="./gdrive_results")