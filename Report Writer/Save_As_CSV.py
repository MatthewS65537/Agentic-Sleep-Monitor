import pickle
import csv
from ablations.Ablation_Defines import names, colors

models = [
    "gemma2:2b @ 1Pass",
    "gemma2:2b @ MoA 1",
    "qwen2.5:3b-instruct @ 1Pass",
    "qwen2.5:3b-instruct @ MoA 5",
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
    avg_ratings.append(round(eval["avg_ratings"]["avg"], 2))
    acc_ratings.append(round(eval["avg_ratings"]["accuracy"], 2))
    clarity_ratings.append(round(eval["avg_ratings"]["clarity"], 2))
    relevance_ratings.append(round(eval["avg_ratings"]["relevancy"], 2))
    style_ratings.append(round(eval["avg_ratings"]["style"], 2))
    quality_ratings.append(round((eval["avg_ratings"]["accuracy"] + eval["avg_ratings"]["clarity"] + eval["avg_ratings"]["relevancy"]) / 3, 2))

# Preparing data for CSV
csv_headers = ['Model', 'Accuracy', 'Clarity', 'Relevancy', 'Style', 'Quality', 'Average']

with open("./results/Ablation_Results.csv", "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)
    for model, acc, clarity, relevancy, style, quality, avg in zip(models, acc_ratings, clarity_ratings, relevance_ratings, style_ratings, quality_ratings, avg_ratings):
        writer.writerow([names.get(model, model), acc, clarity, relevancy, style, quality, avg])

# Preparing data for detailed CSV
detailed_csv_headers = ['Model', 'Accuracy', 'Clarity', 'Relevancy', 'Style', 'Quality', 'Average']

with open("./results/Ablation_Results_Detailed.csv", "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(detailed_csv_headers)
    for model in models:
        with open(f"./results/EVAL_{model}.pkl", "rb") as f:
            eval = pickle.load(f)
        for entry in eval["detailed"]:
            writer.writerow([
                names.get(model, model),
                round(entry["ratings"]["accuracy"], 2),
                round(entry["ratings"]["clarity"], 2),
                round(entry["ratings"]["relevancy"], 2),
                round(entry["ratings"]["style"], 2),
                round((entry["ratings"]["accuracy"] + entry["ratings"]["clarity"] + entry["ratings"]["relevancy"] + entry["ratings"]["style"]) / 4, 2),
                round((entry["ratings"]["accuracy"] + entry["ratings"]["clarity"] + entry["ratings"]["relevancy"] + entry["ratings"]["style"]) / 4, 2)
            ])