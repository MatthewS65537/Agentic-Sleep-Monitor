from Ablation_MoA_General import plot_moa_performance

models = [
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

plot_moa_performance(models, names, "./graphs/GPT4o_MoA_Performance.png", "GPT4o Performance Across MoA Iterations", lower_y=6.75, upper_y=8.5)