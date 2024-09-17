from Ablation_MoA_General import plot_moa_performance

models = [
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

plot_moa_performance(models, names, "./graphs/GPT4o_mini_MoA_Performance.png", "GPT-4o-Mini Performance Across MoA Iterations", lower_y=6.5, upper_y=9.5)