from Ablation_MoA_General import plot_moa_performance

models = [
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

plot_moa_performance(models, names, "./graphs/Gemma2_MoA_Performance.png", "Gemma2-2b Performance Across MoA Iterations", lower_y=5, upper_y=8.5)