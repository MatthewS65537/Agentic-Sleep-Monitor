from Ablation_MoA_General import plot_moa_performance

models = [
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

plot_moa_performance(models, names, "./graphs/Claude35Sonnet_MoA_Performance.png", "Claude 3.5 Sonnet Performance Across MoA Iterations", lower_y=6.5, upper_y=9.5)