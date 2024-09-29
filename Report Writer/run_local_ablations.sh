#!/bin/bash

# Change to the Report Writer directory
cd "$(dirname "$0")"

# Run all ablation scripts
echo "Running Ablation_ALL.py..."
python3.11 -m ablations.Ablation_ALL

echo "Running Ablation_ALL_TableGenerator.py..."
python3.11 -m ablations.Ablation_ALL_TableGenerator

echo "Running Ablation_Cost.py..."
python3.11 -m ablations.Ablation_Cost

echo "Running Ablation_Latency.py..."
python3.11 -m ablations.Ablation_Latency

echo "Running Ablation_MoA_General.py..."
python3.11 -m ablations.Ablation_MoA_General

echo "All ablation scripts have been executed."