#!/bin/bash
#SBATCH --job-name=uncertainty
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate mech_interp_gpu

cd ~/uncertainty_routing

python run_complete_pipeline.py --mode standard --model "Qwen/Qwen2.5-1.5B-Instruct"
#python -u experiment5_trustworthiness.py
#python diagnostic_steering_vectors.py
#python analyze_tradeoff.py
#python detailed_eps20_analysis.py
