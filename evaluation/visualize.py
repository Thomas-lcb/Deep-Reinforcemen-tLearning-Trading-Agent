"""
evaluation/visualize.py — Standalone visualization script.

Loads a saved model and runs a single test episode to generate
an interactive HTML visualization. No retraining needed.

Usage:
    python -m evaluation.visualize --model models/saved/sac_final.zip
    python -m evaluation.visualize --model logs/best_model.zip --algo SAC
"""

import os
import argparse
import yaml
import glob
import numpy as np
import pandas as pd

from stable_baselines3 import SAC, PPO
from env.trading_env import CryptoTradingEnv
from evaluation.visualization import render_visualization

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

ALGOS = {"SAC": SAC, "PPO": PPO}


def load_config(path=None):
    path = path or CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(config):
    data_dir = os.path.join(ROOT_DIR, config["paths"]["processed_data"])
    pattern = os.path.join(data_dir, "*_normalized.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No data found in {data_dir}.")

    data_dict = {}
    print(f"Loading data from {len(files)} files...")
    for f in files:
        pair_key = os.path.basename(f).replace("_normalized.csv", "")
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        data_dict[pair_key] = df
        print(f"  Loaded {pair_key}: {len(df)} rows")

    return data_dict


def run_visualization(args):
    config = load_config(args.config)
    data_dict = load_data(config)

    # Resolve model path
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT_DIR, model_path)
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Detect algo from filename or args
    algo_name = args.algo.upper()
    algo_cls = ALGOS.get(algo_name)
    if algo_cls is None:
        raise ValueError(f"Unknown algo: {algo_name}")

    print(f"Loading model from {model_path} ({algo_name})...")
    device = "cuda" if args.device == "cuda" else "cpu"
    model = algo_cls.load(model_path, device=device)

    # Create test env (no domain randomization)
    env = CryptoTradingEnv(df=data_dict, config=config, mode="test")
    obs, _ = env.reset()

    done = False
    step_count = 0
    print("Running visualization episode...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        if step_count % 5000 == 0:
            print(f"  Step {step_count} | NAV: ${info['portfolio_value']:,.2f} | Return: {info['return_pct']:+.2f}%")

    # Final stats
    final_info = env._get_info()
    print(f"\n{'='*50}")
    print(f"Episode finished after {step_count} steps")
    print(f"Final NAV: ${final_info['portfolio_value']:,.2f}")
    print(f"Return: {final_info['return_pct']:+.2f}%")
    print(f"Total trades: {final_info['n_trades']}")
    print(f"{'='*50}\n")

    # Generate HTML
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(ROOT_DIR, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    render_visualization(env, output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualization from a saved model")
    parser.add_argument("--model", type=str, default="models/saved/sac_final",
                        help="Path to saved model (.zip)")
    parser.add_argument("--algo", type=str, default="SAC", help="RL algorithm (SAC/PPO)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="logs/viz_sac.html",
                        help="Output HTML file path")

    args = parser.parse_args()
    run_visualization(args)
