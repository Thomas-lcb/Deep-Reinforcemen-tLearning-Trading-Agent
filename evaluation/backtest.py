"""
evaluation/backtest.py — Backtest réel : modèle entraîné vs Buy & Hold vs
baseline aléatoire, sur le split test (jamais vu pendant l'entraînement).

Usage:
    python -m evaluation.backtest --level 4 --device cuda
    python -m evaluation.backtest --model models/saved/ppo_curriculum_l4 --level 4
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from env.trading_env import CryptoTradingEnv
from evaluation import metrics
from evaluation.benchmark import buy_and_hold_curve, random_baseline_curve
from evaluation.visualization import render_visualization
from training.curriculum import set_curriculum_level

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

PERIODS_PER_YEAR_1M = 525_600  # 60 * 24 * 365, for 1-minute bars


def extract_test_slice(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    """
    Extracts the chronological test slice (everything after train+val),
    matching the split convention already used by data/download.py and
    training/curriculum.py's load_data().
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df.iloc[val_end:].copy()


def load_test_data(config: dict) -> dict:
    """Loads only the held-out test slice for every pair in config['market']['pairs']."""
    pairs = config["market"]["pairs"]
    timeframe = config["market"]["timeframe"]
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]
    processed_dir = os.path.join(ROOT_DIR, config["paths"]["processed_data"])

    dfs = {}
    for pair in pairs:
        filename = f"{pair.replace('/', '_')}_{timeframe}_normalized.csv"
        path = os.path.join(processed_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}. Run `python -m data.download` first.")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()
        dfs[pair] = extract_test_slice(df, train_ratio, val_ratio)
    return dfs


def run_model_episode(model, dfs: dict, config: dict) -> CryptoTradingEnv:
    """Runs one full deterministic pass over the test set, returns the env (holds all history)."""
    env = CryptoTradingEnv(df=dfs, config=config, mode="test")
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env


def print_comparison_table(results: dict):
    """results: {label: metrics_dict} — prints a simple aligned text table."""
    labels = list(results.keys())
    rows = [
        ("Rendement total", "total_return_pct", "{:+.2f}%"),
        ("Sharpe", "sharpe", "{:.3f}"),
        ("Sortino", "sortino", "{:.3f}"),
        ("Max drawdown", "max_drawdown", "{:.2%}"),
        ("Calmar", "calmar", "{:.3f}"),
        ("Win rate", "win_rate", "{:.1%}"),
        ("Profit factor", "profit_factor", "{:.3f}"),
    ]
    col_width = max(18, max(len(l) for l in labels) + 2)
    header = "Métrique".ljust(20) + "".join(l.ljust(col_width) for l in labels)
    print(header)
    print("-" * len(header))
    for label, key, fmt in rows:
        line = label.ljust(20)
        for res_label in labels:
            value = results[res_label][key]
            cell = "N/A" if (value is None or (isinstance(value, float) and np.isnan(value))) else fmt.format(value)
            line += cell.ljust(col_width)
        print(line)


def run_backtest(args):
    with open(args.config or CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)

    config = set_curriculum_level(base_config, args.level)
    dfs = load_test_data(config)
    print(f"Test set chargé : {sum(len(df) for df in dfs.values())} lignes ({args.level=})")

    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT_DIR, model_path)
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    print(f"Chargement du modèle : {model_path}")
    model = PPO.load(model_path, device=args.device)

    print("Run déterministe du modèle sur le test set...")
    env = run_model_episode(model, dfs, config)
    portfolio_history = env.get_portfolio_history()
    trade_history = env.get_trade_history()
    equity_model = portfolio_history["value"].values

    close_prices_used = env.close_prices[env.lookback_window : env.current_step + 1]

    print("Calcul des références (Buy & Hold, aléatoire)...")
    equity_buyhold = buy_and_hold_curve(
        close_prices_used, initial_capital=env.initial_capital, fee_rate=env.base_fee_rate,
    )
    equity_random, trades_random = random_baseline_curve(dfs, config, seed=123, mode="test")

    empty_trades = pd.DataFrame({"pnl_pct": pd.Series(dtype=float)})
    results = {
        "Modèle": metrics.summarize(equity_model, trade_history, PERIODS_PER_YEAR_1M),
        "Buy & Hold": metrics.summarize(equity_buyhold, empty_trades, PERIODS_PER_YEAR_1M),
        "Aléatoire": metrics.summarize(equity_random, trades_random, PERIODS_PER_YEAR_1M),
    }

    print(f"\n{'='*70}")
    print(f"BACKTEST — Niveau {args.level} — {len(equity_model)} steps sur données jamais vues")
    print(f"{'='*70}")
    print_comparison_table(results)
    print(f"{'='*70}\n")

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(ROOT_DIR, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    render_visualization(
        env, output_path,
        benchmark_curves={"Buy & Hold": equity_buyhold, "Aléatoire": equity_random},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest réel : modèle vs Buy & Hold vs aléatoire")
    parser.add_argument("--model", type=str, default=None,
                        help="Chemin du modèle (.zip). Défaut : models/saved/ppo_curriculum_l{level}")
    parser.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4],
                        help="Niveau de curriculum à backtester (détermine la config : frais, short, etc.)")
    parser.add_argument("--config", type=str, default=None, help="Chemin vers config.yaml")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="logs/backtest.html",
                        help="Chemin du fichier HTML de sortie")

    args = parser.parse_args()
    if args.model is None:
        args.model = f"models/saved/ppo_curriculum_l{args.level}"
    run_backtest(args)
