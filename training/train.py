
import os
import argparse
import glob
import time
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from env.trading_env import CryptoTradingEnv
from training.callbacks import TensorboardCallback, SaveOnBestTrainingRewardCallback

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

def load_config(path=None):
    path = path or CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    """
    Load all processed/normalized CSVs from the data directory.
    Returns a dictionary {pair: dataframe}.
    """
    data_dir = os.path.join(ROOT_DIR, config["paths"]["processed_data"])
    # Look for *_normalized.csv
    pattern = os.path.join(data_dir, "*_normalized.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No data found in {data_dir}. Run data.download first.")
        
    data_dict = {}
    print(f"Loading data from {len(files)} files...")
    
    for f in files:
        # Extract pair name from filename (e.g. BTC_USDT_1h_normalized.csv)
        filename = os.path.basename(f)
        # Assuming format PAIR_TF_normalized.csv
        # We can just use the filename as key or try to reconstruct pair
        pair_key = filename.replace("_normalized.csv", "")
        
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        # Ensure raw prices are present (should be raw_close etc.)
        if "raw_close" not in df.columns and "close" in df.columns:
             # Fallback if raw_close missing (should not happen with new download)
             print(f"Warning: {filename} missing raw_close. Using close (normalized!)")
        
        data_dict[pair_key] = df
        print(f"  Loaded {pair_key}: {len(df)} rows")
        
    return data_dict

LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(data_dict, config, mode="train", rank=0, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = CryptoTradingEnv(df=data_dict, config=config, mode=mode)
        # Each env gets a unique monitor file to avoid corruption with SubprocVecEnv
        monitor_path = os.path.join(LOG_DIR, f"env_{rank}")
        env = Monitor(env, monitor_path)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vec_env(data_dict, config, n_envs=1, seed=42, mode="train"):
    """
    Create a vectorized environment (Dummy or Subproc).
    """
    if n_envs > 1:
        # Create multiple envs
        envs = [make_env(data_dict, config, mode, i, seed) for i in range(n_envs)]
        # Use SubprocVecEnv if possible, else Dummy
        vec_env = SubprocVecEnv(envs)
        print(f"Created {n_envs} parallel environments (SubprocVecEnv).")
    else:
        vec_env = DummyVecEnv([make_env(data_dict, config, mode, 0, seed)])
        print("Created 1 environment (DummyVecEnv).")
    return vec_env

def _parse_learning_rate(lr_value, total_timesteps=None):
    """
    Parse learning rate config value.
    Supports: float (constant), or 'linear_X' (linear decay from X to 0).
    """
    if isinstance(lr_value, str) and lr_value.startswith("linear_"):
        initial_lr = float(lr_value.replace("linear_", ""))
        def linear_schedule(progress_remaining: float) -> float:
            """Linear decay: lr * progress_remaining (1.0 → 0.0)."""
            return initial_lr * progress_remaining
        return linear_schedule
    return float(lr_value)


def get_agent(algo, vec_env, config, device="auto", tensorboard_log=None):
    """
    Initialize the RL agent (SAC or PPO).
    """
    algo = algo.upper()
    policy = "MlpPolicy"
    
    # Hyperparams from config
    train_cfg = config["training"]
    model_params = dict(train_cfg.get(algo.lower(), {}))
    if not model_params:
        model_params = {k: v for k, v in train_cfg.items() if k not in ["n_envs", "domain_randomization", "sac", "ppo"]}
    
    # Parse learning rate schedule
    if "learning_rate" in model_params:
        model_params["learning_rate"] = _parse_learning_rate(model_params["learning_rate"])
    
    # Extract policy_kwargs (net_arch, etc.)
    if "policy_kwargs" in model_params:
        pk = model_params["policy_kwargs"]
        # Ensure net_arch is a list of ints (YAML might parse it weirdly)
        if "net_arch" in pk:
            pk["net_arch"] = [int(x) for x in pk["net_arch"]]
        model_params["policy_kwargs"] = pk
    
    if algo == "SAC":
        return SAC(
            policy,
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            **model_params
        )
    elif algo == "PPO":
        return PPO(
            policy,
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            **model_params
        )
    else:
        raise ValueError(f"Unknown algo {algo}")

def train(args):
    config = load_config(args.config)
    
    # 1. Load Data
    data_dict = load_data(config)
    
    # 2. Create Vectorized Env
    n_envs = config["training"].get("n_envs", 1)
    vec_env = create_vec_env(data_dict, config, n_envs, args.seed, "train")

    # 3. Create Model
    tensorboard_log = os.path.join(ROOT_DIR, "logs", "tensorboard")
    device = "cuda" if args.device == "cuda" else "cpu"
    print(f"Training on {device.upper()}")
    
    model = get_agent(args.algo, vec_env, config, device, tensorboard_log)

    # 4. Callbacks
    eval_callback = SaveOnBestTrainingRewardCallback(
        check_freq=10000 // n_envs,
        log_dir=os.path.join(ROOT_DIR, "logs"),
        verbose=1
    )
    tb_callback = TensorboardCallback()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=os.path.join(ROOT_DIR, "models", "checkpoints"),
        name_prefix=f"{args.algo.lower()}_model"
    )
    
    callbacks = [eval_callback, tb_callback, checkpoint_callback]

    # 5. Train
    total_timesteps = args.steps or config["training"].get("total_timesteps", 1_000_000)
    print(f"Starting training for {total_timesteps} steps...")
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    end_time = time.time()
    
    print(f"Training finished in {end_time - start_time:.2f}s")
    
    # 6. Save Final Model
    save_path = os.path.join(ROOT_DIR, "models", "saved", f"{args.algo.lower()}_final")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    
    vec_env.close()

    # 7. Visualization (optional)
    if args.viz:
        print("Generating visualization...")
        # Create single env for visualization (Test mode)
        # We use a fresh env to record a full episode
        from env.trading_env import CryptoTradingEnv
        from evaluation.visualization import render_visualization
        
        # We need to ensure data_dict is passed correctly (it might be large, but fine for 1 env)
        viz_env = CryptoTradingEnv(df=data_dict, config=config, mode="test")
        
        obs, _ = viz_env.reset()
        done = False
        print("Running visualization episode...")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = viz_env.step(action)
            done = terminated or truncated
            
        render_visualization(viz_env, os.path.join(ROOT_DIR, "logs", f"viz_{args.algo.lower()}.html"))
        print("Visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--algo", type=str, default="SAC", help="RL Algo (SAC, PPO)")
    parser.add_argument("--steps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--optimize", action="store_true", help="Run hyperopt")
    parser.add_argument("--viz", action="store_true", help="Generate visualization after training")
    
    args = parser.parse_args()
    
    if args.optimize:
        print("Hyperopt not implemented yet.")
    else:
        train(args)
