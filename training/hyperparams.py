
import optuna
from optuna.trial import TrialState
import argparse
import os
import yaml
import numpy as np

from training.train import load_data, create_vec_env, get_agent, load_config
from stable_baselines3.common.evaluation import evaluate_policy

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def objective(trial, args, data_dict):
    """
    Optuna objective function for optimizing SAC/PPO hyperparameters.
    """
    # Load base config
    config = load_config(args.config)
    
    # Define search space based on algo
    algo = args.algo.upper()
    train_cfg = config["training"]
    
    if algo == "SAC":
        # Learning rate
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        # Batch size
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        # Buffer size
        buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 500000])
        # Tau (soft update)
        tau = trial.suggest_float("tau", 0.001, 0.05)
        # Gamma
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        # Train freq
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
        # Gradient steps
        gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
        # Ent coef
        ent_coef = "auto" # Keep auto for now, maybe optimize target_entropy?

        # Update config directly in memory
        if "sac" not in train_cfg:
            train_cfg["sac"] = {}
        
        train_cfg["sac"].update({
            "learning_rate": lr,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "ent_coef": ent_coef
        })
        
    elif algo == "PPO":
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        n_epochs = trial.suggest_int("n_epochs", 5, 20)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        
        if "ppo" not in train_cfg:
            train_cfg["ppo"] = {}
            
        train_cfg["ppo"].update({
            "learning_rate": lr,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range
        })

    # Create Env
    # Use fewer envs/short episodes for speed?
    # Actually, we should use same n_envs but maybe fewer steps per trial.
    n_envs = config["training"].get("n_envs", 1)
    
    # We need separate Train and Eval envs?
    # standard practice: train on train set, eval on val set.
    # CryptoTradingEnv mode="train" handles splitting internally?
    # No, Env loads ALL data. Split logic is inside Env?
    # Let's check Env. Env takes 'mode' argument.
    # If mode="train", it uses only training indices.
    # If mode="val", it uses validation indices.
    
    try:
        train_env = create_vec_env(data_dict, config, n_envs=n_envs, seed=args.seed, mode="train")
        eval_env = create_vec_env(data_dict, config, n_envs=1, seed=args.seed + 100, mode="val")
        
        tensorboard_log = None # Disable TB for optimization to save disk? Or keep "logs/optuna"?
        device = "cuda" if args.device == "cuda" else "cpu"
        
        model = get_agent(algo, train_env, config, device, tensorboard_log)
        
        # Train for short duration (e.g. 50k steps)
        # Using PruningCallback? Optuna supports pruning.
        # But for RL it's noisy. Let's just run fixed steps.
        steps = args.steps or 50000
        
        model.learn(total_timesteps=steps)
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
        
        train_env.close()
        eval_env.close()
        
        return mean_reward

    except Exception as e:
        print(f"Trial failed: {e}")
        # Return very low reward?
        return -float('inf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--algo", type=str, default="SAC")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20000, help="Steps per trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--study_name", type=str, default="crypto_rld_study")
    
    args = parser.parse_args()
    
    # Load data ONCE
    config = load_config(args.config)
    data_dict = load_data(config)
    
    storage_path = f"sqlite:///logs/optuna_{args.study_name}.db"
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage_path,
        load_if_exists=True
    )
    
    print(f"Starting optimization with {args.trials} trials...")
    
    study.optimize(lambda trial: objective(trial, args, data_dict), n_trials=args.trials)
    
    print("Optimization finished.")
    print("Best params:")
    print(study.best_params)
    print("Best value:", study.best_value)
    
    # Save best params to YAML
    best_params_path = os.path.join(ROOT_DIR, "config", f"best_params_{args.algo.lower()}.yaml")
    with open(best_params_path, "w") as f:
        yaml.dump(study.best_params, f)
    print(f"Saved best params to {best_params_path}")
