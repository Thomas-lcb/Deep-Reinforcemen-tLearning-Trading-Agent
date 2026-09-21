"""
training/curriculum.py — Apprentissage par paliers progressifs (Curriculum Learning).

L'entraînement par renforcement est souvent plus efficace si la difficulté
augmente progressivement. Ce script définit et exécute 4 niveaux de difficulté
pour le modèle PPO.

Usage:
    python -m training.curriculum --device cuda
"""

import os
import copy
import argparse
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env.trading_env import CryptoTradingEnv
from training.callbacks import get_callbacks

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# W&B (optional)
try:
    import wandb
except ImportError:
    wandb = None


LEVEL_DESCRIPTIONS = {
    1: "Bases du trading (BTC/USDT, 0% frais)",
    2: "Contraintes réelles (BTC/USDT, frais Binance 0.1%)",
    3: "Résilience (BTC/USDT, Domain Randomization)",
    4: "Short-selling (BTC/USDT, vente à découvert autorisée)",
}


def set_curriculum_level(config: dict, level: int):
    """
    Modifie le config global pour correspondre au niveau de difficulté.
    Phase 4: Single pair (BTC/USDT) — difficulty via fees + domain randomization.
    Retourne la configuration modifiée.
    """
    cfg = copy.deepcopy(config)
    
    if level == 1:
        print(f"💡 Niveau 1 : {LEVEL_DESCRIPTIONS[1]}")
        cfg["market"]["pairs"] = ["BTC/USDT"]
        cfg["fees"]["maker"] = 0.0
        cfg["fees"]["taker"] = 0.0
        cfg["training"]["domain_randomization"]["enabled"] = False
        cfg["training"]["total_timesteps"] = 500_000
        # Enforce explicitly: L1 must never leak shorting even if
        # config.yaml is hand-edited to short.enabled=true.
        cfg.setdefault("short", {})["enabled"] = False

    elif level == 2:
        print(f"💡 Niveau 2 : {LEVEL_DESCRIPTIONS[2]}")
        cfg["market"]["pairs"] = ["BTC/USDT"]
        cfg["fees"]["maker"] = 0.001
        cfg["fees"]["taker"] = 0.001
        cfg["training"]["domain_randomization"]["enabled"] = False
        cfg["training"]["total_timesteps"] = 1_000_000
        # Enforce explicitly: L2 must never leak shorting even if
        # config.yaml is hand-edited to short.enabled=true.
        cfg.setdefault("short", {})["enabled"] = False

    elif level == 3:
        print(f"💡 Niveau 3 : {LEVEL_DESCRIPTIONS[3]}")
        cfg["market"]["pairs"] = ["BTC/USDT"]
        cfg["fees"]["maker"] = 0.001
        cfg["fees"]["taker"] = 0.001
        cfg["training"]["domain_randomization"]["enabled"] = True
        cfg["training"]["total_timesteps"] = 1_500_000
        # Enforce explicitly: L3 must never leak shorting even if
        # config.yaml is hand-edited to short.enabled=true.
        cfg.setdefault("short", {})["enabled"] = False

    elif level == 4:
        print(f"💡 Niveau 4 : {LEVEL_DESCRIPTIONS[4]}")
        cfg["market"]["pairs"] = ["BTC/USDT"]
        cfg["fees"]["maker"] = 0.001
        cfg["fees"]["taker"] = 0.001
        cfg["training"]["domain_randomization"]["enabled"] = True
        # setdefault (not a bare cfg["short"]["enabled"]) so this doesn't
        # KeyError when the "short" key is absent from config.yaml — every
        # new config value must have a safe default.
        cfg.setdefault("short", {})["enabled"] = True
        cfg["training"]["total_timesteps"] = 1_500_000
    else:
        raise ValueError(f"Niveau non supporté : {level}")
        
    return cfg


def load_data(config: dict) -> dict:
    """Charge toutes les paires requises par le niveau courant."""
    import pandas as pd

    pairs = config["market"]["pairs"]
    timeframe = config["market"]["timeframe"]
    train_ratio = config["data"]["train_ratio"]
    processed_dir = os.path.join(ROOT_DIR, config["paths"]["processed_data"])
    dfs = {}

    print("Loading data for curriculum level...")
    for pair in pairs:
        filename = f"{pair.replace('/', '_')}_{timeframe}_normalized.csv"
        path = os.path.join(processed_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}. Run `python -m data.download` first.")
        
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.name != 'timestamp' and 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df = df.sort_index()
        
        train_size = int(len(df) * train_ratio)
        df_train = df.iloc[:train_size].copy()
        dfs[pair] = df_train
        
    return dfs


def make_env(dfs, config, rank, run_name, seed=0):
    """Fonction utilitaire pour créer les environnements parallèles avec Monitor."""
    def _init():
        env = CryptoTradingEnv(df=dfs, config=config, mode="train")
        env.reset(seed=seed + rank)
        
        # Le callback lit les fichiers *monitor.csv dans logs/monitor_run_name/
        log_dir = os.path.join(ROOT_DIR, "logs", f"monitor_{run_name}")
        os.makedirs(log_dir, exist_ok=True)
        # Chaque env parallèle aura son propre fichier
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Run PPO Curriculum Learning")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4],
                         help="Run only this single level instead of the full 1→4 curriculum "
                              "(useful for quick validation runs).")
    parser.add_argument("--timesteps", type=int, default=None,
                         help="Override total_timesteps for the run(s) (useful for short "
                              "validation runs instead of the full 500k/1M/1.5M schedule).")
    args = parser.parse_args()

    # Load base config
    config_path = os.path.join(ROOT_DIR, "config", "config.yaml")
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    from training.train import _parse_learning_rate
    # Paramètres globaux
    n_envs = base_config["training"].get("n_envs", 4)
    ppo_cfg = base_config["training"]["ppo"]
    model = None
    previous_model_path = None

    levels_to_run = [args.level] if args.level is not None else range(1, 5)
    if args.level is not None and args.level > 1:
        previous_model_path = os.path.join(ROOT_DIR, "models", "saved", f"ppo_curriculum_l{args.level - 1}")
        if not os.path.exists(previous_model_path + ".zip"):
            raise FileNotFoundError(
                f"--level {args.level} requires a previous checkpoint at {previous_model_path}.zip "
                f"(run level {args.level - 1} first, or omit --level to run the full curriculum)."
            )

    for level in levels_to_run:
        print(f"\n==========================================")
        print(f"🚀 LANCEMENT DU CURRICULUM NIVEAU {level}")
        print(f"==========================================")

        # 1. Configurer le niveau
        curriculum_cfg = set_curriculum_level(base_config, level)
        timesteps = args.timesteps if args.timesteps is not None else curriculum_cfg["training"]["total_timesteps"]
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"PPO_Curriculum_L{level}_{timestamp}"

        # 2. Initialiser W&B pour ce niveau
        if wandb is not None:
            tags = ["curriculum", f"level_{level}", "PPO"]
            if args.timesteps is not None:
                tags.append("quick_validation")
            wandb.init(
                project="RLD-Trading",
                name=run_name,
                group="curriculum_learning",
                tags=tags,
                config={
                    "algo": "PPO",
                    "level": level,
                    "level_description": LEVEL_DESCRIPTIONS[level],
                    "pairs": curriculum_cfg["market"]["pairs"],
                    "timesteps": timesteps,
                    "n_envs": n_envs,
                    "batch_size": ppo_cfg["batch_size"],
                    "n_steps": ppo_cfg["n_steps"],
                    "n_epochs": ppo_cfg["n_epochs"],
                    "learning_rate": ppo_cfg["learning_rate"],
                    "gamma": ppo_cfg["gamma"],
                    "gae_lambda": ppo_cfg["gae_lambda"],
                    "clip_range": ppo_cfg["clip_range"],
                    "ent_coef": ppo_cfg["ent_coef"],
                    "net_arch": ppo_cfg.get("policy_kwargs", {}).get("net_arch", []),
                    "fees_maker": curriculum_cfg["fees"]["maker"],
                    "fees_taker": curriculum_cfg["fees"]["taker"],
                    "domain_randomization": curriculum_cfg["training"]["domain_randomization"]["enabled"],
                    "reward_scale": curriculum_cfg["reward"].get("reward_scale", 1.0),
                    "unrealized_pnl_weight": curriculum_cfg["reward"].get("unrealized_pnl_weight", 0.0),
                    "drawdown_threshold": curriculum_cfg["reward"].get("drawdown_penalty", {}).get("threshold_pct", 0.05),
                    "drawdown_factor": curriculum_cfg["reward"].get("drawdown_penalty", {}).get("penalty_factor", 2.0),
                },
                sync_tensorboard=True,  # Sync TensorBoard logs to W&B automatically
                reinit=True,  # Allow re-init for multi-level runs
            )
            print(f"📊 W&B initialisé : projet=RLD-Trading, run={run_name}")

        # 3. Charger la data (adaptée aux paires de ce niveau)
        dfs = load_data(curriculum_cfg)
        
        # 4. Créer les environnements vectorisés
        env = SubprocVecEnv([make_env(dfs, curriculum_cfg, i, run_name) for i in range(n_envs)], start_method="spawn")

        
        # 5. Créer ou charger le modèle
        if level == 1:
            lr = _parse_learning_rate(ppo_cfg["learning_rate"])
            print(f"Création d'un nouveau modèle PPO (net_arch={ppo_cfg.get('policy_kwargs', {}).get('net_arch', [])})...")
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=lr,
                n_steps=ppo_cfg["n_steps"],
                batch_size=ppo_cfg["batch_size"],
                n_epochs=ppo_cfg["n_epochs"],
                gamma=ppo_cfg["gamma"],
                gae_lambda=ppo_cfg["gae_lambda"],
                clip_range=ppo_cfg["clip_range"],
                ent_coef=ppo_cfg["ent_coef"],
                target_kl=ppo_cfg.get("target_kl"),
                policy_kwargs=ppo_cfg.get("policy_kwargs"),
                device=args.device,
                tensorboard_log=os.path.join(ROOT_DIR, "logs", "tensorboard"),
                verbose=1
            )
        else:
            print(f"Transfert des poids depuis : {previous_model_path}")
            model = PPO.load(previous_model_path, env=env, device=args.device)
            model.tensorboard_log = os.path.join(ROOT_DIR, "logs", "tensorboard")

        # 6. Entraînement
        callbacks = get_callbacks(algo="PPO", run_name=run_name, config=curriculum_cfg)
        
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=run_name,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,  # Réactivation de la barre de progression demandée par l'utilisateur
        )
        
        # 7. Sauvegarde pour le prochain niveau
        save_dir = os.path.join(ROOT_DIR, "models", "saved")
        os.makedirs(save_dir, exist_ok=True)
        previous_model_path = os.path.join(save_dir, f"ppo_curriculum_l{level}")
        model.save(previous_model_path)
        print(f"✅ Niveau {level} terminé. Modèle sauvegardé ({previous_model_path}.zip).\n")

        # 8. Finaliser le run W&B pour ce niveau
        if wandb is not None and wandb.run is not None:
            wandb.finish()

    print("🎉 CURRICULUM LEARNING TERMINÉ ! (L'agent a complété tous les niveaux)")


if __name__ == "__main__":
    main()
