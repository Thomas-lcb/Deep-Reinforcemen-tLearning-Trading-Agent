"""
training/sync_old_runs.py — Synchronise les anciens logs TensorBoard vers W&B.

Usage:
    python -m training.sync_old_runs
"""

import os
import wandb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB_DIR = os.path.join(ROOT_DIR, "logs", "tensorboard")

# Mapping des anciens runs vers des noms lisibles
OLD_RUNS = {
    "SAC_3": {"algo": "SAC", "timesteps": 1_000_000, "description": "Premier run SAC complet (1M steps)"},
    "SAC_4": {"algo": "SAC", "timesteps": 1_000_000, "description": "Run SAC supplémentaire"},
    "PPO_1": {"algo": "PPO", "timesteps": 2_000_000, "description": "Premier run PPO baseline"},
    "PPO_2": {"algo": "PPO", "timesteps": 2_000_000, "description": "Run PPO avec améliorations config"},
}


def main():
    print("🔄 Synchronisation des anciens logs TensorBoard vers W&B...")
    print(f"   Dossier source : {TB_DIR}\n")

    for run_dir_name, meta in OLD_RUNS.items():
        run_path = os.path.join(TB_DIR, run_dir_name)
        
        if not os.path.isdir(run_path):
            print(f"⏭️  {run_dir_name} — dossier introuvable, ignoré.")
            continue

        print(f"📤 Syncing {run_dir_name}...")
        wandb.init(
            project="RLD-Trading",
            name=run_dir_name,
            group="historical_runs",
            config={
                "algo": meta["algo"],
                "timesteps": meta["timesteps"],
                "description": meta["description"],
            },
            sync_tensorboard=True,
            reinit=True,
        )

        # Utilise wandb sync pour les tfevents
        wandb.tensorboard.patch(root_logdir=run_path)
        wandb.finish()
        print(f"   ✅ {run_dir_name} synchronisé.\n")

    print("🎉 Tous les anciens runs ont été envoyés vers W&B !")
    print("   → Rendez-vous sur https://wandb.ai pour les consulter.")


if __name__ == "__main__":
    main()
