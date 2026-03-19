
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

# --- W&B integration (optional) ---
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional metrics in Tensorboard + W&B.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.portfolio_values = []
        self.episode_rewards = []
        self.trades = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'portfolio_value' in info:
                self.logger.record('rollout/portfolio_value', info['portfolio_value'])
            
            if 'trade' in info and info['trade']['type'] != 'hold':
                self.trades.append(info['trade'])
                
            if 'episode' in info:
                pass

        return True

    def _on_rollout_end(self) -> None:
        """This event is triggered before updating the policy."""
        if self.trades:
            pnl_list = [t['pnl_pct'] for t in self.trades if 'pnl_pct' in t]
            if pnl_list:
                win_rate = np.mean([1 if p > 0 else 0 for p in pnl_list])
                avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
                avg_loss = np.mean([p for p in pnl_list if p < 0]) if any(p < 0 for p in pnl_list) else 0
                profit_factor = abs(sum([p for p in pnl_list if p > 0]) / sum([p for p in pnl_list if p < 0])) if any(p < 0 for p in pnl_list) else float('inf')
                
                self.logger.record('trading/win_rate', win_rate)
                self.logger.record('trading/avg_win', avg_win)
                self.logger.record('trading/avg_loss', avg_loss)
                self.logger.record('trading/profit_factor', profit_factor)
                self.logger.record('trading/trades_count', len(self.trades))
            
            self.trades = []


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the training reward.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Filter out any corrupted string values from CSV
                y_clean = []
                for val in y:
                    try:
                        y_clean.append(float(val))
                    except (ValueError, TypeError):
                        pass
                
                if len(y_clean) > 0:
                    mean_reward = np.mean(y_clean[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}.zip")
                        self.model.save(self.save_path)

        return True


# ---------------------------------------------------------------------------
# Factory function — builds the list of callbacks for any training run
# ---------------------------------------------------------------------------

def get_callbacks(algo: str = "PPO", run_name: str = "run", config: dict = None):
    """
    Build and return a CallbackList with TensorBoard + optional W&B callbacks.

    Args:
        algo: Algorithm name (PPO, SAC).
        run_name: Name of the run (used for W&B run name).
        config: Optional config dict to log as hyperparameters in W&B.

    Returns:
        CallbackList ready to pass to model.learn().
    """
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_envs = config["training"].get("n_envs", 1) if config and "training" in config else 1
    
    callbacks = [
        TensorboardCallback(),
        SaveOnBestTrainingRewardCallback(
            check_freq=max(1, 10000 // n_envs),
            log_dir=os.path.join(ROOT_DIR, "logs", f"monitor_{run_name}"),
            verbose=1
        ),
        CheckpointCallback(
            save_freq=max(1, 50000 // n_envs),
            save_path=os.path.join(ROOT_DIR, "models", "checkpoints"),
            name_prefix=f"{algo.lower()}_{run_name}"
        )
    ]

    if WANDB_AVAILABLE:
        wandb_cb = WandbCallback(
            verbose=2,
            model_save_path=None,  # We handle saving ourselves
            model_save_freq=0,
        )
        callbacks.append(wandb_cb)
        print(f"📊 W&B callback activé pour le run '{run_name}'")
    else:
        print("⚠️ wandb non installé — logs TensorBoard uniquement")

    return CallbackList(callbacks)
