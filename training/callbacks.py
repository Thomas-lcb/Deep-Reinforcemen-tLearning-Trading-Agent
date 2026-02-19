
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional metrics in Tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.portfolio_values = []
        self.episode_rewards = []
        self.trades = []
        
    def _on_step(self) -> bool:
        # Access the info dict from the environment
        # In a vectorized env, self.locals['infos'] is a list of dicts
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'portfolio_value' in info:
                self.logger.record('rollout/portfolio_value', info['portfolio_value'])
            
            if 'trade' in info and info['trade']['type'] != 'hold':
                self.trades.append(info['trade'])
                
            # If episode done
            if 'episode' in info:
                # 'episode' key is added by RecordEpisodeStatistics wrapper if used, 
                # OR by the Monitor wrapper. 
                # But our env might not be wrapped yet. 
                # Let's rely on our own info keys if possible
                pass

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Log aggregated trade stats
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
            
            # Reset trades buffer for next rollout to avoid memory leak or stale stats?
            # Or keep cumulative? Usually rollout stats are instantaneous.
            self.trades = []

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
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
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean reward for last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True
