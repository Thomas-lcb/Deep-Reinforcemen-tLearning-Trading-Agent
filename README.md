# RLD Trading Agent 🚀📈

An autonomous cryptocurrency trading agent powered by Deep Reinforcement Learning (DRL), built with PyTorch, Stable-Baselines3, and Gymnasium.

## 🧠 Project Overview

The goal of this project is to train an RL agent capable of trading cryptocurrency pairs on Binance, maximizing returns while strictly managing risk and drawdowns. The agent learns from historical OHLCV data enriched with technical indicators and macro-economic trends.

### Key Milestones & Iterations

1. **Test 1: SAC (Soft Actor-Critic)**
   - **Approach:** Evaluated SAC due to its efficiency in continuous action spaces.
   - **Result:** The agent found a degenerate policy (exploration entropy collapsed), leading to a high artificial portfolio value but poor generalization and negative real rewards. Over-trading heavily impacted returns due to fees.
   - **Solution:** Switched to PPO for more stable, on-policy learning.

2. **Test 2: PPO (Proximal Policy Optimization)**
   - **Approach:** Standard PPO with strict penalties for drawdowns to ensure capital preservation.
   - **Result:** The agent suffered from "learning anxiety." The harsh drawdown penalties in a highly volatile, multi-currency environment caused the agent to stop trading entirely to avoid being penalized, resulting in a slow bleed from time-based holding costs.

3. **Phase 3: Curriculum Learning & Hold Rewards (Current)**
   - **Curriculum Learning:** To counter the agent's anxiety, we implemented a 3-level curriculum (`curriculum.py`):
     - **Level 1 (Kindergarten):** BTC & ETH only, 0% fees. Focuses purely on recognizing basic market structures without the stress of losing capital to fees.
     - **Level 2 (Intermediate):** Adds BNB, SOL, ADA, and introduces real Binance trading fees. Teaches the agent the cost of over-trading (click rarity).
     - **Level 3 (Hard Mode):** Introduces all 10 cryptos, random capital variations, and randomized fees (Domain Randomization) to ensure robust generalization.
   - **Unrealized PNL Reward:** Instead of only rewarding closed trades, the agent receives a continuous reward for holding winning positions (Unrealized PNL). This encourages trend-following behavior and discourages panic selling at the first red candle.
   - **Multi-Environment Batches:** Utilizing `SubprocVecEnv` with `start_method="spawn"`, we run 4 parallel gym environments to gather massive amounts of diverse experience simultaneously, heavily utilizing the GPU for updates.


## 🏗️ Architecture

- **`data/` & `features/`:**
  - `download.py`: Fetches 5 years of 1H OHLCV data for top 10 cryptos via CCXT. Splits into Train (70%), Val (15%), Test (15%).
  - `indicators.py`: Calculates MACD, RSI, Bollinger Bands, ATR, etc.
  - `multi_timeframe.py`: Injects higher timeframe contexts (4H, 1D, 1W) into the 1H base timeframe so the agent perceives macro trends.
  - `normalizer.py`: **Crucial step.** Converts raw prices into sliding Z-Scores. The agent never sees "$65,000", only "+2% above the 30h moving average", making the model asset-agnostic.
- **`env/`:**
  - `trading_env.py`: The Custom Gymnasium environment.
  - `observation.py`: Provides the sliding window (lookback 30h) of normalized states and portfolio context.
  - `action.py`: Continuous action space `[-1, 1]` representing target portfolio percentages, accompanied by a dead-zone `[-0.05, 0.05]` to avoid micro-trades.
  - `reward.py`: Custom reward function heavily penalizing 3% drawdowns and rewarding unrealized PnL, Sharpe ratio, and trend alignment.
- **`training/`:**
  - `curriculum.py`: Manages the curriculum progression and dynamically adjusts difficulties.
  - `train.py`: The core Stable-Baselines3 setup using PPO.
  - `callbacks.py`: Handles model checkpointing and saving the best versions during training.
  - `config/config.yaml`: Unified configuration for hyperparameters, architectures, and market rules.

## 💻 Tech Stack & Requirements

- **Python:** 3.10
- **Deep Learning / RL:** PyTorch (1.13.1+cu116), Stable-Baselines3 (2.4.0), Gymnasium
- **Data processing:** Pandas, NumPy, pandas-ta
- **Market Data:** CCXT
- **Experiment Tracking:** Weights & Biases (W&B), TensorBoard

### Note on Repository Contents
*The trained neural network weights (`.zip` models), the raw/processed datasets (`data/`), and the Python Virtual Environment (`venv/`) are excluded from this repository due to their large size. Only the source code and configuration files are versioned.*

## 🚀 Quick Start

1. **Install dependencies:**
   We recommend using `uv` for fast dependency resolution.
   ```bash
   uv venv --python 3.10 venv
   source venv/bin/activate
   uv pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   uv pip install -r requirements.txt
   ```

2. **Download Historical Data:**
   ```bash
   python -m data.download
   ```

3. **Start Curriculum Training:**
   ```bash
   python -m training.curriculum
   ```
