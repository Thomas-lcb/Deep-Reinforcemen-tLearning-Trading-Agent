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

3. **Phase 3: Curriculum Learning & Hold Rewards**
   - **Curriculum Learning:** To counter the agent's anxiety, we implemented a 3-level curriculum (`curriculum.py`):
     - **Level 1 (Kindergarten):** BTC & ETH only, 0% fees. Focuses purely on recognizing basic market structures without the stress of losing capital to fees.
     - **Level 2 (Intermediate):** Adds BNB, SOL, ADA, and introduces real Binance trading fees. Teaches the agent the cost of over-trading (click rarity).
     - **Level 3 (Hard Mode):** Introduces all 10 cryptos, random capital variations, and randomized fees (Domain Randomization) to ensure robust generalization.
   - **Unrealized PNL Reward:** Instead of only rewarding closed trades, the agent receives a continuous reward for holding winning positions (Unrealized PNL). This encourages trend-following behavior and discourages panic selling at the first red candle.
   - **Multi-Environment Batches:** Utilizing `SubprocVecEnv` with `start_method="spawn"`, we run 4 parallel gym environments to gather massive amounts of diverse experience simultaneously, heavily utilizing the GPU for updates.
   - **Result:** TensorBoard analysis showed that **Level 1 performed well** — stable value loss, explained variance close to 1, and full-length episodes (~30k steps). However, the **transition to Levels 2 and 3 caused a collapse**: `explained_variance` dropped below zero (the critic could no longer predict returns), `ep_len_mean` plummeted (episodes truncated early due to bankruptcy/max drawdown), and `value_loss` became erratic with large spikes. The 10-pair environment in L3 introduced too much noise for the current architecture to generalize. GPU utilization was high and efficient throughout all levels.

4. **Phase 4: Single-Pair High-Frequency Focus**
   - **Single Pair:** Refocused on `BTC/USDT` only — if the agent can't beat fees on one pair, it won't on ten.
   - **1-minute candles:** Switched from `1h` to `1m` timeframe, giving the agent 60× more data points per hour to capture intraday micro-trends, volatility spikes, and local extrema.
   - **Reduced Network:** Architecture downsized from `[512, 512, 256]` to `[256, 256]` to prevent overfitting on noisy minute-level data.
   - **Stable Learning Rate:** Replaced linear decay `0.0007→0` with a constant `3e-4` to avoid destroying learned strategies mid-training.
   - **Multi-Timeframe Context:** Adjusted to `[5m, 15m, 1h]` from the 1m base to provide macro context at the appropriate scales.
   - **Result (Failure):** The agent collapsed under high-frequency market noise and fees. The standard deviation (`train/std`) plummeted to zero instantly in Level 1, indicating premature convergence to a "do nothing" deterministic policy. The value network completely failed to extract signal from 1m noise (`explained_variance` wildly negative). The combination of a high learning rate (`3e-4`) and small batches (2GB VRAM limitation) caused violent, unstable policy updates (huge `approx_kl` spikes and 60-70% `clip_fraction`). Finally, attempting to trade 1m micro-movements caused transaction fees to obliterate the portfolio in Levels 2 & 3, driving `ep_rew_mean` to -4000 and forcing early episode truncation (`ep_len_mean` drop) due to bankruptcy.

## 5. Phase 5: High-Performance Architecture & Deep Features (Proposed)
**Objective:** Evolve from a basic ML approach to an institutional-grade, robust trading agent, respecting the 2GB VRAM hardware limits.

### A. Temporal & Data Resolution
- **Timeframe:** `15m` Candles. 1m is pure noise; 1h lacks intraday execution precision. `15m` is the institutional day-trading sweet spot.
- **Training Duration:** Increase from 2M to 5M-10M timesteps. The agent needs significantly more exposure to market cycles to learn robust strategies instead of overfitting early.

### B. Network Architecture
- **Structure:** Upgrading from `[256, 256]` to `[256, 256, 128]` with orthogonal initialization. A massive network (`[1024, 1024, 512]`) overfits financial noise and crashes the 2GB VRAM with large batches. We must keep it lean but deep enough for feature extraction.
- **Learning Rate:** Dropped to a constant `5e-5` to stabilize gradient updates and prevent the violent policy collapses (KL spikes) observed in Phase 4.

### C. Missing Data Context (The "Edge")
- **Temporal Embeddings:** The agent currently has no concept of time! We will add `hour_of_day` (sin/cos encoding) and `day_of_week`. Asian, London, and NY trading sessions have entirely different volatility profiles.
- **Order Book Imbalance & Funding Rates:** (Future scope if data available). Knowing the leverage tilt (long vs short) via Funding Rates is the most powerful predictor of crypto liquidations.
- **Macro Correlation:** Integrating S&P500 or DXY (US Dollar Index) trends to give the agent macroeconomic context.

### D. Technical Indicator Enhancements
- **Fix ADX Missing Implementation:** `adx` is listed in `config.yaml` but not computed in `indicators.py`. We must implement it to give the agent a "trend strength" filter.
- **VWAP (Volume Weighted Average Price):** The ultimate institutional intraday benchmark, sorely missing from the current feature set.
- **Regime Filter:** Explicitly feeding the agent a "Market Regime" state (Ranging, Trending Up, Trending Down, High Volatility).
- **Action Smoothing & Wider Dead-Zone:** Reward shaping to severely penalize "action flipping" (going from 100% Long to 100% Short in one candle) to mitigate fee bleed, and expanding the neutral dead-zone `[-0.1, 0.1]`.

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
- **Training Hardware:** NVIDIA GPU (Quadro K620, driver version 470.256.02) with 2 Go VRAM (CUDA 11.6) — justifies the `cu116` builds and the reduced batch sizes in `config.yaml`

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
