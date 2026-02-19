# NEXT STEPS — Plan d'exécution du projet

> Référence technique complète : [agent.md](file:///mnt/Data/Projets/Code/RLD_Trading/agent.md)

---

## Phase 1 : Setup & Data Engineering
*cf. agent.md §4 (Architecture) et §6 Phase 1*

- [x] 1.1 — Initialiser la structure du projet (arborescence de fichiers cf. agent.md §4)
- [x] 1.2 — Créer `requirements.txt` avec toutes les dépendances
- [x] 1.3 — Créer `config/config.yaml` (hyperparamètres, frais, seuils de risque)
- [x] 1.4 — Implémenter `data/download.py` : téléchargement OHLCV via CCXT (BTC/USDT, ETH/USDT, SOL/USDT, 1H, 5 ans)
- [x] 1.5 — Implémenter `features/indicators.py` : calcul des indicateurs techniques (cf. agent.md §5.A.2)
- [x] 1.6 — Implémenter `features/normalizer.py` : rolling z-score (cf. agent.md §5.A.1 — anti data-leakage)
- [x] 1.7 — Implémenter `features/multi_timeframe.py` : agrégation 4H et 1D (cf. agent.md §5.A.3)
- [x] 1.8 — Script de split Train/Val/Test (70/15/15, chronologique) — intégré dans `data/download.py`
- [ ] 1.9 — Data augmentation : bruit gaussien, jitter volumes, inversion temporelle

---

## Phase 2 : Environnement Gym
*cf. agent.md §5 (Conception Env RL) et §6 Phase 2*

- [x] 2.1 — Implémenter `env/observation.py` : construction de l'observation space (OHLCV + indicateurs + portefeuille + multi-TF)
- [x] 2.2 — Implémenter `env/action.py` : action space continu avec dead zone `[-0.05, 0.05]` (cf. agent.md §5.B)
- [x] 2.3 — Implémenter `env/reward.py` : reward function avec log-return, pénalité frais/volatilité/drawdown (cf. agent.md §5.C)
- [x] 2.4 — Implémenter `env/trading_env.py` : `CryptoTradingEnv(gym.Env)` principal — `step()`, `reset()`, domain randomization
- [x] 2.5 — Écrire `tests/test_env.py` : tests unitaires de l'env (dimensions obs, bornes actions, frais appliqués)
- [x] 2.6 — Écrire `tests/test_reward.py` : tests de la reward function

---

## Phase 3 : Entraînement
*cf. agent.md §6 Phase 3 et §8 (Techniques avancées)*

- [x] 3.1 — Implémenter `training/train.py` : script d'entraînement SAC + PPO via Stable-Baselines3
- [x] 3.2 — Implémenter `training/callbacks.py` : callbacks Tensorboard, early stopping, checkpoint
- [ ] 3.3 — Implémenter `training/curriculum.py` : curriculum learning (tendance → latéral → chaotique)
- [x] 3.4 — Implémenter `training/hyperparams.py` : recherche via Optuna (learning rate, λ, réseau, etc.)
- [/] 3.5 — Lancer le premier entraînement SAC (2M+ timesteps) *(bug Monitor corrigé, prêt à relancer)*
- [ ] 3.6 — Lancer l'entraînement PPO pour comparaison
- [ ] 3.7 — Analyser les courbes Tensorboard et sélectionner le meilleur modèle

---

## Phase 4 : Backtesting & Visualisation
*cf. agent.md §6 Phase 4*

- [x] 4.1 — Implémenter `evaluation/visualization.py` : Moteur de rendu Plotly (Candlesticks + Trades markers + Portfolio heatmap)
- [ ] 4.2 — Créer `notebooks/demo_replay.ipynb` : Script pour charger un modèle et générer une vidéo/HTML d'un épisode de 100-200 steps
- [ ] 4.3 — Implémenter `evaluation/metrics.py` : Sharpe, Sortino, Calmar, MDD, Win Rate, Profit Factor, etc.
- [ ] 4.4 — Implémenter `evaluation/benchmark.py` : stratégie Buy & Hold + baseline aléatoire
- [ ] 4.5 — Implémenter `evaluation/backtest.py` : exécution du modèle sur données out-of-sample
- [ ] 4.6 — Comparer SAC vs PPO vs Buy & Hold — décider si retour Phase 3

---

## Phase 5 : Paper Trading
*cf. agent.md §6 Phase 5*

- [ ] 5.1 — Implémenter `live/paper_trading.py` : connexion WebSocket temps réel Binance
- [ ] 5.2 — Implémenter `live/risk_manager.py` : stop-loss, circuit breaker, daily limit (cf. agent.md §7)
- [ ] 5.3 — Implémenter `live/notifier.py` : alertes Telegram/Discord
- [ ] 5.4 — Lancer le paper trading pendant 2–4 semaines
- [ ] 5.5 — Analyser l'écart backtest vs temps réel — valider ou retour Phase 3

---

## Phase 6 : Live Trading
*cf. agent.md §6 Phase 6 et §7 (Sécurité)*

- [ ] 6.1 — Implémenter `live/live_trading.py` : exécution réelle
- [ ] 6.2 — Configurer `.env` avec les clés API Binance (read + trade, **pas de withdraw**)
- [ ] 6.3 — Déployer avec capital limité (10–20% du budget)
- [ ] 6.4 — Monitoring continu + alertes
- [ ] 6.5 — Augmentation progressive du capital si performances stables 1+ mois

---

## Phase Bonus : Améliorations V2
*cf. agent.md §9 (Roadmap V2)*

- [ ] V2.1 — Decision Transformer (dépendances temporelles longues)
- [x] V2.2 — Sentiment Analysis : `data/sentiment.py` implémenté (Fear & Greed Index via alternative.me API)
- [ ] V2.3 — Multi-agent (un agent par timeframe)
- [ ] V2.4 — Meta-learning (MAML) pour adaptation aux régimes de marché
- [ ] V2.5 — Inverse RL / RLHF (apprendre la reward d'experts)
- [ ] V2.6 — Ensemble d'agents (vote majoritaire)
- [ ] V2.7 — Order Book features
