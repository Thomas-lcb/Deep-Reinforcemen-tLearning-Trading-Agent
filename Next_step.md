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
- [x] 1.7 — Implémenter `features/multi_timeframe.py` : agrégation 4H, 1D, 1W (macro)
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
- [x] 3.3 — Implémenter `training/curriculum.py` : curriculum learning progressif (3 niveaux de difficulté)
- [x] 3.4 — Implémenter `training/hyperparams.py` : recherche via Optuna (learning rate, λ, réseau, etc.)
- [x] 3.5 — Lancer le premier entraînement SAC (2M+ timesteps) *(Terminé, analysé)*
- [x] 3.6 — Lancer l'entraînement PPO pour comparaison *(Terminé, analysé)*
- [x] 3.7 — Analyser les courbes Tensorboard et sélectionner le meilleur modèle *(PPO conservateur, nécessité de Curriculum)*
- [x] 3.8 — Lancer l'entraînement Curriculum Learning PPO sur 10 paires & 1W *(Terminé)*
- [x] 3.9 — Intégration de Weights & Biases (W&B) pour le suivi des expériences *(Terminé)*

---

## Phase 3-bis : Correctifs issus de l'analyse forensique W&B (Sept 2026)
*cf. REPO_OVERVIEW.md §4 pour le détail chiffré*

- [x] 3b.1 — Fixer `drawdown_penalty` (`env/reward.py`) : ne pénaliser que l'aggravation du drawdown, pas son maintien permanent (tests de non-régression dans `tests/test_reward.py`)
- [x] 3b.2 — `target_kl=0.02` ajouté au PPO (`config.yaml`+`training/curriculum.py`). **Confirmé** par un run de validation complet (Niveau 1, 500k steps, W&B `k1sg9nbk`, 18/09) vs le run précédent sans target_kl (`w4f2hkxx`) : `approx_kl` reste collé à 0.02 (au lieu d'exploser à 0.55), `train/std` ne s'effondre plus (0.92 final au lieu de 0.097), `clip_fraction` sain (0.15 au lieu de 0.59), `ep_rew_mean` inchangé (toujours positif ~2.3). Léger compromis : `explained_variance` un peu moins bon (-0.21 vs +0.31) — attendu, moins de gradient par rollout.
- [x] 3b.3 — Activer la randomisation du point de départ dès L1/L2 (pas seulement L3) et plafonner la longueur d'épisode (`training.max_episode_steps`, `env/trading_env.py`) — sur le dataset 1 an actuel un épisode non plafonné dure ~367k steps (plus long qu'un niveau entier), donc `ep_rew_mean` ne se peuplait jamais et l'agent rejouait toujours la même tranche en L1/L2. Découvert en essayant de valider le fix 3b.1 : le run de validation (75k steps) n'a jamais complété un seul épisode. Tests de non-régression dans `tests/test_env.py::TestMaxEpisodeSteps`.
- [ ] 3b.4 — Réintroduire une décroissance du LR PPO (ou reset du scheduler) lors du transfert de poids entre niveaux de curriculum
- [x] 3b.5 — Run de validation complet (Niveau 1, 500k steps, W&B `w4f2hkxx`, 18/09) avec 3b.1+3b.3 en place : **confirmé** — `ep_rew_mean` enfin peuplé et stable positif (2.41→2.30, vs -397 à -8764 avant), portefeuille à l'équilibre, `explained_variance` s'améliore (-2.3→+0.46). **Toujours pas résolu** : `train/std` s'effondre quand même (0.976→0.097, pire qu'avant à steps égaux), `approx_kl` grimpe à 0.55 et `clip_fraction` à 0.59 — sans surprise, cause distincte (pas de garde-fou PPO). Confirme que 3b.2 est la prochaine étape la mieux étayée.
- [ ] 3b.8 — Perf : `_get_obs()`/`_get_current_volatility()`/`_get_trend_direction()` reconstruisaient un array numpy depuis tout le DataFrame à chaque step (goulot CPU qui laissait le GPU quasi inactif, 0.58% d'utilisation moyenne). **Fixé le 18/09** (`env/trading_env.py`, cache numpy précalculé par reset) : env.step() isolé <25→~7900 steps/s, training réel 16→385 it/s (GPU) puis →1104 it/s après 3b.9.
- [x] 3b.9 — `n_envs` 1→3, `batch_size` PPO 128→256, `n_steps` 2048→4096 pour mieux exploiter le CPU/GPU dispo une fois 3b.8 réglé. Mesuré : GPU 17.6%→21% moyenne (63%→69% pic), CPU 13-15% sur 8 cœurs (pas de saturation), fps 385→1104 it/s.
- [ ] 3b.6 — Retrouver ou reconstruire le code `fee_annealing`/`curriculum_v2` (run crashé du 6 avril, jamais committé) et auditer sa reward avant de le relancer
- [ ] 3b.7 — Auditer `fee_penalty_weight`, `sharpe_bonus`, `unrealized_pnl_weight` avec la même rigueur que `drawdown_penalty` (simulation + test de non-régression)

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
