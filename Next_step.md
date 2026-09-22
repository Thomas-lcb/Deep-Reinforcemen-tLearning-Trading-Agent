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
- [x] 3b.7 (partiel) — `dead_zone` 0.05→0.1 et `fee_penalty_weight` 1.0→20.0 (`config.yaml`). **Mesuré sur L2** (W&B `tofa8beq` vs `hzns806i`) : amélioration réelle mais insuffisante — `win_rate` 9-28%→15-40%, `profit_factor` 0.08-0.32→0.13-0.81 (toujours <1, donc toujours net perdant), mais `trades_count` à peine changé (-2 à -3%). `dead_zone` doublée a un effet marginal tant que `train/std`≈1.0 (la masse de la gaussienne reste hors zone dans les deux cas) — `fee_penalty_weight` semble avoir fait le plus gros du travail. Prochaine piste : `dead_zone` beaucoup plus large (0.3+) et/ou activer `cooldown_steps` (déjà codé dans `env/action.py::apply_cooldown()`, jamais utilisé) pour plafonner la fréquence de façon déterministe plutôt que probabiliste.
- [x] 3b.11 — `cooldown_steps` 0→5 activé (`config.yaml`). **Bug trouvé en testant** : `steps_since_trade` démarrait à 0 dans `__init__`/`reset()`, donc le tout premier trade de CHAQUE épisode était bloqué par le cooldown lui-même — jamais remarqué avant car `cooldown_steps` était toujours à 0. Fixé (`steps_since_trade = cooldown_steps` à l'init). **Mesuré sur L1+L2 frais** (W&B `4ky7kltn`/`surkx7wf` vs `qs15bf2i`/`tofa8beq`) : `trades_count` -82% (~11 500→~2 025, cap déterministe cohérent avec la théorie), et sur L2 : `win_rate` 15-40%→22-49%, `profit_factor` 0.13-0.81→0.23-0.95 (proche de l'équilibre sans le dépasser), plancher du pire épisode -73%→-36%. Bien plus efficace que `dead_zone` seule. Reste net légèrement perdant en moyenne (`profit_factor`<1) — à affiner encore si besoin, ou passer à un vrai backtest (Phase 4) pour trancher.
- [x] 3b.12 — `cooldown_steps` 5→10. **Mesuré sur L2** (W&B `sdjk8b0r` vs `surkx7wf`) : amélioration nette et continue sur tous les axes — `trades_count` -45% (2024→1113), `win_rate` 34.5%→38.1%, `profit_factor` moyenne 0.51→0.65, **pic à 1.43 (franchit 1.0, rentable)**, portefeuille final -24.6%→-9.9%. `train/std`/`approx_kl` restent sains. Plafond de rendement de ce levier pas encore atteint — piste ouverte pour continuer (15-20) ou diversifier (short-selling = vraie cause racine du win_rate<50%, identifié le 21/09 : l'environnement ne permet pas de vendre à découvert sur un marché d'entraînement structurellement baissier à -37%).
- [x] 3b.13 — `cooldown_steps` 10→15. **Mesuré sur L2** (W&B `x5hv5e48` vs `sdjk8b0r`) : **rendement décroissant** — `trades_count` continue de baisser (-31%, 1113→767), et `profit_factor` en pic continue de monter (1.43→1.73), mais `win_rate` (38.1%→37.6%) et `profit_factor` moyen (0.65→0.62) stagnent voire reculent légèrement, contrairement aux deux sauts précédents où tout s'améliorait. Signal de plafond de ce levier — ne pas continuer à le pousser aveuglément, prioriser une piste différente (short-selling ou backtest réel Phase 4) plutôt qu'un nouveau doublement.
- [x] 3b.10 — Fix `pnl_pct` jamais renseigné dans `interpret_action()`/`trading_env.py::step()` → les stats `trading/win_rate`, `avg_win`, `avg_loss`, `profit_factor` n'ont jamais été loggées sur aucun run de l'historique (filtre silencieusement vide). Corrigé + testé (`tests/test_env.py::TestTradePnl`). **Résultat révélé sur le run de vérification (W&B `22zm10uh`)** : ~11 800 trades sur 30k steps (0% frais en L1 = aucun coût à l'overtrading), `win_rate` ~48-52% (quasi pile ou face), `profit_factor` ~0.89-1.11. Le portefeuille proche de l'équilibre observé en 3b.5 n'est donc PAS un comportement défensif appris — c'est du trading fréquent sans edge qui s'annule statistiquement. Pertinent pour 3b.7 : l'agent n'a aucune raison actuelle (en L1, frais nuls) de limiter sa fréquence de trade.

---

## Phase 3-ter : Short-selling (Niveau 4)
*cf. docs/superpowers/specs/2026-09-21-short-selling-design.md et docs/superpowers/plans/2026-09-21-short-selling.md*

- [x] 3c.1 — Implémentation complète (action.py short/cover, trading_env.py exécution+funding+liquidation, observation.py fix unrealized_pnl_pct, curriculum.py Niveau 4). `short.enabled=false` par défaut, L1-L3 inchangés (60 tests existants verts sans modification).
- [x] 3c.2 — Run de validation complet du Niveau 4 (1.5M steps, W&B `w1xxqv8c`, 21/09, après la correction du bug de levier illimité et du garde-fou NaN trouvés en revue finale). **Résultat net positif** : `portfolio_value` termine à 11 090 (+10.9% sur le capital initial, moyenne 9 301), `profit_factor` franchit 1.0 en fin d'entraînement (0.179→1.121, pic 2.561), `win_rate` grimpe à 45.9% en fin de run (pic 54.4%, moyenne 39.7% — nettement mieux que les ~35-38% observés en L2/L3 long-only). `train/std` reste sain (1.07-1.13, pas d'effondrement), `approx_kl` maîtrisé (~0.02), `explained_variance` 0.54-0.70 — le meilleur de tous les niveaux à ce jour. À comparer avec le smoke-test pré-fix (30k steps, run `qsytz83s` en 3c.3) qui finissait à 3 512 (-65%) : la correction du levier illimité (revue finale de branche, cf. spec) explique tout l'écart — le short-selling apporte un vrai bénéfice une fois le bug corrigé, pas une fuite mécanique de capital. Reste ouvert : impact réel de l'asymétrie liquidation-vs-cover (Important #6, différée lors de la revue finale) — pas de métrique dédiée au taux de liquidation forcée vs couverture volontaire encore extraite de ce run, à creuser si on veut affiner encore.
- [x] 3c.3 — Vérification finale (Task 8) : suite de tests complète verte (84 passed, 1 skipped) et smoke-test réel de bout en bout du Niveau 4 (30k steps, `python -m training.curriculum --device cuda --level 4 --timesteps 30000`, exit 0, poids L3 chargés, modèle sauvegardé dans `models/saved/ppo_curriculum_l4.zip`). Run W&B `qsytz83s` (état `finished`, https://wandb.ai/thomas_lcb/RLD-Trading/runs/qsytz83s) : `trading/win_rate`, `trading/trades_count`, `rollout/portfolio_value` bien présents et peuplés dans l'historique (valeurs finales observées : `win_rate` 0.402, `trades_count` 766, `portfolio_value` 3512.33). Confirme que le pipeline fonctionne de bout en bout ; ne remplace pas 3c.2 (run complet 1.5M encore à faire).
- [x] 3c.4 — Note sur la sémantique de `trading/trades_count` : depuis la Task 1 de cette branche, une action "vendre alors qu'on est déjà flat" retourne désormais `type="hold"` au lieu de l'ancien placeholder `type="sell"` à montant nul (fix comportementalement neutre, cf. `env/action.py::interpret_action()`). Conséquence : `trading/trades_count` loggé dans W&B est, à comportement d'agent identique, légèrement plus bas à partir de maintenant qu'il ne l'était sur les runs historiques enregistrés plus haut dans ce fichier (ex. entrées 3b.12/3b.13, `trades_count` 1113/767) — ce n'est pas une régression de comportement, seulement un comptage plus exact, mais ça affecte la comparabilité directe avec ces anciens chiffres.

---

## Phase 4 : Backtesting & Visualisation
*cf. agent.md §6 Phase 4*

- [x] 4.1 — Implémenter `evaluation/visualization.py` : Moteur de rendu Plotly (Candlesticks + Trades markers + Portfolio heatmap). Complété le 22/09 : marqueurs `short`/`cover`/`liquidation` ajoutés (point différé lors de la revue finale du short-selling), overlay des courbes de référence.
- [ ] 4.2 — Créer `notebooks/demo_replay.ipynb` : Script pour charger un modèle et générer une vidéo/HTML d'un épisode de 100-200 steps
- [x] 4.3 — Implémenter `evaluation/metrics.py` : Sharpe, Sortino, Calmar, MDD, Win Rate, Profit Factor (22/09, fonctions pures testées sur cas calculés à la main)
- [x] 4.4 — Implémenter `evaluation/benchmark.py` : Buy & Hold (formule fermée) + baseline aléatoire (rejoue le vrai `CryptoTradingEnv`, mêmes frais/règles — comparaison à armes égales)
- [x] 4.5 — Implémenter `evaluation/backtest.py` : charge un modèle, le fait tourner en déterministe sur le seul split test (15% chronologique jamais vu), compare aux deux références, tableau + HTML.
- [ ] 4.6 — **Résultat du premier backtest réel (Niveau 4, 22/09, 78 812 lignes de test)** : le modèle finit à **-45.16%**, pire que Buy & Hold (**+22.08%**) et que l'aléatoire (**-32.19%**) — Sharpe -25.6, MDD 45.2%, win_rate 32.6%, profit_factor 0.483. **Signal de distribution shift, pas un bug de méthodologie** : la période de train était structurellement baissière (-37%, mesuré le 21/09), la période de test est haussière (Buy & Hold +22% le confirme) — le modèle a probablement appris des comportements défensifs/short adaptés à un marché baissier qui deviennent contre-productifs une fois le régime inversé. Confirme et aggrave un signal déjà vu (évaluation ad hoc à 15 épisodes le 21/09 donnait -20.6%, moins bon que la courbe d'entraînement +10.9% mais pas aussi mauvais). **Conclusion** : toutes les métriques positives observées pendant l'entraînement (L1 à L4, toute la session) mesuraient une performance intra-distribution, pas une vraie généralisation. Reste à faire : comparer L1/L2/L3 sur ce même backtest (le problème est-il propre au short, ou général à toute la stratégie ?), et explorer une exposition à des régimes de marché plus variés pendant l'entraînement (fenêtres temporelles multiples, pas seulement la continuation chronologique actuelle).
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
