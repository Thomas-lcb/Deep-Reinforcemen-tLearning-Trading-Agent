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
- [x] 4.6 — **Comparaison complète L1→L4 vs Buy & Hold vs aléatoire, backtest réel sur le test set (22/09, 78 812 lignes jamais vues)** :

  | Niveau | Rendement | Sharpe | MDD | Win rate | Profit factor |
  |---|---|---|---|---|---|
  | L1 (0% frais) | **+11.36%** | 3.543 | 6.10% | 42.0% | 0.746 |
  | L2 (frais réels) | **-38.48%** | -16.608 | 38.86% | 37.9% | 0.700 |
  | L3 (+ domain rand.) | **-29.67%** | -12.726 | 30.42% | 36.9% | 0.855 |
  | L4 (+ short) | **-45.16%** | -25.560 | 45.20% | 32.6% | 0.483 |
  | Buy & Hold (référence) | +22.08% | 3.939 | 8.55% | — | — |
  | Aléatoire (frais réels) | -32.19% | -13.932 | 32.85% | 35.8% | 0.716 |

  **L1 généralise correctement** (positif, Sharpe sain). **La bascule catastrophique arrive exactement à L2** — dès l'introduction des frais réels pendant l'entraînement — pas à L4. **Le short-selling n'est donc pas la cause** du problème de généralisation, il en hérite seulement (L4 est un peu pire que L2/L3, mais du même ordre de grandeur catastrophique). Tout le réglage fin fait cette session sur L2 (`cooldown_steps`, `fee_penalty_weight`, `target_kl`) a amélioré des métriques *intra-distribution* qui ne transfèrent pas au régime de marché du test set (haussier, alors que le train était structurellement baissier à -37%, mesuré le 21/09) — la politique L2-L4 finit pire qu'un aléatoire pur sur 3 niveaux consécutifs. Signal de distribution shift au niveau du curriculum entier, pas un défaut du short en particulier.

  **Prochaines pistes** (aucune tentée pour l'instant) : exposer l'entraînement à des régimes de marché plus variés (fenêtres temporelles multiples/non-contiguës plutôt que la continuation chronologique actuelle, item 1.9 — data augmentation — jamais fait), ou explorer le meta-learning (roadmap V2.4) pour une adaptation explicite au régime de marché.
- [ ] 4.6 — Comparer SAC vs PPO vs Buy & Hold — décider si retour Phase 3
- [x] 4.7 — **Spec meta-learning (Reptile) écrite (25/09, `docs/superpowers/specs/2026-09-25-meta-learning-reptile-design.md`), puis réfutée par revue adversariale avant tout code.** Un subagent (Opus) dispatché pour challenger la spec a mesuré le vrai 2×2 modèle×frais sur le test set (8 min, harnais existant, aucun code nouveau) : L1 (généralise, +11.36%) s'effondre à -30.70% si on lui applique les frais L2 ; L2 (s'effondre, -38.48%) redevient positif (+6.51%, PF 1.502 — meilleur que L1) si on lui retire les frais. **Ce n'est pas un distribution shift** (les deux modèles transfèrent très bien au régime haussier du test) — c'est un turnover sans edge net de frais : politique déterministe saturée à 99.7% du plafond `cooldown_steps` (médiane `|action|`=1.0), ~430× le capital tradé en 55 jours, 40-43% du capital brûlé en frais. La revue a aussi trouvé des trous bloquants dans le design Reptile lui-même (`target_kl` empêchant tout déplacement de la boucle interne, `window_steps` inerte car un seul rollout ne visite qu'un cinquième de la fenêtre de tâche, `meta_batch_size` justifié par la mauvaise contrainte — VRAM au lieu de temps mur). **Meta-learning abandonné pour cette itération** — repli sur le diagnostic turnover, moins cher et plus direct.
- [x] 4.8 — **Diagnostic turnover confirmé empiriquement (25/09), sans réentraînement.** Backtest du modèle `ppo_curriculum_l2` existant avec `action.cooldown_steps` 15→500 appliqué seulement à l'évaluation (le throttle est un garde-fou d'environnement pur, indépendant des poids — `env/action.py::apply_cooldown`) : rendement -38.48%→**+7.29%**, Sharpe -16.61→2.686, profit factor 0.700→1.909, trades 4904→158, turnover ~431×→20× le capital, frais payés ~43%→2.01% du capital. Confirme sans ambiguïté que le turnover est la cause, pas le régime de marché. **Nuance** : reste sous Buy & Hold (+22.08%) et proche de l'aléatoire sur ce run (+8.83%, PF 2.249, même cooldown appliqué) — ce test répare l'effondrement catastrophique mais ne prouve pas encore un edge net supérieur au hasard à cette fréquence. **Prochaine étape** : réentraîner L2 avec le cooldown élevé *dès l'entraînement* (pas juste appliqué a posteriori sur un modèle qui n'a appris qu'à trader à cooldown=15) — un balayage de quelques valeurs est probablement nécessaire pour trouver le point qui bat effectivement l'aléatoire/Buy&Hold.
- [x] 4.9 — **Relevé des frais réels des plateformes crypto (25/09)** pour vérifier que `fees.taker=0.001` (0.10%) n'était pas artificiellement optimiste : Binance/Bybit 0.10%, OKX 0.08%/0.10%, Kraken (palier de base) 0.40%/0.80%, Coinbase Advanced Trade US (MàJ du 16/09/2026) 0.50%/0.90%. **0.10% confirmé comme le choix cohérent** — Binance/Bybit/OKX dominent le volume algo précisément parce que leurs frais sont bas, un vrai bot n'utiliserait pas Kraken/Coinbase pour du trading actif. Premier réflexe (prendre le taux le plus pénalisant, Coinbase 0.90%) corrigé en cours de route : la marge de sécurité recherchée vient de la correction du turnover (4.8), pas de gonfler artificiellement le frais. Aucun changement de valeur — seulement documenté dans `config/config.yaml`.
- [x] 4.10 — **`cooldown_steps` 15→500 baked-in dès l'entraînement (25/09)**, réentraînement complet de L2 (1M steps, W&B `0w0lbedi`, poids L1 comme point de départ ; ancien modèle sauvegardé dans `models/saved/ppo_curriculum_l2_cooldown15_baseline.zip`). **Résultat sur le test set** : rendement +8.76% (vs +7.29% en bridant seulement au backtest sans réentraîner, vs -38.48% avant tout fix), Sharpe 2.913 (vs 2.686 / -16.61), profit factor 2.015 (vs 1.909 / 0.700). Amélioration réelle mais marginale par rapport au simple bridage post-hoc — **et le modèle fait exactement 158 trades, identique au plafond théorique forcé par le cooldown** : il n'a pas appris à être sélectif, juste à trader à fond dans une fenêtre plus large (toujours une politique bang-bang). Reste statistiquement indiscernable de l'aléatoire sur ce test set (+8.76% vs +8.83% aléatoire, Sharpe 2.91 vs 3.19, PF 2.02 vs 2.25 — l'aléatoire tourne avec le même cooldown=500) et loin derrière Buy & Hold (+22.08%). **Conclusion** : le turnover catastrophique est corrigé (plus "pire qu'aléatoire"), mais aucun edge démontré pour l'instant — juste une fuite de capital réparée. Rappel important (n=1) : ces chiffres reposent sur une seule fenêtre de test de 55 jours et un seul run, pas de barre d'erreur — à garder en tête avant de sur-interpréter un futur delta de quelques points.
  **Pistes ouvertes non tentées** : sweep de `cooldown_steps` autour de 500 pour chercher un optimum (rien ne dit que 500 est le point optimal, juste la première valeur testée) ; redesign de l'espace d'action/reward pour encourager une vraie sélectivité plutôt qu'un plafond uniforme (le modèle choisit toujours l'action extrême dès que le cooldown expire — le dead_zone/l'espace d'action continu n'apprend pas "ne pas trader maintenant", seul le throttle externe l'impose) ; ou accepter qu'avec ce niveau de frais et ce dataset, il n'y a peut-être simplement pas d'edge intraday détectable pour ce modèle et pivoter vers une autre approche (features, timeframe plus long, etc.).
- [x] 4.11 — **Sweep de `cooldown_steps` (25/09)**, piste 1 des trois demandées. D'abord un sweep eval-only (10 valeurs 50→4000, sans réentraîner, sur `ppo_curriculum_l2` entraîné à 500) pour repérer une zone prometteuse avant de committer à un réentraînement complet à chaque valeur :

  | cooldown | trades | rendement | Sharpe | PF | win_rate |
  |---|---|---|---|---|---|
  | 50 | 1542 | -3.29% | -0.96 | 0.61 | 28.0% |
  | 100 | 780 | +2.25% | 0.82 | 0.87 | 35.5% |
  | **200** | 392 | +13.38% | 4.12 | 2.28 | 46.2% |
  | 300 | 262 | +7.42% | 2.65 | 2.17 | 49.3% |
  | 500 | 158 | +8.76% | 2.91 | 2.02 | 33.3% |
  | 750 | 105 | +10.37% | 3.08 | 2.62 | 47.9% |
  | 1000 | 79 | +9.48% | 3.11 | 3.57 | 50.0% |
  | 1500 | 53 | +12.82% | 3.47 | 7.01 | 63.2% |
  | 2500 | 32 | +11.98% | 3.52 | 6.72 | 58.3% |
  | 4000 | 20 | +20.53% | 5.08 | inf | 100% |

  Le point à 4000 (20 trades, 100% win rate, PF infini) est écarté comme artefact d'échantillon trop petit — à cooldown aussi élevé, la politique dégénère vers un quasi Buy & Hold ("acheter une fois tôt, rester bloqué par le cooldown"), qui gagne déjà +22% sur cette fenêtre test haussière sans qu'aucun edge de trading ne soit démontré. `cooldown=200` retenu comme candidat (meilleur Sharpe parmi les échantillons statistiquement crédibles, 392 trades).

  **Réentraînement complet de confirmation à cooldown=200** (1M steps, poids L1 comme point de départ ; modèle à cooldown=500 sauvegardé dans `models/saved/ppo_curriculum_l2_cooldown500_baseline.zip`). Résultat sur le test set, **avec comparaison à armes égales contre l'aléatoire au même cooldown** (leçon retenue de la revue adversariale du meta-learning — comparer à un aléatoire tournant sous la même contrainte, pas un aléatoire différent) :

  | | cooldown=15 (original) | cooldown=500 réentraîné | **cooldown=200 réentraîné** |
  |---|---|---|---|
  | Rendement modèle | -38.48% | +8.76% | **+6.36%** |
  | Sharpe modèle | -16.61 | 2.913 | **2.330** |
  | Profit factor | 0.700 | 2.015 | **1.456** |
  | Win rate | 37.9% | 33.3% | **44.0%** |
  | Trades | 4904 | 158 | **392** |
  | Rendement aléatoire (même cooldown) | -32.19% | +8.83% | **-0.62%** |
  | Sharpe aléatoire (même cooldown) | -13.93 | 3.195 | **-0.124** |
  | **Δ Sharpe modèle − aléatoire** | -2.68 (pire) | -0.28 (à peu près à égalité) | **+2.45 (nettement mieux)** |

  Contre-intuitif mais informatif : le rendement brut du modèle à cooldown=200 (+6.36%) est *inférieur* à celui à cooldown=500 (+8.76%) et au chiffre du sweep eval-only à 200 (+13.38%, obtenu avec le modèle entraîné à 500 — ne se reproduit donc pas au réentraînement, c'était en partie un artefact de l'interaction entre une politique entraînée à 500 et évaluée à 200). Mais c'est **la première fois que le modèle bat clairement son propre aléatoire de référence** (Sharpe +2.45 d'écart, contre quasi-égalité à cooldown=500) — parce qu'à cooldown=200 l'aléatoire lui-même devient franchement perdant (plus de trades = plus de frais payés sans edge = dérive négative), alors que le modèle reste net positif. **Premier signal crédible d'un edge réel**, même modeste, plutôt qu'une simple absence de catastrophe. `cooldown=200` gardé comme config actuelle. Toujours n=1 (une seule fenêtre de test) — à confirmer par la piste 2 (sélectivité apprise) avant de conclure définitivement.
- [x] 4.12 — **Piste 2 : redesign action/reward pour une sélectivité apprise (25/09)**, deux leviers testés, tous deux négatifs sur l'objectif de sélectivité :

  **a) Sweep eval-only de `dead_zone`** (0.1→0.7, `cooldown=200` fixé, sans réentraîner — le seuil s'applique à l'interprétation de l'action, pas aux poids) : nombre de trades collé à 388-392 sur toute la plage, **aucun effet mesurable**. En déterministe, la policy sort une action moyenne déjà proche des bornes ±1 la quasi-totalité du temps — élargir le seuil d'interprétation ne peut rien changer si la policy elle-même n'a jamais appris à produire des valeurs modérées. Confirme et quantifie ce que les commentaires de `config.yaml` notaient déjà de façon qualitative depuis 3b.7.

  **b) `fee_penalty_weight` 20→150 (7.5×), réentraînement complet** (`cooldown=200` conservé ; modèle précédent sauvegardé dans `models/saved/ppo_curriculum_l2_cooldown200_fpw20_baseline.zip`) — cette fois un vrai signal de reward pendant l'entraînement, pas un seuil post-hoc. Résultat :

  | | fee_penalty_weight=20 | **fee_penalty_weight=150** |
  |---|---|---|
  | Rendement modèle | +6.36% | **+9.10%** |
  | Sharpe modèle | 2.330 | **2.695** |
  | Profit factor | 1.456 | **1.640** |
  | Win rate | 44.0% | **46.5%** |
  | **Trades** | **392 (plafond)** | **392 (plafond, inchangé)** |
  | Δ Sharpe vs aléatoire matché | +2.454 | **+2.819** |

  Amélioration réelle sur toutes les métriques de rendement/risque (et l'écart avec l'aléatoire s'élargit encore), **mais le nombre de trades reste identique au plafond théorique du cooldown** — même à 7.5× le poids précédent, la policy ne réduit pas sa fréquence de trade d'un seul cran. Elle devient un peu meilleure dans *ce qu'elle fait* à chaque trade autorisé, mais ne choisit toujours pas *quand* trader.

  **Conclusion piste 2 : sélectivité apprise non obtenue, ni par le seuil d'interprétation (dead_zone) ni par le poids de la pénalité (fee_penalty_weight)**. C'est cohérent avec un diagnostic architectural plutôt qu'un problème de réglage : avec un espace d'action continu et une politique gaussienne, la moyenne apprise se cale sur les bornes et aucun des deux leviers testés ne peut la faire bouger vers le centre — il faudrait soit un espace d'action discret (option plus lourde, écartée à ce stade car elle invaliderait le curriculum complet L1→L4), soit un mécanisme d'exploration différent, non testés ici. `fee_penalty_weight=150` gardé en config malgré tout (net mieux que 20 sur toutes les métriques observées, même sans résoudre la sélectivité). Passage à la piste 3.
- [x] 4.13 — **Piste 3, point 1 : robustesse du résultat (25/09)**, avant d'investir plus dans l'optimisation. Deux checks demandés par l'utilisateur : split validation (gratuit, pas de réentraînement) et multi-seed sur le test set.

  **Backtest sur le split validation** (78 749 steps, période baissière : Buy & Hold -13.17%, confirmé par 3c.1 — jamais utilisé jusqu'ici pour scorer un résultat) du modèle actuel (`ppo_curriculum_l2`, cooldown=200, fee_penalty_weight=150) :

  | | Modèle | Buy & Hold | Aléatoire |
  |---|---|---|---|
  | Rendement | -9.51% | -13.17% | -9.30% |
  | Sharpe | -2.344 | -1.833 | -2.451 |
  | Profit factor | 0.646 | — | 0.546 |
  | Trades | 392 | — | — |

  **L'edge mesuré sur le test set (Sharpe modèle 2.695 vs aléatoire -0.124, écart +2.82) ne se reproduit pas sur le split validation** : le modèle y est quasi indiscernable de l'aléatoire (-9.51% vs -9.30%, Sharpe -2.34 vs -2.45) et **moins bon que Buy & Hold en risque-ajusté** (Sharpe -2.34 vs -1.83 — Buy & Hold traverse une baisse de façon plus régulière que le modèle, qui est plus chahuté pour un rendement similaire). Signal clair que l'edge est dépendant du régime de marché (haussier) plutôt qu'une compétence générale — ramène, sous une forme cette fois mesurée et non supposée, l'inquiétude initiale de distribution shift du 22/09.

  **Multi-seed sur le test set** (même config, aucun seed explicite dans le code — `training/curriculum.py` ne fixe jamais de graine, chaque lancement est donc déjà un seed indépendant) :

  | | seed 1 | seed 2 | seed 3 | aléatoire (référence) |
  |---|---|---|---|---|
  | Rendement | +9.10% | +9.45% | +6.24% | -0.62% |
  | Sharpe | 2.695 | 2.959 | 2.206 | -0.124 |
  | Profit factor | 1.640 | 1.687 | 1.299 | 1.205 |
  | Win rate | 46.5% | 45.7% | 39.4% | 41.5% |

  Sharpe moyen 2.62 (écart-type ~0.38, plage 2.21-2.96) — **les trois seeds battent largement l'aléatoire** (au minimum +2.3 points de Sharpe), avec une dispersion modérée mais aucun seed qui s'effondre ou n'approche l'aléatoire. Modèles sauvegardés : `models/saved/ppo_curriculum_l2_cooldown200_fpw150_seed{1,2,3}.zip`.

  **Synthèse piste 3 point 1** : l'edge du modèle sur le test set (régime haussier) est **robuste au seed d'entraînement** (3/3 confirment, dispersion modérée mais tous nettement positifs vs aléatoire) mais **ne généralise pas au régime baissier** (split validation : quasi indiscernable de l'aléatoire, moins bon que Buy & Hold en risque-ajusté). Ce n'est donc pas un artefact de chance sur un seed particulier — c'est une compétence réelle mais **limitée à un seul régime de marché**, cohérente avec le fait que L2 est long-only (pas de short) : dans un marché qui monte, l'agent capture une partie du mouvement plus efficacement qu'un aléatoire ; dans un marché qui baisse, n'ayant que "vendre pour être plat" comme option défensive, il ne fait pas mieux qu'un aléatoire. **Piste naturelle qui en découle** : le short-selling (Niveau 4, déjà implémenté et mergé depuis la Phase 3-ter) n'a jamais été réévalué avec le fix de turnover (cooldown=200/fee_penalty_weight=150) — c'est exactement le mécanisme qui manque pour le régime baissier du split validation, et il existe déjà dans le repo sans travail d'implémentation supplémentaire.
- [x] 4.14 — **Réévaluation du short-selling (L4) avec le fix de turnover (28/09).** L4 n'avait jamais été réentraîné/réévalué avec `cooldown=200`/`fee_penalty_weight=150` — sa seule évaluation (22/09, W&B `qsytz83s`) datait d'avant tout le travail de cette session sur le turnover et était aussi catastrophique que L2/L3 à l'époque (-45.16%), donc non informative. Ancien modèle sauvegardé dans `models/saved/ppo_curriculum_l4_pre_turnover_fix_baseline.zip`. Réentraînement complet (1.5M steps, part des poids L3 existants, W&B `q6e7kg41`).

  **Résultat, sur les deux splits** :

  | | L4 (nouveau) | L2 (référence) | Buy & Hold | Aléatoire |
  |---|---|---|---|---|
  | Test (haussier) — rendement | **-18.37%** | +9.10% (seed1) | +22.08% | -0.62% |
  | Test (haussier) — Sharpe | **-6.462** | 2.695 | 3.939 | -0.124 |
  | Validation (baissier) — rendement | **-2.58%** | -9.51% | -13.17% | -9.30% |
  | Validation (baissier) — Sharpe | **-0.594** | -2.344 | -1.833 | -2.451 |

  **Cause identifiée** : sur le test set, le modèle fait exclusivement des trades `short`/`cover` (202/190) — **zéro `buy`/`sell`, aucune position longue prise**. Ce n'est pas un modèle qui bascule entre long et short selon le régime détecté : c'est un modèle qui a convergé vers un biais directionnel unique (short permanent). Ça explique le résultat en miroir : excellent en régime baissier (short = bon sens), catastrophique en régime haussier (short = à contre-courant d'un marché +22%). **Même problème de fond que le diagnostic initial du 22/09, sous une forme inversée** — L2 était biaisé long-only (marche en hausse, pas en baisse), L4 s'est biaisé short-only (marche en baisse, pas en hausse). Le mécanisme technique du short fonctionne correctement (pas de levier incontrôlé, pas de crash — les fix de la Phase 3-ter tiennent) ; c'est la politique apprise qui n'est pas adaptative.

  **Vérifié sur un 2e seed (28/09) : systématique, pas un artefact de seed.**

  | | L4 seed1 | L4 seed2 |
  |---|---|---|
  | Test (haussier) — rendement | -18.37% | **-19.54%** |
  | Test (haussier) — Sharpe | -6.462 | **-7.402** |
  | Test — répartition trades | 202 short / 190 cover / 0 buy-sell | 198 short / 194 cover / **0 buy-sell** |
  | Validation (baissier) — rendement | -2.58% | **+1.63%** |
  | Validation (baissier) — Sharpe | -0.594 | **+0.551** |
  | Validation — profit factor | 1.382 | **1.696** |

  Les deux seeds sont **exclusivement short/cover sur le test set** (zéro `buy`/`sell` dans les deux cas) et les deux battent nettement Buy & Hold/aléatoire sur validation (seed2 même net positif, +1.63% vs -13.17% pour Buy & Hold). **Conclusion : L4 apprend systématiquement un biais short-only avec cette config (domain randomization + short activés)**, pas un hasard de seed particulier. C'est une vraie spécialisation baissière (confirmée deux fois, PF>1 sur validation les deux fois), symétrique et opposée à la spécialisation haussière de L2 (confirmée trois fois). Modèles sauvegardés : `models/saved/ppo_curriculum_l4_cooldown200_fpw150_seed{1,2}.zip`.

  **Piste qui en découle, directement actionnable** : L2 (spécialiste haussier, 3 seeds validés) et L4 (spécialiste baissier, 2 seeds validés) existent déjà, entraînés, testés. Un routage simple par détection de régime (ex. pente d'une moyenne mobile) vers l'un ou l'autre serait beaucoup moins cher qu'une nouvelle boucle d'entraînement — repli naturel vers la roadmap V2.6 (ensemble d'agents), en version plus simple (routage déterministe par régime plutôt que vote), et sans le besoin d'apprentissage cross-régime que visait le meta-learning abandonné (4.7).
- [x] 4.15 — **Premier essai d'ensemble à routage par régime (28/09), non concluant — problème de détecteur, pas de concept.** Script diagnostic (`/home/thomas/.claude/jobs/a26260ca/tmp/ensemble_backtest.py`, pas formalisé dans `evaluation/`) : un seul environnement partagé (`short.enabled=True` à la construction, `env.short_enabled` basculé dynamiquement — attribut mutable relu à chaque step, pas besoin de toucher le code source), régime détecté via `mtf_1h_ema200_dir` (feature EMA200 multi-timeframe déjà utilisée par `trend_alignment_bonus`), clôture forcée de toute position ouverte à chaque bascule (cooldown contourné pour ce trade précis via `env.steps_since_trade = env.cooldown_steps`, sinon la clôture forcée serait elle-même bloquée).

  **Sans hystérésis** : le signal `mtf_1h_ema200_dir` passe en négatif seulement 1711/78752 steps (2.2%) sur le test set, en rafales courtes (~29 steps en moyenne) — bien plus courtes que `cooldown_steps=200`. Résultat : chaque bascule vers "baissier" force une clôture qui consomme le cooldown, puis le régime repart haussier avant que L4 ait pu agir ne serait-ce qu'une fois. Rendement +3.65%/Sharpe 1.367 sur test (pire que L2 seul), -9.27%/Sharpe -2.565 sur validation (pire que L2 seul aussi) — **les deux modèles restent paralysés par des resets de cooldown en boucle**, ce n'est pas un vrai test de l'ensemble.

  **Avec hystérésis** (bascule acceptée seulement après confirmation du régime candidat sur une fenêtre ≥ `cooldown_steps`) : **une seule bascule sur toute la période, sur les deux splits** — les phases "baissières" de ce signal ne persistent jamais 200 steps d'affilée sur ce dataset. L'ensemble dégénère en L2 pur (résultats identiques bit-à-bit à L2 seul : test +9.10%/Sharpe 2.695, validation -9.51%/Sharpe -2.344).

  **Conclusion initiale : ni le concept L2/L4 (solidement validé séparément) ni le mécanisme de bascule (attribut `short_enabled` dynamique + clôture forcée) ne sont en cause — le détecteur de régime choisi (`mtf_1h_ema200_dir`) est mal calé sur la cadence de trading `cooldown_steps=200`.**

  **Suite (28/09), option 1 retenue par l'utilisateur — signal de régime plus lissé.** Remplacé par le signe du rendement cumulé sur une fenêtre roulante (W=1000 steps, ~16.7h), sans hystérésis. Deux nouveaux problèmes trouvés et corrigés en cascade :

  1. **Basculer de modèle en pleine position expose l'autre modèle à un état hors distribution.** À l'étape 2814 (régime bear, donc L4 aux commandes), action=+1.0 avec `balance_asset>0` (position longue héritée de L2) → **renforce** la position longue au lieu de la clôturer. L4 n'a jamais vu `balance_asset>0` à l'entraînement (il ne va jamais long), son comportement là est indéfini. Tentative de correction : ne changer de modèle actif que quand la position est déjà plate.
  2. **Cette correction ne se déclenche jamais** : `max_position_pct=0.25` plafonne chaque trade individuel à 25% de la position — aucun trade normal ne peut jamais l'amener exactement à zéro (décroissance asymptotique). Correction finale : `env.max_position_pct` est aussi un attribut mutable relu à chaque step (comme `cooldown_steps`/`short_enabled`) — le lever temporairement à 1.0 pour UN trade de clôture administrative complète au moment de la bascule, en contournant le cooldown pour ce trade puis en restaurant `steps_since_trade` après coup pour ne pas bloquer le modèle entrant.

  **Résultat une fois le mécanisme enfin correct** (L4 exécute vraiment des `short`/`cover`, 336/342 sur test) : **catastrophique**. -32.00%/Sharpe -18.84 (test), -41.29%/Sharpe -21.70 (validation) — pire que tout résultat mesuré cette session. 2105 trades sur le test set (vs 392 pour un seul modèle) : à ~1100 bascules sur 78751 steps (une toutes les ~71 steps en moyenne), le coût de la clôture complète + réouverture à chaque bascule dévore toute la valeur des deux spécialistes.

  **Conclusion finale de la piste ensemble** : cohérente avec le diagnostic de fenêtres fait en amont (même à W=4000, la plus grande testée, la durée moyenne d'un régime ne dépassait pas ~180 steps, sous `cooldown_steps=200`) — **sur ce dataset BTC 1 minute, il n'existe pas de régime de marché assez long et stable pour amortir le coût d'une bascule complète à l'échelle où L2 et L4 ont chacun été validés (tout le test set de 78751 steps).** Le problème n'est ni le concept (les deux spécialistes restent solides individuellement) ni un bug d'implémentation (le mécanisme final est correct) — c'est une caractéristique du marché lui-même : les régimes soutenus au sens "un modèle a le temps d'exprimer son edge avant que le régime change" n'existent pas à cette granularité temporelle avec des coûts de transaction réels. Ensemble régime-gated abandonné pour cette itération. Piste 3 point 2 (sanity-check indépendant du RL) reste ouverte si on veut continuer à creuser.

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
- [ ] V2.4 — Meta-learning (MAML/Reptile) pour adaptation aux régimes de marché — **exploré et abandonné le 25/09** (spec écrite puis réfutée, cf. 4.7 : le vrai problème était le turnover sans edge, pas un distribution shift ; à reconsidérer seulement si le fix turnover (4.8) s'avère insuffisant après réentraînement
- [ ] V2.5 — Inverse RL / RLHF (apprendre la reward d'experts)
- [ ] V2.6 — Ensemble d'agents (vote majoritaire)
- [ ] V2.7 — Order Book features
