# Synthèse et Explication du Repository (RLD_Trading)

Ce document explique le fonctionnement actuel de l'agent de trading par Reinforcement Learning, l'architecture du code, ainsi que les choix d'entraînement effectués jusqu'à présent (février 2026).

---

## 🏗️ 1. Architecture du Code

Le projet est divisé en plusieurs briques logiques qui communiquent entre elles :

### A. Les Données (Dossier `data/` et `features/`)
Avant même de trader, l'agent a besoin de voir le passé.
- `download.py` télécharge l'historique de 10 cryptomonnaies maîtresses (BTC, ETH, SOL...) sur 5 ans en bougies d'1 Heure. Il divise ensuite ce paquet de données : `Train` (70%), `Validation` (15%), et `Test` (15% gardé secret pour la fin).
- Puis, le code enrichit ces bougies nues via `indicators.py` (MACD, RSI, Bandes de Bollinger).
- Ensuite, la macro-économie entre en jeu via `multi_timeframe.py` : l'agent "vit" en 1H, mais on lui injecte les tendances du 4H, du format Journalier (1D) et Hebdomadaire (1W).
- **Très important** : Tout est converti en "Z-Score" roulant par `normalizer.py`. On ne montre pas le vrai prix ("65 000$"), mais on dit à l'agent "le prix actuel est +2% au dessus de la moyenne des dernières 30 heures", ce qui lui permet de trader indifféremment du Dogecoin ou du Bitcoin.

### B. Le Cerveau et son Monde (Dossier `env/`)
C'est le *Gym*, la salle d'entraînement de l'agent.
- L'agent voit (Observe) : Une matrice de 30h de profondeur contenant toutes les valeurs normalisées (`observation.py`), plus le solde de son portefeuille et ses profits latents.
- L'agent agit (Action) : Il choisit un chiffre entre -1 et 1. Une "dead zone" empêche les micro-actions entre -0.05 et 0.05 pour économiser les frais de transactions (`action.py`).
- L'agent est noté (Reward) : C'est le cœur nucléaire, défini dans `reward.py`. L'agent est encouragé s'il engrange un PnL Latent (Hold positif), s'il trade dans le sens de la tendance... Mais il est **violemment puni** si son capital baisse de plus de 3% (Drawdown) ou s'il trade pendant de la haute volatilité stagnante.

### C. L'entraînement (Dossier `training/`)
C'est ici qu'on branche le cerveau (`PPO` ou `SAC`) sur la Gym. On utilise `Stable-Baselines3`.
- `config.yaml` : La tour de contrôle. On y configure la taille de la mémoire (batch), le nombre de coeurs CPU autorisés, la taille des couches de neurones.

---

## 🏋️‍♂️ 2. Historique des Entraînements et Résultats

Nous avons mené deux grandes sessions d'entraînement pour tester la viabilité du projet.

### **Test 1 : SAC (Soft Actor-Critic)**
- **Objectif :** Tester la tuyauterie sur 1 Million de steps. SAC est excellent pour les actions fluides et continues.
- **Résultat :** Après 2h30 d'entraînement, le portefeuille a grimpé artificiellement à 20 000$ (soit +100%), mais la "reward" mathématique était négative.
- **Diagnostic :** SAC s'est trouvé une niche (il a trouvé une seule action qui marchait) et son coefficient "d'exploration" s'est effondré à `0.003`. Il a arrêté de chercher d'autres stratégies et s'est enfermé dans son dogme. De plus, il faisait de l'over-trading (trop de frais).

### **Test 2 : PPO (Proximal Policy Optimization)**
- **Objectif :** PPO est la norme en RL quantitatif. Plus lent, on-policy, mais plus stable. On a aussi ajouté une forte pénalité aux "drawdowns" (pertes subites).
- **Résultat :** L'agent s'est **complètement planté**. Son compte a sombré à **1 800$** (soit -82% de perte).
- **Diagnostic :** Pourquoi cette chute ? La fonction de reward était très stricte (punition dès la moindre baisse de capital), combinée à un environnement fou (10 cryptos aléatoires, frais instables). PPO est devenu **hyper-anxieux**. Ses graphiques d'entropie ont montré qu'il cherchait frénétiquement une martingale, puis il abandonne : il stoppe tout achat, et encaisse perpétuellement le cout du temps qui passe. L'environnement était trop difficile, trop tôt.

---

## 🚀 3. La Solution Actuelle : La Phase 3

Pour contrer cette "anxiété d'apprentissage" constatée de PPO, nous avons bouleversé le code :

1. **Curriculum Learning (`curriculum.py`) :**
   L'agent retournera en classe biberon.
    - *Niveau 1*: Il ne trade que du BTC et de l'ETH, 0% de frais. L'idée est qu'il apprenne juste à "acheter bas, vendre haut" sans stress (500k steps).
    - *Niveau 2*: On ajoute 3 autres monnaies (BNB, SOL, ADA) et on active les frais réels de Binance. Il doit y apprendre la rareté du clic. (1M steps)
    - *Niveau 3*: Le Hard-Mode. Les 10 cryptos, des frais et capitaux qui varient de manière aléatoire (Domain Randomization). Il y transférera les poids appris en classe facile.

2. **Hold récompensé (Unrealized PNL) :**
   L'agent a tendance à vendre désespérément à la moindre bougie rouge. On lui injecte à chaque step une récompense "Unrealized PNL" : plus il conserve longtemps une position dans le vert, plus on flatte son réseau de neurone (il sera alors un vrai "Trend-Follower").

3. **Capacité GPU décuplée :**
   Grâce au paramétrage du fichier config, on a scindé le CPU en 6 "Mondes Gym" parallèles simultanés (`n_envs`). Cela permet de charger un énorme batch au GPU, qui possède désormais un réseau de neurones très lourd : il est passé de 3 petites couches, à un gigantesque millefeuille `[1024, 1024, 512]`. Plus long à entraîner, mais virtuellement capable de retenir une quantité infinie de contextes d'arbitrage croisés.

> **Et la suite ?** Une fois l'entraînement du Curriculum fini, nous construirons un "Backtester" rigide (Phase 4). C'est là que l'agent affrontera le jeu de données qu'il n'a encore jamais vu ("Test Set"), et nous comparerons ses courbes d'équités avec un banal "Buy & Hold" de Bitcoin !

---

## 🔬 4. Analyse forensique W&B (Sept 2026) — pourquoi le curriculum stagnait

Le dépôt local avait dérivé de GitHub (un commit "Phase 4 — Minute trading" jamais récupéré). Une fois resynchronisé, la lecture directe des runs W&B (`mtlfmp5w`, `pexiuym4`, `wwxxuvxq` — les 3 niveaux du 19 mars — et un run `crashed` du 6 avril taggé `fee_annealing`/`curriculum_v2` dont le code n'a jamais été commité) a montré une dégradation progressive et mesurable :

| Niveau | `train/std` (début→fin) | `approx_kl` (pic) | `clip_fraction` | `ep_rew_mean` |
|---|---|---|---|---|
| L1 (0% frais) | 0.99 → 0.26 | 0.61 | jusqu'à 51% | -397 → -146 |
| L2 (frais 0.1%) | 0.26 → 0.025 | 4.34 | jusqu'à 66% | -146 → -141 |
| L3 (+ Domain Rand.) | 0.025 → 0.0105 | **27.9** (pic 82.8) | jusqu'à 74% | -141 → -58 |

**Root cause identifiée et corrigée** : `drawdown_penalty` dans `env/reward.py` se réappliquait à *chaque step* tant que la NAV restait sous son plus-haut historique, sans jamais s'éteindre. Preuve par simulation : un pic +20% suivi d'un retour à la normale, puis 20 000 steps totalement immobiles (NAV inchangée, agent irréprochable), coûtait quand même **-220 de reward cumulée**. Ce terme dominait tous les autres (`dd_factor=5.0` vs `sharpe_weight=0.01`) et rendait la passivité totale mathématiquement optimale — ce qui explique l'effondrement de `train/std` observé ci-dessus. **Fix appliqué** : la pénalité ne taxe plus que l'*aggravation* du drawdown (nouveau plus bas de l'épisode), pas son maintien. Tests de non-régression dans `tests/test_reward.py`.

### Pistes restantes, non encore implémentées (par ordre de priorité probable)

1. **`target_kl` absent du PPO** (`training/curriculum.py`) — rien n'arrête une mise à jour de politique même quand `approx_kl` explose (observé jusqu'à 82.8, une valeur saine est ~0.01-0.03). Fix attendu : ajouter `target_kl≈0.02-0.03` à la construction du modèle.
2. **Pas de randomisation du point de départ en L1/L2** — `domain_randomization` n'est actif qu'en L3, donc l'agent rejoue *exactement* la même tranche de données à chaque épisode en L1/L2 (mémorisation possible plutôt que généralisation), avant de découvrir des points de départ aléatoires seulement au niveau 3 (choc de distribution).
3. **LR constant (3e-4) sans décroissance**, et pas de reset d'optimiseur/scheduler lors du transfert de poids entre niveaux (`PPO.load` + `reset_num_timesteps=False`) — un changement brutal de régime de frais peut encore provoquer un choc alors que le LR reste au même niveau qu'en fin de niveau précédent.
4. **Code `fee_annealing`/`curriculum_v2` perdu** — le run crashé du 6 avril montrait un comportement inverse intéressant (`std` stable ~1.0, `approx_kl` sain <0.05) mais une `ep_rew_mean` catastrophique (-8764 à -1149) malgré un portefeuille correct : signe d'un bug de calibration reward dans cette branche jamais committée. À retrouver ou reconstruire avant d'investir plus de temps dessus.
5. **Bruit intrinsèque du 1-minute BTC** — le README documente déjà que le critique n'extrayait pas de signal exploitable à cette granularité. Même avec une reward saine, l'edge économique net de frais à 1m est peut-être proche de zéro ; la Phase 5 proposée dans le README (passage au 15m) reste la meilleure réponse à ce problème-là si les fixes ci-dessus ne suffisent pas.
6. **Autres termes de reward non audités avec la même rigueur** (`fee_penalty_weight`, `sharpe_bonus`, `unrealized_pnl_weight`) — seul `drawdown_penalty` a un bug démontré ; les autres mériteraient le même traitement (simulation + test de non-régression) avant d'être considérés fiables.
