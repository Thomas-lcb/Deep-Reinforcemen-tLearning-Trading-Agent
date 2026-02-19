# SPÉCIFICATION TECHNIQUE : AGENT DE TRADING AUTONOME (DRL)

## 1. OBJECTIF DU PROJET

Concevoir, entraîner et déployer un agent de **Deep Reinforcement Learning (DRL)** capable de trader de manière autonome sur le marché des crypto-monnaies.

**Objectifs principaux :**
- **Maximisation du Ratio de Sharpe** (rentabilité ajustée au risque), et non du profit brut.
- **Minimisation du Maximum Drawdown** pour assurer la survie long terme.
- **Robustesse** : l'agent doit fonctionner sur des conditions de marché jamais vues (bull, bear, latéral).

---

## 2. CHOIX DU MARCHÉ ET PLATEFORME

| Paramètre | Valeur |
|---|---|
| **Marché** | Crypto-monnaies (Spot uniquement) |
| **Paire principale** | BTC/USDT |
| **Paires secondaires (multi-asset training)** | ETH/USDT, SOL/USDT, BNB/USDT, ADA/USDT, XRP/USDT, DOGE/USDT, DOT/USDT, LTC/USDT, LINK/USDT |
| **Échange** | Binance |
| **Frais de transaction** | 0.1% par trade (0.075% avec BNB) |
| **Levier** | Interdit en V1 |

> [!CAUTION]
> Les frais doivent être déduits à chaque step dans l'environnement de simulation. Si l'agent ignore les frais, la stratégie sera perdante en production.

---

## 3. STACK TECHNIQUE

| Composant | Technologie |
|---|---|
| Langage | Python 3.9+ |
| Algorithmes RL | `Stable-Baselines3` (PPO, SAC) |
| Environnement | `Gymnasium` |
| Données | `Pandas`, `NumPy` |
| Indicateurs techniques | `Pandas-TA` ou `TA-Lib` |
| API Échanges | `CCXT` |
| Temps réel (WebSocket) | `ccxt.pro` ou `python-binance` |
| Monitoring entraînement | `Tensorboard`, `Weights & Biases` |
| Alertes prod | Telegram Bot API / Discord Webhook |

---

## 4. ARCHITECTURE DES FICHIERS

```
RLD_Trading/
├── agent.md                        # CE FICHIER — Spécification technique
├── README.md                       # Documentation utilisateur
├── requirements.txt                # Dépendances Python
├── .env.example                    # Template des variables d'environnement
│
├── config/
│   └── config.yaml                 # Hyperparamètres, frais, limites de risque
│
├── data/
│   ├── raw/                        # Données OHLCV brutes téléchargées
│   ├── processed/                  # Données nettoyées + indicateurs techniques
│   ├── download.py                 # Script de téléchargement CCXT
│   └── sentiment.py                # Fear & Greed Index (alternative.me API)
│
├── env/
│   ├── __init__.py
│   ├── trading_env.py              # CryptoTradingEnv(gym.Env) principal
│   ├── reward.py                   # Fonctions de récompense (modulaire)
│   ├── observation.py              # Construction de l'observation space
│   └── action.py                   # Logique de l'action space + dead zone
│
├── features/
│   ├── __init__.py
│   ├── indicators.py               # Calcul des indicateurs techniques
│   ├── normalizer.py               # Rolling normalization (anti data-leakage)
│   └── multi_timeframe.py          # Agrégation multi-timeframe (1H, 4H, 1D)
│
├── training/
│   ├── __init__.py
│   ├── train.py                    # Script d'entraînement principal
│   ├── hyperparams.py              # Recherche d'hyperparamètres (Optuna)
│   ├── callbacks.py                # Callbacks custom (early stopping, logging)
│   └── curriculum.py               # Curriculum learning (difficulté progressive)
│
├── evaluation/
│   ├── __init__.py
│   ├── backtest.py                 # Backtesting sur données out-of-sample
│   ├── metrics.py                  # Sharpe, Sortino, Calmar, MDD, Win Rate...
│   └── benchmark.py                # Comparaison vs Buy & Hold et autres baselines
│
├── live/
│   ├── __init__.py
│   ├── paper_trading.py            # Paper trading temps réel (Phase 5)
│   ├── live_trading.py             # Exécution réelle (Phase 6)
│   ├── risk_manager.py             # Stop-loss, circuit breaker, daily limits
│   └── notifier.py                 # Alertes Telegram/Discord
│
├── models/
│   └── saved/                      # Modèles entraînés (.zip SB3)
│
├── logs/
│   ├── tensorboard/                # Logs Tensorboard
│   └── trades/                     # Historique des trades (CSV/JSON)
│
├── notebooks/
│   ├── exploration.ipynb           # Analyse exploratoire des données
│   └── results_analysis.ipynb      # Visualisation des résultats de backtest
│
└── tests/
    ├── test_env.py                 # Tests unitaires de l'environnement
    ├── test_reward.py              # Tests de la reward function
    └── test_risk_manager.py        # Tests du risk manager
```

---

## 5. CONCEPTION DE L'ENVIRONNEMENT RL

### A. Espace d'Observation (State Space)

L'agent reçoit une **fenêtre glissante** (ex: 30 dernières bougies) contenant :

**1. Données OHLCV normalisées :**
- Open, High, Low, Close, Volume.
- Normalisées en **pourcentage de variation** (returns) ou via **rolling z-score** sur la fenêtre.

> [!WARNING]
> Ne pas utiliser `MinMaxScaler` global — c'est du **data leakage**. En production, l'agent ne connaît pas les prix futurs. Utiliser une normalisation locale (rolling z-score sur les N dernières bougies).

**2. Indicateurs Techniques (Alpha Factors) :**

| Catégorie | Indicateurs |
|---|---|
| Tendance | MACD, EMA (50, 200) |
| Oscillateurs | RSI, CCI |
| Volatilité | ATR, Bandes de Bollinger |
| Volume | OBV (On-Balance Volume), VWAP |

**3. Données Multi-Timeframe :**
- Direction de l'EMA 200 sur le **Daily** (tendance macro).
- RSI et ATR agrégés sur **4H** (contexte intermédiaire).
- Permet à l'agent d'aligner ses trades 1H avec la tendance de fond.

**4. État du Portefeuille :**
- Solde USDT et solde BTC.
- PnL non-réalisé en % (plus informatif que le prix d'entrée brut).
- Temps écoulé depuis le dernier trade (anti sur-trading).

**5. Features avancées (V2) :**
- Order Book Imbalance (ratio bid/ask).
- Funding Rate (si futures).
- Fear & Greed Index / Sentiment.

### B. Espace d'Action (Action Space)

Espace **Continu** (`Box[-1, 1]`) avec **dead zone** :

| Plage | Action |
|---|---|
| `[-1, -0.05]` | Vendre (proportionnel : -1 = sell 100%) |
| `[-0.05, 0.05]` | **Hold** (dead zone — aucun trade exécuté) |
| `[0.05, 1]` | Acheter (proportionnel : 1 = buy 100% du capital) |

> [!IMPORTANT]
> La dead zone `[-0.05, 0.05]` évite les micro-trades qui seraient dévorés par les frais. Optionnellement, ajouter un **cooldown** d'un minimum de N steps entre deux trades.

### C. Fonction de Récompense (Reward Function)

**Formule principale :**

$$R_t = \log\left(\frac{v_t}{v_{t-1}}\right) - c \cdot |action| - \lambda \cdot \sigma_t$$

| Terme | Description |
|---|---|
| $\log(v_t / v_{t-1})$ | Log-return du portefeuille (plus stable que le delta brut) |
| $c \cdot \|action\|$ | Pénalité proportionnelle aux frais de transaction |
| $\lambda \cdot \sigma_t$ | Pénalité de volatilité (risk aversion) |

**Termes additionnels :**
- **Pénalité de drawdown** : Si le portefeuille perd > X% depuis son pic → pénalité sévère (stop-loss implicite).
- **Bonus de Sharpe glissant** : Petit bonus quand le Sharpe ratio sur les 100 derniers steps s'améliore.
- **Reward shaping** : Faible bonus quand l'agent prend une position alignée avec la tendance (l'EMA macro), pour accélérer l'apprentissage initial.

> [!NOTE]
> Le paramètre $\lambda$ est un hyperparamètre critique. Trop haut → l'agent ne trade jamais. Trop bas → il prend des risques excessifs. À tuner via Optuna.

---

## 6. PLAN D'IMPLÉMENTATION

### Phase 1 : Data Engineering
1. Télécharger l'historique BTC/USDT (+ ETH/USDT, SOL/USDT) en **1H** sur les **5 dernières années** via `CCXT`.
2. Calculer les indicateurs techniques et les features multi-timeframe.
3. Diviser : **Train (70%)**, **Validation (15%)**, **Test (15%)** — split chronologique, pas aléatoire.
4. Appliquer la **rolling normalization** (z-score sur fenêtre glissante).
5. **Data augmentation** : bruit gaussien sur les prix, jitter des volumes, inversion temporelle (transformer un bull market en bear).

### Phase 2 : Construction de l'Environnement Gym
1. Implémenter `CryptoTradingEnv(gym.Env)` dans `env/trading_env.py`.
2. Méthode `step(action)` :
   - Applique la dead zone sur l'action.
   - Exécute l'achat/vente proportionnel.
   - **Soustrait les frais de transaction**.
   - Met à jour la NAV (Net Asset Value).
   - Calcule la récompense (module `reward.py`).
   - Retourne `obs`, `reward`, `terminated`, `truncated`, `info`.
3. **Domain Randomization** : à chaque `reset()`, varier le capital initial (±20%), les frais (0.05%-0.15%), et le point de départ dans les données.

### Phase 3 : Entraînement
1. **Algorithme principal : SAC** (meilleur pour les espaces continus grâce à l'entropie maximale).
2. **Algorithme de comparaison : PPO** (plus conservateur, baseline robuste).
3. Entraînement sur **2M+ timesteps** minimum.
4. **Curriculum Learning** : commencer par les périodes de forte tendance, puis introduire les marchés latéraux.
5. **Prioritized Experience Replay (PER)** pour SAC : prioriser les transitions à forte erreur TD.
6. Monitoring via `Tensorboard` + `Weights & Biases`.
7. Recherche d'hyperparamètres via **Optuna** (learning rate, $\lambda$, taille du réseau, etc.).

### Phase 4 : Backtesting & Visualisation
1.  **Moteur de Backtest** : Tester sur données **out-of-sample** (2024-2025).
2.  **Visualisation (DÉMO)** :
    -   Créer un module `evaluation/visualization.py` utilisant `Plotly`.
    -   **Graphique 1** : Chandelier (Candlestick) + Flèches Achat (Vert) / Vente (Rouge) + Zones de Hold.
    -   **Graphique 2** : Évolution du Portfolio Value (NAV) vs Buy & Hold.
    -   **Export** : Générer un fichier `.html` interactif ou un `.gif` pour partager les meilleures sessions.
3. Métriques obligatoires :

| Métrique | Description |
|---|---|
| Sharpe Ratio | Rendement ajusté au risque |
| Sortino Ratio | Comme Sharpe mais pénalise uniquement la vol. baissière |
| Maximum Drawdown (MDD) | Perte maximale depuis le pic |
| Durée du Drawdown | Temps pour récupérer le pic |
| Calmar Ratio | Rendement annualisé / MDD |
| Win Rate | % de trades gagnants |
| Profit Factor | Gains totaux / Pertes totales |
| Nombre de trades | Détection du sur-trading |
| Durée moyenne d'un trade | Cohérence de la stratégie |

3. Comparer vs **Buy & Hold** et une baseline aléatoire.
4. Si l'agent sous-performe → revoir la reward function et les features.

### Phase 5 : Paper Trading (Validation temps réel)
1. Connecter l'agent au flux **WebSocket temps réel** de Binance.
2. Exécuter les trades **sans argent réel** pendant **2 à 4 semaines**.
3. Comparer les performances temps réel vs backtest.
4. Si écart significatif → **overfitting détecté**, retour en Phase 3.

### Phase 6 : Live Trading
1. Déploiement avec capital limité (10-20% du budget max).
2. Activation de toutes les règles de sécurité (`risk_manager.py`).
3. Monitoring continu + alertes Telegram/Discord.
4. Augmentation progressive du capital si performances stables sur 1+ mois.

---

## 7. SÉCURITÉ ET GESTION DU RISQUE

### Règles Hard-Coded (hors IA)

| Règle | Seuil | Action |
|---|---|---|
| **Stop-Loss par trade** | Perte > 3% | Vente immédiate |
| **Daily Loss Limit** | Perte journalière > 5% | Arrêt total jusqu'au lendemain |
| **Position Size Limit** | Position > 80% du capital | Refus de l'ordre |
| **Circuit Breaker** | Mouvement > 5% en 5 min | Pause de l'agent pendant 1H |
| **Pas de levier** | — | Interdit en V1 |

> [!CAUTION]
> Ces règles sont implémentées dans `live/risk_manager.py` et ne peuvent **jamais** être outrepassées par l'IA. Elles constituent le dernier filet de sécurité.

### Monitoring et Alertes

- **Chaque trade** déclenche une notification (Telegram/Discord) avec : paire, direction, taille, prix, PnL.
- **Anomalies** (circuit breaker, stop-loss, daily limit atteint) déclenchent une alerte prioritaire.
- **Rapport quotidien** automatique : PnL du jour, Sharpe rolling, drawdown courant.

---

## 8. TECHNIQUES D'ENTRAÎNEMENT AVANCÉES

| Technique | Description | Phase |
|---|---|---|
| **Domain Randomization** | Varier capital, frais, point de départ à chaque épisode | V1 |
| **Curriculum Learning** | Difficulté progressive (tendance → latéral → chaotique) | V1 |
| **Ensemble d'agents** | N agents avec seeds différents votent ensemble | V2 |
| **Multi-asset training** | Entraîner sur BTC + ETH + SOL pour patterns généraux | V1 |
| **Data Augmentation** | Bruit gaussien, inversion temporelle, jitter volumes | V1 |

---

## 9. ROADMAP V2 (Améliorations futures)

| Idée | Impact potentiel |
|---|---|
| **Decision Transformer** | Capture mieux les dépendances temporelles longues qu'un MLP |
| **Sentiment Analysis** (Twitter/X, Fear & Greed Index) | Dimension macro pour le decisioning |
| **Multi-agent** (un agent par timeframe) | Meilleure vision multi-échelle |
| **Meta-learning (MAML)** | Adaptation rapide aux changements de régime de marché |
| **Inverse RL / RLHF** | Apprendre la reward function à partir de trades experts |
| **Order Book features** | Pression bid/ask pour le timing d'entrée |