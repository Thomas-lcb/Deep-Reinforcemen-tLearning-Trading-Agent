# Meta-learning (Reptile) pour corriger le distribution shift — Design

Date : 2026-09-25
Statut : approuvé pour implémentation

## 1. Contexte et motivation

Le backtest réel (Phase 4, `evaluation/backtest.py`) a révélé que le curriculum
PPO généralise correctement en L1 (0% frais, +11.36% sur le test set jamais vu,
Sharpe 3.54) mais s'effondre dès L2 (frais réels Binance, -38.48%, Sharpe
-16.6) — pire qu'une politique aléatoire. L3 et L4 héritent du même problème.
La cause retenue : la fenêtre de train (2025-09-18 → 2026-05-31) est
structurellement baissière (-36.9% sur toute la période), la fenêtre de test
est haussière (Buy & Hold +22.08% la confirme). Le curriculum L2+ apprend une
politique adaptée à un seul régime, qui ne transfère pas quand le régime
s'inverse. Tout le réglage fin de cette même session sur L2 (`cooldown_steps`,
`fee_penalty_weight`, `target_kl`) a amélioré des métriques *intra-
distribution* qui ne généralisent pas.

Décision du 25/09 : corriger ça par du meta-learning plutôt que par la piste
moins chère (plus d'historique / inversion temporelle) — l'objectif explicite
est de maximiser la performance, en connaissance du coût plus élevé de cette
option.

## 2. Non-objectifs

- Pas de MAML/FOMAML second-order — le coût mémoire (garder le graphe de calcul
  de la boucle interne pour rétropropager à travers elle) est un risque réel
  d'OOM sur une GTX 960M à 2 Go de VRAM. Reptile (first-order, sans second-order
  gradient) est le choix retenu.
- Pas de RL² / politique récurrente — approche alternative valable mais qui ne
  répond pas à la demande explicite de meta-learning ; à reconsidérer seulement
  si Reptile échoue empiriquement.
- Pas de L3/L4 dans ce premier cycle. Seuls L1 et L2 sont meta-entraînés et
  backtestés : L1 sert de témoin (il généralise déjà — le meta-learning ne
  doit pas le dégrader), L2 est le cas qu'on cherche à corriger. L3/L4 hériteront
  de la méthode dans un cycle ultérieur une fois validée sur ces deux niveaux.
- Pas de continuité de position entre segments du backtest walk-forward (voir
  §6) — chaque segment repart d'un épisode frais, comme le fait déjà
  `evaluation/backtest.py` aujourd'hui pour l'évaluation zero-shot.

## 3. Vue d'ensemble de l'algorithme (Reptile)

Reptile est une approximation first-order de MAML : pas de rétropropagation à
travers la boucle interne, juste une moyenne de directions.

Pour chaque itération externe (*outer iteration*) :
1. Tirer un batch de `meta_batch_size` tâches — une tâche = une fenêtre
   temporelle aléatoire de `window_steps` pas dans le split train (voir §4).
2. Pour chaque tâche : cloner les poids meta courants (`θ`) dans un modèle PPO
   frais (nouvel optimiseur Adam, pas d'état d'optimiseur partagé entre
   tâches), l'entraîner (`model.learn(total_timesteps=inner_steps)`) sur
   l'environnement restreint à cette fenêtre, obtenir les poids adaptés `θ'ᵢ`.
3. Moyenner les déltas du batch : `Δ = mean_i(θ'ᵢ − θ)`.
4. Mettre à jour les poids meta : `θ ← θ + meta_step_size · Δ`.

Le modèle meta final n'est pas directement le produit fini : au déploiement
(backtest), il se fine-tune encore `deploy_adaptation_steps` pas sur les
données les plus récentes disponibles avant d'évaluer la suite (walk-forward,
§6) — c'est ce pas d'adaptation à chaud qui justifie Reptile plutôt qu'un
entraînement classique mieux régularisé.

**Warm start** : le modèle meta part des poids déjà entraînés
(`models/saved/ppo_curriculum_l{1,2}.zip`), pas d'une initialisation aléatoire
— ces modèles ont déjà appris des fondamentaux de trading corrects ; le
problème visé est la généralisation, pas l'apprentissage from scratch. Ça
réduit aussi le budget compute nécessaire.

## 4. Échantillonnage des tâches (`training/task_sampler.py`, nouveau)

```python
def sample_task_window(
    dfs: dict[str, pd.DataFrame],
    window_steps: int,
    lookback_window: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """
    Tire une fenêtre temporelle aléatoire de `window_steps` pas dans chaque
    DataFrame de `dfs`, avec `lookback_window` pas de marge en tête (pour que
    le feature cache de l'environnement ait de l'historique dès le premier
    step de la fenêtre). Même index de départ pour toutes les paires de
    `dfs` (aligne les timestamps si plusieurs paires étaient chargées).
    Ne prélève jamais hors des bornes de `dfs` passé en argument — c'est
    l'appelant qui garantit que `dfs` est déjà borné au split train
    (même convention que `training/curriculum.py::load_data`).
    """
```

Implémentation : un seul `start` tiré dans `[lookback_window, n - window_steps)`
pour toutes les paires (aujourd'hui une seule paire, BTC/USDT), retourne
`df.iloc[start - lookback_window : start + window_steps].copy()` par paire.
Lève `ValueError` si `window_steps + lookback_window >= n` (fenêtre demandée
plus grande que les données disponibles).

## 5. Boucle Reptile (`training/meta_reptile.py`, nouveau)

```python
def inner_adapt(base_state_dict, task_dfs, config, inner_steps, device) -> dict:
    """
    Crée un PPO frais (mêmes hyperparamètres que config['training']['ppo'],
    nouvel optimiseur), charge base_state_dict dans sa policy, l'entraîne
    inner_steps pas sur un DummyVecEnv construit sur task_dfs (mode="train"),
    retourne model.policy.state_dict() (tenseurs .cpu().clone(), pour ne pas
    garder de référence au graphe de calcul du modèle jetable).
    """

def reptile_outer_loop(
    base_model_path: str,
    train_dfs: dict,
    config: dict,
    meta_cfg: dict,   # config['meta']
    device: str,
    run_name: str,
) -> PPO:
    """
    Charge base_model_path comme point de départ, exécute meta_cfg['outer_iterations']
    itérations Reptile (§3), log wandb par itération (récompense moyenne des
    tâches du batch, temps par itération), retourne le modèle meta final.
    """
```

Un `DummyVecEnv` (pas `SubprocVecEnv`) pour l'entraînement interne : les
fenêtres de tâche sont courtes (`inner_steps` par défaut = `n_steps` = 4096,
un seul cycle de collecte+update PPO), le coût de démarrage de sous-processus
par tâche dominerait le temps réel d'entraînement.

Log W&B (projet `RLD-Trading`, groupe `meta_learning`, tags
`["meta", f"level_{level}", "reptile"]`) : une entrée par itération externe
avec la récompense moyenne des tâches du batch et le temps écoulé — permet de
vérifier, comme pour `cooldown_steps` ou `target_kl` plus tôt dans le projet,
que le pas meta converge plutôt que de diverger silencieusement.

## 6. Config (`config/config.yaml`)

Nouvelle section :

```yaml
meta:
  window_steps: 20000            # Longueur d'une tâche = max_episode_steps existant (~14 jours
                                  # de bougies 1m) — même échelle que les épisodes normaux, pour
                                  # que l'inner loop voie une dynamique de marché comparable.
  meta_batch_size: 6              # Tâches par itération externe. Petit délibérément (contrainte
                                   # GPU 2 Go) — à ajuster après le run pilote (§9).
  inner_steps: 4096                # = training.ppo.n_steps : un cycle de collecte+update PPO
                                    # par tâche pour l'adaptation interne.
  outer_iterations: 100            # Nombre de pas meta. Valeur de départ, à revoir après le
                                    # run pilote.
  meta_step_size: 0.3              # epsilon du pas Reptile (θ ← θ + epsilon · Δ). Valeur de
                                    # départ standard dans la littérature Reptile (0.1-0.5).
  deploy_adaptation_steps: 4096    # Pas de fine-tuning au déploiement, avant chaque segment
                                    # walk-forward (§7). Même ordre de grandeur que inner_steps.
  walk_forward_segment_steps: 10000  # ~7 jours de bougies 1m par segment évalué.
```

## 7. Backtest walk-forward (`evaluation/backtest.py`)

Nouveau mode, activé par `--walk-forward` :

```python
def run_walk_forward_backtest(
    model_path: str,
    full_df: pd.DataFrame,     # NON tronqué au test set — walk-forward a besoin
                                # des données antérieures à chaque segment pour l'adaptation
    test_start_idx: int,       # même formule que extract_test_slice (val_end)
    config: dict,
    segment_steps: int,
    adapt_steps: int,
    lookback_window: int,
    device: str,
) -> tuple[np.ndarray, pd.DataFrame]:  # (equity curve compoundée, trade_history concaténé)
```

Pour chaque segment de test `[s, s+segment_steps)` (`s` parcourant
`test_start_idx, test_start_idx+segment_steps, ...` jusqu'à la fin du df) :

1. **Adaptation** : cloner le modèle meta, fine-tuner `adapt_steps` pas sur la
   fenêtre `[max(0, s - adapt_steps - lookback_window), s)` — strictement
   antérieure à `s`, donc causale (peut piocher dans train/val pour les tout
   premiers segments, jamais dans le futur du segment évalué).
2. **Évaluation** : run déterministe du modèle adapté sur
   `[s - lookback_window, s + segment_steps)`, épisode indépendant avec son
   propre `initial_capital` (pas de portage de position entre segments — voir
   non-objectif §2). Récupérer la série de rendements du segment (pas la
   valeur absolue du portefeuille).
3. **Compounding** : concaténer les séries de rendements de tous les segments
   dans l'ordre chronologique, reconstruire UNE courbe d'équité continue en
   partant de `initial_capital` — c'est cette courbe unique qui alimente
   `evaluation/metrics.summarize()`, exactement comme pour le mode zero-shot
   existant.

Ce choix (recomposer via les rendements plutôt que faire porter une position
d'un segment à l'autre) évite de modifier `env/trading_env.py` pour accepter
un état initial arbitraire — chaque segment reste un épisode standard de
l'environnement existant. Limite assumée : une position ouverte à la fin d'un
segment est liquidée par la fin d'épisode plutôt que portée au segment
suivant ; acceptable pour ce premier cycle (raffinement possible plus tard si
ça s'avère être un biais significatif).

Le mode zero-shot existant (`run_backtest`) n'est pas modifié — le nouveau
mode walk-forward est une fonction additionnelle, appelée seulement quand
`--walk-forward` est passé, pour comparer les deux (voir §9).

## 8. CLI

```bash
# Meta-training (par niveau)
python -m training.meta_reptile --level 1 --device cuda
python -m training.meta_reptile --level 2 --device cuda
# Sauvegarde : models/saved/ppo_meta_l{level}.zip

# Backtest walk-forward du modèle meta
python -m evaluation.backtest --level 1 --model models/saved/ppo_meta_l1 --walk-forward --device cuda
python -m evaluation.backtest --level 2 --model models/saved/ppo_meta_l2 --walk-forward --device cuda
```

`--walk-forward` sans argument utilise `meta.walk_forward_segment_steps` et
`meta.deploy_adaptation_steps` de la config ; `--segment-steps`/`--adapt-steps`
permettent de les surcharger pour une exploration rapide sans éditer le YAML.

## 9. Plan de validation empirique

Comparaison à quatre colonnes sur le test set, pour L1 et L2 séparément :

| | L{1,2} zero-shot (existant) | L{1,2} meta zero-shot | L{1,2} meta walk-forward | Buy & Hold |
|---|---|---|---|---|

- **L{1,2} zero-shot** : chiffres déjà mesurés (`Next_step.md`, Phase 4).
- **L{1,2} meta zero-shot** : le modèle meta évalué sans adaptation walk-forward
  (`run_backtest` existant, juste avec `--model ppo_meta_l{level}`) — isole
  l'effet du warm-start + Reptile seul, sans le pas d'adaptation au
  déploiement.
- **L{1,2} meta walk-forward** : le résultat complet de la méthode.

Cette décomposition en 3 colonnes (au lieu d'une seule "meta") permet de
distinguer si un gain observé vient de Reptile lui-même ou du pas
d'adaptation walk-forward — sans ça, un résultat positif ne dirait pas
*pourquoi* ça marche.

**Run pilote avant le run complet** : avant de lancer `outer_iterations=100`
sur L2, lancer un pilote court (`outer_iterations=5` ou 10) pour mesurer le
temps réel par itération sur cette GPU et ajuster `meta_batch_size`/
`outer_iterations` en conséquence — pas de budget deviné, comme pour tous les
autres réglages de ce projet.

## 10. Tests

Tous sans GPU ni données réelles (DataFrames synthétiques, comme le reste du
repo) :

`tests/test_task_sampler.py` (nouveau) :
- Fenêtre retournée a bien `window_steps + lookback_window` lignes.
- `start` toujours dans les bornes de `dfs` passé (jamais hors limites).
- `ValueError` si `window_steps` demandé dépasse les données disponibles.
- Deux appels avec des `rng` différents donnent des fenêtres différentes (pas
  de constante cachée).

`tests/test_meta_reptile.py` (nouveau) :
- Maths du pas Reptile isolées de PPO/SB3 : sur des state_dicts jouets
  (tenseurs simples, pas de vrai réseau), vérifier que
  `mean_i(θ'ᵢ − θ)` et `θ + epsilon · Δ` produisent le résultat attendu pour
  des deltas connus.

`tests/test_backtest.py` (extension) :
- Segmentation walk-forward : couverture complète du test set par les
  segments générés, aucun chevauchement, aucun segment ne commence avant
  `test_start_idx`.
- La fenêtre d'adaptation d'un segment ne dépasse jamais dans le futur de ce
  segment (`adapt_window_end <= segment_start`).
- Compounding des rendements : sur des courbes segment synthétiques connues,
  vérifier que la courbe reconstruite correspond au produit cumulé attendu.

Pas de test d'intégration bout-en-bout avec un vrai modèle PPO dans la suite
automatisée (trop coûteux/lent) — validé manuellement via le run pilote (§9),
comme pour le short-selling (§12 de sa spec) et le reste du réglage de ce
projet.

## 11. Risques et questions ouvertes

- `meta_step_size=0.3`, `inner_steps=4096`, `meta_batch_size=6` sont des
  valeurs de départ raisonnées, pas calibrées — à ajuster empiriquement après
  le run pilote, comme `cooldown_steps` ou `funding_rate_per_step` avant eux.
- La liquidation de position en fin de segment (§7) est une simplification
  qui peut sous-estimer la performance réelle d'une stratégie qui aurait
  porté une position gagnante d'un segment à l'autre — à surveiller dans les
  résultats, pas corrigé dans ce premier cycle.
- Rien ne garantit que Reptile corrige effectivement le distribution shift —
  c'est l'hypothèse testée, pas un acquis. Le plan de validation (§9) est
  construit pour donner une réponse claire même si le résultat est négatif
  (auquel cas RL²/contexte, non retenu ici, redevient la piste suivante).
- Si L1 (déjà généralisant) est dégradé par le meta-training, c'est un signal
  que `meta_step_size` ou `inner_steps` déforment trop le modèle de base — le
  témoin L1 sert précisément à détecter ça tôt.
