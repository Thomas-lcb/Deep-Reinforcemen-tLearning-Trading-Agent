# Short-selling (Niveau 4 du curriculum) — Design

Date : 2026-09-21
Statut : approuvé pour implémentation

## 1. Contexte et motivation

Le curriculum L1→L3 est validé : `target_kl` élimine l'effondrement de `train/std`
observé en mars/avril, `cooldown_steps=15` a fait passer `profit_factor` (L2) d'une
moyenne de 0.18 à un pic de 1.73. Mais l'agent reste net légèrement perdant en
moyenne, et l'analyse de `trading/win_rate` (34-38% en moyenne sur les derniers
runs, très en dessous de 50%) montre que ce n'est plus un problème de fréquence
ou de taille de trade — c'est un problème de **direction** : `env/action.py` ne
permet que d'acheter ou de revendre ce qu'on possède déjà (jamais de vente à
découvert), et le marché d'entraînement (BTC/USDT, 2025-09-18 → 2026-05-31) est
structurellement baissier (-36.9% sur toute la période de train, -1.98% en
moyenne sur des fenêtres de 20 000 steps). Un agent qui ne peut être que long ou
plat, sur un marché qui baisse en moyenne, est mathématiquement désavantagé.

Objectif : permettre à l'agent de vendre à découvert, avec un mécanisme de marge
assez réaliste pour transférer proprement vers un usage réel (marge/perpétuels
type Binance) plus tard, sans construire un moteur d'échange complet aujourd'hui.

## 2. Non-objectifs

- Pas de vrai moteur de marge multi-niveaux (appels de marge progressifs, carnet
  d'ordres, slippage) — hors de portée vu l'état actuel du projet.
- Pas de données de funding rate réelles téléchargées — le coût de financement
  est une approximation constante, pas un signal de marché.
- Pas de levier > 1x (pas d'endettement au-delà du capital disponible).
- L1-L3 ne changent pas de comportement. Le short est strictement additif,
  gated par une nouvelle config `short.enabled` (défaut `false`).

## 3. Modèle de comptage — l'insight clé

`_portfolio_value()` (`env/trading_env.py`) calcule déjà :

```python
return self.balance_usdt + self.balance_asset * current_price
```

Cette formule est **déjà correcte pour un `balance_asset` négatif**, sans aucune
modification : si `balance_asset = -0.1` BTC et que le prix monte de 50 000 à
60 000, `balance_asset * current_price` passe de -5 000 à -6 000 — la valeur du
portefeuille baisse de 1 000, exactement la perte économique réelle d'un short.
Pas besoin d'une comptabilité de marge séparée pour la valorisation : on
autorise simplement `balance_asset` à devenir négatif, et tout le reste
(`_get_info()`, `get_portfolio_history()`, le calcul de reward) continue de
fonctionner sans changement.

Ce qu'il faut ajouter, ce n'est donc pas une nouvelle comptabilité, mais trois
garde-fous absents aujourd'hui : plafonner la taille d'une position courte,
détecter et forcer sa liquidation si la perte dépasse un seuil, et appliquer un
coût de financement périodique tant qu'elle est ouverte.

## 4. Config (`config/config.yaml`)

Nouvelle section :

```yaml
short:
  enabled: false                  # Off par défaut : L1-L3 inchangés. true pour L4 uniquement.
  funding_rate_per_step: 0.000001 # ~0.024%/8h, ordre de grandeur d'un funding rate réel de
                                   # perpétuel BTC (Binance historique ~0.01-0.03%/8h), ramené
                                   # au pas de 1 minute. Sans ce coût, l'agent apprendrait à
                                   # shorter gratuitement — ne transférerait pas au réel.
  maintenance_margin_pct: 0.5     # Liquidation forcée si la perte latente atteint 50% de la
                                   # valeur notionnelle à l'entrée. Avec max_position_pct=0.25,
                                   # pire cas ≈ 12.5% du capital perdu par liquidation — borné.
  liquidation_penalty_pct: 0.01   # Coût supplémentaire à la liquidation forcée vs une clôture
                                   # volontaire, pour inciter l'agent à gérer son risque plutôt
                                   # que compter sur le filet de sécurité.
```

## 5. Interprétation de l'action (`env/action.py`)

`interpret_action()` gagne un paramètre `short_enabled: bool = False`. Quand
`False`, le comportement est **identique bit à bit** à aujourd'hui (les tests
existants ne doivent pas changer). Quand `True`, la branche `action < -dead_zone`
se ramifie selon le signe de `balance_asset` :

| Condition | Type de trade | Logique |
|---|---|---|
| `action > dead_zone`, `balance_asset >= 0` | `buy` | Inchangé — ouvre/augmente le long |
| `action > dead_zone`, `balance_asset < 0` | `cover` | Rachète une fraction du short : `proportion = (action-dead_zone)/(1-dead_zone)` capée par `max_position_pct`, `amount = abs(balance_asset) * proportion` |
| `action < -dead_zone`, `balance_asset > 0` | `sell` | Inchangé — clôture/réduit le long |
| `action < -dead_zone`, `balance_asset <= 0`, `short_enabled=True` | `short` | Ouvre/augmente le short : `proportion` idem, notional = `balance_usdt * proportion` (miroir exact de `buy`), `amount_asset = notional / price` |
| `action < -dead_zone`, `balance_asset <= 0`, `short_enabled=False` | `hold` | Comportement actuel inchangé (`amount_asset = balance_asset * proportion = 0`) |

Frais : `short` calcule les frais comme `sell` (`fee = gross_usdt * fee_rate`,
sur le produit de la vente à découvert). `cover` calcule les frais comme `buy`
(`fee` inclus dans le coût du rachat).

## 6. Exécution (`env/trading_env.py::step()`)

Deux nouvelles branches dans l'exécution du trade, symétriques aux branches
`buy`/`sell` existantes :

- **`short`** : `balance_usdt += net_usdt` (produit net de la vente à découvert),
  `balance_asset -= amount_asset` (devient négatif). Prix d'entrée moyen pondéré
  avec la même formule que `buy`, généralisée en valeur absolue :
  `entry_price = (entry_price * abs(old_balance_asset) + notional) / abs(new_balance_asset)`.
- **`cover`** : PnL réalisé **inversé** par rapport à une vente longue —
  `pnl_pct = (entry_price - current_price) / entry_price` (un short gagne quand
  le prix baisse). `balance_usdt -= (cost + fee)`, `balance_asset += amount`. Si
  `balance_asset` revient à ~0, reset `entry_price = 0.0` (comme aujourd'hui
  pour une clôture longue complète).

**Funding** : à chaque step, si `balance_asset < 0` (short ouvert), déduire
`abs(balance_asset) * current_price * funding_rate_per_step` de `balance_usdt`,
indépendamment de l'action prise ce step-là (coût de portage, pas un frais de
transaction).

**Liquidation forcée** : après l'avancement de `self.current_step` (donc avec
le même `current_price_now` déjà utilisé pour `unrealized_pnl_pct`, pas le
`current_price` d'avant exécution du trade) et avant le calcul de la reward, si
`balance_asset < 0` et `entry_price > 0` : `loss_pct = (current_price_now -
entry_price) / entry_price`. Si `loss_pct >=
maintenance_margin_pct`, forcer une clôture complète au prix courant (même
mécanique qu'un `cover` à 100%), avec un coût additionnel de
`liquidation_penalty_pct` sur le montant racheté, et marquer le trade comme
`type="liquidation"` dans `info["trade"]` pour distinguer une clôture volontaire
d'une clôture forcée dans les futures analyses W&B.

**Asymétrie assumée** : ce garde-fou de liquidation ne s'applique qu'aux
positions courtes. Une position longue entièrement payée (pas d'emprunt) ne
peut perdre au maximum que ce qui a été investi — le garde-fou générique déjà
existant (`current_value < initial_capital * 0.05` → fin d'épisode) reste le
seul filet pour le long, comme aujourd'hui.

## 7. Observation (`env/observation.py`)

Bug à corriger en marge de ce travail : `unrealized_pnl_pct` dans
`get_observation()` ne gère que `balance_asset > 0` (retourne toujours 0.0 pour
une position courte, cachant l'info à l'agent). À corriger :

```python
if entry_price > 0 and balance_asset > 0:
    unrealized_pnl_pct = (asset_price - entry_price) / entry_price
elif entry_price > 0 and balance_asset < 0:
    unrealized_pnl_pct = (entry_price - asset_price) / entry_price
else:
    unrealized_pnl_pct = 0.0
```

`balance_asset_pct = (balance_asset * asset_price) / total_value` est déjà
correct pour une position courte (devient naturellement négatif) — aucun
changement nécessaire. Aucun changement à `build_observation_space` (bornes
déjà `[-inf, inf]`).

## 8. Reward (`env/reward.py` — pas de changement de fichier)

`RewardCalculator.calculate()` est déjà générique : `position_direction` prend
juste une nouvelle valeur possible (`-1.0` au lieu de seulement `0.0`/`1.0`), et
`trend_alignment_bonus` (`trend_direction * position_direction`) fonctionne déjà
correctement pour ce cas (récompense un short quand la tendance macro est
baissière). Seul `trading_env.py::step()` change : `pos_dir = 1.0 if
balance_asset > 1e-10 else (-1.0 if balance_asset < -1e-10 else 0.0)`, et
`unrealized_pnl_pct` calculé avec la même formule inversée que dans
`observation.py` (§7).

Le coût de funding n'a pas de terme de reward dédié dans cette première
itération — il se répercute déjà naturellement dans `log_return` via la baisse
de `balance_usdt`, exactement comme les frais de transaction le font
aujourd'hui. Si l'agent n'apprend pas à limiter la durée de ses shorts, un audit
du type de celui fait sur `fee_penalty_weight` sera la prochaine étape logique
(pas dans le scope de cette itération).

## 9. Curriculum (`training/curriculum.py`)

Nouveau niveau 4, chargeant les poids sauvegardés de L3 :

```python
elif level == 4:
    cfg["market"]["pairs"] = ["BTC/USDT"]
    cfg["fees"]["maker"] = 0.001
    cfg["fees"]["taker"] = 0.001
    cfg["training"]["domain_randomization"]["enabled"] = True  # comme L3
    cfg["short"]["enabled"] = True
    cfg["training"]["total_timesteps"] = 1_500_000  # même budget que L3
```

`LEVEL_DESCRIPTIONS` gagne une entrée `4: "Résilience + short (BTC/USDT, vente à
découvert autorisée)"`. La boucle `for level in levels_to_run` et le choix
`--level` de l'argparse passent de `range(1, 4)`/`[1,2,3]` à `range(1,
5)`/`[1,2,3,4]`. Le chargement de poids (`level > 1` → `PPO.load(...l{level-1})`)
généralise déjà correctement à L4 sans changement.

## 10. Tests

`tests/test_env.py` — nouvelle classe `TestShortSelling` :
- Ouverture d'un short (`balance_asset` devient négatif, `balance_usdt`
  augmente du produit net).
- `pnl_pct` d'un `cover` correctement inversé (gain quand le prix a baissé).
- Liquidation forcée déclenchée quand la perte latente dépasse
  `maintenance_margin_pct`, avec le coût de pénalité appliqué.
- Funding déduit de `balance_usdt` à chaque step tant qu'un short est ouvert.
- **Non-régression** : avec `short.enabled=false` (défaut), `action < -dead_zone`
  sur `balance_asset<=0` reste un `hold` — la suite de tests existante
  (60 tests) doit passer sans aucune modification.

`tests/test_reward.py` — cas `position_direction=-1.0` pour
`trend_alignment_bonus` (déjà couvert par la formule générique, test de
confirmation seulement).

## 11. Compatibilité ascendante

Garantie : `short.enabled=false` (valeur par défaut si la clé est absente de la
config, comme tous les autres paramètres de ce projet) reproduit exactement le
comportement actuel. L1, L2, L3 ne sont pas retouchés. C'est vérifiable
directement par la suite de tests existante, qui ne doit nécessiter aucune
modification.

## 12. Risques et questions ouvertes

- `funding_rate_per_step` et `maintenance_margin_pct` sont des approximations
  raisonnées, pas des valeurs calibrées sur des données réelles de funding —
  à ajuster empiriquement comme tous les autres hyperparamètres de ce projet
  (cf. `cooldown_steps`, `fee_penalty_weight`).
- Le budget de L4 (1.5M steps, calqué sur L3) est une première estimation ; si
  l'agent a besoin de plus de temps pour apprendre le nouveau degré de liberté
  (short), ça se verra dans `ep_rew_mean`/`train/std` du run de validation.
- Cette itération ne dit rien sur si le short *améliore* réellement les
  performances — seulement qu'il devient possible. La validation empirique
  (win_rate, profit_factor sur L4 vs L3) est l'étape suivante après
  l'implémentation.
