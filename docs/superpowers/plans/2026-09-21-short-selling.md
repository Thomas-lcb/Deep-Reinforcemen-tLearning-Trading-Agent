# Short-selling (Niveau 4 du curriculum) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permettre à l'agent de vendre à découvert (short) en plus d'acheter/vendre long, avec marge 1x, coût de financement périodique et liquidation forcée bornant la perte maximale — activé uniquement pour un nouveau Niveau 4 du curriculum, sans changer le comportement de L1-L3.

**Architecture:** `balance_asset` devient signé (négatif = position courte). La formule de valorisation existante (`balance_usdt + balance_asset * price`) gère déjà ce cas correctement sans modification. On ajoute : deux nouveaux types de trade (`short`, `cover`) dans `env/action.py`, deux nouvelles branches d'exécution + un coût de financement périodique + une liquidation forcée dans `env/trading_env.py::step()`, un fix de bug dans `env/observation.py` pour que l'agent voie le PnL latent d'une position courte, et un Niveau 4 dans `training/curriculum.py`.

**Tech Stack:** Python 3.10, Gymnasium, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-short-selling-design.md`

## Global Constraints

- `short.enabled` par défaut à `false` (absent de `config.yaml` avant ce plan, donc `config.get("short", {}).get("enabled", False)`). Avec cette valeur, le comportement doit être **identique bit à bit** à l'état actuel — la suite de tests existante (60 tests dans `tests/test_env.py` + `tests/test_reward.py`) ne doit nécessiter **aucune** modification et doit rester verte après chaque tâche.
- `max_position_pct` (déjà dans `config.yaml`, actuellement `0.25`) plafonne aussi bien l'ouverture d'un short que celle d'un long — pas de nouveau paramètre de taille séparé.
- Toute nouvelle valeur de config a un défaut sûr via `.get(key, default)` — ne jamais supposer que la clé existe dans `config.yaml` (cohérent avec le reste du codebase).
- `entry_price`, `pnl_pct`, `unrealized_pnl_pct` gardent leur signe/sens existant pour une position longue ; pour une position courte, le profit vient de la **baisse** du prix — la formule s'inverse (`(entry_price - price) / entry_price` au lieu de `(price - entry_price) / entry_price`).

---

## Task 1: `env/action.py` — interprétation `short`/`cover`

**Files:**
- Modify: `env/action.py:31-111` (`interpret_action`)
- Test: `tests/test_action.py` (nouveau fichier — `env/action.py` n'a actuellement aucun test dédié, il n'est testé qu'indirectement via `tests/test_env.py`)

**Interfaces:**
- Consumes: rien (fonction pure existante, étendue).
- Produces: `interpret_action(..., short_enabled: bool = False) -> dict` avec `dict["type"]` pouvant désormais valoir `"short"` ou `"cover"` en plus de `"buy"`/`"sell"`/`"hold"`. Pour `"short"` : `amount_usdt` = produit net (après frais, comme `"sell"`). Pour `"cover"` : `amount_usdt` = coût total incluant frais (comme `"buy"`). Consommé par Task 3.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_action.py` :

```python
"""
tests/test_action.py — Tests unitaires de l'interprétation d'action
(env/action.py), y compris le short-selling (short_enabled=True).
"""

import pytest
from env.action import interpret_action, apply_cooldown


class TestLongOnlyUnchanged:
    """short_enabled=False (défaut) doit reproduire le comportement actuel."""

    def test_sell_while_flat_is_hold_without_short(self):
        trade = interpret_action(
            raw_action=-0.8,
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            short_enabled=False,
        )
        assert trade["type"] == "hold"

    def test_buy_unchanged(self):
        trade = interpret_action(
            raw_action=0.8,
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=1.0,
        )
        assert trade["type"] == "buy"
        assert trade["amount_asset"] > 0


class TestShortOpen:
    def test_sell_while_flat_opens_short_when_enabled(self):
        trade = interpret_action(
            raw_action=-0.8,
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=1.0,
            short_enabled=True,
        )
        assert trade["type"] == "short"
        assert trade["amount_asset"] > 0
        assert trade["fee"] > 0
        # amount_usdt = produit NET (après frais), comme "sell"
        gross = trade["amount_asset"] * 50000.0
        assert trade["amount_usdt"] == pytest.approx(gross - trade["fee"], rel=1e-6)

    def test_short_capped_by_max_position_pct(self):
        trade = interpret_action(
            raw_action=-1.0,  # pleine puissance
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=0.25,
            short_enabled=True,
        )
        assert trade["proportion"] == pytest.approx(0.25)
        notional = trade["amount_asset"] * 50000.0
        assert notional == pytest.approx(10000.0 * 0.25, rel=1e-6)

    def test_sell_while_already_short_increases_short(self):
        trade = interpret_action(
            raw_action=-0.8,
            balance_usdt=10000.0,
            balance_asset=-0.05,  # déjà short 0.05 BTC
            asset_price=50000.0,
            short_enabled=True,
        )
        assert trade["type"] == "short"


class TestCover:
    def test_buy_while_short_covers(self):
        trade = interpret_action(
            raw_action=0.8,
            balance_usdt=10000.0,
            balance_asset=-0.1,  # short 0.1 BTC
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=1.0,
            short_enabled=True,
        )
        assert trade["type"] == "cover"
        assert trade["amount_asset"] > 0
        assert trade["amount_asset"] <= 0.1 + 1e-9
        # amount_usdt = cout total INCLUANT les frais, comme "buy"
        cost = trade["amount_asset"] * 50000.0
        assert trade["amount_usdt"] == pytest.approx(cost + trade["fee"], rel=1e-6)

    def test_cover_proportional_to_short_size(self):
        trade = interpret_action(
            raw_action=1.0,  # pleine puissance -> proportion cappee a max_position_pct
            balance_usdt=10000.0,
            balance_asset=-0.2,
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=0.5,
            short_enabled=True,
        )
        assert trade["proportion"] == pytest.approx(0.5)
        assert trade["amount_asset"] == pytest.approx(0.2 * 0.5, rel=1e-6)


class TestCooldownStillWorksWithNewTypes:
    def test_cooldown_blocks_short(self):
        trade = interpret_action(
            raw_action=-0.8,
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            short_enabled=True,
        )
        blocked = apply_cooldown(steps_since_trade=0, cooldown_steps=5, trade=trade)
        assert blocked["type"] == "hold"
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_action.py -v`
Expected: `TestLongOnlyUnchanged` passe déjà (comportement actuel). `TestShortOpen`, `TestCover` échouent avec `TypeError: interpret_action() got an unexpected keyword argument 'short_enabled'`.

- [ ] **Step 3: Implémenter `short`/`cover` dans `env/action.py`**

Remplacer la fonction `interpret_action` (lignes 31-111) en entier par :

```python
def interpret_action(
    raw_action: float,
    balance_usdt: float,
    balance_asset: float,
    asset_price: float,
    dead_zone: float = 0.05,
    fee_rate: float = 0.001,
    max_position_pct: float = 1.0,
    short_enabled: bool = False,
) -> dict:
    """
    Interpret the agent's raw action into a concrete trade.

    Args:
        raw_action: Value in [-1, 1] from the agent.
        balance_usdt: Current USDT balance.
        balance_asset: Current asset quantity. Negative = short position
            (only possible when short_enabled has been True in the past).
        asset_price: Current asset price.
        dead_zone: Actions within [-dead_zone, +dead_zone] are treated as Hold.
        fee_rate: Transaction fee rate (e.g. 0.001 = 0.1%).
        max_position_pct: Maximum proportion of capital per trade (e.g. 0.25 = 25%).
        short_enabled: If False (default), a negative action while flat or
            short does nothing (today's long-only behavior, unchanged bit
            for bit). If True, it opens/increases a short position.

    Returns:
        Dict with keys:
        - 'type': 'buy' | 'sell' | 'short' | 'cover' | 'hold'
        - 'amount_usdt': for 'buy'/'cover', total cash outlay INCLUDING fee.
          for 'sell'/'short', net proceeds AFTER fee.
        - 'amount_asset': asset quantity traded.
        - 'proportion': effective proportion of capital/position used.
        - 'fee': fee for this trade.
    """
    # Clamp action
    action = float(np.clip(raw_action, -1.0, 1.0))

    # Dead zone → Hold
    if abs(action) <= dead_zone:
        return {
            "type": "hold",
            "amount_usdt": 0.0,
            "amount_asset": 0.0,
            "proportion": 0.0,
            "fee": 0.0,
        }

    if action > dead_zone:
        proportion = (action - dead_zone) / (1.0 - dead_zone)
        proportion = min(proportion, max_position_pct)

        if balance_asset < 0:
            # COVER: buy back a fraction of the existing short.
            amount_asset = abs(balance_asset) * proportion
            cost = amount_asset * asset_price
            fee = cost * fee_rate
            amount_usdt = cost + fee

            return {
                "type": "cover",
                "amount_usdt": amount_usdt,
                "amount_asset": amount_asset,
                "proportion": proportion,
                "fee": fee,
            }

        # BUY: scale proportion from 0 to 1 over [dead_zone, 1]
        amount_usdt = balance_usdt * proportion

        # Account for fees: we can only buy (amount / (1 + fee))
        effective_usdt = amount_usdt / (1.0 + fee_rate)
        amount_asset = effective_usdt / asset_price if asset_price > 0 else 0.0
        fee = amount_usdt - effective_usdt

        return {
            "type": "buy",
            "amount_usdt": amount_usdt,
            "amount_asset": amount_asset,
            "proportion": proportion,
            "fee": fee,
        }

    else:
        proportion = (abs(action) - dead_zone) / (1.0 - dead_zone)
        proportion = min(proportion, max_position_pct)

        if balance_asset <= 0:
            if not short_enabled:
                # Unchanged today's behavior: nothing to sell while flat.
                return {
                    "type": "hold",
                    "amount_usdt": 0.0,
                    "amount_asset": 0.0,
                    "proportion": 0.0,
                    "fee": 0.0,
                }

            # SHORT: open/increase a short position.
            notional = balance_usdt * proportion
            amount_asset = notional / asset_price if asset_price > 0 else 0.0
            gross_usdt = amount_asset * asset_price
            fee = gross_usdt * fee_rate
            amount_usdt = gross_usdt - fee

            return {
                "type": "short",
                "amount_usdt": amount_usdt,
                "amount_asset": amount_asset,
                "proportion": proportion,
                "fee": fee,
            }

        # SELL: scale proportion from 0 to 1 over [-1, -dead_zone]
        amount_asset = balance_asset * proportion

        # Revenue after fees
        gross_usdt = amount_asset * asset_price
        fee = gross_usdt * fee_rate
        net_usdt = gross_usdt - fee

        return {
            "type": "sell",
            "amount_usdt": net_usdt,
            "amount_asset": amount_asset,
            "proportion": proportion,
            "fee": fee,
        }
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_action.py -v`
Expected: tous les tests passent (PASS).

- [ ] **Step 5: Vérifier la non-régression complète**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/ -q`
Expected: `61 passed` (60 existants + le nouveau `test_action.py` compte plusieurs tests — le nombre exact peut différer, mais **aucun test existant ne doit échouer**).

- [ ] **Step 6: Commit**

```bash
cd /mnt/Data/Projets/Code/RLD_Trading
git add env/action.py tests/test_action.py
git commit -m "Ajoute l'interprétation short/cover dans interpret_action() (short_enabled, défaut False)"
```

---

## Task 2: `env/observation.py` — fix `unrealized_pnl_pct` pour une position courte

**Files:**
- Modify: `env/observation.py:92-95` (`get_observation`)
- Test: `tests/test_observation.py` (nouveau fichier — `env/observation.py` n'a actuellement aucun test dédié)

**Interfaces:**
- Consumes: rien (fonction pure existante).
- Produces: `get_observation(...)` retourne désormais un `unrealized_pnl_pct` non-nul et correctement inversé pour `balance_asset < 0`. Aucun changement de signature. Consommé indirectement par `env/trading_env.py::_get_obs()` (aucune modification nécessaire côté appelant).

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/test_observation.py` :

```python
"""
tests/test_observation.py — Tests unitaires de la construction
de l'observation (env/observation.py), y compris le PnL latent
d'une position courte.
"""

import numpy as np
import pytest

from env.observation import get_observation


def _base_kwargs(**overrides):
    kwargs = dict(
        market_data=np.zeros((10, 3), dtype=np.float32),
        balance_usdt=5000.0,
        balance_asset=0.0,
        asset_price=50000.0,
        entry_price=0.0,
        steps_since_trade=0,
        initial_capital=10000.0,
        lookback_window=10,
    )
    kwargs.update(overrides)
    return kwargs


class TestUnrealizedPnlPct:
    def test_long_position_positive_pnl(self):
        obs = get_observation(**_base_kwargs(
            balance_asset=0.1, entry_price=40000.0, asset_price=50000.0,
        ))
        # unrealized_pnl_pct est la 3e colonne du vecteur portefeuille
        pnl = obs[0, -2]
        assert pnl == pytest.approx((50000.0 - 40000.0) / 40000.0, rel=1e-5)

    def test_flat_position_zero_pnl(self):
        obs = get_observation(**_base_kwargs(balance_asset=0.0, entry_price=0.0))
        pnl = obs[0, -2]
        assert pnl == pytest.approx(0.0)

    def test_short_position_pnl_inverted(self):
        # Short ouvert a 50000, prix retombe a 45000 -> profit pour le short
        obs = get_observation(**_base_kwargs(
            balance_asset=-0.1, entry_price=50000.0, asset_price=45000.0,
        ))
        pnl = obs[0, -2]
        expected = (50000.0 - 45000.0) / 50000.0
        assert pnl == pytest.approx(expected, rel=1e-5)
        assert pnl > 0  # le prix a baisse, le short est gagnant

    def test_short_position_losing(self):
        # Short ouvert a 50000, prix monte a 55000 -> perte pour le short
        obs = get_observation(**_base_kwargs(
            balance_asset=-0.1, entry_price=50000.0, asset_price=55000.0,
        ))
        pnl = obs[0, -2]
        assert pnl < 0
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_observation.py -v`
Expected: `test_short_position_pnl_inverted` et `test_short_position_losing` échouent (`pnl == 0`, `assert 0.0 > 0` et `assert 0.0 < 0` échouent) puisque `balance_asset > 0` est actuellement requis.

- [ ] **Step 3: Corriger `get_observation()`**

Dans `env/observation.py`, remplacer les lignes 91-95 :

```python
    # Unrealized PnL
    if balance_asset > 0 and entry_price > 0:
        unrealized_pnl_pct = (asset_price - entry_price) / entry_price
    else:
        unrealized_pnl_pct = 0.0
```

par :

```python
    # Unrealized PnL — a long position profits when price rises above entry;
    # a short position (balance_asset < 0) profits when price falls below
    # entry, so the formula inverts.
    if entry_price > 0 and balance_asset > 0:
        unrealized_pnl_pct = (asset_price - entry_price) / entry_price
    elif entry_price > 0 and balance_asset < 0:
        unrealized_pnl_pct = (entry_price - asset_price) / entry_price
    else:
        unrealized_pnl_pct = 0.0
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_observation.py -v`
Expected: tous les tests passent (PASS).

- [ ] **Step 5: Vérifier la non-régression complète**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/ -q`
Expected: aucun test existant cassé.

- [ ] **Step 6: Commit**

```bash
cd /mnt/Data/Projets/Code/RLD_Trading
git add env/observation.py tests/test_observation.py
git commit -m "Fix unrealized_pnl_pct qui ignorait toujours les positions courtes dans l'observation"
```

---

## Task 3: `config/config.yaml` + `env/trading_env.py` — ouverture/clôture d'un short

**Files:**
- Modify: `config/config.yaml` (ajout section `short:`)
- Modify: `env/trading_env.py:117-160` (`__init__` — wiring config), `env/trading_env.py:244-291` (`step()` — appel `interpret_action` + exécution)
- Test: `tests/test_env.py` (nouvelle classe `TestShortSelling`)

**Interfaces:**
- Consumes: `interpret_action(..., short_enabled=...)` (Task 1), `env.action.apply_cooldown` (inchangé).
- Produces: `CryptoTradingEnv` accepte `balance_asset < 0`. Nouveaux attributs d'instance : `self.short_enabled: bool`, `self.funding_rate_per_step: float`, `self.maintenance_margin_pct: float`, `self.liquidation_penalty_pct: float` (consommés par Tasks 4 et 5). `info["trade"]["type"]` peut valoir `"short"`/`"cover"`.

- [ ] **Step 1: Ajouter la section `short:` à `config/config.yaml`**

Ajouter à la fin du fichier (après la section `paths:`) :

```yaml
short:
  enabled: false                  # Off par défaut : L1-L3 inchangés. true pour le Niveau 4 uniquement.
  funding_rate_per_step: 0.000001 # ~0.024%/8h ramené au pas de 1 minute (ordre de grandeur d'un
                                   # funding rate réel de perpétuel BTC, historiquement ~0.01-0.03%/8h
                                   # sur Binance). Sans ce coût, l'agent apprendrait à shorter
                                   # gratuitement — ne transférerait pas à un usage réel.
  maintenance_margin_pct: 0.5     # Liquidation forcée si la perte latente atteint 50% de la valeur
                                   # notionnelle à l'entrée. Avec max_position_pct=0.25, pire cas
                                   # ≈ 12.5% du capital perdu par liquidation — borné.
  liquidation_penalty_pct: 0.01   # Coût supplémentaire à la liquidation forcée vs une clôture
                                   # volontaire, pour inciter l'agent à gérer son risque.
```

- [ ] **Step 2: Écrire les tests qui échouent (`TestShortSelling`)**

Ajouter à `tests/test_env.py`, dans une nouvelle classe (placer après `TestCooldown`) :

```python
class TestShortSelling:
    """
    Regression tests for short-selling, gated by config short.enabled
    (default False — see docs/superpowers/specs/2026-09-21-short-selling-design.md).
    The default `env` fixture loads the real config.yaml, where
    short.enabled=False, so these tests explicitly flip env.short_enabled
    for the duration of the test.
    """

    def test_short_disabled_by_default(self, env):
        assert env.short_enabled is False

    def test_sell_while_flat_opens_short(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([-0.8]))
        assert info["trade"]["type"] == "short"
        assert env.balance_asset < 0
        assert env.balance_usdt > 0  # a recu le produit net de la vente a decouvert

    def test_cover_closes_short_and_reports_pnl(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.reset(seed=0)
        env.step(np.array([-1.0]))  # ouvre un short
        entry_price = env.entry_price
        cover_price = env.close_prices[env.current_step]
        expected_pnl_pct = (entry_price - cover_price) / entry_price

        _, _, _, _, info = env.step(np.array([1.0]))  # couvre entierement

        assert info["trade"]["type"] == "cover"
        assert "pnl_pct" in info["trade"]
        assert info["trade"]["pnl_pct"] == pytest.approx(expected_pnl_pct, rel=1e-6)
        assert env.balance_asset == pytest.approx(0.0, abs=1e-9)
        assert env.entry_price == 0.0

    def test_short_pnl_positive_when_price_falls(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.reset(seed=0)
        env.step(np.array([-1.0]))
        # Force artificiellement une baisse de prix pour un test deterministe
        env.close_prices[env.current_step] = env.entry_price * 0.9
        _, _, _, _, info = env.step(np.array([1.0]))
        assert info["trade"]["pnl_pct"] > 0
```

- [ ] **Step 3: Lancer les tests pour vérifier l'échec**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_env.py -v -k "TestShortSelling"`
Expected: `test_short_disabled_by_default` échoue avec `AttributeError: 'CryptoTradingEnv' object has no attribute 'short_enabled'`. Les autres échouent en cascade.

- [ ] **Step 4: Wirer la config dans `__init__`**

Dans `env/trading_env.py`, juste après le bloc `# --- Domain randomization config ---` (après la ligne `self.dr_fee_range = ...`, avant `# --- Episode length (training only) ---`), insérer :

```python
        # --- Short-selling config ---
        short_cfg = config.get("short", {})
        self.short_enabled = short_cfg.get("enabled", False)
        self.funding_rate_per_step = short_cfg.get("funding_rate_per_step", 0.0)
        self.maintenance_margin_pct = short_cfg.get("maintenance_margin_pct", 0.5)
        self.liquidation_penalty_pct = short_cfg.get("liquidation_penalty_pct", 0.0)
```

- [ ] **Step 5: Passer `short_enabled` à `interpret_action()` et exécuter `short`/`cover`**

Dans `env/trading_env.py::step()`, modifier l'appel à `interpret_action` (bloc `# --- Interpret action ---`) pour ajouter le paramètre :

```python
        # --- Interpret action ---
        trade = interpret_action(
            raw_action=raw_action,
            balance_usdt=self.balance_usdt,
            balance_asset=self.balance_asset,
            asset_price=current_price,
            dead_zone=self.dead_zone,
            fee_rate=self.fee_rate,
            max_position_pct=self.max_position_pct,
            short_enabled=self.short_enabled,
        )
```

Puis, juste après le bloc `elif trade["type"] == "sell" and trade["amount_asset"] > 0:` existant (après son `self._log_trade(trade, current_price)` et avant le `else:` final qui gère le hold), insérer deux nouvelles branches :

```python
        elif trade["type"] == "short" and trade["amount_asset"] > 0:
            notional_at_entry = trade["amount_asset"] * current_price
            total_cost_basis = self.entry_price * abs(self.balance_asset) + notional_at_entry

            self.balance_usdt += trade["amount_usdt"]  # net proceeds
            self.balance_asset -= trade["amount_asset"]  # devient plus negatif

            if self.balance_asset < 0:
                self.entry_price = total_cost_basis / abs(self.balance_asset)

            self.steps_since_trade = 0
            self._log_trade(trade, current_price)

        elif trade["type"] == "cover" and trade["amount_asset"] > 0:
            # Realized P&L of a short is INVERTED vs a long: profits when
            # price falls below entry.
            if self.entry_price > 0:
                trade["pnl_pct"] = (self.entry_price - current_price) / self.entry_price

            self.balance_usdt -= trade["amount_usdt"]  # cost + fee
            self.balance_asset += trade["amount_asset"]  # se rapproche de 0

            if self.balance_asset > -1e-10:
                self.balance_asset = 0.0
                self.entry_price = 0.0

            self.steps_since_trade = 0
            self._log_trade(trade, current_price)

```

Le `else:` final (qui fait `self.steps_since_trade += 1`) reste inchangé et continue de gérer le cas `hold`.

- [ ] **Step 6: Lancer les tests pour vérifier le succès**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_env.py -v -k "TestShortSelling"`
Expected: tous les tests passent (PASS).

- [ ] **Step 7: Vérifier la non-régression complète**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/ -q`
Expected: aucun test existant cassé (60 tests d'avant ce plan doivent toujours passer).

- [ ] **Step 8: Commit**

```bash
cd /mnt/Data/Projets/Code/RLD_Trading
git add config/config.yaml env/trading_env.py tests/test_env.py
git commit -m "Ouverture/clôture de positions courtes dans trading_env.py (short.enabled, défaut False)"
```

---

## Task 4: Coût de financement périodique

**Files:**
- Modify: `env/trading_env.py:293-298` (`step()`, entre l'avancement de `current_step` et le calcul de la reward)
- Test: `tests/test_env.py` (ajout à `TestShortSelling`)

**Interfaces:**
- Consumes: `self.short_enabled`, `self.funding_rate_per_step`, `self.balance_asset`, `self.close_prices` (Task 3).
- Produces: `self.balance_usdt` diminue à chaque step tant qu'un short est ouvert. Consommé par le calcul de `current_value` (déjà existant, aucun changement nécessaire là).

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `TestShortSelling` dans `tests/test_env.py` :

```python
    def test_funding_cost_deducted_while_short_open(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.funding_rate_per_step = 0.001  # valeur elevee pour rendre l'effet mesurable dans le test
        env.reset(seed=0)
        env.step(np.array([-1.0]))  # ouvre le short
        usdt_before = env.balance_usdt
        env.step(np.array([0.0]))  # hold : le funding doit quand meme s'appliquer
        assert env.balance_usdt < usdt_before

    def test_no_funding_cost_while_flat(self, env):
        env.short_enabled = True
        env.funding_rate_per_step = 0.001
        env.cooldown_steps = 0
        env.reset(seed=0)
        usdt_before = env.balance_usdt
        env.step(np.array([0.0]))  # reste a plat
        assert env.balance_usdt == pytest.approx(usdt_before)
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_env.py -v -k "test_funding_cost_deducted_while_short_open"`
Expected: FAIL — `assert env.balance_usdt < usdt_before` échoue (aucun funding appliqué actuellement).

- [ ] **Step 3: Implémenter le funding**

Dans `env/trading_env.py::step()`, juste après `self.current_step += 1` (bloc `# --- Advance time ---`) et avant `# --- Calculate reward ---`, insérer :

```python
        # --- Funding cost (short only) ---
        # Applied every step regardless of this step's action, mirroring a
        # perpetual futures funding payment — the cost of *holding* a short,
        # separate from the transaction fee paid when opening/closing it.
        if self.balance_asset < 0:
            price_for_funding = self.close_prices[min(self.current_step, self.n_steps - 1)]
            funding_cost = abs(self.balance_asset) * price_for_funding * self.funding_rate_per_step
            self.balance_usdt -= funding_cost
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_env.py -v -k "TestShortSelling"`
Expected: tous les tests passent (PASS), y compris `test_no_funding_cost_while_flat`.

- [ ] **Step 5: Vérifier la non-régression complète**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/ -q`
Expected: aucun test existant cassé (`self.funding_rate_per_step` par défaut vient de `config.yaml`, non nul mais minuscule — sans effet mesurable sur les tests existants qui n'ouvrent jamais de short puisque `short.enabled=False` y bloque toute ouverture).

- [ ] **Step 6: Commit**

```bash
cd /mnt/Data/Projets/Code/RLD_Trading
git add env/trading_env.py tests/test_env.py
git commit -m "Applique un coût de financement périodique tant qu'un short est ouvert"
```

---

## Task 5: Liquidation forcée

**Files:**
- Modify: `env/trading_env.py:293-354` (`step()` — entre le funding et le calcul de reward)
- Test: `tests/test_env.py` (ajout à `TestShortSelling`)

**Interfaces:**
- Consumes: `self.maintenance_margin_pct`, `self.liquidation_penalty_pct`, `self.fee_rate`, `self.entry_price`, `self.balance_asset` (Task 3).
- Produces: si liquidation, `self.balance_asset` et `self.entry_price` sont remis à 0, `self.balance_usdt` diminue du coût de rachat + frais + pénalité, et **`trade` (la variable locale utilisée pour `info["trade"]` et le calcul de reward) est remplacée** par un dict `type="liquidation"` incluant `pnl_pct`.

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `TestShortSelling` dans `tests/test_env.py` :

```python
    def test_forced_liquidation_on_large_adverse_move(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.maintenance_margin_pct = 0.5
        env.reset(seed=0)
        env.step(np.array([-1.0]))  # ouvre le short
        entry_price = env.entry_price
        assert env.balance_asset < 0

        # Simule une hausse de prix de +60% (perte latente > 50% du short)
        env.close_prices[env.current_step] = entry_price * 1.6

        _, _, terminated, truncated, info = env.step(np.array([0.0]))  # hold

        assert info["trade"]["type"] == "liquidation"
        assert env.balance_asset == pytest.approx(0.0, abs=1e-9)
        assert env.entry_price == 0.0

    def test_no_liquidation_within_margin(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.maintenance_margin_pct = 0.5
        env.reset(seed=0)
        env.step(np.array([-1.0]))
        entry_price = env.entry_price

        # +10% seulement : bien en dessous du seuil de maintenance de 50%
        env.close_prices[env.current_step] = entry_price * 1.1

        _, _, _, _, info = env.step(np.array([0.0]))

        assert info["trade"]["type"] == "hold"
        assert env.balance_asset < 0  # toujours ouvert
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_env.py -v -k "liquidation"`
Expected: `test_forced_liquidation_on_large_adverse_move` échoue (`info["trade"]["type"] == "hold"`, pas `"liquidation"`).

- [ ] **Step 3: Implémenter la liquidation**

Dans `env/trading_env.py::step()`, juste après le bloc de funding ajouté à la Task 4 (et donc toujours avant `# --- Calculate reward ---`), insérer :

```python
        # --- Forced liquidation (short only) ---
        # A long position that is fully paid for (no borrowing) can lose at
        # most what was invested — the existing bankruptcy check below
        # already covers that. A short's loss is theoretically unbounded, so
        # it needs its own guard: force-close before the loss exceeds the
        # margin committed to that specific position.
        if self.balance_asset < 0 and self.entry_price > 0:
            price_for_liq = self.close_prices[min(self.current_step, self.n_steps - 1)]
            loss_pct = (price_for_liq - self.entry_price) / self.entry_price

            if loss_pct >= self.maintenance_margin_pct:
                liq_amount_asset = abs(self.balance_asset)
                liq_cost = liq_amount_asset * price_for_liq
                liq_fee = liq_cost * self.fee_rate
                liq_penalty = liq_cost * self.liquidation_penalty_pct
                liq_pnl_pct = (self.entry_price - price_for_liq) / self.entry_price

                self.balance_usdt -= (liq_cost + liq_fee + liq_penalty)
                self.balance_asset = 0.0
                self.entry_price = 0.0
                self.steps_since_trade = 0

                trade = {
                    "type": "liquidation",
                    "amount_usdt": liq_cost + liq_fee + liq_penalty,
                    "amount_asset": liq_amount_asset,
                    "proportion": 1.0,
                    "fee": liq_fee + liq_penalty,
                    "pnl_pct": liq_pnl_pct,
                }
                self._log_trade(trade, price_for_liq)
```

Cette liquidation **remplace** la variable `trade` locale utilisée plus bas pour `info["trade"] = trade` et pour `action_magnitude=abs(raw_action) if trade["type"] != "hold" else 0.0` dans l'appel à `self.reward_calc.calculate(...)` — aucune autre ligne du fichier n'a besoin d'être modifiée pour que cette substitution soit prise en compte, car `trade` est déjà lu après ce point dans le code existant.

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_env.py -v -k "TestShortSelling"`
Expected: tous les tests passent (PASS).

- [ ] **Step 5: Vérifier la non-régression complète**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/ -q`
Expected: aucun test existant cassé.

- [ ] **Step 6: Commit**

```bash
cd /mnt/Data/Projets/Code/RLD_Trading
git add env/trading_env.py tests/test_env.py
git commit -m "Ajoute la liquidation forcée d'un short quand la perte latente dépasse maintenance_margin_pct"
```

---

## Task 6: `position_direction`/`unrealized_pnl_pct` pour la reward, cas short

**Files:**
- Modify: `env/trading_env.py:306-312` (`step()`, calcul de `pos_dir`/`unrealized_pnl_pct` avant l'appel à `reward_calc.calculate()`)
- Test: `tests/test_env.py` (ajout à `TestShortSelling`)

**Interfaces:**
- Consumes: `self.balance_asset`, `self.entry_price` (Task 3), `RewardCalculator.calculate()` (inchangé, déjà générique — voir `env/reward.py`).
- Produces: `reward_info["trend_bonus"]` correctement signé pour une position courte.

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `TestShortSelling` dans `tests/test_env.py` :

```python
    def test_short_position_direction_feeds_trend_bonus(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.reset(seed=0)
        env.step(np.array([-1.0]))  # ouvre un short -> position_direction doit valoir -1.0

        # Force artificiellement une tendance macro baissiere (alignee avec le short)
        if env._ema_dir_array is not None:
            env._ema_dir_array[env.current_step] = -1.0

        _, _, _, _, info = env.step(np.array([0.0]))
        # Aligne (short + tendance baissiere) -> bonus positif
        assert info["trend_bonus"] > 0
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_env.py -v -k "test_short_position_direction_feeds_trend_bonus"`
Expected: FAIL — `pos_dir` vaut actuellement `0.0` pour toute position `balance_asset <= 0` (y compris un short), donc `trend_bonus == 0`, pas `> 0`.

- [ ] **Step 3: Corriger le calcul de `pos_dir`/`unrealized_pnl_pct`**

Dans `env/trading_env.py::step()`, remplacer :

```python
        # Position direction: +1 if long (holding asset), 0 if flat
        pos_dir = 1.0 if self.balance_asset > 1e-10 else 0.0

        # Unrealized PNL percentage
        unrealized_pnl_pct = 0.0
        if self.balance_asset > 1e-10 and self.entry_price > 0:
            unrealized_pnl_pct = (current_price_now - self.entry_price) / self.entry_price
```

par :

```python
        # Position direction: +1 long, -1 short, 0 flat
        if self.balance_asset > 1e-10:
            pos_dir = 1.0
        elif self.balance_asset < -1e-10:
            pos_dir = -1.0
        else:
            pos_dir = 0.0

        # Unrealized PNL percentage — inverted for a short (profits when
        # price falls below entry), same formula as env/observation.py.
        unrealized_pnl_pct = 0.0
        if self.entry_price > 0 and self.balance_asset > 1e-10:
            unrealized_pnl_pct = (current_price_now - self.entry_price) / self.entry_price
        elif self.entry_price > 0 and self.balance_asset < -1e-10:
            unrealized_pnl_pct = (self.entry_price - current_price_now) / self.entry_price
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_env.py -v -k "TestShortSelling"`
Expected: tous les tests passent (PASS).

- [ ] **Step 5: Vérifier la non-régression complète**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/ -q`
Expected: aucun test existant cassé (`balance_asset` ne peut jamais être négatif quand `short_enabled=False`, donc les nouvelles branches `elif` ne sont jamais empruntées par les tests existants).

- [ ] **Step 6: Commit**

```bash
cd /mnt/Data/Projets/Code/RLD_Trading
git add env/trading_env.py tests/test_env.py
git commit -m "position_direction/unrealized_pnl_pct gèrent désormais le cas short (-1) dans la reward"
```

---

## Task 7: `training/curriculum.py` — Niveau 4

**Files:**
- Modify: `training/curriculum.py:31-72` (`LEVEL_DESCRIPTIONS`, `set_curriculum_level`), `training/curriculum.py:121-131` (argparse `--level`), `training/curriculum.py:137` (`for level in ...` par défaut)
- Test: `tests/test_curriculum.py` (nouveau fichier — `training/curriculum.py` n'a actuellement aucun test)

**Interfaces:**
- Consumes: `config["short"]["enabled"]` (Task 3).
- Produces: `set_curriculum_level(base_config, 4)` retourne une config avec `short.enabled=True`. Consommé par `main()` (déjà générique, aucun changement supplémentaire nécessaire pour le chargement de poids `PPO.load(...l{level-1}...)`, qui généralise déjà à `level=4`).

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/test_curriculum.py` :

```python
"""
tests/test_curriculum.py — Tests unitaires de set_curriculum_level()
(training/curriculum.py), en particulier le Niveau 4 (short-selling).
"""

import pytest

from training.curriculum import set_curriculum_level, LEVEL_DESCRIPTIONS


@pytest.fixture
def base_config():
    return {
        "market": {"pairs": ["BTC/USDT"]},
        "fees": {"maker": 0.0, "taker": 0.0},
        "training": {
            "total_timesteps": 2_000_000,
            "domain_randomization": {"enabled": True},
        },
        "short": {"enabled": False},
    }


class TestLevel4:
    def test_level_4_enables_short(self, base_config):
        cfg = set_curriculum_level(base_config, 4)
        assert cfg["short"]["enabled"] is True

    def test_level_4_keeps_l3_fees_and_domain_randomization(self, base_config):
        cfg = set_curriculum_level(base_config, 4)
        assert cfg["fees"]["maker"] == 0.001
        assert cfg["fees"]["taker"] == 0.001
        assert cfg["training"]["domain_randomization"]["enabled"] is True

    def test_level_4_has_description(self):
        assert 4 in LEVEL_DESCRIPTIONS

    def test_levels_1_to_3_unaffected(self, base_config):
        for level in (1, 2, 3):
            cfg = set_curriculum_level(base_config, level)
            assert cfg["short"]["enabled"] is False
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_curriculum.py -v`
Expected: FAIL — `set_curriculum_level(base_config, 4)` lève `ValueError: Niveau non supporté : 4` (le `else` existant).

- [ ] **Step 3: Ajouter le Niveau 4**

Dans `training/curriculum.py`, modifier `LEVEL_DESCRIPTIONS` :

```python
LEVEL_DESCRIPTIONS = {
    1: "Bases du trading (BTC/USDT, 0% frais)",
    2: "Contraintes réelles (BTC/USDT, frais Binance 0.1%)",
    3: "Résilience (BTC/USDT, Domain Randomization)",
    4: "Short-selling (BTC/USDT, vente à découvert autorisée)",
}
```

Puis, dans `set_curriculum_level`, ajouter un nouveau `elif` avant le `else: raise ValueError(...)` existant :

```python
    elif level == 4:
        print(f"💡 Niveau 4 : {LEVEL_DESCRIPTIONS[4]}")
        cfg["market"]["pairs"] = ["BTC/USDT"]
        cfg["fees"]["maker"] = 0.001
        cfg["fees"]["taker"] = 0.001
        cfg["training"]["domain_randomization"]["enabled"] = True
        cfg["short"]["enabled"] = True
        cfg["training"]["total_timesteps"] = 1_500_000
```

Enfin, mettre à jour l'argparse et la boucle par défaut dans `main()` :

```python
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4],
                         help="Run only this single level instead of the full 1→4 curriculum "
                              "(useful for quick validation runs).")
```

```python
    levels_to_run = [args.level] if args.level is not None else range(1, 5)
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/test_curriculum.py -v`
Expected: tous les tests passent (PASS).

- [ ] **Step 5: Vérifier la non-régression complète**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/ -q`
Expected: aucun test existant cassé.

- [ ] **Step 6: Commit**

```bash
cd /mnt/Data/Projets/Code/RLD_Trading
git add training/curriculum.py tests/test_curriculum.py
git commit -m "Ajoute le Niveau 4 (short-selling) au curriculum, charge les poids de L3"
```

---

## Task 8: Vérification finale, smoke-test et documentation

**Files:**
- Modify: `Next_step.md`, `REPO_OVERVIEW.md` (optionnel, court ajout)
- Test: suite complète + un run de validation court réel

**Interfaces:**
- Consumes: toutes les tâches précédentes.
- Produces: confirmation que le Niveau 4 tourne réellement de bout en bout (pas seulement au niveau des tests unitaires).

- [ ] **Step 1: Lancer la suite de tests complète**

Run: `cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python -m pytest tests/ -v`
Expected: PASS intégral, y compris les 60 tests d'avant ce plan (inchangés) et tous les nouveaux tests (`test_action.py`, `test_observation.py`, `test_curriculum.py`, `TestShortSelling`).

- [ ] **Step 2: Smoke-test réel du Niveau 4 (nécessite que L3 ait déjà été entraîné au moins une fois — `models/saved/ppo_curriculum_l3.zip` doit exister)**

Run:
```bash
cd /mnt/Data/Projets/Code/RLD_Trading
source venv/bin/activate
ls models/saved/ppo_curriculum_l3.zip  # vérifier que le prérequis existe
python -m training.curriculum --device cuda --level 4 --timesteps 30000
```
Expected: le run se termine sans erreur (`exited with code 0`), charge bien les poids de L3, et le modèle est sauvegardé dans `models/saved/ppo_curriculum_l4.zip`.

- [ ] **Step 3: Vérifier sur W&B que le run a bien loggé des trades de type short/cover/liquidation**

Run (dans un shell Python, après avoir récupéré l'ID du run sur `https://wandb.ai/thomas_lcb/RLD-Trading`) :
```bash
cd /mnt/Data/Projets/Code/RLD_Trading && source venv/bin/activate && python3 -c "
import wandb
api = wandb.Api()
r = api.run('thomas_lcb/RLD-Trading/<RUN_ID>')
hist = r.history(samples=200)
print([c for c in hist.columns if 'trading' in c or 'portfolio' in c])
"
```
Expected: `trading/win_rate`, `trading/trades_count`, `rollout/portfolio_value` sont présents (le pipeline de logging existant, Task pnl_pct de la session précédente, fonctionne sans changement pour `short`/`cover`/`liquidation` puisqu'ils utilisent la même clé `pnl_pct`).

- [ ] **Step 4: Documenter dans `Next_step.md`**

Ajouter une nouvelle sous-section après la Phase 3-bis existante :

```markdown
## Phase 3-ter : Short-selling (Niveau 4)
*cf. docs/superpowers/specs/2026-09-21-short-selling-design.md et docs/superpowers/plans/2026-09-21-short-selling.md*

- [x] 3c.1 — Implémentation complète (action.py short/cover, trading_env.py exécution+funding+liquidation, observation.py fix unrealized_pnl_pct, curriculum.py Niveau 4). `short.enabled=false` par défaut, L1-L3 inchangés (60 tests existants verts sans modification).
- [ ] 3c.2 — Lancer et valider le Niveau 4 complet (1.5M steps), comparer win_rate/profit_factor à L3 pour vérifier que le short améliore réellement les performances, pas seulement qu'il est possible.
```

- [ ] **Step 5: Commit final**

```bash
cd /mnt/Data/Projets/Code/RLD_Trading
git add Next_step.md
git commit -m "Documente la Phase 3-ter (short-selling) dans Next_step.md"
```

---

## Self-Review (fait par l'auteur du plan avant remise)

**Couverture de la spec** : §3 (comptage) → Task 3 step 5 (formule de valorisation déjà correcte, non modifiée, confirmé par les tests). §4 (config) → Task 3 step 1. §5 (action.py) → Task 1. §6 (trading_env.py, exécution/funding/liquidation) → Tasks 3, 4, 5. §7 (observation.py) → Task 2. §8 (reward) → Task 6. §9 (curriculum) → Task 7. §10 (tests) → chaque tâche a ses tests dédiés + Task 8 vérifie l'ensemble. §11 (compatibilité ascendante) → vérifié explicitement à la fin de chaque tâche (step "non-régression complète").

**Cohérence des types** : `interpret_action(..., short_enabled: bool = False) -> dict` (Task 1) est le seul point d'entrée modifié dans `action.py`, et son usage dans `trading_env.py` (Task 3) passe bien `short_enabled=self.short_enabled`. Les clés du dict retourné (`type`, `amount_usdt`, `amount_asset`, `proportion`, `fee`, `pnl_pct` ajouté conditionnellement) sont utilisées de façon cohérente dans toutes les tâches suivantes (Task 3 pour `short`/`cover`, Task 5 pour `liquidation`).

**Pas de placeholder** : chaque step contient soit du code Python complet, soit une commande shell exacte avec le résultat attendu précis.
