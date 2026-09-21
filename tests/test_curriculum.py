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


@pytest.fixture
def base_config_short_true():
    """
    Starts with short.enabled=True (e.g. config.yaml hand-edited), unlike
    base_config. Used to actually exercise set_curriculum_level() forcing
    short.enabled back to False for L1-3, rather than passing vacuously
    because the fixture already had the expected value.
    """
    return {
        "market": {"pairs": ["BTC/USDT"]},
        "fees": {"maker": 0.0, "taker": 0.0},
        "training": {
            "total_timesteps": 2_000_000,
            "domain_randomization": {"enabled": True},
        },
        "short": {"enabled": True},
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

    def test_levels_1_to_3_force_short_disabled_even_if_hand_edited(self, base_config_short_true):
        """
        Regression test: previously only Level 4 touched short.enabled, so
        if config.yaml were ever hand-edited to short.enabled=true, running
        the full curriculum would silently leak shorting into L1-3. The
        fixture here starts with short.enabled=True to actually exercise
        the forcing behavior (unlike test_levels_1_to_3_unaffected, whose
        base_config fixture already has short.enabled=False and so passes
        vacuously).
        """
        for level in (1, 2, 3):
            cfg = set_curriculum_level(base_config_short_true, level)
            assert cfg["short"]["enabled"] is False

    def test_level_4_enables_short_when_key_absent(self):
        """
        Regression test: cfg["short"]["enabled"] = True would raise KeyError
        if the "short" key is absent from config.yaml. set_curriculum_level()
        must use setdefault so every new config value has a safe default.
        """
        config = {
            "market": {"pairs": ["BTC/USDT"]},
            "fees": {"maker": 0.0, "taker": 0.0},
            "training": {
                "total_timesteps": 2_000_000,
                "domain_randomization": {"enabled": True},
            },
            # No "short" key at all.
        }
        cfg = set_curriculum_level(config, 4)
        assert cfg["short"]["enabled"] is True
