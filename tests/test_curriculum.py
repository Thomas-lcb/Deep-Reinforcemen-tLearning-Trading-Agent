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
