"""
tests/test_backtest.py — Tests unitaires de la partie testable sans
données réelles sur disque de evaluation/backtest.py (le découpage
chronologique du split test).
"""

import pandas as pd
import pytest

from evaluation.backtest import extract_test_slice


class TestSplitSlice:
    def test_matches_download_py_convention(self):
        # Same formula as data/download.py: train_end = n*train_ratio,
        # val_end = train_end + n*val_ratio, test = everything after.
        df = pd.DataFrame({"close": range(100)})
        result = extract_test_slice(df, train_ratio=0.70, val_ratio=0.15)
        assert len(result) == 15  # 100 - 70 - 15
        assert result["close"].iloc[0] == 85
        assert result["close"].iloc[-1] == 99

    def test_returns_a_copy_not_a_view(self):
        df = pd.DataFrame({"close": range(10)})
        result = extract_test_slice(df, train_ratio=0.5, val_ratio=0.2)
        result["close"] = 0
        assert df["close"].iloc[-1] != 0
