from __future__ import annotations

import pandas as pd


def validate_not_empty(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} is empty")


def validate_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def validate_null_rate(df: pd.DataFrame, max_null_rate: float = 0.20) -> dict[str, float]:
    rates = df.isna().mean().to_dict()
    bad = {k: v for k, v in rates.items() if v > max_null_rate}
    if bad:
        raise ValueError(f"Null-rate check failed: {bad}")
    return rates
