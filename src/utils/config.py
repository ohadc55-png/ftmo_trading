"""Typed configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MTFTrendConfig:
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    swing_lookback: int = 5


@dataclass
class VolumeConfig:
    ma_period: int = 20
    surge_mult: float = 1.5


@dataclass
class KeyLevelsConfig:
    breakout_buffer_points: float = 5.0
    premarket_start_hour: int = 1
    premarket_end_hour: int = 9


@dataclass
class VWAPConfig:
    rth_start_hour: int = 9
    rth_start_minute: int = 30
    overextended_pct: float = 0.5


@dataclass
class EnginesConfig:
    mtf_trend: MTFTrendConfig = field(default_factory=MTFTrendConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    key_levels: KeyLevelsConfig = field(default_factory=KeyLevelsConfig)
    vwap: VWAPConfig = field(default_factory=VWAPConfig)


@dataclass
class ScoringConfig:
    weight_mtf_trend: float = 3.0
    weight_volume: float = 2.0
    weight_breakout: float = 3.0
    weight_vwap: float = 2.0
    min_score_threshold: float = 6.0
    # Optional BUY-specific weights (None = use default weights)
    buy_weight_mtf_trend: float | None = None
    buy_weight_volume: float | None = None
    buy_weight_breakout: float | None = None
    buy_weight_vwap: float | None = None


@dataclass
class BacktestConfig:
    point_value: float = 20.0
    sl_atr_mult: float = 1.5
    rr_ratio: float = 1.5
    risk_pct: float = 0.50
    max_bars_held: int = 60
    atr_period: int = 14
    use_runner: bool = False
    tp_contracts: int = 1
    runner_contracts: int = 1
    trail_atr_mult: float = 1.5
    runner_max_bars: int = 120


@dataclass
class AppConfig:
    ticker: str = "NQ=F"
    timezone: str = "America/New_York"
    base_timeframe: str = "5m"
    mtf_timeframes: list[str] = field(default_factory=lambda: ["15min", "1h", "4h"])
    engines: EnginesConfig = field(default_factory=EnginesConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _build_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    if data is None:
        return cls()
    fieldnames = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in fieldnames}
    return cls(**filtered)


def load_config(path: str | Path = "config/settings.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    eng_raw = raw.get("engines", {})
    engines = EnginesConfig(
        mtf_trend=_build_dataclass(MTFTrendConfig, eng_raw.get("mtf_trend")),
        volume=_build_dataclass(VolumeConfig, eng_raw.get("volume")),
        key_levels=_build_dataclass(KeyLevelsConfig, eng_raw.get("key_levels")),
        vwap=_build_dataclass(VWAPConfig, eng_raw.get("vwap")),
    )

    scoring = _build_dataclass(ScoringConfig, raw.get("scoring"))
    backtest = _build_dataclass(BacktestConfig, raw.get("backtest"))

    return AppConfig(
        ticker=raw.get("ticker", "NQ=F"),
        timezone=raw.get("timezone", "America/New_York"),
        base_timeframe=raw.get("base_timeframe", "5m"),
        mtf_timeframes=raw.get("mtf_timeframes", ["15min", "1h", "4h"]),
        engines=engines,
        scoring=scoring,
        backtest=backtest,
    )
