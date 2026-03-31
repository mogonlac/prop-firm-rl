"""
preprocess_mes.py — Raw Databento MES 1-min bars → preprocessed CSV (1-min).

This script reads raw GLBX.MDP3 MES 1-minute bars, computes technical
indicators, and outputs data/preprocessed_mes.csv at 1-min resolution.

If preprocessed_mes.csv already exists, this script is not needed.
Kept for reproducibility.

NOTE: An earlier version of this script resampled to 5-min before saving.
That resample has been removed — the output and all downstream environments
expect 1-min bars.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta  # type: ignore

INPUT_PATH  = "data/raw_mes_1min.csv"
OUTPUT_PATH = "data/preprocessed_mes.csv"


def assign_trading_date(ts):
    """CME session date: bars after 5 PM CT → next calendar date."""
    ct = ts.tz_localize("UTC").tz_convert("US/Central") if ts.tzinfo is None else ts.tz_convert("US/Central")
    if ct.hour >= 17:
        return (ct + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return ct.strftime("%Y-%m-%d")


def main():
    print("Loading raw 1-min bars...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["ts_event"])
    df = df.sort_values("ts_event").reset_index(drop=True)

    # Filter out spread instruments
    df = df[df["close"] > 1000].copy()

    ohlcv = df.copy()

    # Assign trading dates
    print("Assigning trading dates...")
    ohlcv["trading_date"] = ohlcv["ts_event"].apply(assign_trading_date)

    # Drop maintenance halt (4-5 PM CT) and post-cutoff (3:10+ PM CT)
    ohlcv["ct_hour"] = ohlcv["ts_event"].dt.tz_localize("UTC").dt.tz_convert("US/Central").dt.hour
    ohlcv["ct_minute"] = ohlcv["ts_event"].dt.tz_localize("UTC").dt.tz_convert("US/Central").dt.minute
    mask_halt = (ohlcv["ct_hour"] == 16)
    mask_cutoff = (ohlcv["ct_hour"] == 15) & (ohlcv["ct_minute"] >= 10)
    ohlcv = ohlcv[~mask_halt & ~mask_cutoff].copy()
    ohlcv = ohlcv.drop(columns=["ct_hour", "ct_minute"])

    # Technical indicators
    print("Computing indicators...")
    c = ohlcv["close"]
    ohlcv["returns"]      = c.pct_change()
    ohlcv["volatility"]   = ohlcv["returns"].rolling(20).std()
    ohlcv["momentum"]     = ta.roc(c, length=10) / 100.0
    rsi_raw               = ta.rsi(c, length=14)
    ohlcv["rsi"]          = rsi_raw / 100.0
    bb = ta.bbands(c, length=20, std=2.0)
    ohlcv["bb_pos"]       = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
    sma50                 = ta.sma(c, length=50)
    sma200                = ta.sma(c, length=200)
    ohlcv["sma_cross"]    = (sma50 - sma200) / sma200
    vol_ma                = ohlcv["volume"].rolling(20).mean()
    ohlcv["volume_ratio"] = ohlcv["volume"] / vol_ma

    # Time features (CT)
    ct_times = ohlcv["ts_event"].dt.tz_localize("UTC").dt.tz_convert("US/Central")
    ohlcv["hour"]        = ct_times.dt.hour / 23.0
    ohlcv["day_of_week"] = ct_times.dt.dayofweek / 6.0

    # Clean up
    ohlcv = ohlcv.dropna().reset_index(drop=True)
    ohlcv["ts_event"] = ohlcv["ts_event"].astype(str)

    cols = ["ts_event", "trading_date", "open", "high", "low", "close", "volume",
            "returns", "volatility", "momentum", "rsi", "bb_pos", "sma_cross",
            "volume_ratio", "hour", "day_of_week"]
    ohlcv[cols].to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(ohlcv)} bars to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
