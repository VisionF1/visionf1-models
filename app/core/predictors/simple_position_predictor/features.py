from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
from typing import Optional
from .constants import CACHED_DATA_PKL, BEFORE_FE_PATH, AFTER_FE_PATH
from app.config import DRIVERS_2025, PREDICTION_CONFIG

# NOTA: Este módulo contiene la lógica movida desde el archivo original.
# He mantenido nombres y comportamiento clave para no romper la API.

def build_base_df() -> pd.DataFrame:
        race = PREDICTION_CONFIG["next_race"]
        w = _active_weather()
        rows = []
        for code, cfg in DRIVERS_2025.items():
            rows.append({
                "driver": code,
                "team": cfg.get("team", ""),
                "rookie": cfg.get("rookie", False),
                "team_change": cfg.get("team_change", False),
                "race_name": race.get("race_name", ""),
                "year": race.get("year", 2025),
                "session_air_temp": w.get("session_air_temp", 25.0),
                "session_track_temp": w.get("session_track_temp", 35.0),
                "session_humidity": w.get("session_humidity", 60.0),
                "session_rainfall": float(w.get("session_rainfall", 0.0)) * 1.0,
            })
        df = pd.DataFrame(rows)
        df.index = df["driver"].values
        return df

def align_columns(X: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    if not order:
        return X
    X = X.copy()
    for col in order:
        if col not in X.columns:
            X[col] = 0.0
    return X[order]


def _active_weather() -> dict:
    return PREDICTION_CONFIG.get("weather", {}).get("prediction", {})


def _compute_weather_performance_stats(X_all: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        def _by_driver(g: pd.DataFrame):
            wet = g[getattr(g, "is_wet", pd.Series([False]*len(g))).astype(bool)]
            dry = g[~getattr(g, "is_wet", pd.Series([False]*len(g))).astype(bool)]
            def _avg_points(x):
                if x.empty:
                    return np.nan
                cols = [c for c in x.columns if c.startswith("points_")]
                if cols:
                    return float(np.nanmean(x[cols].values))
                return float(np.nanmean(x.get("points", pd.Series([np.nan]*len(x))).values))
            return pd.Series({
                "driver": g["driver"].iloc[-1] if "driver" in g.columns else None,
                "driver_avg_points_in_wet": _avg_points(wet),
                "driver_avg_points_in_dry": _avg_points(dry),
            })
        g = X_all.copy()
        out = (g.groupby("driver", as_index=False).apply(_by_driver).reset_index(drop=True))
        out["driver_rain_dry_delta"] = out["driver_avg_points_in_wet"].fillna(0) - out["driver_avg_points_in_dry"].fillna(0)
        return out
    except Exception:
        return None


def apply_weather_adjustment(preds_df: pd.DataFrame, last_weather_stats: Optional[pd.DataFrame]) -> pd.DataFrame:
    scen = PREDICTION_CONFIG.get("active_scenario", "dry")
    if last_weather_stats is None or scen not in ("wet", "dry"):
        return preds_df
    df = preds_df.copy()
    st = last_weather_stats.copy()
    if "driver" not in st.columns:
        return df
    df = df.merge(st[["driver", "driver_rain_dry_delta"]], on="driver", how="left")
    alpha = 0.15
    if scen == "wet":
        df["predicted_position"] = df["predicted_position"] - alpha * df["driver_rain_dry_delta"].fillna(0)
    else:
        df["predicted_position"] = df["predicted_position"] + alpha * df["driver_rain_dry_delta"].fillna(0)
    df.drop(columns=[c for c in ["driver_rain_dry_delta"] if c in df.columns], inplace=True)
    return df


def _add_weather_perf_features(df: pd.DataFrame) -> pd.DataFrame:
    stats = _compute_weather_performance_stats(df)
    return stats if stats is not None else pd.DataFrame(columns=["driver","driver_avg_points_in_rain","driver_avg_points_in_dry","driver_rain_dry_delta"])

