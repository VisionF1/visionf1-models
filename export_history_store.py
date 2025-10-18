#!/usr/bin/env python3
import argparse, sys, pickle, json, math
from pathlib import Path
import pandas as pd
import numpy as np

CACHE_DIR = Path("app/models_cache")
DATASET_CSV = CACHE_DIR / "dataset_before_training_latest.csv"

OUT_PKL = CACHE_DIR / "history_store.pkl"
OUT_JSON = CACHE_DIR / "history_store_meta.json"

ENCODER_FILES = {
    "driver": CACHE_DIR / "driver_encoder.pkl",
    "team": CACHE_DIR / "team_encoder.pkl",
    "race_name": CACHE_DIR / "race_name_encoder.pkl",
    "circuit_type": CACHE_DIR / "circuit_type_encoder.pkl",
}

REQUIRED_COLS = [
    "driver","team","race_name","year","points",
    "session_rainfall","session_humidity","session_air_temp","session_track_temp"
]

def _safe_read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def _build_race_order(df: pd.DataFrame) -> dict:
    """Mapa: year -> [race_name en orden] usando race_time si existe, sino orden de aparición."""
    race_order = {}
    if "race_time" in df.columns:
        # intentamos parsear fechas
        d = df.copy()
        d["__race_dt"] = pd.to_datetime(d["race_time"], errors="coerce")
        for y, g in d.groupby("year", sort=True):
            gg = g.drop_duplicates(["year","race_name"]).copy()
            if gg["__race_dt"].notna().any():
                gg = gg.sort_values(["__race_dt","race_name"])
            else:
                gg = gg.sort_values(["race_name"])
            race_order[int(y)] = gg["race_name"].tolist()
    else:
        for y, g in df.groupby("year", sort=True):
            gg = g.drop_duplicates(["year","race_name"]).copy()
            race_order[int(y)] = gg["race_name"].tolist()
    return race_order

def _seq_with_index(df: pd.DataFrame, race_order: dict) -> pd.DataFrame:
    """Agrega race_index por (year,race_name) según race_order."""
    idx_map = {
        int(y): {rn: i for i, rn in enumerate(rlist, start=1)}
        for y, rlist in race_order.items()
    }
    def idx_row(r):
        return idx_map.get(int(r["year"]), {}).get(r["race_name"], np.nan)
    df = df.copy()
    df["race_index"] = df.apply(idx_row, axis=1)
    # si faltan, ponemos orden de aparición
    mask = df["race_index"].isna()
    if mask.any():
        df.loc[mask, "race_index"] = (
            df[mask].groupby(["year"]).cumcount() + 1
        )
    df["race_index"] = df["race_index"].astype(int)
    return df

def _load_encoder_classes() -> dict:
    enc = {}
    for name, path in ENCODER_FILES.items():
        if path.exists():
            try:
                obj = pickle.load(open(path, "rb"))
                # soporta LabelEncoder o lista cruda
                if hasattr(obj, "classes_"):
                    enc[name] = list(obj.classes_)
                elif isinstance(obj, (list, tuple, np.ndarray)):
                    enc[name] = list(obj)
            except Exception:
                pass
    return enc

def _build_histories(df: pd.DataFrame):
    # driver_history: lista de eventos con orden cronológico
    driver_hist = {}
    for drv, g in df.sort_values(["year","race_index"]).groupby("driver", sort=False):
        driver_hist[drv] = g[["year","race_name","race_index","team","points","session_rainfall"]].to_dict("records")

    # team_history
    team_hist = {}
    for tm, g in df.sort_values(["year","race_index"]).groupby("team", sort=False):
        team_hist[tm] = g[["year","race_name","race_index","points"]].to_dict("records")

    # promedios rain/dry por driver
    rain_avgs = {}
    for drv, g in df.groupby("driver", sort=False):
        dry = g.loc[g["session_rainfall"].fillna(0) <= 0.0, "points"]
        wet = g.loc[g["session_rainfall"].fillna(0) > 0.0, "points"]
        rain_avgs[drv] = {
            "dry": float(dry.mean()) if len(dry) else 0.0,
            "rain": float(wet.mean()) if len(wet) else 0.0,
        }

    # defaults globales
    defaults = {
        "points_last_3": float(df["points"].rolling(3).mean().dropna().mean()) if len(df) else 0.0,
        "driver_competitiveness": float(df.groupby("driver")["points"].mean().mean()) if len(df) else 0.0,
        "team_competitiveness": float(df.groupby("team")["points"].mean().mean()) if len(df) else 0.0,
    }
    return driver_hist, team_hist, rain_avgs, defaults

def main(argv):
    ap = argparse.ArgumentParser(description="Exporta history_store.pkl para inferencia offline (sin FastF1/Ergast).")
    ap.add_argument("--dataset", default=str(DATASET_CSV), help="CSV histórico (dataset_before_training_latest.csv)")
    ap.add_argument("--out-pkl", default=str(OUT_PKL), help="Salida PKL del history store")
    ap.add_argument("--out-json", default=str(OUT_JSON), help="Salida JSON con metadatos")
    args = ap.parse_args(argv)

    csv_path = Path(args.dataset)
    if not csv_path.exists():
        print(f"❌ No existe {csv_path}.")
        sys.exit(2)

    df = _safe_read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"⚠️  Faltan columnas no críticas en el histórico: {missing} (continuo con lo disponible)")

    # orden de carreras por año e índice
    race_order = _build_race_order(df)
    df = _seq_with_index(df, race_order)

    # encoders (clases)
    encoders = _load_encoder_classes()

    # mapping circuito si lo tuvieras
    circuit_map = {}
    if "circuit_type" in df.columns:
        circuit_map = (
            df.drop_duplicates(["race_name"]) [["race_name","circuit_type"]]
              .set_index("race_name")["circuit_type"].to_dict()
        )

    # histories + defaults
    driver_hist, team_hist, rain_avgs, defaults = _build_histories(df)

    store = {
        "encoders": encoders,                    # classes por categoría (si existen)
        "race_order": race_order,                # year -> [race_name...]
        "driver_history": driver_hist,           # driver -> list of events
        "team_history": team_hist,               # team -> list of events
        "driver_rain_dry_avgs": rain_avgs,       # driver -> {dry, rain}
        "circuit_type_map": circuit_map,         # opcional
        "defaults": defaults,
        "features_final": [
            'driver_encoded','team_encoded','race_name_encoded','year',
            'driver_competitiveness','team_competitiveness','points_last_3',
            'session_air_temp','session_track_temp','session_humidity','session_rainfall',
            'heat_index','weather_difficulty_index','circuit_type_encoded',
            'driver_avg_points_in_rain','driver_avg_points_in_dry','driver_rain_dry_delta'
        ],
    }

    # guardar
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(store, f)
    meta = {
        "rows": int(len(df)),
        "drivers": int(df["driver"].nunique()),
        "teams": int(df["team"].nunique()),
        "years": sorted(map(int, df["year"].dropna().unique())),
        "has_encoders": list(encoders.keys()),
        "has_circuit_map": bool(circuit_map),
    }
    Path(args.out_json).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"history_store.pkl creado → {args.out_pkl}")
    print(f"meta → {args.out_json}")
if __name__ == "__main__":
    main(sys.argv[1:])
