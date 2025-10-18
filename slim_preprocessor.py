#!/usr/bin/env python3
from pathlib import Path
import pickle, math
import pandas as pd
import numpy as np

def _heat_index_celsius(air_c, rh):
    T = air_c * 9/5 + 32.0
    R = rh
    HI = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R
          - 0.00683783*T*T - 0.05481717*R*R + 0.00122874*T*T*R
          + 0.00085282*T*R*R - 0.00000199*T*T*R*R)
    HI = np.where(T < 80, 0.5*(T + 61.0 + ((T-68.0)*1.2) + (R*0.094)), HI)
    return (HI - 32.0) * 5/9

def _weather_difficulty_index(track_c, rh, rain):
    t = np.clip((track_c - 25.0)/25.0, -2, 2)
    h = np.clip(rh/100.0, 0, 1.5)
    r = 1.5 if rain >= 0.5 else 0.0
    return float(0.6*t + 0.4*h + r)

class SlimHistoryStore:
    def __init__(self, path="app/models_cache/history_store.pkl"):
        self.path = Path(path)
        with self.path.open("rb") as f:
            store = pickle.load(f)
        self.encoders = store.get("encoders", {})
        self.race_order = store.get("race_order", {})
        self.driver_history = store.get("driver_history", {})
        self.team_history = store.get("team_history", {})
        self.driver_rain_dry_avgs = store.get("driver_rain_dry_avgs", {})
        self.circuit_type_map = store.get("circuit_type_map", {})
        self.defaults = store.get("defaults", {})
        self.features_final = store.get("features_final", [])

        # Mapeos string->índice tal cual entrenaste; unknown=0
        self._enc_maps = {}
        for name, classes in self.encoders.items():
            # Aseguramos tipo str para las claves
            m = {str(v): i for i, v in enumerate(classes)}
            self._enc_maps[name] = m

    def encode(self, colname, value, unknown_index=0):
        classes = self.encoders.get(colname)
        if not classes:
            return unknown_index
        return self._enc_maps[colname].get(str(value), unknown_index)

    def get_race_index(self, year, race_name):
        order = self.race_order.get(int(year), [])
        try:
            return order.index(race_name) + 1
        except ValueError:
            return len(order) + 1

    def _prior_events(self, seq, year, race_name):
        if not seq:
            return []
        target_idx = self.get_race_index(year, race_name)
        return [x for x in seq if int(x["year"]) < int(year) or
                (int(x["year"]) == int(year) and int(x["race_index"]) < int(target_idx))]

    def _rolling_avg_points(self, seq, window):
        if not seq:
            return None
        pts = [float(x.get("points", 0.0)) for x in seq]
        if len(pts) == 0:
            return None
        return float(np.mean(pts[-window:] if len(pts) >= window else pts))

    def points_last_k(self, drv, year, race_name, k=3):
        seq = self._prior_events(self.driver_history.get(drv, []), year, race_name)
        pts = [float(x.get("points", 0.0)) for x in seq[-k:]]
        if not pts:
            return float(self.defaults.get("points_last_3", 0.0))
        return float(sum(pts))

    def driver_competitiveness(self, drv, year, race_name, window=10):
        seq = self._prior_events(self.driver_history.get(drv, []), year, race_name)
        v = self._rolling_avg_points(seq, window)
        if v is None or math.isnan(v):
            v = float(self.defaults.get("driver_competitiveness", 0.0))
        return float(v)

    def team_competitiveness(self, team, year, race_name, window=10):
        seq = self._prior_events(self.team_history.get(team, []), year, race_name)
        v = self._rolling_avg_points(seq, window)
        if v is None or math.isnan(v):
            v = float(self.defaults.get("team_competitiveness", 0.0))
        return float(v)

    def driver_rain_dry(self, drv):
        d = self.driver_rain_dry_avgs.get(drv, {"dry":0.0,"rain":0.0})
        dry = float(d.get("dry", 0.0))
        wet = float(d.get("rain", 0.0))
        return dry, wet, float(dry - wet)

    def circuit_type_guess(self, race_name):
        return self.circuit_type_map.get(race_name, "street")

class SlimPreprocessor:
    """Reconstruye las 17 features finales a partir de RAW + history_store."""
    def __init__(self, history_store_path="app/models_cache/history_store.pkl",
                 feature_names_path="app/models_cache/feature_names.pkl", quiet=False):
        self.store = SlimHistoryStore(history_store_path)
        self.quiet = quiet
        self.feature_names = self._load_feature_names(feature_names_path)

    def _load_feature_names(self, p):
        p = Path(p)
        if p.exists():
            with p.open("rb") as f:
                n = pickle.load(f)
            return list(n) if isinstance(n, (list,tuple)) else None
        return None

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()

        # saneos mínimos
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2025).astype(int)
        for col in ["session_air_temp","session_track_temp","session_humidity","session_rainfall"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "session_rainfall" in df.columns:
            m = df["session_rainfall"].fillna(0).astype(str).str.lower().str.strip()
            df["session_rainfall"] = m.replace({"true":1,"wet":1,"1":1,"false":0,"dry":0,"0":0}).astype(float)
        if "circuit_type" in df.columns:
            m = df["circuit_type"].astype(str).str.lower().str.strip()
            df["circuit_type"] = m.replace({"urban":"street","city":"street","road":"permanent","classic":"permanent"})
        else:
            df["circuit_type"] = df["race_name"].apply(lambda rn: self.store.circuit_type_guess(str(rn)))

        rows = []
        for _, r in df.iterrows():
            drv = str(r["driver"])
            team = str(r["team"])
            race_name = str(r["race_name"])
            year = int(r["year"])

            de = self.store.encode("driver", drv, 0)
            te = self.store.encode("team", team, 0)
            re = self.store.encode("race_name", race_name, 0)
            ce = self.store.encode("circuit_type", str(r.get("circuit_type","street")), 0)

            pl3 = self.store.points_last_k(drv, year, race_name, k=3)
            dcomp = self.store.driver_competitiveness(drv, year, race_name, window=10)
            tcomp = self.store.team_competitiveness(team, year, race_name, window=10)

            air = float(r.get("session_air_temp", 25.0))
            track = float(r.get("session_track_temp", 35.0))
            hum = float(r.get("session_humidity", 60.0))
            rain = float(r.get("session_rainfall", 0.0))

            hi = float(_heat_index_celsius(air, hum))
            wdi = float(_weather_difficulty_index(track, hum, rain))

            dry, wet, delta = self.store.driver_rain_dry(drv)

            row = {
                'driver_encoded': de,
                'team_encoded': te,
                'race_name_encoded': re,
                'year': year,
                'driver_competitiveness': dcomp,
                'team_competitiveness': tcomp,
                'points_last_3': pl3,
                'session_air_temp': air,
                'session_track_temp': track,
                'session_humidity': hum,
                'session_rainfall': rain,
                'heat_index': hi,
                'weather_difficulty_index': wdi,
                'circuit_type_encoded': ce,
                'driver_avg_points_in_rain': wet,
                'driver_avg_points_in_dry': dry,
                'driver_rain_dry_delta': delta,
            }
            rows.append(row)

        X = pd.DataFrame(rows)

        # alinear al orden entrenado si existe
        if self.feature_names:
            for c in self.feature_names:
                if c not in X.columns:
                    X[c] = 0
            X = X.reindex(columns=self.feature_names, fill_value=0)

        if not self.quiet:
            print(f"✅ SlimPreprocessor → {X.shape[0]} filas × {X.shape[1]} cols")
        return X
