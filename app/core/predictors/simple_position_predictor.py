import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd

from app.config import DRIVERS_2025, PREDICTION_CONFIG, PENALTIES
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
from app.core.adapters.progressive_adapter import ProgressiveAdapter


MODEL_FILES = {
    "RandomForest": "app/models_cache/randomforest_model.pkl",
    "XGBoost": "app/models_cache/xgboost_model.pkl",
    "GradientBoosting": "app/models_cache/gradientboosting_model.pkl",
}

AFTER_FE_PATH = "app/models_cache/dataset_after_feature_engineering_latest.csv"
BEFORE_FE_PATH = "app/models_cache/dataset_before_training_latest.csv"
CACHED_DATA_PKL = "app/models_cache/cached_data.pkl"
FEATURE_NAMES_PKL = "app/models_cache/feature_names.pkl"
INFERENCE_OUT = "app/models_cache/inference_dataset_latest.csv"

MANIFEST_PATHS = [
    "app/models_cache/inference_manifest.json",
    "/mnt/data/inference_manifest.json",
]


class SimplePositionPredictor:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.dp = EnhancedDataPreparer(quiet=True)
        self.adapter = ProgressiveAdapter()

        # cargar manifiesto si existe
        self.manifest = self._load_inference_manifest()

        # elegir modelo (prioriza el indicado en manifiesto)
        forced_model_name = None
        if self.manifest and isinstance(self.manifest.get("best_model_name"), str):
            forced_model_name = self.manifest["best_model_name"]
            self._log(f"🧭 Manifiesto sugiere modelo: {forced_model_name}")

        if forced_model_name:
            self.model = self._load_model(forced_model_name) or self.load_best_model() or self._load_model(None)
        else:
            self.model = self.load_best_model() or self._load_model(None)

        # orden de features: manifiesto > pkl legacy
        self.feature_order = self._feature_names_from_manifest() or self._load_trained_feature_names()

    # -------------------- utils --------------------
    def _log(self, msg: str):
        if not self.quiet:
            print(msg)


    def _feature_names_from_manifest(self) -> list[str]:
        if not self.manifest:
            return []
        names = self.manifest.get("feature_names") or []
        if isinstance(names, (list, tuple)):
            return [str(x) for x in names]
        return []

   # --- helpers de manifest/encoders ---

    def _load_inference_manifest(self):
        try:
            with open("app/models_cache/inference_manifest.json", "r", encoding="utf-8") as f:
                m = json.load(f)
            return m
        except Exception:
            return None

    def _manifest_dir(self) -> str:
        return os.path.dirname("app/models_cache/inference_manifest.json")

    def _get_encoder_spec(self, key: str):
        if not getattr(self, "manifest", None):
            return None
        enc = self.manifest.get("encoders", {})
        return enc.get(key)

    def _resolve_encoder(self, key: str):
        """Devuelve un encoder listo para usar (dict/list/LabelEncoder/OrdinalEncoder o cargado desde .pkl)."""
        if not hasattr(self, "_enc_cache"):
            self._enc_cache = {}

        if key in self._enc_cache:
            return self._enc_cache[key]

        spec = self._get_encoder_spec(key)
        enc_obj = None

        if isinstance(spec, (dict, list, tuple)):
            enc_obj = spec
        elif isinstance(spec, str) and spec.endswith(".pkl"):
            path = spec
            if not os.path.exists(path):
                # probar relativo al manifiesto
                cand = os.path.join(self._manifest_dir(), os.path.basename(spec))
                if os.path.exists(cand):
                    path = cand
            try:
                with open(path, "rb") as f:
                    enc_obj = pickle.load(f)
                self._log(f"📦 Encoder '{key}' cargado (tipo={type(enc_obj).__name__})")
            except Exception as e:
                self._log(f"❌ No pude cargar encoder {key} desde {spec}: {e}")
                enc_obj = None
        else:
            enc_obj = spec  # None u otro

        self._enc_cache[key] = enc_obj
        return enc_obj

    def _encode_value(self, key: str, raw_value):
        """Codifica raw_value con el encoder (maneja LabelEncoder/OrdinalEncoder/dict/list)."""
        enc = self._resolve_encoder(key)
        if enc is None:
            return None

        import re, unicodedata
        def strip_accents(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
        def norm_up(s: str) -> str:
            return strip_accents(str(s).strip().upper())
        def norm_lo(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", strip_accents(str(s).strip().lower()))

        # dict
        if isinstance(enc, dict):
            if raw_value in enc: return enc[raw_value]
            if str(raw_value) in enc: return enc[str(raw_value)]
            # normalizado
            nmap = {norm_lo(k): v for k, v in enc.items()}
            return nmap.get(norm_lo(raw_value))

        # list/tuple
        if isinstance(enc, (list, tuple)):
            vals = list(enc)
            for candidate in (raw_value, norm_up(raw_value)):
                if candidate in vals: return vals.index(candidate)
                if str(candidate) in vals: return vals.index(str(candidate))
            nvals = [norm_lo(v) for v in vals]
            nl = norm_lo(raw_value)
            return nvals.index(nl) if nl in nvals else None

        # LabelEncoder
        classes = getattr(enc, "classes_", None)
        if classes is not None:
            vals = [norm_up(v) for v in list(classes)]  # p.ej. ['ALB','ALO',...,'VER']
            target = norm_up(raw_value)                 # asegurar 'VER','LEC',...
            if target in vals:
                return vals.index(target)
            return None

        # OrdinalEncoder
        cats = getattr(enc, "categories_", None)
        if cats is not None:
            arr = cats[0] if len(cats) == 1 else cats[0]
            vals = [norm_up(v) for v in list(arr)]
            target = norm_up(raw_value)
            if target in vals:
                return vals.index(target)
            nvals = [norm_lo(v) for v in list(arr)]
            nl = norm_lo(raw_value)
            return nvals.index(nl) if nl in nvals else None

        return None

    def _decode_series(self, key: str, code_series: pd.Series) -> pd.Series:
        """Decodifica códigos a raw con el encoder."""
        enc = self._resolve_encoder(key)
        if enc is None:
            return pd.Series([None] * len(code_series), index=code_series.index)

        # dict invertido
        if isinstance(enc, dict):
            inv = {}
            for k, v in enc.items():
                inv[v] = k
                inv[str(v)] = k
            return code_series.map(lambda x: inv.get(x, inv.get(str(x))))

        # list/tuple
        if isinstance(enc, (list, tuple)):
            vals = list(enc)
            def _idx(x):
                try:
                    i = int(x);  return vals[i] if 0 <= i < len(vals) else None
                except Exception:
                    return None
            return code_series.map(_idx)

        # LabelEncoder
        classes = getattr(enc, "classes_", None)
        if classes is not None:
            vals = [str(x) for x in list(classes)]
            def _le(x):
                try:
                    i = int(x);  return vals[i] if 0 <= i < len(vals) else None
                except Exception:
                    return None
            return code_series.map(_le)

        # OrdinalEncoder
        cats = getattr(enc, "categories_", None)
        if cats is not None:
            arr = cats[0] if len(cats) == 1 else cats[0]
            vals = [str(x) for x in list(arr)]
            def _oe(x):
                try:
                    i = int(x);  return vals[i] if 0 <= i < len(vals) else None
                except Exception:
                    return None
            return code_series.map(_oe)

        return pd.Series([None] * len(code_series), index=code_series.index)

    def _load_trained_feature_names(self):
        try:
            if os.path.exists(FEATURE_NAMES_PKL):
                with open(FEATURE_NAMES_PKL, "rb") as f:
                    names = pickle.load(f)
                if isinstance(names, (list, tuple)):
                    return [str(x) for x in names]
                if hasattr(names, "tolist"):
                    return [str(x) for x in names.tolist()]
                return [str(names)]
        except Exception as e:
            self._log(f"⚠️ No se pudieron cargar feature_names entrenadas: {e}")
        return []

    def _load_model(self, name: str | None):
        candidates = []
        if name and name in MODEL_FILES:
            candidates.append((name, MODEL_FILES[name]))
        for n, p in MODEL_FILES.items():
            if not any(p == c[1] for c in candidates):
                candidates.append((n, p))
        for n, path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        model = pickle.load(f)
                    print(f"✅ Modelo cargado: {n}")
                    return model
                except Exception as e:
                    print(f"⚠️  Error cargando {n}: {e}")
        print("❌ No se encontró ningún modelo entrenado.")
        return None

    # -------------------- selección por métricas --------------------
    def load_best_model(self):
        metrics = self._load_training_metrics("app/models_cache/training_results.pkl")
        if not metrics:
            return None
        best_name = self._select_best_model_from_metrics(metrics)
        if not best_name:
            return None
        return self._load_model(best_name)

    def _load_training_metrics(self, path: str):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            self._log(f"⚠️  No se pudieron leer métricas: {e}")
        return None

    def _select_best_model_from_metrics(self, metrics_dict: dict) -> str | None:
        candidates = []
        for name, m in metrics_dict.items():
            if not isinstance(m, dict) or ("error" in m):
                continue
            cv = m.get("cv_mse_mean", float("inf"))
            over = m.get("overfitting_score", 1.0)
            r2 = m.get("test_r2", 0.0)
            comp = (cv / 100.0) + max(0.0, over - 1.0) * 2.0 + (1.0 - max(0.0, min(1.0, r2))) * 10.0
            candidates.append((comp, name))
        if not candidates:
            return None
        candidates.sort()
        best = candidates[0][1]
        print(f"🎯 Mejor modelo por métricas: {best}")
        return best

    def _active_weather(self) -> dict:
        scen_key = PREDICTION_CONFIG.get("active_scenario", "dry")
        return PREDICTION_CONFIG.get("weather_scenarios", {}).get(scen_key, {})

    def _build_base_df(self) -> pd.DataFrame:
        race = PREDICTION_CONFIG["next_race"]
        w = self._active_weather()
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

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        order = self.feature_order or []
        if not order:
            return X
        X = X.copy()
        for col in order:
            if col not in X.columns:
                X[col] = 0.0
        X = X[order]
        return X

    @staticmethod
    def _deterministic_eps(driver_code: str) -> float:
        h = hashlib.sha1(driver_code.encode("utf-8")).hexdigest()
        return (int(h[:6], 16) % 1000) * 1e-6

    # -------------------- carga de dataset para inferencia --------------------
    # --- núcleo: history-augmented + híbrido + dedupe seguro ---

    def _load_inference_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara X de 20 filas (los pilotos de base_df) usando TODO el histórico para cálculos de FE.
        Extrae las filas “actuales” por (year, race_name_encoded, driver_encoded) o
        por última aparición del driver, completando faltantes con SOLO base_df.
        """
        year = int(base_df["year"].iloc[0])
        race_name = str(base_df["race_name"].iloc[0])

        # 1) cargar histórico
        hist_df = None
        if os.path.exists(CACHED_DATA_PKL):
            try:
                hist_df = pickle.load(open(CACHED_DATA_PKL, "rb"))
                if not isinstance(hist_df, pd.DataFrame):
                    hist_df = pd.DataFrame(hist_df)
            except Exception as e:
                self._log(f"⚠️ No se pudo cargar histórico (PKL): {e}")

        if (hist_df is None or hist_df.empty) and os.path.exists(BEFORE_FE_PATH):
            try:
                hist_df = pd.read_csv(BEFORE_FE_PATH)
            except Exception as e:
                self._log(f"⚠️ No se pudo leer histórico (CSV): {e}")

        # sin histórico → solo base_df
        if hist_df is None or hist_df.empty:
            self._log("ℹ️ Sin histórico: genero features SOLO con base_df")
            X_only, _, _, _ = self.dp.prepare_enhanced_features(base_df.copy(), inference=True)
            X_only = pd.DataFrame(X_only) if not isinstance(X_only, pd.DataFrame) else X_only
            if X_only.shape[0] == 0:
                raise ValueError("prepare_enhanced_features(base_df) devolvió 0 filas.")
            if X_only.shape[0] != base_df.shape[0]:
                self._log(f"⚠️ Cardinalidad inesperada: X_only={X_only.shape[0]} vs base={base_df.shape[0]}. Ajusto.")
                X_only = X_only.iloc[: base_df.shape[0]].copy()
            X_only.index = base_df.index
            return self._align_columns(X_only)

        # 2) normalizar columnas del histórico
        for c in ["driver", "team", "year", "race_name", "points"]:
            if c not in hist_df.columns:
                hist_df[c] = np.nan

        # 3) combo histórico + actuales
        base_marked = base_df[["driver", "team", "year", "race_name"]].copy()
        base_marked["points"] = np.nan
        base_marked["is_current"] = 1
        hist_marked = hist_df[["driver", "team", "year", "race_name", "points"]].copy()
        hist_marked["is_current"] = 0
        combo_df = pd.concat([hist_marked, base_marked], ignore_index=True)

        # 4) FE sobre combo
        X_all, _, _, _ = self.dp.prepare_enhanced_features(combo_df.copy(), inference=True)
        X_all = pd.DataFrame(X_all) if not isinstance(X_all, pd.DataFrame) else X_all

        # ===== extracción de 20 filas =====
        cur = pd.DataFrame()
        used_hybrid = False

        # ayudar con encodings
        race_code = None
        if getattr(self, "manifest", None):
            try:
                race_code = self._encode_value("race_name", race_name)
            except Exception:
                race_code = None

        has_year = "year" in X_all.columns
        has_rne  = "race_name_encoded" in X_all.columns
        has_dre  = "driver_encoded" in X_all.columns

        def _base_driver_codes():
            codes = []
            for d in base_df.index.tolist():  # 'VER', 'LEC', ...
                enc_val = self._encode_value("driver", d)  # con LabelEncoder UPPER
                if enc_val is not None:
                    codes.append(enc_val)
            return set(codes)

        # Intento 1: year & race_name_encoded & driver_encoded
        if (has_year and has_rne and has_dre) and (race_code is not None):
            base_codes = _base_driver_codes()
            self._log(f"🧭 base_driver_codes detectados: {len(base_codes)}")
            cur_try = X_all[(X_all["year"] == year) & (X_all["race_name_encoded"] == race_code)].copy()
            self._log(f"🔎 (ideal) filtro year & race_name_encoded -> filas={len(cur_try)}")
            if base_codes:
                cur_try = cur_try[cur_try["driver_encoded"].isin(base_codes)].copy()
                self._log(f"🔎 + driver_encoded in base_set -> filas={len(cur_try)}")
            if len(cur_try) > len(base_df):
                cur_try = (cur_try.reset_index(drop=True)
                                .drop_duplicates(subset=["driver_encoded"], keep="last"))
                self._log(f"🔎 depurado a última por driver_encoded -> filas={len(cur_try)}")
            if len(cur_try) == len(base_df):
                cur = cur_try.copy()

        # Intento 2: última por driver_encoded (y filtrar a base_set).
        if cur.empty and has_dre:
            base_codes = _base_driver_codes()
            self._log(f"🧭 base_driver_codes detectados: {len(base_codes)}")
            last_per_driver = (X_all.reset_index(drop=True)
                                .drop_duplicates(subset=["driver_encoded"], keep="last"))
            self._log(f"🔎 última por driver_encoded -> únicos={len(last_per_driver)}")

            cur_try = last_per_driver.copy()
            if base_codes:
                cur_try = cur_try[cur_try["driver_encoded"].isin(base_codes)].copy()
                self._log(f"🔎 + filtro base_set -> filas={len(cur_try)}")

            # Fallback si base_codes está vacío: usar decode y filtrar por abreviaturas de base_df
            if (len(cur_try) == 0) and ("driver_encoded" in last_per_driver.columns):
                try:
                    dec = self._decode_series("driver", last_per_driver["driver_encoded"])
                    last_per_driver = last_per_driver.copy()
                    last_per_driver["driver"] = dec
                    # quedarnos con los 20 de base_df por abreviatura directa
                    cur_try = last_per_driver[last_per_driver["driver"].isin(base_df.index)].copy()
                    self._log(f"🔎 fallback decode driver → filtro por base_df -> filas={len(cur_try)}")
                except Exception as e:
                    self._log(f"ℹ️ No pude decodificar driver_encoded para fallback: {e}")

            # Completar faltantes con SOLO base_df
            missing = []
            have_set = set()
            if "driver" in cur_try.columns:
                have_set = set(cur_try["driver"].dropna().tolist())
            elif "driver_encoded" in cur_try.columns:
                # mapear a abreviatura para saber si está
                try:
                    dec2 = self._decode_series("driver", cur_try["driver_encoded"])
                    have_set = set(dec2.dropna().tolist())
                    cur_try = cur_try.copy()
                    cur_try.insert(0, "driver", dec2)
                except Exception:
                    have_set = set()

            for d in base_df.index.tolist():
                if d not in have_set:
                    missing.append(d)

            if missing:
                self._log(f"🧩 Híbrido: faltan {len(missing)} pilotos → completo con SOLO base_df")
                X_only, _, _, _ = self.dp.prepare_enhanced_features(base_df.loc[missing].copy(), inference=True)
                X_only = pd.DataFrame(X_only) if not isinstance(X_only, pd.DataFrame) else X_only
                if "driver" not in X_only.columns:
                    X_only.insert(0, "driver", missing)
                cur = pd.concat([cur_try, X_only], ignore_index=True)
                used_hybrid = True
            else:
                cur = cur_try.copy()

        # Intento 3: is_current
        if cur.empty and "is_current" in X_all.columns:
            cur_try = X_all[X_all["is_current"] == 1].copy()
            self._log(f"🔎 por is_current -> filas={len(cur_try)}")
            if len(cur_try) == len(base_df):
                cur = cur_try.copy()

        # Fallback final
        if cur.empty:
            self._log("⚠️ No pude aislar actuales; uso SOLO base_df")
            X_only, _, _, _ = self.dp.prepare_enhanced_features(base_df.copy(), inference=True)
            cur = pd.DataFrame(X_only) if not isinstance(X_only, pd.DataFrame) else X_only
            if "driver" not in cur.columns:
                cur.insert(0, "driver", base_df.index.tolist())

        # ===== índice por driver + dedupe seguro =====
        if "driver" not in cur.columns and "driver_encoded" in cur.columns:
            try:
                dec = self._decode_series("driver", cur["driver_encoded"])
                cur.insert(0, "driver", dec)
            except Exception:
                cur.insert(0, "driver", [None]*len(cur))

        # Si existen duplicados de driver, quedarnos con la última
        if "driver" in cur.columns:
            # dedupe por driver ANTES de set_index
            cur = cur.copy()
            cur = cur.dropna(subset=["driver"])
            cur = cur[~cur["driver"].duplicated(keep="last")]

            cur = cur.set_index("driver", drop=True)

            # por las dudas, si aún quedaron duplicados en el índice
            if cur.index.has_duplicates:
                cur = cur[~cur.index.duplicated(keep="last")]

        else:
            # como último recurso, forzar índice con base_df
            if cur.shape[0] != base_df.shape[0]:
                cur = cur.iloc[: base_df.shape[0]].copy()
            cur.index = base_df.index

        # normalizar cardinalidad y orden según base_df (ahora sí sin duplicados)
        if cur.shape[0] != base_df.shape[0] or not cur.index.equals(base_df.index):
            self._log(f"🔧 Normalizo cardinalidad/orden (cur={cur.shape[0]} vs base={base_df.shape[0]})")
            # reindex seguro: como ya deduplicamos, no explota
            cur = cur.reindex(base_df.index)

        # alinear columnas al orden del modelo
        cur = self._align_columns(cur)
        self._log(f"📦 FE history-augmented{' (híbrido)' if used_hybrid else ''} → filas={len(cur)}, cols={cur.shape[1]}")
        return cur



    # -------------------- API principal --------------------
    def predict_positions_2025(self) -> pd.DataFrame:
        print("🎯 Prediciendo posiciones (pre-race)")

        base_df = self._build_base_df()
        X = self._load_inference_features(base_df)

        # Guardar dataset de inferencia con metadatos
        try:
            X_with_meta = X.copy()
            if "team" not in X_with_meta.columns:
                X_with_meta.insert(0, "team", base_df.loc[X.index, "team"].values)
            if "driver" not in X_with_meta.columns:
                X_with_meta.insert(0, "driver", X.index)
            X_with_meta.to_csv(INFERENCE_OUT, index=False)
            print(f"💾 Dataset de inferencia guardado: {INFERENCE_OUT} (shape={X_with_meta.shape})")
        except Exception as e:
            print(f"⚠️ No se pudo guardar dataset de inferencia: {e}")

        # Predicción
        if X is None or X.shape[0] == 0:
            self._log("⚠️ X vacío: uso predicción determinista como respaldo")
            drivers = base_df.index.tolist()
            y_hat = np.array([10.0 + self._deterministic_eps(d) for d in drivers])
            idx = drivers
        else:
            idx = X.index
            if self.model is None:
                y_hat = np.array([10.0 + self._deterministic_eps(d) for d in idx])
            else:
                y_hat = self.model.predict(X.values)

        out = pd.DataFrame({
            "driver": idx,
            "team": base_df.loc[idx, "team"].values,
            "rookie": base_df.loc[idx, "rookie"].values,
            "team_change": base_df.loc[idx, "team_change"].values,
            "predicted_position": y_hat.astype(float),
        })

        out["predicted_position"] = out.apply(
            lambda r: float(r["predicted_position"]) + self._deterministic_eps(r["driver"]), axis=1
        )

        # Adaptaciones progresivas (si están activadas)
        if PENALTIES.get("use_progressive", False):
            race_no = int(PREDICTION_CONFIG["next_race"].get("race_number", 1))
            out = self.adapter.apply_progressive_penalties(out, race_no)

        # Ranking final
        out = out.sort_values("predicted_position", ascending=True).reset_index(drop=True)
        out["final_position"] = np.arange(1, len(out) + 1)

        # Heurística de confianza
        out["confidence"] = out.apply(
            lambda x: max(60, 100 - abs(x["predicted_position"] - x["final_position"]) * 10), axis=1
        ).round(1)

        # Tipificado de piloto
        def _type(row):
            if row.get("rookie", False):
                return "🆕 Rookie"
            if row.get("team_change", False):
                return "🔄 Cambio equipo"
            return "👤 Veterano"

        out["driver_type"] = out.apply(_type, axis=1)

        return out[["final_position", "driver", "team", "driver_type", "predicted_position", "confidence"]]

    # -------------------- retrocompatibilidad --------------------
    def show_realistic_predictions(self, predictions_df: pd.DataFrame) -> None:
        current_race_name = PREDICTION_CONFIG["next_race"].get("race_name", "Carrera Desconocida")
        print(f"\n{'='*100}")
        print(f"🏆 PREDICCIONES 2025 - {current_race_name}")
        print(f"{'='*100}")
        print(f"{'Pos':<4} {'Piloto':<6} {'Equipo':<16} {'Tipo':<20} {'Pred':<8} {'Conf.':<6}")
        print("-" * 100)

        for _, row in predictions_df.iterrows():
            print(
                f"P{int(row['final_position']):<3} {row['driver']:<6} {str(row['team'])[:16]:<16} {row['driver_type']:<20} "
                f"{float(row['predicted_position']):<8.3f} {float(row['confidence']):<6.1f}"
            )



    def _compute_permutation_importance_on_preds(self, X: pd.DataFrame, n_repeats: int = 10, random_state: int = 42) -> pd.Series:
        """
        Importancia por permutación SIN y_true:
        mide el cambio medio absoluto en la predicción al permutar cada feature.
        Compatible con cualquier modelo ya entrenado.
        """
        rng = np.random.RandomState(random_state)
        X_aligned = self._align_columns(X)
        cols = list(X_aligned.columns)
        baseline = self.model.predict(X_aligned.values).astype(float)

        changes = np.zeros(len(cols), dtype=float)
        for rep in range(n_repeats):
            for j, col in enumerate(cols):
                Xp = X_aligned.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                preds = self.model.predict(Xp.values).astype(float)
                changes[j] += np.mean(np.abs(preds - baseline))
        changes /= float(n_repeats)
        return pd.Series(changes, index=cols)

    def explain_feature_importance(self, top_k: int = 25,
                                csv_path: str = "app/models_cache/feature_importances.csv",
                                png_path: str = "app/models_cache/feature_importances.png",
                                n_repeats: int = 10) -> pd.DataFrame:
        """
        1) Reconstruye X (history-augmented) para la próxima carrera.
        2) Calcula importancias:
        - Si el modelo tiene .feature_importances_ -> nativo.
        - Si no, usa permutación sobre predicciones (no necesita y_true).
        3) Guarda CSV + un gráfico PNG y devuelve el DataFrame ordenado.
        """
        # 1) armar X de la próxima carrera
        base_df = self._build_base_df()
        X = self._load_inference_features(base_df)
        X = self._align_columns(X)

        # 2) importancias
        if hasattr(self.model, "feature_importances_") and self.model.feature_importances_ is not None:
            importances = pd.Series(self.model.feature_importances_, index=X.columns, dtype=float)
            method = "native"
        else:
            importances = self._compute_permutation_importance_on_preds(X, n_repeats=n_repeats)
            method = f"permutation_preds_{n_repeats}"

        # 3) ordenar y normalizar a %
        fi = (importances
            .sort_values(ascending=False)
            .to_frame(name="importance"))
        fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100.0).round(2)

        # 4) guardar CSV
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fi.reset_index(names="feature").to_csv(csv_path, index=False)

        # 5) gráfico simple (top_k)
        try:
            import matplotlib.pyplot as plt
            top = fi.head(top_k)
            plt.figure()
            plt.barh(top.index[::-1], top["importance"].iloc[:top_k][::-1])
            plt.title(f"Feature Importance ({method}) — top {top_k}")
            plt.xlabel("importance")
            plt.ylabel("feature")
            plt.tight_layout()
            plt.savefig(png_path, dpi=140)
        except Exception as e:
            self._log(f"⚠️ No pude generar gráfico de importancias: {e}")

        self._log(f"💾 Feature importances guardadas: {csv_path}")
        self._log(f"🖼️ Gráfico guardado: {png_path}")
        return fi
