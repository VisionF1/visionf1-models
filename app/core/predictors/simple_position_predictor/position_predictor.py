from __future__ import annotations
import os, pickle, hashlib
import numpy as np
import pandas as pd
from typing import Optional

from app.config import DRIVERS_2025, PREDICTION_CONFIG, PENALTIES
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
from app.core.adapters.progressive_adapter import ProgressiveAdapter

from .manifest import InferenceManifest
from .encoders import EncoderResolver
from .model_selector import ModelSelector
from .features import build_base_df, align_columns, _active_weather, _compute_weather_performance_stats, apply_weather_adjustment
from .constants import INFERENCE_OUT

class SimplePositionPredictor:
    """Orquestador de inferencia (versión modular). Mantiene API del original."""
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.dp = EnhancedDataPreparer(quiet=True)
        self.adapter = ProgressiveAdapter()
        self.manifest = InferenceManifest(quiet=quiet); self.manifest.load()
        self.enc = EncoderResolver(self.manifest)
        self.selector = ModelSelector(quiet=quiet)
        self.model = None
        self.feature_order: list[str] = []
        self._last_weather_stats: Optional[pd.DataFrame] = None

        # Cargar modelo + features entrenadas
        name, model, feats = self.selector.load_best_model()
        if model is not None:
            self.model = getattr(model, "model", model)  # soportar wrappers
        if feats:
            self.feature_order = list(feats)

    # -------------------- helpers de logging --------------------
    def _log(self, msg: str):
        if not self.quiet:
            print(msg)

    # -------------------- helpers diversos --------------------
    def _deterministic_eps(self, driver_code: str) -> float:
        h = int(hashlib.md5(str(driver_code).encode("utf-8")).hexdigest(), 16)
        return ((h % 1000) / 1000.0 - 0.5) * 0.05

    # -------------------- construcción de X con histórico --------------------
    def _load_inference_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        year = int(base_df["year"].iloc[0]); race_name = str(base_df["race_name"].iloc[0])
        # 1) histórico
        hist_df = None
        try:
            if os.path.exists("app/models_cache/cached_data.pkl"):
                hist_df = pickle.load(open("app/models_cache/cached_data.pkl", "rb"))
                if not isinstance(hist_df, pd.DataFrame):
                    hist_df = pd.DataFrame(hist_df)
        except Exception as e:
            self._log(f"⚠️ No se pudo cargar histórico (PKL): {e}")
        if (hist_df is None or hist_df.empty) and os.path.exists("app/models_cache/dataset_before_training_latest.csv"):
            try:
                hist_df = pd.read_csv("app/models_cache/dataset_before_training_latest.csv")
            except Exception as e:
                self._log(f"⚠️ No se pudo cargar histórico (CSV): {e}")
        if hist_df is None:
            hist_df = pd.DataFrame([])

        # 2) FE del histórico (usa el mismo pipeline de training en modo inference)
        X_all, _, _, _ = self.dp.prepare_enhanced_features(hist_df.copy(), inference=True)
        X_all = pd.DataFrame(X_all) if not isinstance(X_all, pd.DataFrame) else X_all

        # 3) stats climáticos para ajuste post-modelo
        try:
            self._last_weather_stats = _compute_weather_performance_stats(X_all)
        except Exception:
            self._last_weather_stats = None

        # 4) Seleccionar fila “actual” por driver y rellenar faltantes con base_df
        cur = pd.DataFrame([])
        if "driver" in X_all.columns:
            last_per_driver = (X_all.reset_index(drop=True)
                                .drop_duplicates(subset=["driver"], keep="last"))
            cur_try = last_per_driver.copy()
            base_set = set(base_df.index)
            cur_try = cur_try[cur_try["driver"].isin(base_set)].copy()
            # completar faltantes con base_df sólo para columnas base
            if len(cur_try) < len(base_df):
                missing = [d for d in base_df.index if d not in set(cur_try["driver"]) ]
                fill = base_df.loc[missing].copy(); fill["driver"] = fill.index
                cur_try = pd.concat([cur_try, fill], axis=0, ignore_index=True)
            # Index por driver
            cur = cur_try.set_index("driver", drop=True)

        # Fallback si quedó desalineado
        if cur.shape[0] != base_df.shape[0]:
            cur = cur.reindex(base_df.index)
        # Alinear columnas al orden del entrenamiento
        cur = align_columns(cur, self.feature_order)
        return cur

    # -------------------- API pública --------------------
    def predict_positions_2025(self) -> pd.DataFrame:
        if self.model is None and self.feature_order:
            self._log("⚠️ No hay modelo entrenado — usaré predicción determinista de respaldo")
        # 1) base de 20 pilotos
        base_df = build_base_df()
        # 2) X con histórico
        X = self._load_inference_features(base_df)

        # 2.5) Guardado de dataset de inferencia (debug)
        X_with_meta = X.copy()
        if "driver" not in X_with_meta.columns:
            X_with_meta.insert(0, "driver", X.index)
        if "team" not in X_with_meta.columns and "team" in base_df.columns:
            X_with_meta.insert(1, "team", base_df.loc[X.index, "team"].values)
        stats = getattr(self, "_last_weather_stats", None)
        try:
            if stats is not None and not stats.empty:
                X_with_meta = X_with_meta.merge(stats, on="driver", how="left")
            X_with_meta.to_csv(INFERENCE_OUT, index=False)
            self._log(f"💾 Dataset de inferencia guardado: {INFERENCE_OUT} (shape={X_with_meta.shape})")
        except Exception as e:
            self._log(f"⚠️ No se pudo guardar dataset de inferencia: {e}")

        # 3) Predecir
        if X.empty or self.model is None:
            idx = base_df.index
            y_hat = np.array([10.0 + self._deterministic_eps(d) for d in idx])
        else:
            idx = X.index
            y_hat = self.model.predict(X.values).astype(float)
        out = pd.DataFrame({
            "driver": idx,
            "team": base_df.loc[idx, "team"].values,
            "rookie": base_df.loc[idx, "rookie"].values,
            "team_change": base_df.loc[idx, "team_change"].values,
            "predicted_position": y_hat.astype(float),
        })
        # jit tiny noise para estabilidad determinista por piloto
        out["predicted_position"] = out.apply(
            lambda r: float(r["predicted_position"]) + self._deterministic_eps(r["driver"]), axis=1
        )
        # 4) Ajuste por clima
        try:
            out = apply_weather_adjustment(out, self._last_weather_stats)
        except Exception as e:
            self._log(f"ℹ️ Ajuste climático omitido: {e}")
        # 4.5) Penalizaciones progresivas (si aplica)
        from app.config import PENALTIES, PREDICTION_CONFIG
        try:
            if PENALTIES.get("use_progressive", False):
                race_no = int(PREDICTION_CONFIG.get("next_race", {}).get("race_number", 1))
                out = self.adapter.apply_progressive_penalties(out, race_no)
        except Exception as e:
            self._log(f"ℹ️ Penalizaciones progresivas omitidas: {e}")

        # 5) Ordenar y enriquecer salida
        out = out.sort_values("predicted_position", ascending=True).reset_index(drop=True)
        out["final_position"] = np.arange(1, len(out) + 1)
        # confianza heurística como en original
        out["confidence"] = out.apply(
            lambda x: max(60, 100 - abs(float(x["predicted_position"]) - float(x["final_position"])) * 10), axis=1
        ).round(1)
        # Tipificado de piloto
        def _type(row):
            if row.get("rookie", False):
                return "🆕 Rookie"
            if row.get("team_change", False):
                return "🔄 Cambio equipo"
            return "👤 Veterano"
        out["driver_type"] = out.apply(_type, axis=1)
        # columnas finales
        cols = ["final_position", "driver", "team", "driver_type", "predicted_position", "confidence"]
        out = out[[c for c in cols if c in out.columns]]
        return out

    # -------------------- Importancias de features (incluido en este archivo) --------------------
    def _compute_permutation_importance_on_preds(self, X: pd.DataFrame, n_repeats: int = 10) -> pd.Series:
        rng = np.random.default_rng(42)
        baseline = self.model.predict(X.values).astype(float)
        cols = list(X.columns)
        changes = np.zeros(len(cols), dtype=float)
        for _ in range(n_repeats):
            for j, col in enumerate(cols):
                Xp = X.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                preds = self.model.predict(Xp.values).astype(float)
                changes[j] += np.mean(np.abs(preds - baseline))
        changes /= float(n_repeats)
        import pandas as pd
        return pd.Series(changes, index=cols)

    def explain_feature_importance(self, top_k: int = 25,
                                   csv_path: str = "app/models_cache/feature_importances.csv",
                                   png_path: str = "app/models_cache/feature_importances.png",
                                   n_repeats: int = 10) -> pd.DataFrame:
        import pandas as pd
        # 1) armar X actual
        base_df = build_base_df(self.dp, self.adapter, 2025,
                                PREDICTION_CONFIG.get("next_race", {}).get("race_name", ""),
                                PREDICTION_CONFIG.get("active_scenario", "dry"),
                                quiet=self.quiet)
        X = self._load_inference_features(base_df)
        X = align_columns(X, self.feature_order)
        # 2) importancias
        if hasattr(self.model, "feature_importances_") and self.model.feature_importances_ is not None:
            importances = pd.Series(self.model.feature_importances_, index=X.columns, dtype=float)
            method = "native"
        else:
            importances = self._compute_permutation_importance_on_preds(X, n_repeats=n_repeats)
            method = f"permutation_preds_{n_repeats}"
        fi = (importances.rename("importance").sort_values(ascending=False)
                                .reset_index().rename(columns={"index": "feature"}))
        try:
            fi.to_csv(csv_path, index=False)
            # gráfico opcional
            import matplotlib.pyplot as plt
            top = fi.head(top_k)
            plt.figure()
            plt.barh(top["feature"][::-1], top["importance"].iloc[:top_k][::-1])
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

    # -------------------- Pretty print realistic --------------------
    def show_realistic_predictions(self, predictions_df: pd.DataFrame) -> None:
        current_race_name = PREDICTION_CONFIG.get("next_race", {}).get("race_name", "Carrera Desconocida")
        print(f"{'='*100}")
        print(f"🏆 PREDICCIONES 2025 - {current_race_name}")
        print(f"{'='*100}")
        print(f"{'Pos':<4} {'Piloto':<6} {'Equipo':<16} {'Tipo':<20} {'Pred':<8} {'Conf.':<6}")
        print("-" * 100)
        for _, row in predictions_df.iterrows():
            print(
                f"P{int(row['final_position']):<3} {row['driver']:<6} {str(row['team'])[:16]:<16} {row['driver_type']:<20} "
                f"{float(row['predicted_position']):<8.3f} {float(row['confidence']):<6.1f}"
            )