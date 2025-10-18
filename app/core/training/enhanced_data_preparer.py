import warnings
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import pickle

from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

warnings.filterwarnings('ignore')


class InferencePreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesador sklearn-compatible y pickeable que usa EnhancedDataPreparer
    para generar **features pre-race** a partir de columnas RAW.
    - fit: no aprende nada (los encoders ya fueron persistidos durante el train)
    - transform: aplica `prepare_enhanced_features(inference=True)` y devuelve X numérico
    """
    def __init__(self, quiet: bool = False, de_emphasize_team: bool = True, team_deemphasis_factor: float = 0.4):
        self.quiet = quiet
        self.de_emphasize_team = de_emphasize_team
        self.team_deemphasis_factor = max(0.0, min(1.0, team_deemphasis_factor))
        self._inner: Optional[EnhancedDataPreparer] = None  # se inicializa perezoso
        self.feature_names_out_: Optional[List[str]] = None

    def _get_inner(self) -> "EnhancedDataPreparer":
        if self._inner is None:
            self._inner = EnhancedDataPreparer(
                quiet=self.quiet,
                de_emphasize_team=self.de_emphasize_team,
                team_deemphasis_factor=self.team_deemphasis_factor,
            )
        return self._inner

    def fit(self, X, y=None):
        # No necesitamos ajustar nada para inferencia (los encoders están guardados)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        prep = self._get_inner()
        Xnum, _, _, feature_names = prep.prepare_enhanced_features(X, inference=True)

        # -------- ALINEACIÓN CRÍTICA POR feature_names.pkl --------
        try:
            fn_path = Path("app/models_cache/feature_names.pkl")
            if fn_path.exists():
                with fn_path.open("rb") as f:
                    trained = pickle.load(f)
                if isinstance(trained, (list, tuple)):
                    trained = list(trained)
                    # crear faltantes y reordenar
                    for c in trained:
                        if c not in Xnum.columns:
                            Xnum[c] = 0
                    Xnum = Xnum.reindex(columns=trained, fill_value=0)
                    if not self.quiet:
                        print(f"✅ Alineado al orden de entrenamiento ({len(trained)} features)")
            else:
                if not self.quiet:
                    print("⚠️  No encontré feature_names.pkl; se usa el orden generado por FE.")
        except Exception as e:
            if not self.quiet:
                print(f"⚠️  No se pudo alinear por feature_names.pkl: {e}")

        self.feature_names_out_ = list(Xnum.columns)
        return Xnum

    # compat para sklearn >=1.0 (introspección)
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_ or [], dtype=object)


class EnhancedDataPreparer:
    def __init__(self, quiet: bool = False, de_emphasize_team: bool = True, team_deemphasis_factor: float = 0.4):
        self.quiet = quiet
        self.feature_engineer = AdvancedFeatureEngineer(quiet=self.quiet)
        self.label_encoder = LabelEncoder()  # (legacy, no lo usamos directamente)
        self.feature_names: List[str] = []
        self.de_emphasize_team = de_emphasize_team
        self.team_deemphasis_factor = max(0.0, min(1.0, team_deemphasis_factor))

        # Exponer un preprocesador para pipeline RAW (lo construimos bajo demanda)
        self.preprocessor: Optional[InferencePreprocessor] = None

    # ---------- utils ----------
    def _log(self, msg: str):
        if not self.quiet:
            print(msg)

    def _load_encoder(self, col: str) -> Optional[LabelEncoder]:
        p = Path(f"app/models_cache/{col}_encoder.pkl")
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def _save_encoder(self, col: str, le: LabelEncoder):
        p = Path(f"app/models_cache/{col}_encoder.pkl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(le, f)

    def _transform_with_encoder(self, le: LabelEncoder, series: pd.Series, unknown_index: int = 0) -> np.ndarray:
        """Transforma usando un LabelEncoder sin modificar sus clases.
        Cualquier valor desconocido se mapea a unknown_index (por defecto 0)."""
        vals = series.astype(str).fillna("")
        classes = list(le.classes_)
        mapping = {c: i for i, c in enumerate(classes)}
        if not (0 <= unknown_index < len(classes)):
            unknown_index = 0
        return np.array([mapping.get(v, unknown_index) for v in vals], dtype=int)

    def _fit_or_transform_categorical(self, df: pd.DataFrame, col: str, inference: bool) -> pd.DataFrame:
        if col not in df.columns:
            return df

        df = df.dropna(subset=[col]).copy()
        df[col] = df[col].astype(str)

        if inference:
            le = self._load_encoder(col)
            if le is None:
                # >>>>>>>>>>> LOG CRÍTICO: estamos cayendo a fallback
                if not self.quiet:
                    print(f"⚠️  Fallback encoder para '{col}': NO se encontró app/models_cache/{col}_encoder.pkl")
                le = LabelEncoder().fit(df[col])  # solo para no romper; mapea “a su manera”
            else:
                if not self.quiet:
                    print(f"📦 Encoder '{col}' cargado (clases={len(le.classes_)})")
            df[f"{col}_encoded"] = self._transform_with_encoder(le, df[col], unknown_index=0)
        else:
            # ENTRENAMIENTO: ajustamos y guardamos
            le = self._load_encoder(col)
            if le is None:
                le = LabelEncoder().fit(df[col])
            else:
                prev = list(le.classes_)
                new_vals = sorted(set(df[col].unique()) - set(prev))
                if new_vals:
                    le = LabelEncoder()
                    le.classes_ = np.array(prev + new_vals, dtype=object)
            df[f"{col}_encoded"] = self._transform_with_encoder(le, df[col], unknown_index=0)
            self._save_encoder(col, le)

        return df

    def _process_weather_features(self, df: pd.DataFrame):
        defaults = {
            'session_air_temp': 25.0,
            'session_track_temp': 35.0,
            'session_humidity': 60.0,
            'session_rainfall': 0.0,
        }
        for c, v in defaults.items():
            if c not in df.columns:
                df[c] = v
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(v)

        if 'heat_index' not in df.columns:
            air = df['session_air_temp']
            track = df['session_track_temp']
            hum = df['session_humidity']
            hi = (0.6 * air + 0.4 * track) / 100.0 - (hum - 50.0) * 0.001
            df['heat_index'] = np.clip(hi, 0.0, 1.0)

        if 'weather_difficulty_index' not in df.columns:
            df['weather_difficulty_index'] = (df['session_humidity'] / 100.0) + (df['session_rainfall'] * 3.0)

        return df

    # ---------- main ----------
    def prepare_enhanced_features(self, df: pd.DataFrame, inference: bool = False):
        """Genera el set de features pre-race seguro. En inference reutiliza encoders guardados."""
        df = df.copy()

        # Feature engineering pre-race
        df = self.feature_engineer.create_circuit_compatibility_features(df)
        df = self.feature_engineer.create_momentum_features(df)  # sólo histórico (usa shift(1))
        df = self._process_weather_features(df)
        df = self.feature_engineer.create_weather_performance_features(df)

        # Categóricas (usando encoders persistentes si inference=True)
        df = self._fit_or_transform_categorical(df, "team", inference)
        df = self._fit_or_transform_categorical(df, "race_name", inference)
        df = self._fit_or_transform_categorical(df, "driver", inference)
        df = self._fit_or_transform_categorical(df, "circuit_type", inference)

        # Selección final de features pre-race safe
        requested_features = [
            'driver_encoded', 'team_encoded', 'race_name_encoded', 'year',
            'driver_competitiveness', 'team_competitiveness',
            'driver_skill_factor', 'team_strength_factor', 'driver_team_synergy',
            'driver_weather_skill', 'overtaking_ability',
            'points_last_3',
            'session_air_temp', 'session_track_temp', 'session_humidity', 'session_rainfall',
            'heat_index', 'weather_difficulty_index',
            'circuit_type_encoded', 'driver_avg_points_in_rain',
            'driver_avg_points_in_dry', 'driver_rain_dry_delta'
        ]

        available = [f for f in requested_features if f in df.columns]
        X = df[available].copy()

        # Target si está presente (sólo entreno)
        y = None
        if not inference:
            for col in ['final_position', 'race_position', 'position']:
                if col in df.columns:
                    y = df[col].copy()
                    break

        self.feature_names = list(X.columns)
        return X, y, None, self.feature_names

    def prepare_training_data(self, df: pd.DataFrame):
        X, y, label_encoder, feature_names = self.prepare_enhanced_features(df, inference=False)
        if X is None or y is None:
            return None, None, None, None, None

        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, shuffle=True)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        return X_train, X_test, y_train, y_test, feature_names

    # ---------- Exponer preprocesador RAW para el pipeline ---------
    def get_inference_preprocessor(self) -> InferencePreprocessor:
        """Devuelve un preprocesador sklearn pickeable que transforma RAW → features finales.
        Se usa por `pipeline.export_full_pipeline` para que el .pkl espere columnas crudas.
        """
        if self.preprocessor is None:
            self.preprocessor = InferencePreprocessor(
                quiet=self.quiet,
                de_emphasize_team=self.de_emphasize_team,
                team_deemphasis_factor=self.team_deemphasis_factor,
            )
        return self.preprocessor

    def list_expected_raw_inputs(self) -> List[str]:
        """Lista mínima de columnas RAW esperadas para inferencia.
        (El feature engineering añade derivadas y los encoders mapean categóricas).
        """
        # Núcleo mínimo
        cols = [
            'driver', 'team', 'race_name', 'year',
            'session_air_temp', 'session_track_temp', 'session_humidity', 'session_rainfall',
            'circuit_type'
        ]
        # La ingeniería avanzada puede usar históricos internos; no es necesario aportarlos aquí
        return cols
