"""
Enhanced Data Preparer con Feature Engineering Avanzado (pre-race safe)
Integra el feature engineering avanzado en el pipeline de preparación de datos
"""

import os
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.config import VALID_TEAMS, DATA_IMPORTANCE
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

warnings.filterwarnings('ignore')


def debug_print_df(df: pd.DataFrame, cols: Optional[List[str]] = None, title: str = "🔍 Debug DataFrame"):
    if os.getenv('VISIONF1_DEBUG', '0') != '1':
        return
    prev = (
        pd.get_option('display.max_columns'),
        pd.get_option('display.width'),
        pd.get_option('display.max_colwidth'),
        pd.get_option('display.max_rows'),
    )
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    print("\n" + title)
    print(df.head())
    num = (df[cols] if cols else df).select_dtypes(include=[np.number])
    stats = num.agg(['mean', 'min', 'max']).T if len(df) <= 1 else num.describe().T
    print(stats)
    pd.set_option('display.max_columns', prev[0])
    pd.set_option('display.width', prev[1])
    pd.set_option('display.max_colwidth', prev[2])
    pd.set_option('display.max_rows', prev[3])


class EnhancedDataPreparer:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.feature_engineer = AdvancedFeatureEngineer(quiet=self.quiet)
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.use_advanced_features = True
        self.year_weights = {
            2022: DATA_IMPORTANCE.get("2022_weight", 0.10),
            2023: DATA_IMPORTANCE.get("2023_weight", 0.15),
            2024: DATA_IMPORTANCE.get("2024_weight", 0.30),
            2025: DATA_IMPORTANCE.get("2025_weight", 0.50),
        }
        self.sample_years: Optional[pd.Series] = None
        self.train_years: Optional[pd.Series] = None
        self.test_years: Optional[pd.Series] = None
        self.train_indices = None
        self.test_indices = None

    def _log(self, msg: str):
        if self.quiet and os.getenv('VISIONF1_DEBUG', '0') != '1':
            return
        print(msg)

    def prepare_enhanced_features(self, df: pd.DataFrame):
        original_shape = df.shape
        self._log(f"📊 Datos originales: {original_shape}")

        # 1) Normalizar equipos
        self._log("\nAplicando mapeo histórico de equipos...")
        df = self._apply_team_mapping(df)

        # 2) Filtrar equipos válidos
        before_filter = len(df)
        df = df[df['team'].isin(VALID_TEAMS)]
        after_filter = len(df)
        self._log(f"   🗑️  Eliminados {before_filter - after_filter} registros sin equipo válido")
        self._log(f"   ✅ Registros restantes: {after_filter}")

        # 3) Feature engineering avanzado (pre-race safe ya en engineer)
        self._log("\n🔧 Aplicando feature engineering avanzado (pre-race safe)...")
        df = self.feature_engineer.create_all_advanced_features(df)

        # 4) Definir conjuntos de features
        base_features = ['team_encoded', 'driver_encoded', 'total_laps', 'fp3_best_time']
        weather_features = ['session_air_temp', 'session_track_temp', 'session_humidity', 'session_rainfall']
        historical_features = ['team_avg_position_2024', 'team_avg_position_2023', 'team_avg_position_2022']
        race_weekend_features = ['expected_grid_position', 'fp3_rank', 'overtaking_ability', 'fp_time_consistency', 'fp_time_improvement']

        all_features: List[str] = base_features + weather_features + historical_features
        all_features.extend([f for f in race_weekend_features if f in df.columns])

        if self.use_advanced_features:
            created = [f for f in self.feature_engineer.created_features if f in df.columns]
            advanced_safe = [
                f for f in created if any(x in f for x in [
                    'fp1_gap_to_fastest', 'avg_position_last_3', 'avg_quali_last_3', 'points_last_3',
                    'heat_index', 'temp_deviation_from_ideal', 'weather_difficulty_index', 'team_track_avg_position',
                    'sector', 'driver_rain_advantage', 'driver_skill_factor', 'team_strength_factor', 'driver_team_synergy',
                    'driver_competitiveness', 'team_competitiveness', 'driver_weather_skill', 'overtaking_ability',
                    'fp_time_consistency', 'fp_time_improvement', 'driver_track_avg_position', 'expected_grid_position', 'fp3_rank'
                ])
            ]
            all_features.extend(advanced_safe)

        # Encoding de team
        if 'team' in df.columns:
            df['team_encoded'] = self.label_encoder.fit_transform(df['team'])
            self._log(f"   ✅ team_encoded: {df['team'].nunique()} equipos únicos")

        # Encoding de driver
        if 'driver' in df.columns:
            from pathlib import Path
            import pickle
            enc_path = Path("app/models_cache/label_encoder.pkl")
            if enc_path.exists():
                try:
                    with open(enc_path, 'rb') as f:
                        driver_encoder = pickle.load(f)
                    if len(getattr(driver_encoder, 'classes_', [])) > 15:
                        self._log("   🔧 Usando encoder de pilotos guardado")
                        df['driver_encoded'] = -1
                        known = [d for d in df['driver'].unique() if d in driver_encoder.classes_]
                        for d in known:
                            df.loc[df['driver'] == d, 'driver_encoded'] = driver_encoder.transform([d])[0]
                        df = df[df['driver_encoded'] != -1].copy()
                        self.label_encoder = driver_encoder
                    else:
                        raise ValueError("Encoder guardado no parece de pilotos")
                except Exception as e:
                    self._log(f"   ⚠️ Encoder guardado inválido: {e}. Creando nuevo...")
                    df['driver_encoded'] = LabelEncoder().fit_transform(df['driver'])
                    self.label_encoder = LabelEncoder().fit(df['driver'])
            else:
                df['driver_encoded'] = LabelEncoder().fit_transform(df['driver'])
                self.label_encoder = LabelEncoder().fit(df['driver'])
            self._log(f"   ✅ driver_encoded: {df['driver'].nunique()} pilotos únicos")

        # 5) Performance histórico y derivados driver/team
        df = self._calculate_team_historical_performance(df)
        df = self._create_driver_team_derived_features(df)

        derived_features = [
            'driver_competitiveness', 'team_competitiveness', 'driver_skill_factor', 'team_strength_factor',
            'driver_team_synergy', 'driver_weather_skill', 'driver_rain_advantage'
        ]
        all_features.extend([f for f in derived_features if f in df.columns])

        # 6) Selección final de features
        available = [f for f in all_features if f in df.columns]
        self._log(f"   📊 Base: {len([f for f in base_features if f in df.columns])} | Meteo: {len([f for f in weather_features if f in df.columns])} | Hist: {len([f for f in historical_features if f in df.columns])}")
        if self.use_advanced_features:
            adv_count = len([f for f in available if f not in base_features + weather_features + historical_features])
            self._log(f"   🚀 Avanzadas seguras: {adv_count}")

        # 7) Construir X
        dynamic = []
        if 'team_encoded' in df.columns:
            dynamic.append('team_encoded')
        if 'driver_encoded' in df.columns:
            dynamic.append('driver_encoded')
        final_features: List[str] = []
        seen = set()
        for c in (available + dynamic):
            if c not in seen:
                final_features.append(c)
                seen.add(c)
        X = df[final_features].copy()
        X = self._ensure_31_features(X)

        # Años para export
        if 'year' in df.columns:
            try:
                self.sample_years = pd.to_numeric(df.loc[X.index, 'year'], errors='coerce')
            except Exception:
                self.sample_years = df.loc[X.index, 'year']

        # 8) Target y limpieza
        target_columns = ['final_position', 'race_position', 'position']
        y = None
        target_col = None
        for col in target_columns:
            if col in df.columns:
                y = df[col].copy()
                target_col = col
                break
        if y is None:
            self._log(f"   ❌ No se encontró variable objetivo en: {target_columns}")
            return None, None, None, None
        self._log(f"   ✅ Target: {target_col}")

        X, y = self._clean_missing_data(X, y)
        self.feature_names = list(X.columns)
        self._log(f"\n✅ FEATURES MEJORADAS COMPLETADAS | X{X.shape} y{len(y)}")
        debug_print_df(X, title="🔍 X (pre-return) - enhanced_data_preparer")
        return X, y, self.label_encoder, self.feature_names

    def _apply_team_mapping(self, df: pd.DataFrame):
        mapping = {
            'AlphaTauri': 'Racing Bulls', 'Alpha Tauri': 'Racing Bulls', 'RB': 'Racing Bulls',
            'Alfa Romeo': 'Kick Sauber', 'Sauber': 'Kick Sauber',
            'Aston Martin Aramco Cognizant F1 Team': 'Aston Martin',
            'Mercedes-AMG Petronas F1 Team': 'Mercedes',
            'Oracle Red Bull Racing': 'Red Bull Racing',
            'Red Bull Racing Honda RBPT': 'Red Bull Racing',
            'Scuderia Ferrari': 'Ferrari', 'McLaren F1 Team': 'McLaren',
            'BWT Alpine F1 Team': 'Alpine', 'Williams Racing': 'Williams',
            'MoneyGram Haas F1 Team': 'Haas F1 Team', 'Haas': 'Haas F1 Team',
            'Kick Sauber': 'Kick Sauber'
        }
        if 'team' in df.columns:
            df['team'] = df['team'].replace(mapping)
            self._log(f"   ✅ Equipos tras mapeo: {df['team'].nunique()} únicos")
        return df

    def _calculate_team_historical_performance(self, df: pd.DataFrame):
        if 'team' in df.columns and 'race_position' in df.columns and 'year' in df.columns:
            self._log("   🔢 Calculando performance histórico de equipos...")
            for year in [2024, 2023, 2022]:
                col = f'team_avg_position_{year}'
                year_data = df[df['year'] == year]
                if len(year_data) > 0:
                    team_perf = year_data.groupby('team')['race_position'].mean()
                    df[col] = df['team'].map(team_perf).fillna(df['race_position'].mean())
                else:
                    df[col] = df['race_position'].mean()
        return df

    def _create_driver_team_derived_features(self, df: pd.DataFrame):
        self._log("   🚀 Creando features derivadas driver/team (data-driven)...")
        hist_df = df.copy()

        if 'year' in hist_df.columns:
            hist_df['year'] = pd.to_numeric(hist_df['year'], errors='coerce')
            hist_df = hist_df[~hist_df['year'].isna()].copy()
            hist_df['year'] = hist_df['year'].astype(int)

        needed_cols = {'driver', 'team', 'year', 'race_position'}
        if not needed_cols.issubset(hist_df.columns):
            df['driver_competitiveness'] = df.get('driver_competitiveness', pd.Series(0.75, index=df.index))
            df['team_competitiveness'] = df.get('team_competitiveness', pd.Series(0.65, index=df.index))
            df['driver_weather_skill'] = df.get('driver_weather_skill', pd.Series(0.75, index=df.index))
            df['driver_skill_factor'] = df['driver_competitiveness']
            df['team_strength_factor'] = df['team_competitiveness']
            df['driver_team_synergy'] = df['driver_competitiveness'] * df['team_competitiveness']
            if 'session_rainfall' in df.columns:
                try:
                    rain = df['session_rainfall'].astype(float)
                except Exception:
                    rain = (df['session_rainfall'] == True).astype(float)
                df['driver_rain_advantage'] = df['driver_weather_skill'] * rain
            return df

        unique_years = sorted(hist_df['year'].dropna().unique())

        def entity_year_means(hdf: pd.DataFrame, entity_col: str, target: str) -> pd.DataFrame:
            g = hdf.groupby([entity_col, 'year'])[target].mean().reset_index()
            g.rename(columns={target: 'mean_position'}, inplace=True)
            return g

        def year_weight(y: int) -> float:
            return float(self.year_weights.get(int(y), 0.05))

        def build_prior_weighted_table(hdf: pd.DataFrame, entity_col: str, target: str, wet_only: bool = False) -> pd.DataFrame:
            h = hdf.copy()
            if wet_only and 'session_rainfall' in h.columns:
                try:
                    h = h[(h['session_rainfall'].astype(float) > 0)]
                except Exception:
                    h = h[(h['session_rainfall'] == True)]
            means = entity_year_means(h, entity_col, target)
            rows = []
            ent_values = means[entity_col].unique()
            all_years = sorted(means['year'].unique())
            for cy in unique_years:
                prev_years = [y for y in all_years if y < cy]
                if not prev_years:
                    continue
                prev = means[means['year'].isin(prev_years)]
                prev = prev.assign(year_weight=prev['year'].map(year_weight))
                for ent in ent_values:
                    ent_prev = prev[prev[entity_col] == ent]
                    if len(ent_prev) == 0:
                        continue
                    w = ent_prev['year_weight'].sum()
                    if w == 0:
                        continue
                    weighted_mean = (ent_prev['mean_position'] * ent_prev['year_weight']).sum() / w
                    rows.append({entity_col: ent, 'current_year': cy, 'weighted_hist_pos': weighted_mean})
            return pd.DataFrame(rows)

        driver_prior = build_prior_weighted_table(hist_df, 'driver', 'race_position', wet_only=False)
        team_prior = build_prior_weighted_table(hist_df, 'team', 'race_position', wet_only=False)
        driver_wet_prior = build_prior_weighted_table(hist_df, 'driver', 'race_position', wet_only=True)

        for tbl in (driver_prior, team_prior, driver_wet_prior):
            if not tbl.empty and 'current_year' in tbl.columns:
                try:
                    tbl['current_year'] = pd.to_numeric(tbl['current_year'], errors='coerce').astype('Int64')
                except Exception:
                    pass

        def inverse_minmax(series: pd.Series) -> pd.Series:
            s = series.astype(float)
            if s.empty:
                return s
            s_min, s_max = s.min(), s.max()
            if s_max == s_min:
                return pd.Series(0.5, index=series.index)
            return (s_max - s) / (s_max - s_min)

        df = df.copy()
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['current_year'] = df['year'].astype('Int64')

        if not driver_prior.empty:
            driver_prior = driver_prior.copy()
            driver_prior['driver_competitiveness'] = inverse_minmax(driver_prior['weighted_hist_pos'])
            df = df.merge(driver_prior[['driver', 'current_year', 'driver_competitiveness']], how='left', on=['driver', 'current_year'])
        if not team_prior.empty:
            team_prior = team_prior.copy()
            team_prior['team_competitiveness'] = inverse_minmax(team_prior['weighted_hist_pos'])
            df = df.merge(team_prior[['team', 'current_year', 'team_competitiveness']], how='left', on=['team', 'current_year'])
        if not driver_wet_prior.empty:
            driver_wet_prior = driver_wet_prior.copy()
            driver_wet_prior['driver_weather_skill'] = inverse_minmax(driver_wet_prior['weighted_hist_pos'])
            df = df.merge(driver_wet_prior[['driver', 'current_year', 'driver_weather_skill']], how='left', on=['driver', 'current_year'])

        if 'driver_competitiveness' not in df.columns:
            df['driver_competitiveness'] = 0.75
        if 'team_competitiveness' not in df.columns:
            df['team_competitiveness'] = 0.65
        df['driver_competitiveness'] = df['driver_competitiveness'].fillna(0.5)
        df['team_competitiveness'] = df['team_competitiveness'].fillna(0.5)
        df['driver_weather_skill'] = df.get('driver_weather_skill', pd.Series(0.5, index=df.index)).fillna(0.5)

        df['driver_skill_factor'] = df['driver_competitiveness']
        df['team_strength_factor'] = df['team_competitiveness']
        df['driver_team_synergy'] = df['driver_competitiveness'] * df['team_competitiveness']

        if 'session_rainfall' in df.columns:
            try:
                rain = df['session_rainfall'].astype(float)
            except Exception:
                rain = (df['session_rainfall'] == True).astype(float)
            df['driver_rain_advantage'] = df['driver_weather_skill'] * rain

        if 'grid_to_race_change' in df.columns and 'driver_competitiveness' in df.columns:
            df['adjusted_position_change'] = df['grid_to_race_change'] * df['driver_competitiveness']

        df.drop(columns=['current_year'], inplace=True, errors='ignore')

        if os.getenv('VISIONF1_DEBUG', '0') == '1':
            try:
                grp = df.groupby(['driver', 'team', 'year'])[['driver_skill_factor', 'team_strength_factor']].nunique()
                offenders = grp[(grp['driver_skill_factor'] > 1) | (grp['team_strength_factor'] > 1)].reset_index()
                self._log(f"   🔎 DEBUG factores por temporada: grupos con variación dentro del mismo año = {len(offenders)}")
                if len(offenders) > 0:
                    self._log(offenders.head(12).to_string(index=False))
            except Exception:
                pass

        self._log("   ✅ Features derivadas driver/team creadas (data-driven)")
        return df

    def _process_weather_features(self, df: pd.DataFrame):
        return df

    def _clean_missing_data(self, X: pd.DataFrame, y: pd.Series):
        self._log("\n🧹 Limpiando datos...")
        missing_info = X.isnull().sum()
        cols_missing = missing_info[missing_info > 0]
        if len(cols_missing) > 0:
            for col, cnt in cols_missing.items():
                if 'position' in col:
                    X[col] = X[col].fillna(X[col].median())
                elif any(k in col for k in ['temp', 'humidity']):
                    X[col] = X[col].fillna(X[col].mean())
                elif 'rainfall' in col:
                    X[col] = X[col].fillna(0)
                elif any(k in col for k in ['time', 'pace']):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].median())
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median(numeric_only=True))
        valid = ~y.isnull()
        X, y = X[valid], y[valid]
        self._log(f"   ✅ Datos limpios: X{X.shape}")
        return X, y

    def _ensure_31_features(self, X: pd.DataFrame) -> pd.DataFrame:
        target_features = [
            'driver_encoded', 'team_encoded', 'driver_skill_factor', 'team_strength_factor',
            'driver_team_synergy', 'driver_competitiveness', 'team_competitiveness',
            'session_air_temp', 'session_track_temp', 'session_humidity', 'session_rainfall',
            'driver_weather_skill', 'driver_rain_advantage', 'fp1_gap_to_fastest', 'points_last_3',
            'heat_index', 'weather_difficulty_index', 'team_track_avg_position', 'sector_consistency',
            'overtaking_ability', 'total_laps', 'fp3_best_time', 'expected_grid_position', 'fp3_rank'
        ]
        result = pd.DataFrame(index=X.index)
        for f in target_features:
            if f in X.columns:
                result[f] = X[f]
            else:
                if any(k in f for k in ['skill', 'competitiveness', 'synergy']):
                    val = 0.75
                elif 'strength' in f:
                    val = 0.5
                elif 'position' in f:
                    val = 10.0
                elif 'temp' in f:
                    val = 25.0
                elif 'humidity' in f:
                    val = 50.0
                elif 'rainfall' in f:
                    val = 0.0
                elif any(k in f for k in ['pace', 'time']):
                    val = 90.0
                elif 'gap' in f:
                    val = 1.0
                elif 'rank' in f:
                    val = 10.0
                elif 'points' in f:
                    val = 5.0
                elif 'index' in f or 'consistency' in f or 'improvement' in f:
                    val = 0.5
                elif 'change' in f or 'ability' in f:
                    val = 0.0
                elif 'laps' in f:
                    val = 58.0
                elif 'tyre' in f or 'life' in f:
                    val = 25.0
                else:
                    val = 0.0
                result[f] = val
        self._log(f"   ✅ Features finales aseguradas: {len(result.columns)}")
        return result

    def prepare_training_data(self, df: pd.DataFrame):
        result = self.prepare_enhanced_features(df)
        if not (isinstance(result, tuple) and len(result) == 4):
            self._log("❌ Error: formato inesperado de prepare_enhanced_features")
            return None, None, None, None, None
        X, y, label_encoder, feature_names = result
        if X is None or y is None:
            self._log("❌ Error en preparación de features")
            return None, None, None, None, None
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, shuffle=True)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        try:
            years_series = getattr(self, 'sample_years', None)
            if years_series is not None:
                self.train_years = years_series.iloc[train_idx].reset_index(drop=True)
                self.test_years = years_series.iloc[test_idx].reset_index(drop=True)
        except Exception:
            self.train_years = None
            self.test_years = None
        self.train_indices = train_idx
        self.test_indices = test_idx
        self._log(f"✅ Split: train {X_train.shape}, test {X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_names
