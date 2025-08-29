"""
Advanced Feature Engineering para F1 Predictions
Genera características avanzadas para mejorar el rendimiento de los modelos
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import os

class AdvancedFeatureEngineer:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.created_features: List[str] = []
        self.pca_weather = None
        # Tipos de circuitos
        self.circuit_types = {
            'Monaco Grand Prix': 'street', 'Singapore Grand Prix': 'street', 'Azerbaijan Grand Prix': 'street', 'Las Vegas Grand Prix': 'street',
            'Italian Grand Prix': 'power', 'Belgian Grand Prix': 'power', 'British Grand Prix': 'power', 'Canadian Grand Prix': 'power',
            'Hungarian Grand Prix': 'technical', 'Spanish Grand Prix': 'technical', 'Austrian Grand Prix': 'technical',
            'Australian Grand Prix': 'hybrid', 'Japanese Grand Prix': 'hybrid', 'Dutch Grand Prix': 'hybrid', 'Brazilian Grand Prix': 'hybrid',
            'Mexican Grand Prix': 'hybrid', 'United States Grand Prix': 'hybrid', 'Qatar Grand Prix': 'hybrid',
            'Saudi Arabian Grand Prix': 'hybrid', 'Bahrain Grand Prix': 'hybrid', 'Emilia Romagna Grand Prix': 'hybrid',
            'Miami Grand Prix': 'hybrid', 'French Grand Prix': 'hybrid', 'Portuguese Grand Prix': 'hybrid', 'Turkish Grand Prix': 'hybrid',
            'Russian Grand Prix': 'hybrid', 'Abu Dhabi Grand Prix': 'hybrid'
        }

    def _log(self, msg: str):
        if self.quiet and os.getenv('VISIONF1_DEBUG', '0') != '1':
            return
        print(msg)

    def create_performance_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'fp1_best_time' in df.columns:
            df['fp1_gap_to_fastest'] = df['fp1_best_time'] - df.groupby(['race_name', 'year'])['fp1_best_time'].transform('min')
            self.created_features.append('fp1_gap_to_fastest')
        if 'quali_position' in df.columns and 'team' in df.columns:
            team_best_quali = df.groupby(['team', 'race_name', 'year'])['quali_position'].min().reset_index()
            team_best_quali.rename(columns={'quali_position': 'team_best_quali'}, inplace=True)
            team_best_quali['team_quali_rank'] = team_best_quali.groupby(['race_name', 'year'])['team_best_quali'].rank()
            df = df.merge(team_best_quali[['team', 'race_name', 'year', 'team_quali_rank']], on=['team', 'race_name', 'year'], how='left')
            df['team_avg_quali'] = df.groupby(['team', 'race_name', 'year'])['quali_position'].transform('mean')
            self.created_features.extend(['team_quali_rank', 'team_avg_quali'])
        if 'grid_position' in df.columns and 'team' in df.columns:
            df['team_vs_field_grid'] = df['grid_position'] - df.groupby(['race_name', 'year'])['grid_position'].transform('mean')
            self.created_features.append('team_vs_field_grid')
        self._log(f"   ✅ {len([f for f in self.created_features if f in df.columns])} features de rendimiento relativo creadas")
        return df

    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(['driver', 'year', 'race_name'])
        if 'race_position' in df.columns:
            df['avg_position_last_3'] = df.groupby('driver')['race_position'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
            df['avg_position_last_3'] = df['avg_position_last_3'].fillna(10.5)
            self.created_features.extend(['avg_position_last_3'])
        if 'quali_position' in df.columns:
            df['avg_quali_last_3'] = df.groupby('driver')['quali_position'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
            df['avg_quali_last_3'] = df['avg_quali_last_3'].fillna(10.5)
            self.created_features.append('avg_quali_last_3')
        if 'points' in df.columns:
            df['points_last_3'] = df.groupby('driver')['points'].transform(lambda x: x.rolling(3, min_periods=1).sum().shift(1))
            df['points_last_3'] = df['points_last_3'].fillna(0)
            self.created_features.extend(['points_last_3'])
        self._log(f"   ✅ {len([f for f in ['avg_position_last_3', 'avg_quali_last_3', 'points_last_3'] if f in df.columns])} features de momentum creadas")
        return df

    def create_weather_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'session_air_temp' in df.columns and 'session_humidity' in df.columns:
            df['heat_index'] = df['session_air_temp'] + (df['session_humidity'] / 100) * 5
            self.created_features.append('heat_index')
        if 'session_track_temp' in df.columns and 'session_air_temp' in df.columns:
            df['temp_differential'] = df['session_track_temp'] - df['session_air_temp']
            self.created_features.append('temp_differential')
        weather_data = pd.DataFrame()
        if 'session_air_temp' in df.columns:
            weather_data['air_temperature'] = df['session_air_temp']
        if 'session_track_temp' in df.columns:
            weather_data['track_temperature'] = df['session_track_temp']
        if 'session_humidity' in df.columns:
            weather_data['humidity'] = df['session_humidity']
        if 'session_rainfall' in df.columns:
            weather_data['precipitation'] = df['session_rainfall']
        if 'session_wind_speed' in df.columns:
            weather_data['wind_speed'] = df['session_wind_speed']
        elif 'wind_speed' in df.columns:
            weather_data['wind_speed'] = df['wind_speed']
        else:
            weather_data['wind_speed'] = 10.0
        if len(weather_data.columns) >= 3:
            weather_data_clean = weather_data.fillna(weather_data.mean())
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            weather_normalized = scaler.fit_transform(weather_data_clean)
            if self.pca_weather is None:
                self.pca_weather = PCA(n_components=1)
                weather_pca = self.pca_weather.fit_transform(weather_normalized)
            else:
                weather_pca = self.pca_weather.transform(weather_normalized)
            df['weather_difficulty_index'] = weather_pca.flatten()
            self.created_features.append('weather_difficulty_index')
            self._log(f"   🌤️ PCA weather con {len(weather_data.columns)} features: {list(weather_data.columns)}")
        else:
            weather_components = []
            if 'session_rainfall' in df.columns:
                weather_components.append(df['session_rainfall'] * 3)
            if 'session_humidity' in df.columns:
                weather_components.append((df['session_humidity'] - 50).abs() / 25)
            if 'session_air_temp' in df.columns:
                weather_components.append((df['session_air_temp'] - 25).abs() / 15)
            if weather_components:
                df['weather_difficulty_index'] = sum(weather_components)
                self.created_features.append('weather_difficulty_index')
        if 'session_air_temp' in df.columns:
            df['temp_deviation_from_ideal'] = (df['session_air_temp'] - 22).abs()
            self.created_features.append('temp_deviation_from_ideal')
        self._log(f"   ✅ {len([f for f in ['heat_index', 'temp_differential', 'weather_difficulty_index', 'temp_deviation_from_ideal'] if f in df.columns])} features meteorológicos creadas")
        return df

    def create_circuit_compatibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['circuit_type'] = df['race_name'].map(self.circuit_types).fillna('hybrid')
        self.created_features.append('circuit_type')
        if 'race_position' in df.columns:
            df = df.sort_values(['driver', 'race_name', 'year'])
            df['driver_track_avg_position'] = df.groupby(['driver', 'race_name'])['race_position'].transform(lambda x: x.expanding().mean().shift(1))
            df['team_track_avg_position'] = df.groupby(['team', 'race_name'])['race_position'].transform(lambda x: x.expanding().mean().shift(1))
            df['driver_track_avg_position'] = df['driver_track_avg_position'].fillna(df.groupby('driver')['race_position'].transform('mean'))
            df['team_track_avg_position'] = df['team_track_avg_position'].fillna(df.groupby('team')['race_position'].transform('mean'))
            self.created_features.extend(['driver_track_avg_position', 'team_track_avg_position'])
        self._log("   ✅ Features de compatibilidad circuito creadas")
        return df

    def create_sector_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("🔧 Creando features de sectores mejoradas (FP1-FP3)...")
        df = df.copy()
        for sector_num in [1, 2, 3]:
            sector_cols = []
            for session in ['fp1', 'fp2', 'fp3']:
                col_name = f'{session}_sector{sector_num}'
                if col_name in df.columns:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    sector_cols.append(col_name)
            if len(sector_cols) >= 2:
                df[f'sector{sector_num}_avg_practice'] = df[sector_cols].mean(axis=1)
                df[f'sector{sector_num}_std_practice'] = df[sector_cols].std(axis=1)
                if f'fp1_sector{sector_num}' in sector_cols and f'fp3_sector{sector_num}' in sector_cols:
                    df[f'sector{sector_num}_trend'] = df[f'fp3_sector{sector_num}'] - df[f'fp1_sector{sector_num}']
                    self.created_features.append(f'sector{sector_num}_trend')
                self.created_features.extend([f'sector{sector_num}_avg_practice', f'sector{sector_num}_std_practice'])
                gap_col = f'sector{sector_num}_gap_to_best'
                df[gap_col] = df[f'sector{sector_num}_avg_practice'] - df.groupby(['race_name', 'year'])[f'sector{sector_num}_avg_practice'].transform('min')
                self.created_features.append(gap_col)
        speed_cols = ['fp1_max_speed_i1', 'fp1_max_speed_i2', 'fp1_max_speed_fl', 'fp1_max_speed_st']
        for speed_col in speed_cols:
            if speed_col in df.columns:
                advantage_col = f'{speed_col}_advantage'
                df[advantage_col] = df[speed_col] - df.groupby(['race_name', 'year'])[speed_col].transform('mean')
                self.created_features.append(advantage_col)
        sector_avg_cols = [f'sector{i}_avg_practice' for i in [1, 2, 3] if f'sector{i}_avg_practice' in df.columns]
        if len(sector_avg_cols) >= 2:
            df['sector_consistency'] = df[sector_avg_cols].std(axis=1)
            self.created_features.append('sector_consistency')
        self._log(f"   ✅ Features de sectores robustas creadas: {len([f for f in self.created_features if 'sector' in f])} features")
        return df

    def create_teammate_comparison_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("🔧 Creando features de comparación con compañero...")
        df = df.copy()
        if 'quali_position' in df.columns and 'team' in df.columns:
            team_quali_mean = df.groupby(['race_name', 'year', 'team'])['quali_position'].transform('mean')
            team_count = df.groupby(['race_name', 'year', 'team'])['quali_position'].transform('count')
            df['delta_teammate_quali'] = np.where(team_count == 2, df['quali_position'] - team_quali_mean, np.nan)
            self.created_features.append('delta_teammate_quali')
        if 'race_position' in df.columns and 'team' in df.columns:
            team_race_mean = df.groupby(['race_name', 'year', 'team'])['race_position'].transform('mean')
            team_count = df.groupby(['race_name', 'year', 'team'])['race_position'].transform('count')
            df['delta_teammate_race'] = np.where(team_count == 2, df['race_position'] - team_race_mean, np.nan)
            self.created_features.append('delta_teammate_race')
        if 'delta_teammate_quali' in df.columns:
            df['avg_delta_teammate_quali'] = df.groupby('driver')['delta_teammate_quali'].transform(lambda x: x.expanding().mean().shift(1))
            df['avg_delta_teammate_quali'] = df['avg_delta_teammate_quali'].fillna(0)
            self.created_features.append('avg_delta_teammate_quali')
        self._log("   ✅ Features de comparación teammate creadas")
        return df

    def create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("🔧 Creando features de consistencia...")
        df = df.copy()
        fp_time_cols = ['fp1_best_time', 'fp2_best_time', 'fp3_best_time']
        if all(col in df.columns for col in fp_time_cols):
            df['fp_time_consistency'] = df[fp_time_cols].std(axis=1)
            df['fp_time_improvement'] = df['fp1_best_time'] - df['fp3_best_time']
            self.created_features.extend(['fp_time_consistency', 'fp_time_improvement'])
        if 'race_position' in df.columns:
            df['driver_position_std'] = df.groupby('driver')['race_position'].transform('std')
            self.created_features.append('driver_position_std')
        self._log("   ✅ Features de consistencia creadas")
        return df

    def create_strategic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("🔧 Creando features estratégicos...")
        df = df.copy()
        if 'grid_position' in df.columns and 'race_position' in df.columns:
            df['grid_to_race_change'] = df['grid_position'] - df['race_position']
            df = df.sort_values(['driver', 'year', 'race_name'])
            df['overtaking_ability'] = df.groupby('driver')['grid_to_race_change'].transform(lambda x: x.expanding().mean().shift(1))
            self.created_features.extend(['grid_to_race_change', 'overtaking_ability'])
        if 'quali_position' in df.columns and 'race_position' in df.columns:
            df['quali_vs_race_delta'] = df['quali_position'] - df['race_position']
            self.created_features.append('quali_vs_race_delta')
        if 'race_position' in df.columns and 'points' in df.columns:
            df['points_efficiency'] = df['points'] / (21 - df['race_position'].clip(1, 20))
            self.created_features.append('points_efficiency')
        self._log("   ✅ Features estratégicos creadas")
        return df

    def create_pre_race_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'fp3_best_time' in df.columns:
            df['fp3_rank'] = df.groupby(['race_name', 'year'])['fp3_best_time'].rank(method='average')
            self.created_features.append('fp3_rank')
        if 'fp3_rank' in df.columns:
            if 'avg_quali_last_3' in df.columns:
                df['expected_grid_position'] = 0.7 * df['fp3_rank'] + 0.3 * df['avg_quali_last_3']
            else:
                df['expected_grid_position'] = df['fp3_rank']
            self.created_features.append('expected_grid_position')
        return df

    def create_all_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        original_cols = len(df.columns)
        self.created_features = []
        self._log("🚀 INICIANDO FEATURE ENGINEERING MEJORADO")
        self._log("=" * 60)
        df = self.create_performance_relative_features(df)
        df = self.create_momentum_features(df)
        df = self.create_weather_advanced_features(df)
        df = self.create_circuit_compatibility_features(df)
        df = self.create_sector_speed_features(df)
        df = self.create_teammate_comparison_features(df)
        df = self.create_consistency_features(df)
        df = self.create_strategic_features(df)
        df = self.create_pre_race_proxies(df)
        self._log("\n🧹 Limpiando features con demasiados valores faltantes...")
        threshold = 0.5
        features_to_drop: List[str] = []
        for feature in list(self.created_features):
            if feature in df.columns:
                missing_pct = df[feature].isnull().mean()
                if missing_pct > threshold:
                    features_to_drop.append(feature)
                    self._log(f"   ❌ {feature}: {missing_pct:.1%} faltantes - marcado para eliminación")
        if features_to_drop:
            self._log(f"   🗑️  Eliminando {len(features_to_drop)} features con >{int(threshold*100)}% faltantes:")
            for feature in features_to_drop:
                self._log(f"      - {feature}")
                if feature in self.created_features:
                    self.created_features.remove(feature)
            df = df.drop(columns=[f for f in features_to_drop if f in df.columns])
        self._log("\n🔧 Rellenando valores faltantes...")
        for feature in self.created_features:
            if feature in df.columns and df[feature].isnull().any():
                if 'avg_position' in feature or 'position' in feature:
                    df[feature] = df[feature].fillna(df.groupby('driver')[feature].transform('median'))
                    df[feature] = df[feature].fillna(df[feature].median())
                elif 'points' in feature:
                    df[feature] = df[feature].fillna(0)
                elif any(k in feature for k in ['gap', 'advantage', 'delta']):
                    df[feature] = df[feature].fillna(0)
                else:
                    df[feature] = df[feature].fillna(df[feature].median())
        new_cols = len(df.columns)
        created_count = len([f for f in self.created_features if f in df.columns])
        self._log("\n✅ FEATURE ENGINEERING MEJORADO COMPLETADO")
        self._log("=" * 60)
        self._log(f"   📊 Features originales: {original_cols}")
        self._log(f"   🆕 Features creadas: {created_count}")
        self._log(f"   📈 Total features: {new_cols}")
        inc = ((new_cols - original_cols) / original_cols * 100) if original_cols else 0
        self._log(f"   🎯 Incremento: {inc:.1f}%")
        self._log("\n🚀 MEJORAS IMPLEMENTADAS:")
        self._log("   🔧 Sectores robustos: FP1-FP3 combinados")
        self._log("   🌤️ Weather PCA: Componente principal climático")
        self._log("   👥 Driver vs Teammate: Comparación intra-equipo")
        return df

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        groups = {
            'performance_relative': [f for f in self.created_features if any(x in f for x in ['gap_to', 'vs_field', 'team_quali_rank'])],
            'momentum': [f for f in self.created_features if any(x in f for x in ['last_3', 'last_5', 'trend'])],
            'weather': [f for f in self.created_features if any(x in f for x in ['heat_index', 'temp_', 'weather_'])],
            'circuit_compatibility': [f for f in self.created_features if any(x in f for x in ['track_avg', 'circuit_type', '_avg'])],
            'sector_speed': [f for f in self.created_features if any(x in f for x in ['sector', 'speed_', 'advantage'])],
            'teammate_comparison': [f for f in self.created_features if any(x in f for x in ['delta_teammate', 'vs_teammate'])],
            'consistency': [f for f in self.created_features if any(x in f for x in ['consistency', 'improvement', '_std'])],
            'strategic': [f for f in self.created_features if any(x in f for x in ['change', 'ability', 'delta', 'efficiency'])]
        }
        return groups
