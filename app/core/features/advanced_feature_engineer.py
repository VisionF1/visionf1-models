"""
Advanced Feature Engineering para F1 Predictions
Genera características avanzadas para mejorar el rendimiento de los modelos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.circuit_types = {
            # Circuitos urbanos/callejeros - requieren precisión
            'Monaco Grand Prix': 'street',
            'Singapore Grand Prix': 'street', 
            'Azerbaijan Grand Prix': 'street',
            'Las Vegas Grand Prix': 'street',
            
            # Circuitos de potencia - largas rectas
            'Italian Grand Prix': 'power',
            'Belgian Grand Prix': 'power',
            'British Grand Prix': 'power',
            'Canadian Grand Prix': 'power',
            
            # Circuitos técnicos - muchas curvas
            'Hungarian Grand Prix': 'technical',
            'Spanish Grand Prix': 'technical',
            'Austrian Grand Prix': 'technical',
            
            # Circuitos híbridos - balance
            'Australian Grand Prix': 'hybrid',
            'Japanese Grand Prix': 'hybrid',
            'Dutch Grand Prix': 'hybrid',
            'Brazilian Grand Prix': 'hybrid',
            'Mexican Grand Prix': 'hybrid',
            'United States Grand Prix': 'hybrid',
            'Qatar Grand Prix': 'hybrid',
            'Saudi Arabian Grand Prix': 'hybrid',
            'Bahrain Grand Prix': 'hybrid',
            'Emilia Romagna Grand Prix': 'hybrid',
            'Miami Grand Prix': 'hybrid',
            'French Grand Prix': 'hybrid',
            'Portuguese Grand Prix': 'hybrid',
            'Turkish Grand Prix': 'hybrid',
            'Russian Grand Prix': 'hybrid',
            'Abu Dhabi Grand Prix': 'hybrid'
        }
        
        self.created_features = []
    
    def create_performance_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de rendimiento relativo"""
        print("🔧 Creando features de rendimiento relativo...")
        
        df = df.copy()
        
        # Performance relativo al mejor del fin de semana
        if 'quali_best_time' in df.columns:
            df['quali_gap_to_pole'] = df['quali_best_time'] - df.groupby(['race_name', 'year'])['quali_best_time'].transform('min')
            self.created_features.append('quali_gap_to_pole')
        
        if 'fp1_best_time' in df.columns:
            df['fp1_gap_to_fastest'] = df['fp1_best_time'] - df.groupby(['race_name', 'year'])['fp1_best_time'].transform('min')
            self.created_features.append('fp1_gap_to_fastest')
        
        # Posición relativa en equipo y ranking del equipo
        if 'quali_position' in df.columns and 'team' in df.columns:
            # Mejor posición de clasificación por equipo en cada carrera
            team_best_quali = df.groupby(['team', 'race_name', 'year'])['quali_position'].min().reset_index()
            team_best_quali.rename(columns={'quali_position': 'team_best_quali'}, inplace=True)
            
            # Ranking de equipos basado en su mejor clasificación
            team_best_quali['team_quali_rank'] = team_best_quali.groupby(['race_name', 'year'])['team_best_quali'].rank()
            
            # Merge back con el dataframe original
            df = df.merge(team_best_quali[['team', 'race_name', 'year', 'team_quali_rank']], 
                         on=['team', 'race_name', 'year'], how='left')
            
            # También crear la posición promedio del equipo para información adicional
            df['team_avg_quali'] = df.groupby(['team', 'race_name', 'year'])['quali_position'].transform('mean')
            self.created_features.extend(['team_quali_rank', 'team_avg_quali'])
        
        # Performance del equipo vs el campo
        if 'grid_position' in df.columns and 'team' in df.columns:
            df['team_vs_field_grid'] = df['grid_position'] - df.groupby(['race_name', 'year'])['grid_position'].transform('mean')
            self.created_features.append('team_vs_field_grid')
        
        print(f"   ✅ {len([f for f in self.created_features if f in df.columns])} features de rendimiento relativo creadas")
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de momentum y tendencia con valores por defecto para historial insuficiente"""
        print("🔧 Creando features de momentum...")
        
        df = df.copy()
        # Ordenar por driver y fecha para rolling calculations
        df = df.sort_values(['driver', 'year', 'race_name'])
        
        # Rendimiento en últimas carreras
        if 'race_position' in df.columns:
            df['avg_position_last_3'] = df.groupby('driver')['race_position'].transform(
                lambda x: x.rolling(3, min_periods=1).mean().shift(1)
            )
            df['avg_position_last_5'] = df.groupby('driver')['race_position'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
            
            # Si no tiene historial, usar posición media del grid (10.5 para 20 pilotos)
            df['avg_position_last_3'] = df['avg_position_last_3'].fillna(10.5)
            df['avg_position_last_5'] = df['avg_position_last_5'].fillna(10.5)
            
            self.created_features.extend(['avg_position_last_3', 'avg_position_last_5'])
        
        if 'quali_position' in df.columns:
            df['avg_quali_last_3'] = df.groupby('driver')['quali_position'].transform(
                lambda x: x.rolling(3, min_periods=1).mean().shift(1)
            )
            # Si no tiene historial, usar posición media de clasificación
            df['avg_quali_last_3'] = df['avg_quali_last_3'].fillna(10.5)
            self.created_features.append('avg_quali_last_3')
        
        # Puntuaciones acumuladas
        if 'points' in df.columns:
            df['points_last_3'] = df.groupby('driver')['points'].transform(
                lambda x: x.rolling(3, min_periods=1).sum().shift(1)
            )
            df['points_last_5'] = df.groupby('driver')['points'].transform(
                lambda x: x.rolling(5, min_periods=1).sum().shift(1)
            )
            
            # Si no tiene historial de puntos, usar 0
            df['points_last_3'] = df['points_last_3'].fillna(0)
            df['points_last_5'] = df['points_last_5'].fillna(0)
            
            self.created_features.extend(['points_last_3', 'points_last_5'])
        
        # Tendencia de rendimiento
        if 'race_position' in df.columns:
            def calculate_trend(series):
                if len(series) < 2:
                    return 0
                # Calcular pendiente simple
                x = np.arange(len(series))
                return np.polyfit(x, series, 1)[0]
            
            df['position_trend_last_5'] = df.groupby('driver')['race_position'].transform(
                lambda x: x.rolling(5, min_periods=2).apply(calculate_trend).shift(1)
            )
            
            # Si no tiene suficiente historial para tendencia, usar 0 (neutral)
            df['position_trend_last_5'] = df['position_trend_last_5'].fillna(0)
            
            self.created_features.append('position_trend_last_5')
        
        print(f"   ✅ {len([f for f in ['avg_position_last_3', 'avg_position_last_5', 'avg_quali_last_3', 'points_last_3', 'points_last_5', 'position_trend_last_5'] if f in df.columns])} features de momentum creadas")
        return df
    
    def create_weather_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features meteorológicos avanzados"""
        print("🔧 Creando features meteorológicos avanzados...")
        
        df = df.copy()
        
        # Índice de calor y confort
        if 'session_air_temp' in df.columns and 'session_humidity' in df.columns:
            df['heat_index'] = df['session_air_temp'] + (df['session_humidity'] / 100) * 5
            self.created_features.append('heat_index')
        
        if 'session_track_temp' in df.columns and 'session_air_temp' in df.columns:
            df['temp_differential'] = df['session_track_temp'] - df['session_air_temp']
            self.created_features.append('temp_differential')
        
        # Índice de dificultad climática
        weather_components = []
        if 'session_rainfall' in df.columns:
            weather_components.append(df['session_rainfall'] * 3)  # Lluvia es el factor más importante
        if 'session_humidity' in df.columns:
            weather_components.append((df['session_humidity'] - 50).abs() / 25)  # Humedad extrema
        if 'session_air_temp' in df.columns:
            weather_components.append((df['session_air_temp'] - 25).abs() / 15)  # Temperatura extrema
        
        if weather_components:
            df['weather_difficulty_index'] = sum(weather_components)
            self.created_features.append('weather_difficulty_index')
        
        # Condiciones vs temperatura ideal
        if 'session_air_temp' in df.columns:
            df['temp_deviation_from_ideal'] = (df['session_air_temp'] - 22).abs()  # 22°C considerado ideal
            self.created_features.append('temp_deviation_from_ideal')
        
        print(f"   ✅ {len([f for f in ['heat_index', 'temp_differential', 'weather_difficulty_index', 'temp_deviation_from_ideal'] if f in df.columns])} features meteorológicos creadas")
        return df
    
    def create_circuit_compatibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de compatibilidad piloto-circuito"""
        print("🔧 Creando features de compatibilidad circuito...")
        
        df = df.copy()
        
        # Clasificación de circuitos
        df['circuit_type'] = df['race_name'].map(self.circuit_types).fillna('hybrid')
        self.created_features.append('circuit_type')
        
        # Rendimiento histórico por circuito
        if 'race_position' in df.columns:
            # Histórico del piloto en este circuito (excluyendo carrera actual)
            df = df.sort_values(['driver', 'race_name', 'year'])
            
            # Método más seguro para calcular medias históricas
            df['driver_track_avg_position'] = df.groupby(['driver', 'race_name'])['race_position'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            
            # Histórico del equipo en este circuito
            df['team_track_avg_position'] = df.groupby(['team', 'race_name'])['race_position'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            
            # Rellenar NaN con promedio general
            df['driver_track_avg_position'] = df['driver_track_avg_position'].fillna(
                df.groupby('driver')['race_position'].transform('mean')
            )
            df['team_track_avg_position'] = df['team_track_avg_position'].fillna(
                df.groupby('team')['race_position'].transform('mean')
            )
            
            self.created_features.extend(['driver_track_avg_position', 'team_track_avg_position'])
        
        # Especialización por tipo de circuito
        if 'race_position' in df.columns:
            for circuit_type in ['street', 'power', 'technical', 'hybrid']:
                mask = df['circuit_type'] == circuit_type
                if mask.sum() > 0:
                    col_name = f'driver_{circuit_type}_avg'
                    # Calcular promedio por tipo de circuito para cada piloto
                    type_avg = df[mask].groupby('driver')['race_position'].transform('mean')
                    df[col_name] = 0.0  # Inicializar columna
                    df.loc[mask, col_name] = type_avg[mask]
                    
                    # Rellenar con promedio general del piloto para otros tipos
                    df[col_name] = df[col_name].replace(0, np.nan)
                    df[col_name] = df[col_name].fillna(df.groupby('driver')['race_position'].transform('mean'))
                    self.created_features.append(col_name)
        
        print(f"   ✅ Features de compatibilidad circuito creadas")
        return df
    
    def create_sector_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de sectores y velocidades"""
        print("🔧 Creando features de sectores y velocidades...")
        
        df = df.copy()
        
        # Dominancia por sector (gap al mejor)
        sector_cols = ['fp1_sector1', 'fp1_sector2', 'fp1_sector3']
        for i, sector_col in enumerate(sector_cols, 1):
            if sector_col in df.columns:
                gap_col = f'sector{i}_gap_to_best'
                df[gap_col] = df[sector_col] - df.groupby(['race_name', 'year'])[sector_col].transform('min')
                self.created_features.append(gap_col)
        
        # Velocidades relativas
        speed_cols = ['fp1_max_speed_i1', 'fp1_max_speed_i2', 'fp1_max_speed_fl', 'fp1_max_speed_st']
        for speed_col in speed_cols:
            if speed_col in df.columns:
                advantage_col = f'{speed_col}_advantage'
                df[advantage_col] = df[speed_col] - df.groupby(['race_name', 'year'])[speed_col].transform('mean')
                self.created_features.append(advantage_col)
        
        # Perfil de velocidad del piloto (fortaleza en sectores)
        if all(col in df.columns for col in sector_cols):
            # Calcular en qué sector es mejor cada piloto (solo donde no hay NaN)
            sector_data = df[sector_cols].dropna()
            if len(sector_data) > 0:
                best_sectors = sector_data.idxmin(axis=1).str.replace('fp1_sector', '')
                worst_sectors = sector_data.idxmax(axis=1).str.replace('fp1_sector', '')
                
                # Crear columnas inicializadas con NaN
                df['best_sector'] = np.nan
                df['worst_sector'] = np.nan
                df['sector_consistency'] = np.nan
                
                # Asignar valores solo para filas válidas
                df.loc[sector_data.index, 'best_sector'] = best_sectors.astype(float)
                df.loc[sector_data.index, 'worst_sector'] = worst_sectors.astype(float)
                df.loc[sector_data.index, 'sector_consistency'] = df.loc[sector_data.index, sector_cols].std(axis=1)
                
                self.created_features.extend(['best_sector', 'worst_sector', 'sector_consistency'])
        
        print(f"   ✅ Features de sectores y velocidades creadas")
        return df
    
    def create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de consistencia"""
        print("🔧 Creando features de consistencia...")
        
        df = df.copy()
        
        # Consistencia en entrenamientos libres
        fp_time_cols = ['fp1_best_time', 'fp2_best_time', 'fp3_best_time']
        if all(col in df.columns for col in fp_time_cols):
            df['fp_time_consistency'] = df[fp_time_cols].std(axis=1)
            df['fp_time_improvement'] = df['fp1_best_time'] - df['fp3_best_time']  # Mejora de FP1 a FP3
            self.created_features.extend(['fp_time_consistency', 'fp_time_improvement'])
        
        # Mejora durante el fin de semana
        if 'fp1_best_time' in df.columns and 'quali_best_time' in df.columns:
            df['fp1_to_quali_improvement'] = df['fp1_best_time'] - df['quali_best_time']
            self.created_features.append('fp1_to_quali_improvement')
        
        if 'quali_best_time' in df.columns and 'race_best_lap_time' in df.columns:
            df['quali_to_race_improvement'] = df['quali_best_time'] - df['race_best_lap_time']
            self.created_features.append('quali_to_race_improvement')
        
        # Consistencia histórica del piloto
        if 'race_position' in df.columns:
            df['driver_position_std'] = df.groupby('driver')['race_position'].transform('std')
            self.created_features.append('driver_position_std')
        
        print(f"   ✅ Features de consistencia creadas")
        return df
    
    def create_strategic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features estratégicos"""
        print("🔧 Creando features estratégicos...")
        
        df = df.copy()
        
        # Features de grid position vs performance
        if 'grid_position' in df.columns and 'race_position' in df.columns:
            df['grid_to_race_change'] = df['grid_position'] - df['race_position']  # Positivo = ganó posiciones
            df['overtaking_ability'] = df.groupby('driver')['grid_to_race_change'].transform('mean')
            self.created_features.extend(['grid_to_race_change', 'overtaking_ability'])
        
        # Análisis de quali vs carrera
        if 'quali_position' in df.columns and 'race_position' in df.columns:
            df['quali_vs_race_delta'] = df['quali_position'] - df['race_position']
            self.created_features.append('quali_vs_race_delta')
        
        # Features de puntuación vs posición
        if 'race_position' in df.columns and 'points' in df.columns:
            df['points_efficiency'] = df['points'] / (21 - df['race_position'].clip(1, 20))  # Eficiencia de puntos
            self.created_features.append('points_efficiency')
        
        print(f"   ✅ Features estratégicos creadas")
        return df
    
    def create_all_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea todas las features avanzadas"""
        print("\n🚀 CREANDO FEATURES AVANZADAS")
        print("=" * 50)
        
        original_cols = len(df.columns)
        self.created_features = []
        
        # Aplicar todas las transformaciones
        df = self.create_performance_relative_features(df)
        df = self.create_momentum_features(df)
        df = self.create_weather_advanced_features(df)
        df = self.create_circuit_compatibility_features(df)
        df = self.create_sector_speed_features(df)
        df = self.create_consistency_features(df)
        df = self.create_strategic_features(df)
        
        # Limpiar features con demasiados NaN
        print(f"\n🧹 Limpiando features con demasiados valores faltantes...")
        threshold = 0.7  # Eliminar si más del 70% son NaN
        features_to_drop = []
        
        for feature in self.created_features:
            if feature in df.columns:
                missing_pct = df[feature].isnull().sum() / len(df)
                if missing_pct > threshold:
                    features_to_drop.append(feature)
                    
        if features_to_drop:
            print(f"   🗑️  Eliminando {len(features_to_drop)} features con >{threshold*100}% faltantes:")
            for feature in features_to_drop:
                print(f"      - {feature}")
                self.created_features.remove(feature)
            df = df.drop(columns=features_to_drop)
        
        # Rellenar NaN restantes con estrategias inteligentes
        print(f"\n🔧 Rellenando valores faltantes...")
        for feature in self.created_features:
            if feature in df.columns and df[feature].isnull().sum() > 0:
                if 'avg_position' in feature or 'position' in feature:
                    # Para posiciones, usar mediana del piloto
                    df[feature] = df[feature].fillna(df.groupby('driver')[feature].transform('median'))
                    df[feature] = df[feature].fillna(df[feature].median())
                elif 'points' in feature:
                    # Para puntos, usar 0
                    df[feature] = df[feature].fillna(0)
                elif 'gap' in feature or 'advantage' in feature:
                    # Para gaps, usar 0 (sin ventaja/desventaja)
                    df[feature] = df[feature].fillna(0)
                else:
                    # Para otros, usar mediana general
                    df[feature] = df[feature].fillna(df[feature].median())
        
        new_cols = len(df.columns)
        created_count = len([f for f in self.created_features if f in df.columns])
        
        print(f"\n✅ FEATURE ENGINEERING COMPLETADO")
        print(f"   📊 Features originales: {original_cols}")
        print(f"   🆕 Features creadas: {created_count}")
        print(f"   📈 Total features: {new_cols}")
        print(f"   🎯 Incremento: {((new_cols - original_cols) / original_cols * 100):.1f}%")
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Devuelve features agrupadas por categoría para análisis"""
        groups = {
            'performance_relative': [f for f in self.created_features if any(x in f for x in ['gap_to', 'vs_field', 'team_quali_rank'])],
            'momentum': [f for f in self.created_features if any(x in f for x in ['last_3', 'last_5', 'trend'])],
            'weather': [f for f in self.created_features if any(x in f for x in ['heat_index', 'temp_', 'weather_'])],
            'circuit_compatibility': [f for f in self.created_features if any(x in f for x in ['track_avg', 'circuit_type', '_avg'])],
            'sector_speed': [f for f in self.created_features if any(x in f for x in ['sector', 'speed_', 'advantage'])],
            'consistency': [f for f in self.created_features if any(x in f for x in ['consistency', 'improvement', '_std'])],
            'strategic': [f for f in self.created_features if any(x in f for x in ['change', 'ability', 'delta', 'efficiency'])]
        }
        return groups
